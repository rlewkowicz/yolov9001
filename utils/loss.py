import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_iou import bbox_iou_aligned
from torch.nn.utils.rnn import pad_sequence
from .boxes import BoundingBox
from .geometry import dfl_expectation
from utils.logging import get_logger
from .assigner import TaskAlignedAssigner
from core.config import get_config

def _dfl_bounds(reg_max: int, eps: float) -> tuple[float, float]:
    tiny = 1e-6
    upper = float(reg_max - 1) - max(float(eps), tiny)
    return 0.0, upper

class DetectionLoss(nn.Module):
    def __init__(self, model, imgsz=640):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model

        config = get_config(hyp=getattr(model, 'hyp', None))
        self.hyp = config.to_dict()

        self.nc = model.nc
        self.reg_max = model.reg_max
        assert int(self.reg_max) >= 2, "reg_max must be >= 2 for DFL to be well-defined"
        self.strides = model.strides

        if self.strides is None:
            raise RuntimeError("Model strides not set. Call attach_runtime(model) first.")

        self.nl = len(self.strides)
        self.imgsz = imgsz

        loss_weights = config.loss_weights
        self.box_weight = loss_weights['box']
        self.cls_weight = loss_weights['cls']
        self.dfl_weight = loss_weights['dfl']
        self.obj_weight = float(config.get('obj_weight', 1.0))

        self.iou_type = config.get('iou_type', 'GIoU')
        self.l1_weight = config.get('l1_weight', 0.0)
        self.l1_beta = float(config.get('l1_beta', 3.0))
        self.l1_iou_gate = float(config.get('l1_iou_gate', 0.5))

        self.dfl_eps = config.get('dfl_eps', 1e-3)
        self.dfl_label_smooth = config.get('dfl_label_smooth', 0.0)
        self.dfl_strict_targets = config.get('dfl_strict_targets', False)
        self.dfl_clip_tolerance = float(config.get('dfl_clip_tolerance', 0.01))

        self.vfl_alpha = config.get('vfl_alpha', 0.75)
        self.vfl_gamma = config.get('vfl_gamma', 2.0)

        self.assign_topk = config.get('assign_topk', 10)
        self.assign_radius = config.get('assign_radius', 2.5)
        self.assign_alpha = config.get('assign_alpha', 0.5)
        self.assign_beta = config.get('assign_beta', 6.0)
        self.decode_centered = config.get('decode_centered', True)

        self.qfl_beta = config.get('qfl_beta', 2.0)  # QFL beta parameter

        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        if not hasattr(model, "dfl_decoder"):
            raise AttributeError("Model must have a dfl_decoder attribute for DetectionLoss.")
        self.dfl_decoder = model.dfl_decoder
        self.dfl_decoder.reg_max = self.reg_max
        self.dfl_decoder.strides = self.strides
        self.dfl_decoder.centered = self.decode_centered

        from utils.geometry import _assert_ltrb_order_is_consistent
        if str(self.device).startswith('cuda') and torch.cuda.is_available():
            try:
                _assert_ltrb_order_is_consistent()
            except AssertionError as e:
                raise RuntimeError(f"Detect head LTRB order mismatch: {e}")

        self.assign_mode = config.get('assign_mode', 'ult')
        self.assigner = TaskAlignedAssigner(
            mode=self.assign_mode,
            num_classes=self.nc,
            topk=self.assign_topk,
            alpha=self.assign_alpha,
            beta=self.assign_beta,
            lambda_iou=float(config.get('assign_lambda_iou', 3.0)),
            topq=int(config.get('assign_topq', 10)),
            center_radius=float(self.assign_radius),
            input_is_logits=True,
            min_stride_alpha=float(config.get('min_stride_box_alpha', 0.0)),
            filter_tiny_gt=bool(config.get('filter_tiny_gt', True)),
            objless_target_iou=float(config.get('objless_target_iou', 0.0)),
            objless_weight_floor=float(config.get('objless_weight_floor', 0.05)),
        )
        self.dino_proto_enabled = False
        self.dino_proto_temp: float = 0.3
        self.dino_proto_weight: float = 0.05
        self.dino_class_prototypes: torch.Tensor | None = None  # [C,Ct]
        dino_cfg = config.get('dino', {}) if isinstance(config.get('dino', {}), dict) else {}
        self.salreg_enabled = bool(dino_cfg.get('reg_weight_with_saliency', False))
        self.salreg_floor = float(dino_cfg.get('reg_weight_floor', 0.2))
        self.salreg_power = float(dino_cfg.get('reg_weight_power', 1.0))
        self.centroid_enabled = bool(dino_cfg.get('centroid_align', False))
        self.centroid_weight = float(dino_cfg.get('centroid_weight', 0.05))
        try:
            self.centroid_max_epochs = int(dino_cfg.get('centroid_max_epochs', 0))
        except Exception:
            self.centroid_max_epochs = 0
        try:
            self.objfor_decay_epochs = int(dino_cfg.get('objfor02_decay_epochs', 0))
        except Exception:
            self.objfor_decay_epochs = 0
        # Objfor02 logging knobs
        try:
            self.objfor_log_every = int(dino_cfg.get('objfor02_log_every', 0))
        except Exception:
            self.objfor_log_every = 0
        try:
            self.objfor_lowprior_thresh = float(dino_cfg.get('objfor02_lowprior_thresh', 0.2))
        except Exception:
            self.objfor_lowprior_thresh = 0.2
        self._objfor_log_counter = 0

    def pack_targets(self, gt_labels_list, gt_bboxes_list):
        """
        Vectorized packing of variable-length GT lists into padded tensors without Python loops.
        Returns:
          lbl: [B, Mmax, 1] long, -1 padded
          box: [B, Mmax, 4] float32, 0 padded
          msk: [B, Mmax, 1] bool, True where valid
        """
        device = self.device
        lbl_pad = pad_sequence(gt_labels_list, batch_first=True, padding_value=-1).to(device=device)
        box_pad = pad_sequence(gt_bboxes_list, batch_first=True,
                               padding_value=0.0).to(device=device)
        msk = (lbl_pad >= 0).unsqueeze(-1)
        lbl = lbl_pad.unsqueeze(-1).long()
        box = box_pad.to(torch.float32)
        return lbl, box, msk

    def forward(self, preds, targets):
        if len(preds) != self.nl:
            raise ValueError(
                f"Loss function expects {self.nl} prediction tensors, but got {len(preds)}. "
                "This is likely due to a mismatch between the model's architecture and the "
                "strides configured in the loss function. Ensure they are consistent."
            )

        bs = preds[0].shape[0]
        pred_concat = torch.cat([xi.reshape(bs, self.reg_max * 4 + self.nc, -1) for xi in preds], 2)
        pred_concat = torch.nan_to_num(pred_concat, nan=0.0, posinf=30.0, neginf=-30.0)
        pred_distri, pred_scores = pred_concat.split((self.reg_max * 4, self.nc), 1)

        assert pred_distri.shape[1] == self.reg_max * 4, \
            f"Expected {self.reg_max*4} reg channels, got {pred_distri.shape[1]}"

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchors, anchor_strides = self.dfl_decoder.get_anchors(preds)
        N = anchors.shape[0]
        assert pred_scores.shape[1] == N and pred_distri.shape[1] == N, \
            f"Anchor count {N} != predictions N {pred_scores.shape[1]}; check level order/shapes."

        gt_labels, gt_bboxes_px = self.preprocess(targets, bs)

        pred_dist_grid = dfl_expectation(
            pred_distri.view(bs, -1, 4, self.reg_max),
            self.reg_max,
            tau=float(self.hyp.get("dfl_tau", 1.0))
        )
        pred_bboxes_grid = BoundingBox.dist2bbox(pred_dist_grid, anchors.unsqueeze(0), xywh=False)

        stride_b_flat = anchor_strides.view(1, -1, 1)
        pred_bboxes_px = pred_bboxes_grid.detach() * stride_b_flat
        anchors_px = anchors * anchor_strides

        gt_labels_b, gt_bboxes_b, gt_mask_b = self.pack_targets(gt_labels, gt_bboxes_px)

        pd_obj = None
        try:
            if bool(self.hyp.get('dino', {}).get('objfor02', False)):
                det_for_bias = getattr(self.model, 'detect_layer', None)
                if det_for_bias is not None and getattr(
                    det_for_bias, 'last_dino_obj_flat', None
                ) is not None:
                    prior = det_for_bias.last_dino_obj_flat  # [B,N] in [0,1]
                    prior = prior.clamp(1e-4, 1.0 - 1e-4)
                    logits = torch.logit(prior).to(dtype=pred_scores.dtype)
                    # Optional per-step decay of prior influence via logit scaling
                    # First prefer anchored decay started by trainer on patience
                    if hasattr(self.model, '_objfor_decay_len') and float(getattr(self.model, '_objfor_decay_len', 0)) > 0:
                        try:
                            e = float(getattr(self.model, 'epoch', 0) or 0)
                            nb = float(getattr(self.model, '_epoch_nb', 0) or 0)
                            bi = float(getattr(self.model, '_epoch_batch_idx', 0) or 0)
                            if nb > 0:
                                e = e + (bi / nb)
                            e0 = float(getattr(self.model, '_objfor_decay_start_e', 0) or 0)
                            dd = float(getattr(self.model, '_objfor_decay_len', 0) or 0)
                            scale = max(0.0, 1.0 - ((e - e0) / max(dd, 1e-12)))
                            logits = logits * float(scale)
                        except Exception:
                            pass
                    elif self.objfor_decay_epochs and self.objfor_decay_epochs > 0:
                        try:
                            e = float(getattr(self.model, 'epoch', 0) or 0)
                            nb = float(getattr(self.model, '_epoch_nb', 0) or 0)
                            bi = float(getattr(self.model, '_epoch_batch_idx', 0) or 0)
                            if nb > 0:
                                e = e + (bi / nb)
                            scale = max(0.0, 1.0 - (e / float(self.objfor_decay_epochs)))
                            logits = logits * float(scale)
                        except Exception:
                            pass
                    pd_obj = logits.unsqueeze(-1)
        except Exception:
            pd_obj = None

        assign = self.assigner(
            pred_scores.detach(),  # [B, N, C], logits OK (use_sigmoid=True handles sigmoid inside)
            pred_bboxes_px,  # [B, N, 4] in pixels
            anchors_px,  # [N, 2] in pixels
            gt_labels_b,  # [B, M, 1]
            gt_bboxes_b,  # [B, M, 4]
            gt_mask_b,  # [B, M, 1] bool
            pd_objectness=pd_obj,  # DINO prior logits for cost bias if available
            anc_strides=anchor_strides  # [N, 1], used by SimOTA/mixed center prior
        )

        # Optional: lightweight logging of objfor02 effect on SimOTA positives
        if self.objfor_log_every and bool(self.hyp.get('dino', {}).get('objfor02', False)):
            try:
                self._objfor_log_counter += 1
            except Exception:
                self._objfor_log_counter = 1
            if (self._objfor_log_counter % max(1, int(self.objfor_log_every))) == 0:
                try:
                    assign_noprior = self.assigner(
                        pred_scores.detach(),
                        pred_bboxes_px,
                        anchors_px,
                        gt_labels_b,
                        gt_bboxes_b,
                        gt_mask_b,
                        pd_objectness=None,
                        anc_strides=anchor_strides
                    )
                    fg_with = assign["reg"]["fg_mask"].to(torch.int32)
                    fg_no = assign_noprior["reg"]["fg_mask"].to(torch.int32)
                    suppressed = int((fg_no & (1 - fg_with)).sum().item())
                    helped = int((fg_with & (1 - fg_no)).sum().item())
                    total_with = int(fg_with.sum().item())
                    total_no = int(fg_no.sum().item())
                    lowp = None
                    try:
                        det = getattr(self.model, 'detect_layer', None)
                        if det is not None and getattr(det, 'last_dino_obj_flat', None) is not None:
                            prior = det.last_dino_obj_flat  # [B,N]
                            thr = float(self.objfor_lowprior_thresh)
                            lowp = int((assign["reg"]["fg_mask"] & (prior < thr)).sum().item())
                    except Exception:
                        lowp = None
                    log = get_logger()
                    log.basic('simota/pos_with_prior', float(total_with))
                    log.basic('simota/pos_no_prior', float(total_no))
                    log.basic('simota/suppressed', float(suppressed))
                    log.basic('simota/helped', float(helped))
                    if lowp is not None:
                        log.basic('simota/lowprior_pos', float(lowp))
                except Exception as e:
                    try:
                        get_logger().debug('simota/objfor02_log_error', str(e))
                    except Exception:
                        pass

        target_bboxes_px = assign["reg"]["target_bboxes"]  # [B, N, 4] pixels
        fg_mask = assign["reg"]["fg_mask"]  # [B, N] bool
        target_bboxes_grid = target_bboxes_px / stride_b_flat
        target_scores_cls = assign["cls"]["target_scores"]  # [B, N, C] floats in [0,1]
        target_scores_cls = target_scores_cls.clamp_(0.0, 1.0).nan_to_num_(0.0)

        if fg_mask.any():
            with torch.no_grad():
                target_ltrb_unclamped = BoundingBox.bbox2dist(
                    anchors, target_bboxes_grid, self.reg_max
                )
                lo, hi = _dfl_bounds(self.reg_max, self.dfl_eps)
                mask_valid_dfl = (target_ltrb_unclamped.min(dim=-1)[0] >= lo) & \
                                 (target_ltrb_unclamped.max(dim=-1)[0] <= hi)
                if (~mask_valid_dfl).any():
                    get_logger().debug(
                        "loss/dfl_oob_fraction", {
                            "oob_frac": float((~mask_valid_dfl & fg_mask).float().mean().item()),
                            "reg_max": int(self.reg_max)
                        }
                    )

        with torch.no_grad():
            try:
                fg_mask_cls = assign["cls"]["fg_mask"]  # [B,N]
                matched_gt = assign["cls"]["matched_gt"]  # [B,N]
                B, N, C = pred_scores.shape
                M = gt_bboxes_b.shape[1]
                b_inds = torch.arange(B, device=self.device)[:, None].expand(B, N)
                m_inds = matched_gt.clamp_min(0).clamp_max(max(M - 1, 0))
                gt_matched = gt_bboxes_b[b_inds, m_inds]  # [B,N,4]
                iou_aligned = bbox_iou_aligned(pred_bboxes_px, gt_matched,
                                               iou_type="IoU").clamp_(0.0, 1.0)
                y_iou = iou_aligned * fg_mask_cls.to(iou_aligned.dtype)  # [B,N]
                gt_lab_full = gt_labels_b[b_inds, m_inds, 0].clamp_min(0).clamp_max(self.nc - 1)
                q = torch.zeros_like(target_scores_cls)  # [B,N,C]
                q.scatter_(2, gt_lab_full.unsqueeze(-1), y_iou.unsqueeze(-1))
                q_anchor = q.max(dim=2).values  # [B,N]
                Npos = (q_anchor > 0).sum().clamp_min(1)
            except Exception:
                q = target_scores_cls
                q_anchor = q.max(dim=2).values
                Npos = (q_anchor > 0).sum().clamp_min(1)

        sal_w = None
        if self.salreg_enabled:
            try:
                det = getattr(self.model, 'detect_layer', None)
                if det is not None and getattr(det, 'last_dino_obj_flat', None) is not None:
                    prior = det.last_dino_obj_flat.clamp(0.0, 1.0)  # [B,N]
                    floor = 0.0 if self.salreg_floor < 0.0 else (
                        1.0 if self.salreg_floor > 1.0 else self.salreg_floor
                    )
                    if self.salreg_power != 1.0:
                        prior = prior.pow(self.salreg_power)
                    sal_w = floor + (1.0 - floor) * prior
            except Exception:
                sal_w = None

        loss_box = self.box_loss(pred_bboxes_px, target_bboxes_px, q_anchor, fg_mask, Npos, sal_w)
        # Use classification mask for proper normalization in BCE path
        loss_cls = self.cls_loss(pred_scores, q, fg_mask_cls)
        loss_dfl = self.dfl_loss(
            pred_distri, target_bboxes_grid, q_anchor, fg_mask, anchors, Npos, sal_w
        )

        loss_centroid = torch.zeros((), device=self.device)
        if self.centroid_enabled and self.centroid_weight > 0:
            epoch = getattr(self.model, 'epoch', None)
            if (self.centroid_max_epochs <= 0) or (
                isinstance(epoch, (int, float)) and int(epoch) < int(self.centroid_max_epochs)
            ):
                try:
                    det = getattr(self.model, 'detect_layer', None)
                    sal_levels = getattr(
                        det, 'last_dino_obj_levels', None
                    ) if det is not None else None
                except Exception:
                    sal_levels = None
                if sal_levels is not None and isinstance(sal_levels,
                                                         (list,
                                                          tuple)) and len(sal_levels) == self.nl:
                    level_sizes = [xi.shape[-2] * xi.shape[-1] for xi in preds]
                    offsets = [0]
                    for n in level_sizes[:-1]:
                        offsets.append(offsets[-1] + n)
                    img_diag2 = float(self.imgsz) * float(self.imgsz) * 2.0
                    total = torch.zeros((), device=self.device)
                    count = torch.zeros((), device=self.device)
                    for li in range(self.nl):
                        Nl = level_sizes[li]
                        if Nl <= 0:
                            continue
                        off = offsets[li]
                        HiWi = sal_levels[li].shape[-2:]  # (Hi,Wi)
                        Hi, Wi = int(HiWi[0]), int(HiWi[1])
                        stride_l = float(self.strides[li])
                        S = sal_levels[li].to(device=self.device, dtype=torch.float32)  # [B,Hi,Wi]
                        B = S.shape[0]
                        Xc = (
                            torch.arange(Wi, device=self.device,
                                         dtype=torch.float32).view(1, 1, Wi) + 0.5
                        ) * stride_l
                        Yc = (
                            torch.arange(Hi, device=self.device,
                                         dtype=torch.float32).view(1, Hi, 1) + 0.5
                        ) * stride_l
                        SX = S * Xc
                        SY = S * Yc
                        pad = (1, 0, 1, 0)  # (w_left, w_right, h_top, h_bottom)
                        I0 = torch.cumsum(
                            torch.cumsum(torch.nn.functional.pad(S, pad, value=0.0), dim=1), dim=2
                        )
                        IX = torch.cumsum(
                            torch.cumsum(torch.nn.functional.pad(SX, pad, value=0.0), dim=1), dim=2
                        )
                        IY = torch.cumsum(
                            torch.cumsum(torch.nn.functional.pad(SY, pad, value=0.0), dim=1), dim=2
                        )
                        for b in range(B):
                            mask_b = fg_mask[b, off:off + Nl]
                            if not mask_b.any():
                                continue
                            tb = target_bboxes_px[b, off:off + Nl][mask_b]  # [P,4]
                            pb = pred_bboxes_px[b, off:off + Nl][mask_b]
                            cx = 0.5 * (pb[:, 0] + pb[:, 2])
                            cy = 0.5 * (pb[:, 1] + pb[:, 3])
                            x0 = (tb[:, 0] / stride_l).floor().clamp(0, Wi - 1).long()
                            y0 = (tb[:, 1] / stride_l).floor().clamp(0, Hi - 1).long()
                            x1 = (tb[:, 2] / stride_l).ceil().clamp(1, Wi).long()
                            y1 = (tb[:, 3] / stride_l).ceil().clamp(1, Hi).long()
                            x1 = torch.maximum(x1, x0 + 1)
                            y1 = torch.maximum(y1, y0 + 1)
                            I0b = I0[b]
                            IXb = IX[b]
                            IYb = IY[b]
                            Wip = Wi + 1
                            idx = lambda A, yy, xx: A.view(-1)[(yy * Wip + xx).long()]
                            sumS = idx(I0b, y1, x1) - idx(I0b, y0, x1) - idx(I0b, y1,
                                                                             x0) + idx(I0b, y0, x0)
                            sumX = idx(IXb, y1, x1) - idx(IXb, y0, x1) - idx(IXb, y1,
                                                                             x0) + idx(IXb, y0, x0)
                            sumY = idx(IYb, y1, x1) - idx(IYb, y0, x1) - idx(IYb, y1,
                                                                             x0) + idx(IYb, y0, x0)
                            valid = sumS > 1e-6
                            if valid.any():
                                cx_sal = torch.where(
                                    valid, sumX / (sumS + 1e-6), cx
                                )  # fallback to pred cx if no mass
                                cy_sal = torch.where(valid, sumY / (sumS + 1e-6), cy)
                                d2 = (cx - cx_sal).pow(2) + (cy - cy_sal).pow(2)
                                total = total + (d2 / img_diag2).sum()
                                count = count + valid.to(count.dtype).sum()
                    if count > 0:
                        loss_centroid = (total / Npos.to(total.dtype)) * float(self.centroid_weight)

        loss_obj = torch.zeros((), device=self.device)
        obj_branch = assign.get('obj', None) if isinstance(assign, dict) else None

        det = getattr(self.model, 'detect_layer', None)
        if self.assign_mode == 'mixed' and obj_branch is not None:
            tgt_obj = obj_branch.get('target_objectness', None)
            if tgt_obj is not None:
                conf_logits = None
                try:
                    if det is not None and getattr(det, 'last_obj_logits_flat', None) is not None:
                        conf_logits = det.last_obj_logits_flat  # [B,N]
                except Exception:
                    conf_logits = None
                if conf_logits is None:
                    conf_logits = pred_scores.max(dim=2).values  # [B,N]
                bce_elem = F.binary_cross_entropy_with_logits(
                    conf_logits, tgt_obj.to(conf_logits.dtype), reduction='none'
                )  # [B,N]

                target_iou = float(self.hyp.get('objless_target_iou', 0.0))
                floor_w = float(self.hyp.get('objless_weight_floor', 0.05))
                if target_iou > 0.0:
                    with torch.no_grad():
                        obj_fg_mask = obj_branch.get('fg_mask', None)
                        obj_matched = obj_branch.get('matched_gt', None)
                        if obj_fg_mask is None or obj_matched is None:
                            weight_pos = None
                        else:
                            B, N, _ = pred_bboxes_px.shape
                            gt_boxes = gt_bboxes_b  # [B,M,4] in pixels
                            M = gt_boxes.shape[1]
                            b_inds = torch.arange(B, device=self.device).view(B, 1).expand(B, N)
                            m_inds = obj_matched.clamp_min(0).clamp_max(max(M - 1, 0))
                            gt_matched = gt_boxes[b_inds, m_inds]  # [B,N,4]
                            iou_aligned = bbox_iou_aligned(pred_bboxes_px,
                                                           gt_matched).clamp_(0.0, 1.0)
                            f = 0.0 if floor_w < 0.0 else (1.0 if floor_w > 1.0 else floor_w)
                            w = f + (1.0 - f) * (iou_aligned /
                                                 max(target_iou, 1e-12)).clamp(0.0, 1.0)
                            weight_pos = w  # [B,N]
                    if 'weight_pos' in locals() and weight_pos is not None:
                        pos = (tgt_obj > 0.5).to(bce_elem.dtype)
                        bce_elem = bce_elem * (pos * weight_pos + (1.0 - pos))

                loss_obj = bce_elem.mean()

        eff_l1_w = float(self.l1_weight) * float(getattr(self, 'l1_ramp_scale', 1.0))
        if eff_l1_w > 0 and fg_mask.any():
            weight, denom = self._quality_weight_and_norm(target_scores_cls, fg_mask)
            if 'sal_w' in locals() and sal_w is not None:
                try:
                    weight = (weight * sal_w[fg_mask].to(weight.dtype))
                except Exception:
                    pass
            pred_bboxes_px_l1 = pred_bboxes_grid[fg_mask] * stride_b_flat.expand(bs, -1,
                                                                                 -1)[fg_mask]
            target_bboxes_px_l1 = target_bboxes_grid[fg_mask] * stride_b_flat.expand(bs, -1,
                                                                                     -1)[fg_mask]
            huber = F.smooth_l1_loss(
                pred_bboxes_px_l1, target_bboxes_px_l1, beta=float(self.l1_beta), reduction='none'
            ).sum(dim=-1)
            if float(self.l1_iou_gate) > 0.0:
                with torch.no_grad():
                    iou_aligned = bbox_iou_aligned(pred_bboxes_px_l1,
                                                   target_bboxes_px_l1).clamp_(0.0, 1.0)
                    gate = (iou_aligned > float(self.l1_iou_gate)).to(huber.dtype)
            else:
                gate = torch.ones_like(huber)
            loss_box += eff_l1_w * (((huber * gate) * weight).sum() / denom)

        loss_box *= self.box_weight
        loss_cls *= self.cls_weight
        loss_dfl *= self.dfl_weight
        loss_obj *= self.obj_weight

        loss_clsproto = torch.zeros((), device=self.device)
        if self.dino_proto_enabled and (self.dino_class_prototypes is not None):
            try:
                prototypes = self.dino_class_prototypes.to(device=self.device)
                fg_mask_cls = assign["cls"]["fg_mask"]  # [B,N]
                matched_gt = assign["cls"]["matched_gt"]  # [B,N]
                B, N, C = pred_scores.shape
                gt_labels_b, gt_bboxes_b, gt_mask_b = self.pack_targets(gt_labels, gt_bboxes_px)
                b_inds = torch.arange(B, device=self.device)[:, None].expand(B, N)
                m_inds = matched_gt.clamp_min(0)
                gt_lab_full = gt_labels_b[b_inds, m_inds,
                                          0].clamp_min(0).clamp_max(self.nc - 1)  # [B,N]
                if fg_mask_cls.any():
                    pos_idx = fg_mask_cls
                    labels_pos = gt_lab_full[pos_idx].long()  # [P]
                    logits_pos = pred_scores[pos_idx]  # [P,C]
                    proto_full = prototypes  # [C,Ct]
                    valid = torch.linalg.norm(proto_full, dim=1) > 1e-6
                    if valid.sum() >= 2:
                        proto = torch.nn.functional.normalize(proto_full[valid], dim=1)  # [Cv,Ct]
                        valid_idx = torch.where(valid)[0]
                        class_to_valid = torch.full((C, ), -1, device=self.device, dtype=torch.long)
                        class_to_valid[valid_idx] = torch.arange(
                            valid_idx.numel(), device=self.device
                        )
                        keep = (class_to_valid[labels_pos] >= 0)
                        if keep.any():
                            labels_kept = labels_pos[keep]
                            logits_kept = logits_pos[keep]
                            mapped = class_to_valid[labels_kept]
                            true_proto = proto[mapped]  # [P_kept,Ct]
                            sim = torch.matmul(true_proto, proto.transpose(0, 1))  # [P_kept,Cv]
                            q = torch.softmax(sim / max(1e-6, float(self.dino_proto_temp)), dim=1)
                            logp = torch.log_softmax(logits_kept[:, valid], dim=1)
                            loss_clsproto = torch.nn.functional.kl_div(
                                logp, q, reduction='batchmean'
                            ) * float(self.dino_proto_weight)
            except Exception as e:
                get_logger().debug("loss/clsproto_error", str(e))

        total_loss = loss_box + loss_cls + loss_dfl + loss_obj + loss_clsproto + loss_centroid
        loss_items = {
            'box':
                loss_box.detach().item(),
            'cls':
                loss_cls.detach().item(),
            'dfl':
                loss_dfl.detach().item(),
            'dino_centroid':
                loss_centroid.detach().item()
                if self.centroid_enabled and loss_centroid.numel() else 0.0,
            **({
                'obj': loss_obj.detach().item()
            } if self.assign_mode == 'mixed' and obj_branch is not None else {}),
            **({
                'dino_clsproto': loss_clsproto.detach().item()
            } if self.dino_proto_enabled and (self.dino_class_prototypes is not None) else {}),
        }
        return total_loss, loss_items

    def preprocess(self, targets, bs):
        if targets.numel() == 0:
            empty_lbl = torch.empty(0, dtype=torch.long, device=self.device)
            empty_box = torch.empty(0, 4, dtype=torch.float32, device=self.device)
            return [empty_lbl] * bs, [empty_box] * bs
        t = targets.to(self.device, non_blocking=True)
        b = t[:, 0].long()
        cls = t[:, 1].long()
        xywh = t[:, 2:6] * float(self.imgsz)
        x, y, w, h = xywh.unbind(1)
        xyxy = torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
        order = torch.argsort(b)
        b_s, cls_s, xyxy_s = b[order], cls[order], xyxy[order]
        counts = torch.bincount(b_s, minlength=bs).tolist()
        cls_list = torch.split(cls_s, counts)
        xyxy_list = torch.split(xyxy_s, counts)
        return list(cls_list), list(xyxy_list)

    def _quality_weight_and_norm(self, target_scores, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].float()
        w_sum = weight.sum()
        denom = torch.where(w_sum > 1e-6, w_sum, fg_mask.sum().clamp_min(1).to(weight.dtype))
        return weight, denom

    def box_loss(
        self,
        pred_bboxes,
        target_bboxes,
        q_anchor,
        fg_mask,
        Npos,
        sal_w: torch.Tensor | None = None
    ):
        if fg_mask.sum() == 0:
            return torch.zeros((), device=self.device)

        boxes_for_iou = pred_bboxes[fg_mask]
        targets_for_iou = target_bboxes[fg_mask]
        assert boxes_for_iou.numel(
        ) % 4 == 0 and targets_for_iou.shape == boxes_for_iou.shape, f"IoU arg mismatch: pred {boxes_for_iou.shape} vs tgt {targets_for_iou.shape}"

        iou = bbox_iou_aligned(
            boxes_for_iou, targets_for_iou, iou_type=self.iou_type, img_wh=(self.imgsz, self.imgsz)
        )
        loss_iou = 1.0 - iou

        weight = q_anchor[fg_mask].float()
        if sal_w is not None:
            try:
                weight = weight * sal_w[fg_mask].to(weight.dtype)
            except Exception:
                pass
        denom = Npos.to(weight.dtype)
        loss_iou = (loss_iou * weight).sum() / denom

        return loss_iou

    def cls_loss(self, pred_scores, target_scores, fg_mask):
        """
        Stable classification loss using BCE-with-logits for numerical stability.
        Implements a variant of Varifocal Loss (VFL), Quality Focal Loss (QFL), or
        standard Binary Cross Entropy (BCE) based on the `cls_type` hyperparameter.
        """
        cls_type = self.hyp.get("cls_type", "vfl")

        if cls_type == "bce":
            loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
            num_pos = fg_mask.sum().clamp_min(1)
            return loss.sum() / num_pos

        alpha = float(self.vfl_alpha)
        gamma = float(self.vfl_gamma)
        pred_prob = pred_scores.sigmoid()

        if cls_type == "qfl":
            beta = float(self.qfl_beta)
            y = target_scores
            sigma = pred_prob  # sigmoid(pred_scores)
            mod = (y - sigma).abs().pow(beta)
            bce = F.binary_cross_entropy_with_logits(pred_scores, y, reduction='none')
            loss_sum = (bce * mod).sum()
            num_pos = (y.max(dim=2).values > 0).sum().clamp_min(1)
            return loss_sum / num_pos.to(loss_sum.dtype)
        else:  # Varifocal Loss (matches vfl.pdf Eq. (2))
            q = target_scores
            p = pred_prob
            bce = F.binary_cross_entropy_with_logits(pred_scores, q, reduction='none')
            w_pos = q
            w_neg = alpha * p.pow(gamma)
            weight = torch.where(q > 0, w_pos, w_neg)
            loss_sum = (bce * weight).sum()
            num_pos = (q.max(dim=2).values > 0).sum().clamp_min(1)
            return loss_sum / num_pos.to(loss_sum.dtype)

    def dfl_loss(
        self,
        pred_dist,
        target_bboxes,
        q_anchor,
        fg_mask,
        anchors,
        Npos,
        sal_w: torch.Tensor | None = None
    ):
        num_fg = fg_mask.sum()
        if num_fg == 0:
            return torch.zeros((), device=self.device)

        target_ltrb_grid = BoundingBox.bbox2dist(
            anchors, target_bboxes, self.reg_max
        )  # [A,4], un-clamped

        pred_dist_fg = pred_dist[fg_mask]
        target_ltrb_fg = target_ltrb_grid[fg_mask]

        assert pred_dist_fg.dim() == 2, f"Expected 2D pred_dist_fg, got {pred_dist_fg.dim()}D"
        assert pred_dist_fg.shape[
            1
        ] == 4 * self.reg_max, f"Expected {4*self.reg_max} channels, got {pred_dist_fg.shape[1]}"
        assert target_ltrb_fg.shape == (
            num_fg, 4
        ), f"Expected target shape ({num_fg}, 4), got {target_ltrb_fg.shape}"

        with torch.no_grad():
            lo, hi = _dfl_bounds(self.reg_max, self.dfl_eps)
            lower_ok = target_ltrb_fg >= lo
            upper_ok = target_ltrb_fg <= hi
            inrange = (lower_ok & upper_ok).float().mean()
            oob_frac = 1.0 - inrange.item()
            if oob_frac > 0:
                get_logger().debug(
                    "loss/dfl_oob_fraction",
                    {"oob_frac": float(oob_frac), "reg_max": int(self.reg_max)}
                )
            if self.dfl_strict_targets and oob_frac > self.dfl_clip_tolerance:
                raise AssertionError(
                    f"DFL targets out-of-range fraction {oob_frac:.4f} exceeded tolerance "
                    f"{self.dfl_clip_tolerance:.4f}. Check anchor scaling or bbox2dist."
                )

        N = pred_dist_fg.shape[0]
        pred = pred_dist_fg.view(N, 4, self.reg_max).permute(1, 0,
                                                             2).contiguous().view(-1, self.reg_max)
        tgt = target_ltrb_fg.T.contiguous().view(
            -1
        )  # unclamped; build_dfl_targets will clamp internally

        assert pred.shape == (num_fg * 4, self.reg_max), \
            f"Pred shape mismatch: expected ({num_fg*4}, {self.reg_max}), got {pred.shape}"
        assert tgt.shape == (num_fg * 4,), \
            f"Target shape mismatch: expected ({num_fg*4},), got {tgt.shape}"

        weight = q_anchor[fg_mask].float()
        if sal_w is not None:
            try:
                weight = weight * sal_w[fg_mask].to(weight.dtype)
            except Exception:
                pass
        denom = Npos.to(weight.dtype)
        return self.dist_focal_loss(pred, tgt, weight, denom)

    def build_dfl_targets(self, target, smooth=None):
        if smooth is None:
            smooth = self.dfl_label_smooth
        N = target.shape[0]
        device = target.device
        tiny = 1e-6
        upper = float(self.reg_max - 1) - max(float(self.dfl_eps), tiny)
        target = target.clamp_(0.0, upper)
        li = target.floor().long()
        ri = (li + 1).clamp_(max=self.reg_max - 1)
        wl = (ri.float() - target).unsqueeze(1)
        wr = 1.0 - wl
        base = torch.zeros(N, self.reg_max, device=device, dtype=torch.float32)
        base.scatter_add_(1, li.unsqueeze(1), wl)
        base.scatter_add_(1, ri.unsqueeze(1), wr)
        if smooth and smooth > 0:
            base = (1 - smooth) * base + smooth / self.reg_max
        return base

    def dist_focal_loss(self, pred_dist, target, weight, normalizer):
        """Distribution Focal Loss for regression - optimized version using KL divergence.
        
        Args:
            pred_dist: [N*4, reg_max] distribution logits
            target: [N*4] continuous target values in [0, reg_max-1]
            weight: [N] quality score weights for each foreground anchor
            normalizer: Normalization factor (sum of weights)
        """
        pred32 = pred_dist.float()
        target_dist = self.build_dfl_targets(target, smooth=self.dfl_label_smooth)  # fp32
        log_probs = F.log_softmax(pred32, dim=1)
        loss_vec = F.kl_div(log_probs, target_dist, reduction='none').sum(dim=1)  # fp32
        w = weight.repeat(4).float()
        denom = normalizer + 1e-9
        loss = (loss_vec * w).sum() / denom
        return loss  # leave fp32; AMP handles cast
