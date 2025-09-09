"""
utils/postprocess.py

YOLO model post-processing utilities.
"""
import torch
import json
from typing import Tuple, Optional, List

from .geometry import decode_distances, DFLDecoder

class Postprocessor:
    """
    Centralized postprocessor class to manage NMS thresholds, DFL decoder, and configuration.
    Prevents configuration drift between training, validation, and inference.
    """
    def __init__(self, cfg, nc, device, model=None, decoder=None):
        """
        Initialize postprocessor with unified configuration.
        
        Args:
            cfg: YOLOConfig or dict with postprocess settings
            nc: Number of classes
            device: Torch device
            model: Optional model instance to source feat_shapes from
            decoder: Optional DFLDecoder instance
        """
        self.nc = nc
        self.device = device
        self._model = None

        if hasattr(cfg, 'postprocess_config'):
            pp = cfg.postprocess_config
        else:
            pp = cfg

        required_keys = (
            'conf_thresh', 'iou_thresh', 'pre_nms_topk', 'post_nms_topk', 'class_agnostic_nms'
        )
        for k in required_keys:
            if k not in pp:
                raise KeyError(f"postprocess missing {k} in config")

        self.conf_thresh = float(pp['conf_thresh'])
        self.iou_thresh = float(pp['iou_thresh'])
        self.pre_nms_topk = int(pp['pre_nms_topk'])
        self.post_nms_topk = int(pp['post_nms_topk'])
        self.class_agnostic_nms = bool(pp['class_agnostic_nms'])
        self.nms_free = bool(pp.get('nms_free', False))
        self.use_objectness_eval = bool(pp.get('use_objectness_eval', False))

        if decoder is None:
            raise ValueError("Postprocessor must be initialized with a DFLDecoder instance.")
        self.decoder = decoder
        self.model = model  # use property to sync if provided

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        self._model = m
        if m is None:
            return
        if hasattr(m, 'strides'):
            self.decoder.strides = torch.as_tensor(
                m.strides, dtype=torch.float32, device=self.device
            )
        if hasattr(m, 'detect_layer') and hasattr(m.detect_layer, 'reg_max'):
            self.decoder.reg_max = int(m.detect_layer.reg_max)
        elif hasattr(m, 'reg_max'):
            self.decoder.reg_max = int(m.reg_max)
        if hasattr(m, 'hyp'):
            self.decoder.tau = float(m.hyp.get('dfl_tau', 1.0))

    @torch.no_grad()
    def __call__(self, outputs, img_size, feat_shapes=None):
        """
        Process model outputs with configured NMS and decoding.
        """
        if feat_shapes is None and hasattr(self.model, "detect_layer"
                                          ) and hasattr(self.model.detect_layer, "last_shapes"):
            feat_shapes = self.model.detect_layer.last_shapes

        if feat_shapes is None:
            raise ValueError(
                "postprocess: feat_shapes must be provided. "
                "Pass the Detect layer's cached last_shapes for this forward."
            )

        model_strides_t = getattr(getattr(self.model, 'detect_layer', None), 'strides', None)
        if model_strides_t is None:
            raise ValueError(
                "Postprocessor: model.detect_layer.strides not set. Call attach_runtime(model)."
            )
        model_strides = list(
            map(
                int, (
                    model_strides_t.tolist()
                    if isinstance(model_strides_t, torch.Tensor) else model_strides_t
                )
            )
        )

        inferred = []
        for (h, w) in feat_shapes:
            s_h = float(img_size) / float(h)
            s_w = float(img_size) / float(w)
            if abs(s_h - s_w) > 1e-3:
                raise AssertionError(
                    f"Non-square stride implied by H={h}, W={w} at img_size={img_size}: s_h={s_h}, s_w={s_w}"
                )
            inferred.append(int(round(s_h)))

        from utils.logging import get_logger
        get_logger().log_text(
            "runtime/strides",
            json.dumps({
                "model_strides": model_strides, "inferred_strides": inferred, "feat_shapes":
                    feat_shapes
            },
                       indent=2)
        )

        if inferred != model_strides:
            raise AssertionError(
                f"Stride drift detected: model {model_strides} vs inferred {inferred}."
            )

        self.decoder.strides = torch.as_tensor(
            model_strides, dtype=torch.float32, device=self.device
        )
        strides = model_strides

        anchor_points, stride_tensor = self.decoder.get_anchors(feat_shapes)
        N = outputs.shape[2]
        assert anchor_points.shape[0] == N and stride_tensor.shape[0] == N, \
            f"Anchor count {anchor_points.shape[0]} != predictions N {N}; check level order/shapes."

        for i, (h, w) in enumerate(feat_shapes):
            expected = img_size / float(strides[i])
            tol = 1.1
            if not (
                expected / tol <= h <= expected * tol and expected / tol <= w <= expected * tol
            ):
                raise AssertionError(
                    f"Grid mismatch at level {i}: got {h}x{w}, expected ~{expected:.0f}x{expected:.0f}"
                )

        if self.model is not None and hasattr(self.model, 'detect_layer'
                                             ) and hasattr(self.model.detect_layer, 'reg_max'):
            reg_max = self.model.detect_layer.reg_max
            self.decoder.reg_max = reg_max
        else:
            reg_max = self.decoder.reg_max

        if self.model is not None and hasattr(self.model, 'hyp'):
            self.decoder.tau = float(self.model.hyp.get("dfl_tau", 1.0))

        obj_logits = None
        try:
            det = getattr(self.model, 'detect_layer', None)
            if det is not None and getattr(det, 'last_obj_logits_flat', None) is not None:
                obj_logits = det.last_obj_logits_flat  # [B, N]
        except Exception:
            obj_logits = None

        return postprocess_detections(
            outputs=outputs,
            img_size=img_size,
            nc=self.nc,
            device=self.device,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh,
            reg_max=reg_max,
            pre_nms_topk=self.pre_nms_topk,
            post_nms_topk=self.post_nms_topk,
            class_agnostic_nms=self.class_agnostic_nms,
            decoder=self.decoder,
            feat_shapes=feat_shapes,
            obj_logits=obj_logits,
            nms_free=self.nms_free,
            use_objectness_eval=self.use_objectness_eval,
        )

@torch.no_grad()
def postprocess_detections(
    outputs: torch.Tensor,
    img_size: int,
    nc: int,
    device: torch.device,
    conf_thresh: float = None,
    iou_thresh: float = None,
    reg_max: int = 16,
    strides: list = None,
    pre_nms_topk: int = None,
    post_nms_topk: int = None,
    class_agnostic_nms: bool = None,
    decoder: Optional[DFLDecoder] = None,
    feat_shapes: Optional[List[Tuple[int, int]]] = None,
    obj_logits: Optional[torch.Tensor] = None,
    nms_free: bool = False,
    use_objectness_eval: bool = False
) -> List:
    """
    Postprocess model outputs to get final detections.
    Handles DFL decoding and NMS with configurable thresholds.
    Everything stays on GPU.
    
    Args:
        outputs: Model outputs [bs, nc+reg_max*4, N]
        img_size: Input image size
        nc: Number of classes
        device: Torch device
        conf_thresh: Confidence threshold 
        iou_thresh: IoU threshold for NMS
        reg_max: DFL reg_max parameter
        strides: Feature strides
        pre_nms_topk: Keep top-k before NMS
        post_nms_topk: Keep top-k after NMS
        class_agnostic_nms: Apply class-agnostic NMS
    
    Returns:
        List of detections as dicts with tensors: {'boxes':, 'scores':, 'class_ids':}
    """
    if iou_thresh is None or conf_thresh is None:
        raise ValueError("Must supply iou_thresh and conf_thresh")
    if decoder is None:
        raise ValueError("Pass a configured DFLDecoder via Postprocessor")
    if strides is None:
        strides = decoder.strides

    try:
        if getattr(decoder, 'reg_max', None) != int(reg_max):
            decoder.reg_max = int(reg_max)
        if hasattr(decoder, 'to'):
            decoder.to(outputs.device)
    except Exception:
        pass

    conf_thresh = float(conf_thresh)
    iou_thresh = float(iou_thresh)

    assert 0.0 < iou_thresh <= 1.0, f"Invalid iou_thresh={iou_thresh}, must be in (0, 1]"
    assert 0.0 <= conf_thresh <= 1.0, f"Invalid conf_thresh={conf_thresh}, must be in [0, 1]"
    assert decoder is not None, "Pass a configured DFLDecoder via Postprocessor"
    assert feat_shapes is not None, "feat_shapes must be provided to postprocess_detections"
    dfl_decoder = decoder

    tau = float(getattr(decoder, 'tau', 1.0))

    anchor_points, stride_tensor = dfl_decoder.get_anchors(feat_shapes)
    N = outputs.shape[2]
    assert anchor_points.shape[0] == N and stride_tensor.shape[0] == N, \
        f"Anchor count {anchor_points.shape[0]} != predictions N {N}; check level order/shapes."

    num_reg = reg_max * 4
    reg = outputs[:, :num_reg, :].permute(0, 2, 1).contiguous()
    cls = outputs[:, num_reg:, :].contiguous()
    boxes_b = decode_distances(
        anchor_points, stride_tensor, reg, reg_max, tau=tau, decoder=dfl_decoder
    )
    scores_b, class_ids_b = cls.sigmoid().max(1)
    if use_objectness_eval and (obj_logits is not None):
        obj_scores = obj_logits.sigmoid()  # [B,N]
        scores_b = scores_b * obj_scores

    keep_mask = scores_b > conf_thresh  # [B, N]

    B = outputs.size(0)
    device = outputs.device

    if pre_nms_topk:
        k = min(int(pre_nms_topk), scores_b.size(1))
        masked_scores = scores_b.masked_fill(~keep_mask, float('-inf'))
        topk_scores, topk_idx = torch.topk(masked_scores, k, dim=1)  # [B, k]
        topk_classes = class_ids_b.gather(1, topk_idx)  # [B, k]
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, 4)  # [B, k, 4]
        topk_boxes = boxes_b.gather(1, topk_idx_exp)  # [B, k, 4]
        batch_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, k)  # [B, k]
        flat_boxes = topk_boxes.reshape(-1, 4)
        flat_scores = topk_scores.reshape(-1)
        flat_classes = topk_classes.reshape(-1)
        flat_batch = batch_ids.reshape(-1)
    else:
        sel_list = []
        for b in range(B):
            sel = torch.where(keep_mask[b])[0]
            if sel.numel() == 0:
                continue
            sel_list.append((
                boxes_b[b, sel], scores_b[b, sel], class_ids_b[b, sel],
                torch.full((sel.numel(), ), b, device=device, dtype=torch.long)
            ))
        if not sel_list:
            return [
                dict(
                    boxes=torch.empty(0, 4, device=device),
                    scores=torch.empty(0, device=device),
                    class_ids=torch.empty(0, dtype=torch.long, device=device)
                ) for _ in range(B)
            ]
        flat_boxes = torch.cat([x[0] for x in sel_list], dim=0)
        flat_scores = torch.cat([x[1] for x in sel_list], dim=0)
        flat_classes = torch.cat([x[2] for x in sel_list], dim=0)
        flat_batch = torch.cat([x[3] for x in sel_list], dim=0)

    if nms_free:
        detections = []
        for b in range(B):
            mask_b = flat_batch == b
            if mask_b.sum() == 0:
                detections.append(
                    dict(
                        boxes=torch.empty(0, 4, device=device),
                        scores=torch.empty(0, device=device),
                        class_ids=torch.empty(0, dtype=torch.long, device=device)
                    )
                )
                continue
            boxes_b_sel = flat_boxes[mask_b].to(torch.float32)
            scores_b_sel = flat_scores[mask_b].to(torch.float32)
            classes_b_sel = flat_classes[mask_b].to(torch.long)
            detections.append(dict(boxes=boxes_b_sel, scores=scores_b_sel, class_ids=classes_b_sel))
        return detections

    try:
        from torchvision.ops import batched_nms, nms  # type: ignore
    except Exception:
        batched_nms = None

        def nms(
            boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
        ) -> torch.Tensor:  # type: ignore
            if boxes.numel() == 0:
                return torch.empty((0, ), dtype=torch.long, device=boxes.device)
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1).clamp_min_(0) * (y2 - y1).clamp_min_(0)
            order = scores.sort(descending=True).indices
            keep = []
            while order.numel() > 0:
                i = order[0]
                keep.append(i)
                if order.numel() == 1:
                    break
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = (xx2 - xx1).clamp_min_(0)
                h = (yy2 - yy1).clamp_min_(0)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
                mask = iou <= iou_threshold
                order = order[1:][mask]
            return torch.stack(keep) if keep else torch.empty((0, ),
                                                              dtype=torch.long,
                                                              device=boxes.device)

    detections = []
    for b in range(B):
        mask_b = flat_batch == b
        if mask_b.sum() == 0:
            detections.append(
                dict(
                    boxes=torch.empty(0, 4, device=device),
                    scores=torch.empty(0, device=device),
                    class_ids=torch.empty(0, dtype=torch.long, device=device)
                )
            )
            continue

        boxes_b_sel = flat_boxes[mask_b].to(torch.float32)
        scores_b_sel = flat_scores[mask_b].to(torch.float32)
        classes_b_sel = flat_classes[mask_b].to(torch.long)

        if batched_nms is not None:
            idxs = torch.zeros_like(classes_b_sel) if class_agnostic_nms else classes_b_sel
            keep_idx = batched_nms(boxes_b_sel, scores_b_sel, idxs, iou_thresh)
        else:
            keep_sets = []
            if class_agnostic_nms:
                keep_sets.append(nms(boxes_b_sel, scores_b_sel, iou_thresh))
            else:
                ucls = classes_b_sel.unique()
                for c in ucls:
                    m = classes_b_sel == c
                    keep_c = nms(boxes_b_sel[m], scores_b_sel[m], iou_thresh)
                    keep_sets.append(torch.where(m)[0][keep_c])
            keep_idx = torch.cat(keep_sets, dim=0) if len(keep_sets) else torch.empty(
                0, dtype=torch.long, device=device
            )

        if post_nms_topk and keep_idx.numel() > post_nms_topk:
            _, order = scores_b_sel[keep_idx].sort(descending=True)
            keep_idx = keep_idx[order[:post_nms_topk]]

        det_boxes = boxes_b_sel[keep_idx]
        det_scores = scores_b_sel[keep_idx]
        det_classes = classes_b_sel[keep_idx]

        detections.append(dict(boxes=det_boxes, scores=det_scores, class_ids=det_classes))

    return detections
