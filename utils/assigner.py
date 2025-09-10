from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from .box_iou import pairwise_box_iou

class TaskAlignedAssigner(nn.Module):
    """
    One class, three behaviors:
      - mode="ult": Ultralytics/TOOD-style TaskAlignedAssigner (alignment p^α · IoU^β, center-in-box, top-k)
      - mode="simota": SimOTA dynamic-k (BCE + λ·(-log IoU) cost, center prior, least-cost selection)
      - mode="mixed": MLA-style dual assignment: O2M (SimOTA) for regression + O2O (Hungarian) for classification,
                      both using the same cost as SimOTA (Electronics 13(4856)).

    Box format: xyxy. Inputs (B = batch, N = anchors, M = max GTs, C = classes):
      pd_scores:    (B,N,C)  probabilities if use_sigmoid=True, else logits (will be sigmoided)
      pd_bboxes:    (B,N,4)  decoded boxes (xyxy)
      anc_points:   (N,2)    anchor centers (x,y)
      gt_labels:    (B,M,1)  class id per GT (pad doesn't matter; see mask_gt)
      gt_bboxes:    (B,M,4)  GT boxes (xyxy)
      mask_gt:      (B,M,1)  1 if GT valid else 0
    Optional:
      pd_objectness:(B,N,1)  objectness probs/logits; multiplied into pd_scores if provided (biasing costs)
      anc_strides:  (N,1)    per-anchor stride for center prior (SimOTA/mixed). If None, use inside-box only.

    Returns a dict with regression and classification targets; in mixed mode,
    classification targets come from O2O matching.
    """
    def __init__(
        self,
        mode: str = "ult",
        num_classes: int = 80,
        topk: int = 13,  # used by ULT
        alpha: float = 1.0,  # used by ULT (alignment exponent on class prob)
        beta: float = 6.0,  # used by ULT (alignment exponent on IoU)
        lambda_iou: float = 3.0,  # cost weight (SimOTA/MLA)
        topq: int = 10,  # dynamic-k: sum of top-q IoUs (SimOTA/MLA)
        center_radius: float = 2.5,  # center prior radius in strides (SimOTA/MLA)
        input_is_logits: bool = True,
        eps: float = 1e-9,
        min_stride_alpha: float = 0.0,  # per-stride tiny GT filter; 0 disables
        filter_tiny_gt: bool = True,
        objless_target_iou:
        float = 0.0,  # Linear ramp: weight = floor + (1-floor)*clamp(IoU/t,0,1); 0 disables
        objless_weight_floor: float = 0.05,  # Minimum weight at IoU=0 (0..1)
    ):
        super().__init__()
        assert mode in ("ult", "simota", "mixed")
        self.mode = mode
        self.num_classes = num_classes

        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.lambda_iou = lambda_iou
        self.topq = topq
        self.center_radius = center_radius
        self.input_is_logits = input_is_logits
        self.eps = eps
        self.min_stride_alpha = float(min_stride_alpha)
        self.filter_tiny_gt = bool(filter_tiny_gt)
        self.objless_target_iou = float(objless_target_iou)
        self.objless_weight_floor = float(objless_weight_floor)

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,  # (B,N,C)
        pd_bboxes: torch.Tensor,  # (B,N,4)
        anc_points: torch.Tensor,  # (N,2)
        gt_labels: torch.Tensor,  # (B,M,1)
        gt_bboxes: torch.Tensor,  # (B,M,4)
        mask_gt: torch.Tensor,  # (B,M,1)
        pd_objectness: Optional[torch.Tensor] = None,  # (B,N,1)
        anc_strides: Optional[torch.Tensor] = None,  # (N,1)
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        B, N, C = pd_scores.shape
        _, M, _ = gt_labels.shape
        device = pd_scores.device
        mask_gt_bool = mask_gt.bool().squeeze(-1)  # (B,M)

        if self.input_is_logits:
            pd_scores = pd_scores.sigmoid()
            if pd_objectness is not None:
                pd_objectness = pd_objectness.sigmoid()

        if M == 0 or mask_gt_bool.sum() == 0:
            return self._empty_return(pd_scores, pd_bboxes, device)

        if self.mode == "ult":
            return self._tal_assign(
                pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt_bool, anc_strides
            )
        elif self.mode == "simota":
            return self._simota_assign(
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt_bool,
                pd_objectness=pd_objectness,
                anc_strides=anc_strides
            )
        else:  # mixed (MLA): O2M for regression (SimOTA), O2O for classification (Hungarian)
            o2m = self._simota_assign(
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt_bool,
                pd_objectness=pd_objectness,
                anc_strides=anc_strides
            )
            o2o_cls = self._o2o_for_classification(
                pd_scores,
                pd_bboxes,
                gt_labels,
                gt_bboxes,
                mask_gt_bool,
                pd_objectness=pd_objectness,
                anc_strides=anc_strides
            )
            # Reuse classification O2O matches for objectness to avoid an extra Hungarian pass
            cls_branch = o2o_cls["cls"]
            obj_branch = {
                "target_objectness": cls_branch["fg_mask"].float(),
                "fg_mask": cls_branch["fg_mask"],
                "matched_gt": cls_branch["matched_gt"],
            }
            return {"reg": o2m["reg"], "cls": cls_branch, "obj": obj_branch}

    def _tal_assign(
        self,
        pd_scores,
        pd_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt_bool,
        anc_strides=None
    ):
        B, N, C = pd_scores.shape
        _, M, _ = gt_labels.shape

        mask_in_gts = self._centers_in_boxes(anc_points, gt_bboxes)  # (B,M,N)
        if self.filter_tiny_gt and (anc_strides is not None) and self.min_stride_alpha > 0.0:
            min_wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
            min_side = torch.minimum(min_wh[..., 0], min_wh[..., 1])  # (B,M)
            thresh = self.min_stride_alpha * anc_strides.view(1, 1, -1)  # (1,1,N)
            allow = (min_side.unsqueeze(-1) > thresh)  # (B,M,N)
            mask_in_gts = mask_in_gts & allow

        valid_mask = (mask_in_gts & mask_gt_bool.unsqueeze(-1))
        align, ious = self._tal_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, valid_mask)

        topk_mask = self._topk_mask(align, self.topk, valid_rows=mask_gt_bool)  # (B,M,N) bool
        mask_pos = (topk_mask & valid_mask).float()

        tgt_gt_idx, fg_mask, mask_pos = self._resolve_conflicts(mask_pos, ious)

        target_labels, target_bboxes = self._gather_targets(gt_labels, gt_bboxes, tgt_gt_idx)

        align = align * mask_pos  # (B,M,N)
        max_align_per_anchor = align.amax(dim=1, keepdim=True) + self.eps  # (B,1,N)
        scaled = (align / max_align_per_anchor) * (ious * mask_pos)  # (B,M,N)
        per_anchor_scale = scaled.sum(dim=1, keepdim=False)  # (B,N)

        onehot = self._one_hot(target_labels, self.num_classes)  # (B,N,C)
        cls_targets = onehot * per_anchor_scale.unsqueeze(-1)

        return {
            "reg": {
                "target_labels": target_labels,
                "target_bboxes": target_bboxes,
                "fg_mask": fg_mask.bool(),
                "matched_gt": tgt_gt_idx,
            }, "cls": {
                "target_scores": cls_targets,
                "fg_mask": fg_mask.bool(),
                "matched_gt": tgt_gt_idx,
            }
        }

    def _simota_assign(
        self,
        pd_scores,
        pd_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt_bool,
        pd_objectness=None,
        anc_strides: Optional[torch.Tensor] = None
    ):
        B, N, C = pd_scores.shape

        cand_mask = self._center_prior(
            anc_points, gt_bboxes, anc_strides, self.center_radius
        )  # (B,M,N)
        if self.filter_tiny_gt and (anc_strides is not None) and self.min_stride_alpha > 0.0:
            min_wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
            min_side = torch.minimum(min_wh[..., 0], min_wh[..., 1])  # (B,M)
            thresh = self.min_stride_alpha * anc_strides.view(1, 1, -1)  # (1,1,N)
            allow = (min_side.unsqueeze(-1) > thresh)  # (B,M,N)
            cand_mask = cand_mask & allow

        a1, a2 = gt_bboxes.unsqueeze(2).chunk(2, dim=-1)
        b1, b2 = pd_bboxes.unsqueeze(1).chunk(2, dim=-1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(-1)
        ious = inter / (((a2 - a1).prod(-1) + (b2 - b1).prod(-1) - inter).clamp_min(self.eps))

        if pd_objectness is not None:
            cls_prob_all = pd_scores * pd_objectness
        else:
            cls_prob_all = pd_scores
        gt_ids = gt_labels.clamp_min(0).squeeze(-1)  # (B,M)
        cls_for_gt = torch.gather(
            cls_prob_all.unsqueeze(1).expand(B, gt_ids.shape[1], N, C), 3,
            gt_ids.view(B, -1, 1, 1).expand(B, gt_ids.shape[1], N, 1)
        ).squeeze(-1)

        cls_cost = -torch.log(cls_for_gt.clamp_min(self.eps))
        iou_cost = -torch.log(ious.clamp_min(self.eps))
        cost = cls_cost + self.lambda_iou * iou_cost

        cost = cost.masked_fill(~(cand_mask & mask_gt_bool.unsqueeze(-1)), float('inf'))

        dynamic_k = self._dynamic_k(ious, cand_mask, mask_gt_bool, topq=self.topq)  # (B,M) long

        Kmax = int(min(self.topq, N)) if self.topq > 0 else 0
        pos_mask = torch.zeros_like(cost, dtype=torch.bool)  # (B,M,N)
        if Kmax > 0:
            safe_cost = torch.where(
                torch.isfinite(cost), cost,
                torch.tensor(float('inf'), device=cost.device, dtype=cost.dtype)
            )
            scores = -safe_cost  # larger is better
            topv, topi = torch.topk(scores, k=Kmax, dim=2, largest=True)
            k_per_row = dynamic_k.clamp_min(0).clamp_max(Kmax).unsqueeze(-1)  # (B,M,1)
            ar = torch.arange(Kmax, device=cost.device).view(1, 1, Kmax)
            sel = ar < k_per_row  # (B,M,Kmax)
            pos_mask.scatter_(2, topi, sel)

        multi = pos_mask.sum(dim=1) > 1  # (B,N) bool
        if multi.any():
            best_gt = cost.argmin(dim=1)  # (B,N)
            keep = torch.zeros_like(pos_mask)
            keep.scatter_(1, best_gt.unsqueeze(1), True)
            pos_mask = pos_mask & keep

        fg_mask = pos_mask.any(dim=1)  # (B,N)
        matched_gt_idx = pos_mask.float().argmax(dim=1)  # (B,N)
        target_labels, target_bboxes = self._gather_targets(gt_labels, gt_bboxes, matched_gt_idx)
        cls_targets = self._one_hot(target_labels, self.num_classes) * fg_mask.unsqueeze(-1)

        return {
            "reg": {
                "target_labels": target_labels,
                "target_bboxes": target_bboxes,
                "fg_mask": fg_mask,
                "matched_gt": matched_gt_idx,
            }, "cls": {
                "target_scores": cls_targets,
                "fg_mask": fg_mask,
                "matched_gt": matched_gt_idx,
            }
        }

    def _o2o_for_confidence(
        self,
        pd_scores: torch.
        Tensor,  # (B,N,C) probabilities (if input_is_logits=True, already sigmoided above)
        pd_bboxes: torch.Tensor,  # (B,N,4)
        gt_labels: torch.Tensor,  # (B,M,1)
        gt_bboxes: torch.Tensor,  # (B,M,4)
        mask_gt_bool: torch.Tensor,  # (B,M)
        pd_objectness: Optional[torch.Tensor] = None,  # (B,N,1)
        anc_strides: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        One-to-one matching for confidence head as in MLA (Electronics 2024, Eq. (1)–(4)).
        Cost(i,j) = Cscore(i,j) + lambda_iou * Cloc(i,j), where
          Cscore(i,j) = -log( (pd_scores * pd_objectness)[i, j_cls] )
          Cloc(i,j)   = -log( IoU(pd_bboxes[i], gt_bboxes[j]) )
        Returns targets for an objectness/confidence head.
        """
        B, N, C = pd_scores.shape

        if pd_objectness is not None:
            cls_prob_all = pd_scores * pd_objectness  # (B,N,C)
        else:
            cls_prob_all = pd_scores

        a1, a2 = gt_bboxes.unsqueeze(2).chunk(2, dim=-1)
        b1, b2 = pd_bboxes.unsqueeze(1).chunk(2, dim=-1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(-1)
        ious = inter / (((a2 - a1).prod(-1) + (b2 - b1).prod(-1) - inter).clamp_min(self.eps))

        gt_ids = gt_labels.clamp_min(0).squeeze(-1)  # (B,M)
        B, N, C = cls_prob_all.shape
        cls_for_gt = torch.gather(
            cls_prob_all.unsqueeze(1).expand(B, gt_ids.shape[1], N, C), 3,
            gt_ids.view(B, -1, 1, 1).expand(B, gt_ids.shape[1], N, 1)
        ).squeeze(-1).clamp_min(self.eps)

        eps = self.eps
        device_type = 'cuda' if str(cls_prob_all.device).startswith('cuda') else 'cpu'
        with torch.autocast(device_type=device_type, enabled=True):
            if (pd_objectness is None) and (self.objless_target_iou > 0.0):
                t = float(self.objless_target_iou)
                f = float(self.objless_weight_floor)
                f = 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
                w = f + (1.0 - f) * (ious / max(t, 1e-12)).clamp(0.0, 1.0)
                cls_for_gt = (cls_for_gt * w).clamp_min(eps)
            cls_cost = -torch.log(cls_for_gt.clamp_min(eps))
            iou_cost = -torch.log(ious.clamp_min(eps))
            cost = cls_cost + self.lambda_iou * iou_cost

        if self.filter_tiny_gt and (anc_strides is not None) and self.min_stride_alpha > 0.0:
            min_wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
            min_side = torch.minimum(min_wh[..., 0], min_wh[..., 1])  # (B,M)
            thresh = self.min_stride_alpha * anc_strides.view(1, 1, -1)  # (1,1,N)
            allow = (min_side.unsqueeze(-1) > thresh)  # (B,M,N)
            cost = cost.masked_fill(~allow, float('inf'))
        cost = cost.masked_fill(~mask_gt_bool.unsqueeze(-1), float('inf'))

        pos_mask = torch.zeros_like(cost, dtype=torch.bool)  # (B,M,N)
        for b in range(B):
            rows = torch.where(mask_gt_bool[b])[0]
            if rows.numel() == 0:
                continue
            cb = cost[b, rows]  # (R,N)
            row_has_any = torch.isfinite(cb).any(dim=1)
            if not row_has_any.any():
                continue
            sub_rows = torch.where(row_has_any)[0]
            cb_valid = cb[sub_rows]
            r_sub, c_idx = self._hungarian(cb_valid)
            if r_sub.numel() > 0:
                r_idx = sub_rows[r_sub]
            else:
                r_idx = r_sub
            if r_idx.numel() > 0:
                pos_mask[b, rows[r_idx], c_idx] = True

        obj_fg_mask = pos_mask.any(dim=1)  # (B,N)
        obj_matched_gt = pos_mask.float().argmax(dim=1)  # (B,N)
        target_objectness = obj_fg_mask.float()  # (B,N)

        return {
            "obj": {
                "target_objectness": target_objectness,
                "fg_mask": obj_fg_mask,
                "matched_gt": obj_matched_gt,
            }
        }

    def _o2o_for_classification(
        self,
        pd_scores: torch.
        Tensor,  # (B,N,C) probabilities (if input_is_logits=True, already sigmoided above)
        pd_bboxes: torch.Tensor,  # (B,N,4)
        gt_labels: torch.Tensor,  # (B,M,1)
        gt_bboxes: torch.Tensor,  # (B,M,4)
        mask_gt_bool: torch.Tensor,  # (B,M)
        pd_objectness: Optional[torch.Tensor] = None,  # (B,N,1)
        anc_strides: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        One-to-one matching for classification branch as in MLA.
        Cost(i,j) = -log( p_cls(i,j) ) + lambda_iou * (-log IoU(i,j)).
        If pd_objectness is provided, p_cls is multiplied by it to bias costs.
        Returns one-hot class targets only for matched anchors.
        """
        B, N, C = pd_scores.shape

        if pd_objectness is not None:
            cls_prob_all = pd_scores * pd_objectness  # (B,N,C)
        else:
            cls_prob_all = pd_scores

        a1, a2 = gt_bboxes.unsqueeze(2).chunk(2, dim=-1)
        b1, b2 = pd_bboxes.unsqueeze(1).chunk(2, dim=-1)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(-1)
        ious = inter / (((a2 - a1).prod(-1) + (b2 - b1).prod(-1) - inter).clamp_min(self.eps))

        gt_ids = gt_labels.clamp_min(0).squeeze(-1)  # (B,M)
        B, N, C = cls_prob_all.shape
        cls_for_gt = torch.gather(
            cls_prob_all.unsqueeze(1).expand(B, gt_ids.shape[1], N, C), 3,
            gt_ids.view(B, -1, 1, 1).expand(B, gt_ids.shape[1], N, 1)
        ).squeeze(-1).clamp_min(self.eps)  # (B,M,N)

        cls_cost = -torch.log(cls_for_gt.clamp_min(self.eps))
        iou_cost = -torch.log(ious.clamp_min(self.eps))
        cost = cls_cost + self.lambda_iou * iou_cost

        if self.filter_tiny_gt and (anc_strides is not None) and self.min_stride_alpha > 0.0:
            min_wh = (gt_bboxes[..., 2:4] - gt_bboxes[..., 0:2]).clamp_min(0.0)
            min_side = torch.minimum(min_wh[..., 0], min_wh[..., 1])  # (B,M)
            thresh = self.min_stride_alpha * anc_strides.view(1, 1, -1)  # (1,1,N)
            allow = (min_side.unsqueeze(-1) > thresh)  # (B,M,N)
            cost = cost.masked_fill(~allow, float('inf'))
        cost = cost.masked_fill(~mask_gt_bool.unsqueeze(-1), float('inf'))

        pos_mask = torch.zeros_like(cost, dtype=torch.bool)  # (B,M,N)
        for b in range(B):
            rows = torch.where(mask_gt_bool[b])[0]
            if rows.numel() == 0:
                continue
            cb = cost[b, rows]  # (R,N)
            row_has_any = torch.isfinite(cb).any(dim=1)
            if not row_has_any.any():
                continue
            sub_rows = torch.where(row_has_any)[0]
            cb_valid = cb[sub_rows]
            r_sub, c_idx = self._hungarian(cb_valid)
            if r_sub.numel() > 0:
                r_idx = sub_rows[r_sub]
            else:
                r_idx = r_sub
            if r_idx.numel() > 0:
                pos_mask[b, rows[r_idx], c_idx] = True

        cls_fg_mask = pos_mask.any(dim=1)  # (B,N)
        cls_matched_gt = pos_mask.float().argmax(dim=1)  # (B,N)
        target_labels, _ = self._gather_targets(gt_labels, gt_bboxes, cls_matched_gt)
        cls_targets = self._one_hot(target_labels, self.num_classes) * cls_fg_mask.unsqueeze(-1)

        return {
            "cls": {
                "target_scores": cls_targets,
                "fg_mask": cls_fg_mask,
                "matched_gt": cls_matched_gt,
            }
        }

    def _empty_return(self, pd_scores, pd_bboxes, device):
        B, N, C = pd_scores.shape
        zerosN = torch.zeros((B, N), device=device, dtype=torch.bool)
        return {
            "reg": {
                "target_labels":
                    torch.full((B, N), self.num_classes, device=device, dtype=torch.long),
                "target_bboxes":
                    torch.zeros((B, N, 4), device=device, dtype=pd_bboxes.dtype),
                "fg_mask":
                    zerosN,
                "matched_gt":
                    torch.zeros((B, N), device=device, dtype=torch.long),
            },
            "cls": {
                "target_scores": torch.zeros((B, N, C), device=device, dtype=pd_scores.dtype),
                "fg_mask": zerosN,
                "matched_gt": torch.zeros((B, N), device=device, dtype=torch.long),
            },
        }

    def _centers_in_boxes(self, anc_points, gt_bboxes):
        B, M, _ = gt_bboxes.shape
        N = anc_points.shape[0]
        lt = gt_bboxes[:, :, :2].unsqueeze(2)  # (B,M,1,2)
        rb = gt_bboxes[:, :, 2:4].unsqueeze(2)  # (B,M,1,2)
        ap = anc_points.view(1, 1, N, 2).expand(B, M, N, 2)
        deltas = torch.cat([ap - lt, rb - ap], dim=-1)  # (B,M,N,4)
        return (deltas.amin(dim=-1) > 0)

    def _tal_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, valid_mask):
        B, N, C = pd_scores.shape
        M = gt_labels.shape[1]
        device = pd_scores.device
        overlaps = torch.zeros((B, M, N), device=device, dtype=pd_bboxes.dtype)
        clsprob = torch.zeros((B, M, N), device=device, dtype=pd_scores.dtype)

        for b in range(B):
            rows = torch.where(valid_mask[b].any(dim=-1))[0]
            if rows.numel() == 0:
                continue
            cls_col = pd_scores[b, :, gt_labels[b, rows, 0].clamp_min(0)].transpose(0, 1)  # (R,N)
            clsprob[b, rows] = cls_col
            overlaps[b, rows] = pairwise_box_iou(gt_bboxes[b, rows], pd_bboxes[b])  # .squeeze(0)

        align = (
            clsprob.clamp_min(self.eps).pow(self.alpha) *
            overlaps.clamp_min(self.eps).pow(self.beta)
        )
        return align, overlaps

    def _topk_mask(self, metric, k: int, valid_rows: torch.Tensor):
        B, M, N = metric.shape
        mask = torch.zeros_like(metric, dtype=torch.bool)
        if k <= 0:
            return mask
        for b in range(B):
            rows = torch.where(valid_rows[b])[0]
            if rows.numel() == 0:
                continue
            mb = metric[b, rows]  # (R,N)
            k_eff = min(k, N)
            topk = torch.topk(mb, k_eff, dim=-1, largest=True).indices  # (R,k)
            mask[b, rows] = torch.zeros_like(mb, dtype=torch.bool).scatter(-1, topk, True)
        return mask

    def _resolve_conflicts(self, mask_pos, overlaps):
        B, M, N = mask_pos.shape
        fg_mask = mask_pos.sum(dim=1)  # (B,N)
        if (fg_mask.max() > 1).item():
            multi = (fg_mask.unsqueeze(1) > 1).expand(-1, M, -1)
            winner = overlaps.argmax(dim=1)  # (B,N)
            keep = torch.zeros_like(mask_pos)
            keep.scatter_(1, winner.unsqueeze(1), 1.0)
            mask_pos = torch.where(multi, keep, mask_pos)
            fg_mask = mask_pos.sum(dim=1)
        target_gt_idx = mask_pos.argmax(dim=1)
        return target_gt_idx, fg_mask.bool(), mask_pos

    def _gather_targets(self, gt_labels, gt_bboxes, target_gt_idx):
        B, M, _ = gt_labels.shape
        device = gt_labels.device
        batch_base = torch.arange(B, device=device)[:, None] * M
        flat_idx = (target_gt_idx + batch_base).view(-1)
        flat_labels = gt_labels.view(-1, 1)[flat_idx]
        flat_boxes = gt_bboxes.view(-1, 4)[flat_idx]
        return flat_labels.view(B, -1).long(), flat_boxes.view(B, -1, 4)

    def _one_hot(self, labels: torch.Tensor, num_classes: int):
        B, N = labels.shape
        out = torch.zeros((B, N, num_classes), device=labels.device, dtype=torch.float32)
        valid = (labels >= 0) & (labels < num_classes)
        if valid.any():
            i = torch.nonzero(valid, as_tuple=False)
            out[i[:, 0], i[:, 1], labels[valid]] = 1.0
        return out

    def _center_prior(self, anc_points, gt_bboxes, anc_strides, radius: float):
        inside_box = self._centers_in_boxes(anc_points, gt_bboxes)  # (B,M,N)
        if anc_strides is None:
            return inside_box
        B, M, _ = gt_bboxes.shape
        N = anc_points.shape[0]
        strides = anc_strides.view(1, 1, N)  # (1,1,N)
        centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0  # (B,M,2)
        half = radius * strides  # (1,1,N)
        cx = centers[..., 0].unsqueeze(-1)  # (B,M,1)
        cy = centers[..., 1].unsqueeze(-1)  # (B,M,1)
        x_ok = (anc_points[None, None, :, 0]
                >= (cx - half)) & (anc_points[None, None, :, 0] <= (cx + half))
        y_ok = (anc_points[None, None, :, 1]
                >= (cy - half)) & (anc_points[None, None, :, 1] <= (cy + half))
        return inside_box & (x_ok & y_ok)

    def _dynamic_k(self, ious, cand_mask, mask_gt_bool, topq: int):
        B, M, N = ious.shape
        if topq <= 0:
            return torch.zeros((B, M), dtype=torch.long, device=ious.device)
        vals = torch.where(cand_mask, ious, torch.zeros_like(ious))  # (B,M,N)
        q = min(topq, N)
        top_vals, _ = torch.topk(vals, k=q, dim=2)
        dynk = top_vals.sum(dim=2).clamp_min(1.0).round().long()
        dynk = torch.where(mask_gt_bool, dynk, torch.zeros_like(dynk))
        dynk.clamp_min_(1)
        return dynk

    def _hungarian(self, cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hybrid assignment:
          - Greedy torch on device for large problems (R>64 or N>10000)
          - SciPy linear_sum_assignment for small problems (CPU), fallback to greedy
        """
        R, N = cost.shape
        d = cost.device
        if R == 0 or N == 0:
            return (
                torch.empty(0, dtype=torch.long, device=d),
                torch.empty(0, dtype=torch.long, device=d),
            )

        def greedy(c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            used = torch.zeros(c.shape[1], dtype=torch.bool, device=c.device)
            r_sel: list[int] = []
            c_sel: list[int] = []
            big = torch.tensor(1e12, device=c.device, dtype=c.dtype)
            for r in range(c.shape[0]):
                row = c[r] + used.to(c.dtype) * big
                j = int(torch.argmin(row).item())
                if not torch.isfinite(row[j]):
                    continue
                used[j] = True
                r_sel.append(r)
                c_sel.append(j)
            return (
                torch.tensor(r_sel, dtype=torch.long, device=c.device),
                torch.tensor(c_sel, dtype=torch.long, device=c.device),
            )

        if (R > 64) or (N > 2000):
            return greedy(cost)
        try:
            from scipy.optimize import linear_sum_assignment
            C = cost.detach().float().cpu().numpy()
            r_idx, c_idx = linear_sum_assignment(C)
            return (
                torch.as_tensor(r_idx, dtype=torch.long, device=d),
                torch.as_tensor(c_idx, dtype=torch.long, device=d),
            )
        except Exception:
            return greedy(cost)
