import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import bbox_iou


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """Select positive anchor centers that fall inside GT boxes.

    Args:
        xy_centers (Tensor): (num_anchors, 2)
        gt_bboxes (Tensor): (B, M, 4) as (x1, y1, x2, y2)

    Returns:
        Tensor: (B, M, num_anchors) boolean mask
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, dim=2)  # (B*M,1,2) each
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
        bs, n_boxes, n_anchors, -1
    )
    return bbox_deltas.amin(3).gt(eps)  # no in-place op on temps


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """If an anchor is assigned to multiple GTs, keep the GT with the highest IoU.

    Args:
        mask_pos   (Tensor): (B, M, A)
        overlaps   (Tensor): (B, M, A)
        n_max_boxes (int)

    Returns:
        target_gt_idx (Tensor): (B, A) indices of chosen GT per anchor
        fg_mask       (Tensor): (B, A)
        mask_pos      (Tensor): (B, M, A)
    """
    fg_mask = mask_pos.sum(-2)  # (B, A), float dtype

    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (B, M, A)
        max_overlaps_idx = overlaps.argmax(1)  # (B, A) best GT index per anchor
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes).permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-2)

    target_gt_idx = mask_pos.argmax(-2)  # (B, A)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Assign targets.

        Args:
            pd_scores  (Tensor): (B, A, C)
            pd_bboxes  (Tensor): (B, A, 4)
            anc_points (Tensor): (A, 2)
            gt_labels  (Tensor): (B, M, 1)
            gt_bboxes  (Tensor): (B, M, 4)
            mask_gt    (Tensor): (B, M, 1)

        Returns:
            target_labels (Tensor): (B, A)
            target_bboxes (Tensor): (B, A, 4)
            target_scores (Tensor): (B, A, C)
            fg_mask       (Tensor): (B, A) bool
        """
        B = pd_scores.size(0)
        M = gt_bboxes.size(1)

        if M == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx, device=device),
                torch.zeros_like(pd_bboxes, device=device),
                torch.zeros_like(pd_scores, device=device),
                torch.zeros_like(pd_scores[..., 0], device=device).bool(),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, M)

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)   # (B, M, 1)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # (B, M, 1)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        )  # (B, A, 1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)  # (B, M, A)

        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),  # (B, M, topk)
        )

        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        """Compute aligned metric (score^alpha * IoU^beta) and IoU overlaps.

        Returns:
            align_metric (Tensor): (B, M, A)
            overlaps     (Tensor): (B, M, A)
        """
        device = pd_scores.device
        gt_labels = gt_labels.to(torch.long, non_blocking=True)

        B = pd_scores.size(0)
        M = gt_labels.size(1)

        b_idx = torch.arange(B, device=device)[:, None].expand(-1, M)     # (B, M)
        c_idx = gt_labels.squeeze(-1).to(device=device)                   # (B, M)
        bbox_scores = pd_scores[b_idx, :, c_idx]                          # (B, M, A)

        overlaps = bbox_iou(
            gt_bboxes.unsqueeze(2),         # (B, M, 1, 4)
            pd_bboxes.unsqueeze(1),         # (B, 1, A, 4)
            xywh=False,
            CIoU=True
        ).squeeze(3).clamp_min(0)  # no in-place

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics   (Tensor): (B, M, A)
            topk_mask (Tensor or None): (B, M, topk)

        Returns:
            is_in_topk (Tensor): (B, M, A) in {0,1}
        """
        B, M, A = metrics.shape

        topk_vals, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)  # (B, M, K)
        if topk_mask is None:
            topk_mask = (topk_vals.max(-1, keepdim=True).values > self.eps).expand(-1, -1, self.topk)

        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))

        ones = torch.ones_like(topk_idxs, dtype=torch.int32)
        counts = torch.zeros(B, M, A, dtype=torch.int32, device=metrics.device)
        counts.scatter_add_(dim=-1, index=topk_idxs, src=ones)

        is_in_topk = (counts == 1).to(metrics.dtype)
        return is_in_topk

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels     (Tensor): (B, M, 1)
            gt_bboxes     (Tensor): (B, M, 4)
            target_gt_idx (Tensor): (B, A)
            fg_mask       (Tensor): (B, A)

        Returns:
            target_labels (Tensor): (B, A)
            target_bboxes (Tensor): (B, A, 4)
            target_scores (Tensor): (B, A, C)
        """
        B, M, _ = gt_bboxes.shape
        device = gt_bboxes.device

        batch_ind = torch.arange(B, device=device, dtype=torch.int64)[:, None]  # (B,1)
        flat_idx = target_gt_idx + batch_ind * M                                 # (B, A)

        flat_labels = gt_labels.long().flatten()          # (B*M,)
        flat_bboxes = gt_bboxes.reshape(-1, 4)            # (B*M, 4)

        target_labels = flat_labels[flat_idx]             # (B, A)
        target_bboxes = flat_bboxes[flat_idx]             # (B, A, 4)

        target_labels = target_labels.clamp_min(0)
        target_scores = F.one_hot(target_labels, self.num_classes).to(gt_bboxes.dtype)  # (B, A, C)

        fg_scores_mask = fg_mask[:, :, None].expand(-1, -1, self.num_classes)           # (B, A, C)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.zeros_like(target_scores))

        return target_labels, target_bboxes, target_scores
