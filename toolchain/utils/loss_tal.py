import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel

class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t)**self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = (1.0 - iou) * bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return (loss_iou, loss_dfl)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1),
                target_left.view(-1),
                reduction="none",
            ).view(target_left.shape) * weight_left
        )
        loss_right = (
            F.cross_entropy(
                pred_dist.view(-1, self.reg_max + 1),
                target_right.view(-1),
                reduction="none",
            ).view(target_left.shape) * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)

class RN_ComputeLoss(nn.Module):
    def __init__(self, model, use_dfl=True):
        super().__init__()
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction="none"
        )
        if h["fl_gamma"] > 0:
            BCEcls = FocalLoss(BCEcls, h["fl_gamma"])
        m = de_parallel(model).model[-1]
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.nl = m.nl
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = use_dfl
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        i = targets[:, 0]
        (_, counts) = i.unique(return_counts=True)
        out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            (b, a, c) = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _calculate_loss_for_branch(self, feats, targets, imgsz):
        loss = torch.zeros(3, device=self.device)
        (pred_distri,
         pred_scores) = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats],
                                  2).split((self.reg_max * 4, self.nc), 1)

        if not torch.isfinite(pred_distri).all() or not torch.isfinite(pred_scores).all():
            raise ValueError(
                "Error: NaN or inf found in model predictions. "
                "This is the primary sign of an exploding gradient. "
                "Try lowering your learning rate (`lr0`)."
            )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        local_strides = torch.tensor([imgsz[0] / f.shape[2] for f in feats], device=self.device)
        (anchor_points, stride_tensor) = make_anchors(feats, local_strides, 0.5)
        targets_preprocessed = self.preprocess(
            targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        (gt_labels, gt_bboxes) = targets_preprocessed.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pixel_anchor_points = anchor_points * stride_tensor
        pixel_pred_dist = pred_distri.detach()
        if self.use_dfl:
            (b, a, c) = pixel_pred_dist.shape
            pixel_pred_dist = (
                pixel_pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                    self.proj.type(pixel_pred_dist.dtype)
                )
            )
        pixel_pred_dist *= stride_tensor
        pred_bboxes_for_assigner = dist2bbox(pixel_pred_dist, pixel_anchor_points, xywh=False)
        (target_labels, target_bboxes, target_scores, fg_mask) = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_bboxes /= stride_tensor

        target_scores_sum = target_scores.sum().clamp(min=1)

        loss[1] = (self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum)
        if fg_mask.sum():
            pred_bboxes_decoded = self.bbox_decode(anchor_points, pred_distri)
            (loss[0], loss[2]) = self.bbox_loss(
                pred_distri,
                pred_bboxes_decoded,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        return loss

    def __call__(self, p, targets, img=None, epoch=0):
        feats = p[1] if isinstance(p, tuple) else p
        imgsz = (
            torch.tensor(feats[0][0].shape[2:], device=self.device, dtype=torch.float32) *
            self.stride[0]
        )
        loss_aux = self._calculate_loss_for_branch(feats=feats[0], targets=targets, imgsz=imgsz)
        loss_main = self._calculate_loss_for_branch(feats=feats[1], targets=targets, imgsz=imgsz)
        loss = torch.zeros(3, device=self.device)
        loss[0] = 0.25 * loss_aux[0] + 1.0 * loss_main[0]
        loss[1] = 0.25 * loss_aux[1] + 1.0 * loss_main[1]
        loss[2] = 0.25 * loss_aux[2] + 1.0 * loss_main[2]
        loss[0] *= self.hyp.get("box", 7.5)
        loss[1] *= self.hyp.get("cls", 0.5)
        loss[2] *= self.hyp.get("dfl", 1.5)
        batch_size = feats[0][0].shape[0]
        return (loss.sum(), loss.detach())