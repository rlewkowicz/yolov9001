"""
Model validation metrics.
"""
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import torch

from utils.box_iou import pairwise_box_iou  # at top
from utils.logging import get_logger

def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Delegates to unified IoU implementation in utils.box_iou.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values.
    """
    return pairwise_box_iou(box1, box2, eps=eps)

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec

@torch.no_grad()
def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear 1D interpolation akin to numpy.interp, vectorized in torch.
    Assumes xp is 1D ascending. Clamps outside-range to end values.
    """
    x = x.flatten()
    xp = xp.flatten().contiguous()
    fp = fp.flatten().contiguous()
    n = xp.numel()
    if n == 0:
        return torch.zeros_like(x)
    idx = torch.searchsorted(xp, x, right=True)
    idx0 = (idx - 1).clamp(min=0, max=n - 1)
    idx1 = idx.clamp(min=0, max=n - 1)
    x0 = xp[idx0]
    x1 = xp[idx1]
    y0 = fp[idx0]
    y1 = fp[idx1]
    denom = (x1 - x0).clamp_min(1e-12)
    w = torch.where((x1 > x0), (x - x0) / denom, torch.zeros_like(x))
    return y0 + w * (y1 - y0)

@torch.no_grad()
def _compute_ap_torch(
    recall: torch.Tensor, precision: torch.Tensor, eps: float = 1e-16
) -> torch.Tensor:
    """Compute AP using precision envelope and 101-point interpolation entirely in torch.
    recall, precision: 1D tensors
    Returns: scalar AP (torch scalar on same device)
    """
    device = recall.device
    if recall.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=torch.float64)
    recall = recall.to(torch.float64)
    precision = precision.to(torch.float64)
    mrec = torch.cat([
        torch.tensor([0.0], device=device, dtype=torch.float64), recall,
        torch.tensor([1.0], device=device, dtype=torch.float64)
    ])
    mpre = torch.cat([
        torch.tensor([1.0], device=device, dtype=torch.float64), precision,
        torch.tensor([0.0], device=device, dtype=torch.float64)
    ])
    mpre = torch.flip(torch.cummax(torch.flip(mpre, dims=[0]), dim=0).values, dims=[0])
    x = torch.linspace(0, 1, 101, device=device, dtype=torch.float64)
    y = _interp1d(x, mrec, mpre)
    ap = torch.trapz(y, x)
    return ap

def process_batch(
    detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor
) -> torch.Tensor:
    """
    Ultralytics-equivalent correctness computation per prediction across T IoU thresholds.
    detections: [N, 6] -> [x1, y1, x2, y2, conf, cls]
    labels    : [M, 5] -> [cls, x1, y1, x2, y2]
    iouv      : [T]    -> IoU thresholds in ascending order
    Returns:
      correct: [N, T] bool
    """
    device = detections.device
    T = int(iouv.numel())
    assert iouv.dim() == 1 and T > 0, "iouv must be 1D with at least one threshold"
    assert bool(torch.all(iouv[:-1] <= iouv[1:])), "iouv must be sorted ascending"
    correct = torch.zeros((detections.shape[0], T), dtype=torch.bool, device=device)
    if labels.shape[0] == 0 or detections.shape[0] == 0:
        return correct

    iou = box_iou(labels[:, 1:], detections[:, :4])
    y, x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
                      )  # y=gt idx, x=pred idx
    if y.numel():
        ious = iou[y, x]
        order_iou = torch.argsort(ious, descending=True)
        y, x, ious = y[order_iou], x[order_iou], ious[order_iou]
        matches = torch.stack((y, x, ious), dim=1)  # [K,3]

        K = matches.shape[0]
        rank = torch.arange(K, device=device)  # rank by IoU-desc order
        key_pred = matches[:, 1].to(torch.long) * K + rank
        ord_pred = torch.argsort(key_pred)
        pred_sorted = matches[ord_pred]
        first_mask = torch.ones_like(ord_pred, dtype=torch.bool, device=device)
        first_mask[1:] = pred_sorted[1:, 1] != pred_sorted[:-1, 1]
        keep_pred_idx = ord_pred[first_mask]
        kept_rank = rank[keep_pred_idx]
        ord_restore = torch.argsort(kept_rank)
        matches = matches[keep_pred_idx[ord_restore]]

        K2 = matches.shape[0]
        rank2 = torch.arange(K2, device=device)  # rank within current subset, still IoU-desc
        key_gt = matches[:, 0].to(torch.long) * K2 + rank2
        ord_gt = torch.argsort(key_gt)
        gt_sorted = matches[ord_gt]
        first_mask2 = torch.ones_like(ord_gt, dtype=torch.bool, device=device)
        first_mask2[1:] = gt_sorted[1:, 0] != gt_sorted[:-1, 0]
        keep_gt_idx = ord_gt[first_mask2]
        matches = matches[keep_gt_idx]

        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def process_batch_batched(
    detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor
) -> torch.Tensor:
    """
    Batched correctness computation across T IoU thresholds.
    detections: [K, 7] -> [x1, y1, x2, y2, conf, cls, batch]
    labels    : [L, 6] -> [batch, cls, x1, y1, x2, y2]
    iouv      : [T]    -> IoU thresholds in ascending order
    Returns:
      correct: [K, T] bool, aligned with detections rows
    """
    device = detections.device
    T = int(iouv.numel())
    assert iouv.dim() == 1 and T > 0, "iouv must be 1D with at least one threshold"
    assert bool(torch.all(iouv[:-1] <= iouv[1:])), "iouv must be sorted ascending"

    if detections.numel() == 0:
        return torch.zeros((0, T), dtype=torch.bool, device=device)

    assert detections.shape[1] == 7, "detections must be [x1,y1,x2,y2,conf,cls,batch]"
    assert labels.shape[1] == 6, "labels must be [batch,cls,x1,y1,x2,y2]"

    det_b = detections[:, 6].to(torch.long)
    det_cls = detections[:, 5].to(torch.long)
    correct = torch.zeros((detections.shape[0], T), dtype=torch.bool, device=device)

    if labels.numel() == 0:
        return correct

    lab_b = labels[:, 0].to(torch.long)
    lab_cls = labels[:, 1].to(torch.long)

    uniq = torch.unique(torch.cat([det_b, lab_b], dim=0))
    for b in uniq.tolist():
        det_mask = det_b == b
        lab_mask = lab_b == b
        if not det_mask.any():
            continue
        d_b = detections[det_mask]
        if not lab_mask.any():
            continue
        l_b = labels[lab_mask]
        iou = box_iou(l_b[:, 2:6], d_b[:, 0:4])
        y, x = torch.where((iou >= iouv[0]) & (l_b[:, 1:2] == d_b[:, 5].view(1, -1)))
        if y.numel():
            ious = iou[y, x]
            order_iou = torch.argsort(ious, descending=True)
            y, x, ious = y[order_iou], x[order_iou], ious[order_iou]
            matches = torch.stack((y, x, ious), dim=1)

            K = matches.shape[0]
            rank = torch.arange(K, device=device)
            key_pred = matches[:, 1].to(torch.long) * K + rank
            ord_pred = torch.argsort(key_pred)
            pred_sorted = matches[ord_pred]
            first_mask = torch.ones_like(ord_pred, dtype=torch.bool, device=device)
            first_mask[1:] = pred_sorted[1:, 1] != pred_sorted[:-1, 1]
            keep_pred_idx = ord_pred[first_mask]
            kept_rank = rank[keep_pred_idx]
            ord_restore = torch.argsort(kept_rank)
            matches = matches[keep_pred_idx[ord_restore]]

            K2 = matches.shape[0]
            rank2 = torch.arange(K2, device=device)
            key_gt = matches[:, 0].to(torch.long) * K2 + rank2
            ord_gt = torch.argsort(key_gt)
            gt_sorted = matches[ord_gt]
            first_mask2 = torch.ones_like(ord_gt, dtype=torch.bool, device=device)
            first_mask2[1:] = gt_sorted[1:, 0] != gt_sorted[:-1, 0]
            keep_gt_idx = ord_gt[first_mask2]
            matches = matches[keep_gt_idx]

            det_idx_global = torch.nonzero(det_mask, as_tuple=False).squeeze(1)
            global_x = det_idx_global[matches[:, 1].long()]
            correct[global_x] = matches[:, 2:3] >= iouv
    return correct

@torch.no_grad()
def ap_per_class(
    tp: torch.Tensor,
    conf: torch.Tensor,
    pred_cls: torch.Tensor,
    target_cls: torch.Tensor,
    eps: float = 1e-16
):
    """
    Pure-PyTorch implementation (GPU-native) of AP/PR computation.
    Args:
      tp: [N, T] bool tensor
      conf: [N] float tensor
      pred_cls: [N] long tensor
      target_cls: [M] long tensor
    Returns (torch tensors on same device):
      p, r, p_best, r_best, f1_best, ap (nc x T), unique_classes (nc,)
    """
    input_was_numpy = False
    if isinstance(tp, np.ndarray):
        input_was_numpy = True
        tp = torch.from_numpy(tp)
        conf = torch.from_numpy(conf)
        pred_cls = torch.from_numpy(pred_cls)
        target_cls = torch.from_numpy(target_cls)
    device = conf.device if isinstance(conf, torch.Tensor) else torch.device('cpu')
    if tp.numel() == 0 or target_cls.numel() == 0:
        T = tp.shape[1] if tp.ndim == 2 and tp.shape[1] else 1
        unique_classes = torch.unique(target_cls.to(torch.long))
        nc = int(unique_classes.numel())
        if nc == 0:
            return (
                torch.zeros(0, dtype=torch.float64,
                            device=device), torch.zeros(0, dtype=torch.float64, device=device),
                torch.zeros(0, dtype=torch.float64,
                            device=device), torch.zeros(0, dtype=torch.float64, device=device),
                torch.zeros(0, dtype=torch.float64,
                            device=device), torch.zeros((0, T), dtype=torch.float64,
                                                        device=device), unique_classes
            )

    i = torch.argsort(conf, descending=True)
    tp = tp[i]
    conf = conf[i]
    pred_cls = pred_cls[i].to(torch.long)

    unique_classes, nt = torch.unique(target_cls.to(torch.long), return_counts=True)
    nc = int(unique_classes.numel())
    T = int(tp.shape[1])

    ap = torch.zeros((nc, T), dtype=torch.float64, device=device)
    px = torch.linspace(0, 1, 1000, device=device, dtype=torch.float64)
    p_curve = torch.zeros((nc, px.numel()), dtype=torch.float64, device=device)
    r_curve = torch.zeros((nc, px.numel()), dtype=torch.float64, device=device)
    f1_curve = torch.zeros((nc, px.numel()), dtype=torch.float64, device=device)

    for ci, c in enumerate(unique_classes.tolist()):
        m = (pred_cls == int(c))
        n_l = int(nt[ci].item())
        n_p = int(m.sum().item())
        if n_p == 0 or n_l == 0:
            continue
        tp_ci = tp[m]
        tpc = tp_ci.to(torch.float64).cumsum(0)
        fpc = (~tp_ci).to(torch.float64).cumsum(0)
        recall_ci = tpc / (n_l + eps)
        precision_ci = tpc / (tpc + fpc + eps)

        for j in range(T):
            if tpc.shape[0] > 0 and (tpc[-1, j] >= (n_l - 1e-12)) and (fpc[-1, j] <= 1e-12):
                ap[ci, j] = 1.0
            else:
                ap[ci, j] = _compute_ap_torch(recall_ci[:, j], precision_ci[:, j])

        if recall_ci.shape[0] > 0:
            r_curve[ci] = _interp1d(px, recall_ci[:, 0], recall_ci[:, 0]).clamp(0.0, 1.0)
            p_curve[ci] = _interp1d(px, recall_ci[:, 0], precision_ci[:, 0]).clamp(0.0, 1.0)
            denom = (p_curve[ci] + r_curve[ci]).clamp_min(1e-12)
            f1_curve[ci] = 2.0 * p_curve[ci] * r_curve[ci] / denom

    i_best = int(torch.argmax(f1_curve.mean(0)).item())
    p_best = p_curve[:, i_best]
    r_best = r_curve[:, i_best]
    f1_best = f1_curve[:, i_best]

    p = p_best.clone()
    r = r_best.clone()
    if input_was_numpy:
        return (
            p.detach().cpu().numpy(),
            r.detach().cpu().numpy(),
            p_best.detach().cpu().numpy(),
            r_best.detach().cpu().numpy(),
            f1_best.detach().cpu().numpy(),
            ap.detach().cpu().numpy(),
            unique_classes.detach().cpu().numpy().astype(int),
        )
    return p, r, p_best, r_best, f1_best, ap, unique_classes

class Metric:
    def __init__(self) -> None:
        self.p = torch.empty(0, dtype=torch.float64)
        self.r = torch.empty(0, dtype=torch.float64)
        self.f1 = torch.empty(0, dtype=torch.float64)
        self.all_ap = torch.empty(0, 0, dtype=torch.float64)
        self.ap_class_index = torch.empty(0, dtype=torch.long)
        self.nc = 0
        self.iouv = None  # stores IoU thresholds as a 1D torch tensor

    def _idx_for_iou(self, target_iou: float) -> int:
        assert self.iouv is not None and (len(self.iouv) > 0), "Metric.iouv not set"
        if isinstance(self.iouv, np.ndarray):
            return int(np.argmin(np.abs(self.iouv - float(target_iou))))
        else:
            d = torch.abs(self.iouv.to(torch.float64) - float(target_iou))
            return int(torch.argmin(d).item())

    @property
    def ap50(self):
        if (isinstance(self.all_ap, torch.Tensor) and self.all_ap.numel() == 0) or \
           (isinstance(self.all_ap, np.ndarray) and self.all_ap.size == 0):
            return np.array([], dtype=np.float64)
        j = self._idx_for_iou(0.50)
        A = self.all_ap
        if isinstance(A, torch.Tensor):
            return A[:, j].detach().cpu().numpy()
        return A[:, j]

    @property
    def ap(self):
        A = self.all_ap
        if isinstance(A, torch.Tensor):
            return A.mean(1).detach().cpu().numpy() if A.numel() else np.array([], dtype=np.float64)
        else:
            return A.mean(1) if A.size else np.array([], dtype=np.float64)

    @property
    def mp(self):
        P = self.p
        if isinstance(P, torch.Tensor):
            return float(P.mean().item()) if P.numel() else 0.0
        else:
            return float(P.mean()) if P.size else 0.0

    @property
    def mr(self):
        R = self.r
        if isinstance(R, torch.Tensor):
            return float(R.mean().item()) if R.numel() else 0.0
        else:
            return float(R.mean()) if R.size else 0.0

    @property
    def map50(self):
        A = self.all_ap
        if (isinstance(A, torch.Tensor) and
            A.numel() == 0) or (isinstance(A, np.ndarray) and A.size == 0):
            return 0.0
        j = self._idx_for_iou(0.50)
        if isinstance(A, torch.Tensor):
            return float(A[:, j].mean().item())
        else:
            return float(A[:, j].mean())

    @property
    def map75(self):
        A = self.all_ap
        if (isinstance(A, torch.Tensor) and
            A.numel() == 0) or (isinstance(A, np.ndarray) and A.size == 0):
            return 0.0
        j = self._idx_for_iou(0.75)
        if isinstance(A, torch.Tensor):
            return float(A[:, j].mean().item())
        else:
            return float(A[:, j].mean())

    @property
    def map(self):
        A = self.all_ap
        if isinstance(A, torch.Tensor):
            return float(A.mean().item()) if A.numel() else 0.0
        else:
            return float(A.mean()) if A.size else 0.0

    def mean_results(self):
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int):
        p = self.p[i].item() if isinstance(self.p, torch.Tensor) else self.p[i]
        r = self.r[i].item() if isinstance(self.r, torch.Tensor) else self.r[i]
        ap50 = self.ap50[i]
        ap = self.ap[i]
        return float(p), float(r), float(ap50), float(ap)

    @property
    def maps(self):
        if isinstance(self.all_ap, torch.Tensor):
            base = float(self.all_ap.mean().item()) if self.all_ap.numel() else 0.0
        else:
            base = float(self.all_ap.mean()) if self.all_ap.size else 0.0
        maps = np.zeros(self.nc, dtype=np.float64) + base
        idx = self.ap_class_index.detach().cpu().numpy() if isinstance(
            self.ap_class_index, torch.Tensor
        ) else np.asarray(self.ap_class_index, dtype=int)
        ap_vals = self.ap
        for i, c in enumerate(idx.tolist() if hasattr(idx, 'tolist') else idx):
            maps[int(c)] = float(ap_vals[i])
        return maps

    def fitness(self):
        w = torch.tensor([0.0, 0.0, 0.1, 0.9], dtype=torch.float64)
        vals = torch.tensor(self.mean_results(), dtype=torch.float64)
        return float((vals * w).sum().item())

    def update(self, results: tuple):
        (p, r, f1, all_ap, ap_class_index) = results[2:7]
        self.p = p
        self.r = r
        self.f1 = f1
        self.all_ap = all_ap
        self.ap_class_index = ap_class_index

def compute_confusion_matrix(
    pred_cls: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_mat: torch.Tensor,
    iou_thr: float = 0.5,
    nc: int = 80
) -> torch.Tensor:
    """
    Simple confusion matrix:
    - For each gt, find the best pred of the same class above IoU threshold.
    - Count TP on matched pairs; FP are unmatched preds; FN are unmatched gts.
    - Last row/col are often background (optional convention).
    Returns: (nc+1, nc+1) long tensor where last index is 'background'.
    """
    device = pred_cls.device
    cm = torch.zeros(nc + 1, nc + 1, dtype=torch.int64, device=device)
    if gt_cls.numel() == 0 and pred_cls.numel() == 0:
        return cm
    if gt_cls.numel() == 0:
        if pred_cls.numel():
            counts = torch.bincount(pred_cls.long(), minlength=nc)
            cm[nc, :nc] += counts.to(cm.dtype)
        return cm
    if pred_cls.numel() == 0:
        counts = torch.bincount(gt_cls.long(), minlength=nc)
        cm[:nc, nc] += counts.to(cm.dtype)
        return cm

    iou = iou_mat.clone().T.contiguous()  # [G, P]
    G, P = iou.shape
    cls_eq = (gt_cls.view(-1, 1) == pred_cls.view(1, -1))
    iou[~cls_eq] = -1.0

    used_g = torch.zeros(G, dtype=torch.bool, device=device)
    used_p = torch.zeros(P, dtype=torch.bool, device=device)
    while True:
        max_iou, best_p = iou.max(dim=1)  # [G]
        ok = (max_iou >= float(iou_thr)) & (~used_g)
        if not ok.any():
            break
        rows = torch.where(ok)[0]
        for r in rows.tolist():
            j = int(best_p[r])
            if used_p[j]:
                continue
            g = int(gt_cls[r].item())
            p = int(pred_cls[j].item())
            cm[g, p] += 1
            used_g[r] = True
            used_p[j] = True
            iou[r, :] = -1.0
            iou[:, j] = -1.0

    if (~used_p).any():
        counts_fp = torch.bincount(pred_cls[~used_p].long(), minlength=nc)
        cm[nc, :nc] += counts_fp.to(cm.dtype)
    if (~used_g).any():
        counts_fn = torch.bincount(gt_cls[~used_g].long(), minlength=nc)
        cm[:nc, nc] += counts_fn.to(cm.dtype)

    return cm

class DetMetrics:
    def __init__(self, names: Dict[int, str] = {}, num_iou_thrs: int = 10) -> None:
        self.names = names
        self.box = Metric()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        self.num_iou_thrs = num_iou_thrs
        self.set_iouv(torch.linspace(0.5, 0.95, num_iou_thrs))

    def set_iouv(self, iouv: torch.Tensor):
        """Set IoU thresholds with sorted-unique guard."""
        iouv = iouv.flatten().to(torch.float32)
        iouv = torch.unique(iouv)
        iouv, _ = torch.sort(iouv)
        assert iouv.numel() > 0, "iouv must have at least one threshold"
        self.iouv = iouv

    def process(self):
        if not self.stats['tp']:
            return
        if len(set(len(v) for v in self.stats.values())) > 1:
            logger = get_logger()
            lengths = {k: len(v) for k, v in self.stats.items()}
            logger.debug(
                "DetMetrics.process", f"Inconsistent stats lengths, skipping update: {lengths}"
            )
            return
        stats = {
            k: torch.cat(v, 0) if len(v) else torch.empty(0, device=self.iouv.device)
            for k, v in self.stats.items()
        }
        if stats["tp"].numel() == 0:
            return
        results = ap_per_class(
            stats["tp"].to(torch.bool), stats["conf"], stats["pred_cls"], stats["target_cls"]
        )
        self.box.nc = len(self.names)
        self.box.update(results)
        self.box.iouv = self.iouv

        if getattr(self.box, "all_ap", None) is not None and self.box.all_ap.numel():
            avg_per_thr = self.box.all_ap.mean(0)
            idx50 = self.box._idx_for_iou(0.50)
            if float(avg_per_thr[idx50].item()) + 1e-6 < float(avg_per_thr.mean().item()):
                msg = {
                    "msg":
                        "AP @0.5 lower than mean(AP @0.5:0.95) — threshold mapping likely out-of-sync",
                    "ap_per_thr":
                        avg_per_thr.detach().cpu().tolist(),
                    "iouv":
                        self.box.iouv.detach().cpu().tolist()
                        if hasattr(self.box, "iouv") and self.box.iouv is not None else None,
                    "idx_for_0.5":
                        int(idx50),
                }
                get_logger().warning("val/metric_sanity", msg)

    @property
    def keys(self) -> List[str]:
        """Default keys for scalar logging. vulture: ignore[unused-property]"""
        return [
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i: int):
        return self.box.class_result(i)

    @property
    def maps(self):
        return self.box.maps

    @property
    def fitness(self):
        return self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

class ConfusionMatrixTorch:
    """
    Accumulates a confusion matrix for detection in the same convention used here:
      - Matrix shape: (nc+1, nc+1), last index is background.
      - Rows are TRUE classes; columns are PREDICTED classes.
      - A matched pair increments cm[gt, pred]; unmatched preds increment cm[nc, pred] (FP),
        unmatched gts increment cm[gt, nc] (FN).
    All ops are torch-only. Matching is class-aware and IoU-thresholded.
    """
    def __init__(
        self,
        nc: int,
        iou_thr: float = 0.5,
        conf: float = 0.0,
        device: Optional[torch.device] = None
    ):
        self.nc = int(nc)
        self.iou_thr = float(iou_thr)
        self.conf = float(conf)
        self.device = torch.device("cpu") if device is None else device
        self._cm = torch.zeros(self.nc + 1, self.nc + 1, dtype=torch.int64, device=self.device)

    def reset(self):
        self._cm.zero_()

    @torch.no_grad()
    def update(
        self,
        pred_boxes: torch.Tensor,  # [P,4] xyxy
        pred_scores: torch.Tensor,  # [P]
        pred_cls: torch.Tensor,  # [P] long
        gt_boxes: torch.Tensor,  # [G,4] xyxy
        gt_cls: torch.Tensor  # [G] long
    ):
        if pred_boxes is None or pred_boxes.numel() == 0:
            pred_boxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
            pred_scores = torch.empty((0, ), device=self.device, dtype=torch.float32)
            pred_cls = torch.empty((0, ), device=self.device, dtype=torch.long)
        if gt_boxes is None or gt_boxes.numel() == 0:
            gt_boxes = torch.empty((0, 4), device=self.device, dtype=torch.float32)
            gt_cls = torch.empty((0, ), device=self.device, dtype=torch.long)

        assert pred_boxes.shape[-1] == 4 and gt_boxes.shape[-1] == 4, "Boxes must be [*,4] xyxy"
        pred_boxes = pred_boxes.to(self.device)
        pred_scores = pred_scores.to(self.device)
        pred_cls = pred_cls.to(self.device).long().clamp_min(0)
        gt_boxes = gt_boxes.to(self.device)
        gt_cls = gt_cls.to(self.device).long().clamp_min(0)

        if self.conf > 0.0 and pred_scores.numel():
            keep = pred_scores >= self.conf
            if keep.any():
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_cls = pred_cls[keep]
            else:
                pred_boxes = pred_boxes[:0]
                pred_scores = pred_scores[:0]
                pred_cls = pred_cls[:0]

        if pred_cls.numel():
            pred_cls = pred_cls.clamp_max(self.nc - 1)
        if gt_cls.numel():
            gt_cls = gt_cls.clamp_max(self.nc - 1)

        if pred_boxes.numel() and gt_boxes.numel():
            iou_mat = box_iou(pred_boxes, gt_boxes)
        else:
            iou_mat = torch.empty((pred_boxes.shape[0], gt_boxes.shape[0]), device=self.device)

        cm = compute_confusion_matrix(pred_cls, gt_cls, iou_mat, iou_thr=self.iou_thr, nc=self.nc)
        self._cm += cm.to(self._cm.dtype)

    @property
    def array(self) -> torch.Tensor:
        return self._cm

    def cpu(self) -> "ConfusionMatrixTorch":
        self._cm = self._cm.cpu()
        self.device = torch.device("cpu")
        return self

    def to(self, device: torch.device) -> "ConfusionMatrixTorch":
        self._cm = self._cm.to(device)
        self.device = device
        return self

@torch.no_grad()
def process_batch_area_bins(
    detections: torch.Tensor,  # [N,6]: x1,y1,x2,y2,conf,cls
    labels: torch.Tensor,  # [M,5]: cls,x1,y1,x2,y2
    iouv: torch.Tensor,  # [T]
    label_bin_ids: torch.Tensor,  # [M] long in [0,B)
    num_bins: int,
) -> torch.Tensor:
    """
    Compute correctness per IoU threshold for each area bin in a single image.
    Returns: [B, N, T] bool tensor.
    """
    device = detections.device
    T = int(iouv.numel())
    N = detections.shape[0]
    B = int(num_bins)
    out = torch.zeros((B, N, T), dtype=torch.bool, device=device)
    if labels.shape[0] == 0 or N == 0:
        return out
    iou = box_iou(labels[:, 1:5], detections[:, :4])  # [M,N]
    min_thr = float(iouv[0].item())
    m = (iou >= min_thr) & (labels[:, 0:1].long() == detections[:, 5].long())
    y, x = torch.where(m)
    if y.numel() == 0:
        return out
    ious = iou[y, x]
    bins = label_bin_ids[y].long().clamp_min(0)
    order = torch.argsort(ious, descending=True)
    y = y[order]
    x = x[order]
    ious = ious[order]
    bins = bins[order]
    K = y.numel()
    rank = torch.arange(K, device=device)
    Npred = N
    Ngt = labels.shape[0]
    key_pred = bins * Npred + x
    ord_pred = torch.argsort(key_pred * K + rank)
    sorted_pred = key_pred[ord_pred]
    first_pred = torch.ones_like(sorted_pred, dtype=torch.bool)
    first_pred[1:] = sorted_pred[1:] != sorted_pred[:-1]
    keep_pred_idx = ord_pred[first_pred]
    kept_rank = rank[keep_pred_idx]
    ord_restore = torch.argsort(kept_rank)
    y = y[keep_pred_idx[ord_restore]]
    x = x[keep_pred_idx[ord_restore]]
    ious = ious[keep_pred_idx[ord_restore]]
    bins = bins[keep_pred_idx[ord_restore]]
    K2 = y.numel()
    if K2 == 0:
        return out
    rank2 = torch.arange(K2, device=device)
    key_gt = bins * Ngt + y
    ord_gt = torch.argsort(key_gt * K2 + rank2)
    sorted_gt = key_gt[ord_gt]
    first_gt = torch.ones_like(sorted_gt, dtype=torch.bool)
    first_gt[1:] = sorted_gt[1:] != sorted_gt[:-1]
    keep_gt_idx = ord_gt[first_gt]
    y = y[keep_gt_idx]
    x = x[keep_gt_idx]
    ious = ious[keep_gt_idx]
    bins = bins[keep_gt_idx]
    thr_ok = ious.view(-1, 1) >= iouv.view(1, -1)  # [K', T]
    out[bins, x] = thr_ok
    return out

@torch.no_grad()
def process_batch_area_bins_batched(
    detections: torch.Tensor,  # [K,7]: x1,y1,x2,y2,conf,cls,batch
    labels: torch.Tensor,  # [L,6]: batch,cls,x1,y1,x2,y2
    iouv: torch.Tensor,  # [T]
    label_bin_ids: torch.Tensor,  # [L] long bin id
    num_bins: int,
) -> torch.Tensor:
    """
    Batched variant: computes [B, K, T] correctness, preserving detection indexing.
    """
    device = detections.device
    B = int(num_bins)
    K = detections.shape[0]
    T = int(iouv.numel())
    out = torch.zeros((B, K, T), dtype=torch.bool, device=device)
    if K == 0:
        return out
    if labels.numel() == 0:
        return out
    det_b = detections[:, 6].long()
    lab_b = labels[:, 0].long()
    uniq = torch.unique(torch.cat([det_b, lab_b], 0))
    for b in uniq.tolist():
        det_mask = det_b == b
        lab_mask = lab_b == b
        if not det_mask.any():
            continue
        idx_det_global = torch.nonzero(det_mask, as_tuple=False).squeeze(1)
        d_b = detections[det_mask][:, :6]
        if not lab_mask.any():
            continue
        l_b = labels[lab_mask][:, 1:6]
        bins_b = label_bin_ids[lab_mask]
        out_b = process_batch_area_bins(d_b, l_b, iouv, bins_b, num_bins=B)
        out[:, idx_det_global, :] = out_b
    return out

def plot_confusion_matrix(
    cm: torch.Tensor,
    names: Optional[List[str]] = None,
    normalize: Optional[Literal["true", "pred",
                                None]] = "true",  # "true"(row), "pred"(col), or None
    title: str = "Confusion Matrix"
):
    """
    Minimal plotting helper; imports matplotlib lazily.
    Rows are True classes, Columns are Predicted classes.

    vulture: ignore[unused-function] — utility for analysis/visualization.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    A = cm.float().cpu().numpy()
    if normalize == "true":
        denom = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / denom
    elif normalize == "pred":
        denom = A.sum(axis=0, keepdims=True) + 1e-12
        A = A / denom
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(A, cmap="Blues", vmin=0.0, vmax=np.nanmax(A) if A.size else 1.0)
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ticks = names + ["background"] if names and len(names) == (cm.shape[0] - 1) else None
    if ticks:
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_yticks(np.arange(len(ticks)))
        ax.set_xticklabels(ticks, rotation=90)
        ax.set_yticklabels(ticks)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def plot_confusion_matrix_pretty(
    cm: torch.Tensor,
    names: Optional[List[str]] = None,
    normalize: Optional[Literal["true", "pred", None]] = "true",
    drop_background: bool = True,
    topk: Optional[int] = None,
    title: str = "Confusion Matrix"
):
    import numpy as np, matplotlib.pyplot as plt
    A = cm.float().cpu().numpy()
    nc = A.shape[0] - 1
    if drop_background:
        A = A[:nc, :nc]
        names_ = names or [str(i) for i in range(nc)]
    else:
        names_ = (names or [str(i) for i in range(nc)]) + ["background"]
    if topk:
        support = A.sum(1)
        idx = np.argsort(-support)[:topk]
        A = A[idx][:, idx]
        names_ = [names_[i] for i in idx]
    if normalize == "true":
        denom = A.sum(axis=1, keepdims=True) + 1e-12
        A = A / denom
        vmax = 1.0
    elif normalize == "pred":
        denom = A.sum(axis=0, keepdims=True) + 1e-12
        A = A / denom
        vmax = 1.0
    else:
        vmax = np.percentile(A, 99) if A.size else 1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(A, vmin=0.0, vmax=vmax, cmap="Blues")
    ax.set_title(title)
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Predicted")
    ax.set_xticks(np.arange(len(names_)))
    ax.set_yticks(np.arange(len(names_)))
    ax.set_xticklabels(names_, rotation=90)
    ax.set_yticklabels(names_)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

def cm_stats(cm: torch.Tensor):
    """Compute per-class precision/recall/FP/FN. vulture: ignore[unused-function]"""
    A = cm.long()
    nc = A.shape[0] - 1
    inter = A[:nc, :nc]
    tp = inter.diag()
    fp = A[nc, :nc]
    fn = A[:nc, nc]
    pred_total = inter.sum(0) + fp
    gt_total = inter.sum(1) + fn
    precision = (tp.float() / pred_total.clamp_min(1)).cpu().numpy()
    recall = (tp.float() / gt_total.clamp_min(1)).cpu().numpy()
    return precision, recall, fp.cpu().numpy(), fn.cpu().numpy()

def normalize_confusion_matrix(
    cm: torch.Tensor,
    mode: Literal["true", "pred"] = "true",
    drop_background: bool = True,
    to_percent: bool = True,
) -> np.ndarray:
    """
    Return a normalized confusion matrix as a NumPy array.
    - mode="true": row-normalize by ground-truth counts (per-class recall distribution)
    - mode="pred": column-normalize by predicted counts (per-class precision distribution)
    - drop_background: exclude the last row/column which contains FP/FN aggregates
    - to_percent: scale to [0,100] percentages for readability
    """

    A = cm.detach().float().cpu().numpy()
    nc = A.shape[0] - 1
    if drop_background:
        A = A[:nc, :nc]
    if mode == "true":
        denom = A.sum(axis=1, keepdims=True) + 1e-12
    elif mode == "pred":
        denom = A.sum(axis=0, keepdims=True) + 1e-12
    else:
        raise ValueError("mode must be 'true' or 'pred'")
    A = A / denom
    if to_percent:
        A = A * 100.0
    return A

def per_class_precision_recall(cm: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper to extract per-class precision and recall (excluding background)."""
    p, r, _, _ = cm_stats(cm)
    return p, r
