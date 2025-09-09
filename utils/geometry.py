"""
utils/geometry.py

Shared geometric utilities for YOLO models.
"""
import torch
from .boxes import BoundingBox

def dfl_expectation(logits, reg_max, tau=1.0, eps=1e-6, bins=None):
    """
    Stable DFL expectation computation with temperature and epsilon for numerical stability.
    
    Args:
        logits: Distribution logits [..., reg_max]
        reg_max: Number of bins
        tau: Temperature for softmax (higher = smoother)
        eps: Small epsilon for numerical stability
        bins: Precomputed bins tensor
    
    Returns:
        expectation: Expected distance values in bin units
    """
    x = logits.float()  # force fp32 for stability
    p = torch.softmax(x / max(tau, 1e-3), dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    b = bins if bins is not None else torch.arange(reg_max, device=x.device, dtype=torch.float32)
    return (p * b).sum(dim=-1)

def decode_distances(anchor_points, stride_tensor, pred_dist, reg_max, tau=1.0, decoder=None):
    """
    Unified DFL distance decoding for training and inference.
    Uses stable dfl_expectation for numerically robust decoding.
    
    Args:
        anchor_points: Anchor points in grid coordinates [N, 2]
        stride_tensor: Stride values per anchor [N, 1]
        pred_dist: Distance distribution logits [bs, N, 4*reg_max]
        reg_max: Number of bins per side
        tau: Temperature for softmax stability
        decoder: DFLDecoder instance to source cached bins from
    
    Returns:
        pred_bboxes: Decoded bounding boxes in xyxy format [bs, N, 4] in pixel space
    """
    bs, na, _ = pred_dist.shape
    logits = pred_dist.reshape(bs, na, 4, reg_max)
    bins = getattr(decoder, "_bins", None) if decoder else None
    dist = dfl_expectation(logits, reg_max, tau=tau, bins=bins).to(torch.float32)

    anchor_points_px = anchor_points * stride_tensor
    dist_px = dist * stride_tensor.view(1, na, 1)

    boxes = BoundingBox.dist2bbox(dist_px, anchor_points_px.unsqueeze(0), xywh=False)
    return boxes.to(torch.float32)

class DFLDecoder:
    def __init__(self, reg_max=16, strides=None, centered=True, device='cuda', tau=1.0):
        self.device = torch.device(device)
        self.centered = centered
        self._strides = torch.as_tensor(
            strides or [8, 16, 32], device=self.device, dtype=torch.float32
        )
        self._reg_max = int(reg_max)
        self.tau = float(tau)
        self._cache = {}
        self._bins_cache = None

    @property
    def strides(self):
        return self._strides

    @strides.setter
    def strides(self, v):
        self._strides = torch.as_tensor(v, device=self.device, dtype=torch.float32)
        self._cache.clear()  # invalidate anchors cache because stride_tensor depends on strides

    @property
    def reg_max(self):
        return self._reg_max

    @reg_max.setter
    def reg_max(self, value):
        value = int(value)
        assert value >= 2, "DFLDecoder.reg_max must be >= 2"
        if self._reg_max != value:
            self._reg_max = value
            self._bins_cache = None  # keeps bins in sync

    @property
    def _bins(self):
        if self._bins_cache is None:
            self._bins_cache = torch.arange(self.reg_max, device=self.device, dtype=torch.float32)
        return self._bins_cache

    def to(self, device):
        self.device = device
        self._strides = self._strides.to(device)
        if self._bins_cache is not None:
            self._bins_cache = self._bins_cache.to(device)
        self._cache.clear()
        return self

    def get_anchors(self, feats_or_shapes):
        """
        Generate anchors from features or feature shapes.
        Handles both tensor inputs and shape tuple inputs for broader compatibility.
        """
        if not feats_or_shapes:
            return torch.empty(0, 2, device=self.device), torch.empty(0, 1, device=self.device)

        if hasattr(feats_or_shapes[0], 'shape'):
            self.device = feats_or_shapes[0].device
            shapes = [f.shape for f in feats_or_shapes]
        else:
            shapes = feats_or_shapes
        dtype = torch.float32
        key = (tuple(shapes), self.centered, "fp32")

        if key in self._cache:
            return self._cache[key]

        anchor_points, stride_tensor = [], []
        for i, shape in enumerate(shapes):
            h, w = shape[-2:]  # Works for both tensor shapes (b,c,h,w) and tuples (h,w)
            stride = self._strides[i]

            sx = torch.arange(w, device=self.device, dtype=dtype)
            sy = torch.arange(h, device=self.device, dtype=dtype)

            if self.centered:
                sx += 0.5
                sy += 0.5

            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=self.device))

        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        self._cache[key] = (anchor_points, stride_tensor)
        return anchor_points, stride_tensor

def _assert_ltrb_order_is_consistent():
    """
    Debug check to ensure LTRB channel ordering is consistent.
    This is a sanity check that can be run during model init to detect
    wrong side order statistically.
    """
    import torch
    from .boxes import BoundingBox
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reg_max = 16
    na = 1024  # anchors
    ax = torch.linspace(32, 608, int(na**0.5), device=device)
    ay = torch.linspace(32, 608, int(na**0.5), device=device)
    gy, gx = torch.meshgrid(ay, ax, indexing='ij')
    anchor_points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)[:na]  # [na,2]

    def one_side_logits(pref_bin):
        x = -2.0 * torch.ones(reg_max, device=device)
        x[pref_bin] = 2.0
        return x

    L = one_side_logits(0)  # small
    T = one_side_logits(0)  # small
    R = one_side_logits(8)  # larger
    B = one_side_logits(8)  # larger

    sides = torch.stack([L, T, R, B], dim=0).reshape(-1)  # [4*reg_max]
    pred = sides.unsqueeze(0).unsqueeze(0).repeat(1, na, 1).contiguous()  # [1,na,4*reg_max]

    from .geometry import dfl_expectation
    logits = pred.reshape(1, na, 4, reg_max)
    dist = dfl_expectation(logits, reg_max, tau=1.0).float()  # [1,na,4]

    boxes = BoundingBox.dist2bbox(dist, anchor_points.unsqueeze(0), xywh=False)  # [1,na,4]
    w = (boxes[..., 2] - boxes[..., 0]).mean()
    h = (boxes[..., 3] - boxes[..., 1]).mean()

    if not (w > 0 and h > 0):
        raise RuntimeError(
            "LTRB order mismatch: decoded width/height not positive; check head channel grouping."
        )
