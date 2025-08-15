import torch

from utils.general import check_version

TORCH_1_10 = check_version(torch.__version__, "1.10.0")

import torch

def make_anchors(feats, strides, grid_cell_offset: float = 0.5):
    device = feats[0].device
    dtype = feats[0].dtype

    if not isinstance(strides, torch.Tensor):
        strides = torch.as_tensor(strides, device=device, dtype=dtype)
    else:
        strides = strides.to(device=device, dtype=dtype)

    anchor_points = []
    stride_tensor = []

    for i, f in enumerate(feats):
        h, w = f.shape[-2], f.shape[-1]            # SymInts, ok for compile
        y = torch.arange(h, device=device)
        x = torch.arange(w, device=device)
        gy, gx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((gx, gy), -1).reshape(-1, 2).to(dtype)
        grid = grid + torch.tensor(grid_cell_offset, device=device, dtype=dtype)

        anchor_points.append(grid)                 # (h*w, 2), cell coords (not scaled)
        s = torch.ones((h * w, 1), device=device, dtype=dtype) * strides[i]
        stride_tensor.append(s)                    # (h*w, 1), per-loc stride (tensorized)

    return torch.cat(anchor_points, 0), torch.cat(stride_tensor, 0)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)
