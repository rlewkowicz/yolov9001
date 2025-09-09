"""
utils/augment.py

Image augmentation utilities for training.
"""
import cv2
from typing import Any, Tuple, Optional, TYPE_CHECKING, List, Dict
import numpy as np
import torch
import math
from dataclasses import dataclass
import torch.nn.functional as F

if TYPE_CHECKING:
    from utils.dataloaders import MultiFormatDataset

@dataclass
class LetterboxMeta:
    r: float  # scale applied to original image
    new_hw: Tuple[int, int]  # (new_h, new_w) before padding
    canvas_hw: Tuple[int, int]  # (Hc, Wc) after padding/stride
    pad: Tuple[int, int, int, int]  # (left, top, right, bottom)

def compute_letterbox(
    h0: int,
    w0: int,
    img_size: int | Tuple[int, int],
    *,
    center: bool = True,
    scaleup: bool = False,
    pad_to_stride: Optional[int] = None
) -> LetterboxMeta:
    if isinstance(img_size, int):
        tgt_h, tgt_w = img_size, img_size
    else:
        tgt_h, tgt_w = img_size

    r = min(tgt_h / h0, tgt_w / w0)
    if not scaleup:
        r = min(r, 1.0)

    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    base_Wc = tgt_w
    base_Hc = tgt_h

    if pad_to_stride is not None and pad_to_stride > 1:
        Wc = int(math.ceil(base_Wc / pad_to_stride) * pad_to_stride)
        Hc = int(math.ceil(base_Hc / pad_to_stride) * pad_to_stride)
    else:
        Wc, Hc = base_Wc, base_Hc

    total_dw = Wc - new_w
    total_dh = Hc - new_h

    if center:
        left = total_dw // 2
        right = total_dw - left
        top = total_dh // 2
        bottom = total_dh - top
    else:
        left, top = 0, 0
        right, bottom = total_dw, total_dh

    return LetterboxMeta(
        r=r,
        new_hw=(new_h, new_w),
        canvas_hw=(Hc, Wc),
        pad=(left, top, right, bottom),
    )

@torch.no_grad()
def _transform_boxes_xyxy_by_H_batched(
    boxes_list: List[torch.Tensor], H: torch.Tensor, out_w: int, out_h: int
):
    device = H.device
    B = H.shape[0]
    counts = [(b.shape[0] if (b is not None and b.numel()) else 0) for b in boxes_list]
    K = int(sum(counts))
    if K == 0:
        empty_b = torch.zeros((0, 4), device=device, dtype=torch.float32)
        empty_k = torch.zeros((0, ), device=device, dtype=torch.bool)
        return [empty_b.clone() for _ in range(B)], [empty_k.clone() for _ in range(B)]

    b_flat = torch.cat([
        boxes_list[i].to(torch.float32) if counts[i] > 0 else torch.zeros((0, 4), device=device)
        for i in range(B)
    ],
                       dim=0).to(device)
    batch_id = torch.cat([
        torch.full((counts[i], ), i, device=device, dtype=torch.long)
        for i in range(B) if counts[i] > 0
    ],
                         dim=0)

    corners = torch.stack([
        b_flat[:, [0, 1]], b_flat[:, [2, 1]], b_flat[:, [2, 3]], b_flat[:, [0, 3]]
    ],
                          dim=1)
    ones = torch.ones((K, 4, 1), device=device, dtype=torch.float32)
    corners_h = torch.cat([corners, ones], dim=2)  # [K,4,3]
    H_sel = H[batch_id]  # [K,3,3]
    tc = corners_h @ H_sel.transpose(1, 2)  # [K,4,3]
    z = tc[:, :, 2:3].clamp_min(1e-6)
    xy = tc[:, :, :2] / z  # [K,4,2]

    p = xy
    p_next = torch.roll(p, shifts=-1, dims=1)
    e = p_next - p
    edge_angles = -torch.atan2(e[..., 1], e[..., 0])
    c = torch.cos(edge_angles)
    s = torch.sin(edge_angles)
    px = p[..., 0].unsqueeze(1)
    py = p[..., 1].unsqueeze(1)
    cx = c.unsqueeze(-1)
    sx = s.unsqueeze(-1)
    xr = cx * px - sx * py
    yr = sx * px + cx * py
    xmin, _ = xr.min(dim=2)
    xmax, _ = xr.max(dim=2)
    ymin, _ = yr.min(dim=2)
    ymax, _ = yr.max(dim=2)
    w = (xmax - xmin).clamp_min(0)
    h = (ymax - ymin).clamp_min(0)
    area = w * h
    best_idx = area.argmin(dim=1)

    def gather(batched, idx):
        return batched.gather(1, idx.view(-1, 1)).squeeze(1)

    w_best = gather(w, best_idx)
    h_best = gather(h, best_idx)
    xmin_best = gather(xmin, best_idx)
    xmax_best = gather(xmax, best_idx)
    ymin_best = gather(ymin, best_idx)
    ymax_best = gather(ymax, best_idx)
    ang_best = gather(edge_angles, best_idx)

    c_inv = torch.cos(-ang_best)
    s_inv = torch.sin(-ang_best)
    cx_r = 0.5 * (xmin_best + xmax_best)
    cy_r = 0.5 * (ymin_best + ymax_best)
    half_w = 0.5 * w_best
    half_h = 0.5 * h_best
    corners_r = torch.stack([
        torch.stack([-half_w, -half_h], dim=-1),
        torch.stack([half_w, -half_h], dim=-1),
        torch.stack([half_w, half_h], dim=-1),
        torch.stack([-half_w, half_h], dim=-1),
    ],
                            dim=1)
    cx0 = c_inv * cx_r - s_inv * cy_r
    cy0 = s_inv * cx_r + c_inv * cy_r
    cbi = c_inv.view(-1, 1)
    sbi = s_inv.view(-1, 1)
    xr2 = cbi * corners_r[..., 0] - sbi * corners_r[..., 1]
    yr2 = sbi * corners_r[..., 0] + cbi * corners_r[..., 1]
    xr2 = xr2 + cx0.view(-1, 1)
    yr2 = yr2 + cy0.view(-1, 1)
    x1_outer = xr2.min(dim=1).values
    y1_outer = yr2.min(dim=1).values
    x2_outer = xr2.max(dim=1).values
    y2_outer = yr2.max(dim=1).values

    c_abs = torch.cos(ang_best).abs()
    s_abs = torch.sin(ang_best).abs()
    denom1 = (h_best * c_abs + w_best * s_abs).clamp_min(1e-6)
    denom2 = (w_best * c_abs + h_best * s_abs).clamp_min(1e-6)
    w_in = (w_best * h_best) / denom1
    h_in = (w_best * h_best) / denom2
    x1_inner = cx0 - 0.5 * w_in
    y1_inner = cy0 - 0.5 * h_in
    x2_inner = cx0 + 0.5 * w_in
    y2_inner = cy0 + 0.5 * h_in

    alpha = 0.35
    x1 = x1_outer * (1.0 - alpha) + x1_inner * alpha
    y1 = y1_outer * (1.0 - alpha) + y1_inner * alpha
    x2 = x2_outer * (1.0 - alpha) + x2_inner * alpha
    y2 = y2_outer * (1.0 - alpha) + y2_inner * alpha

    out = torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(out_w))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(out_h))
    wv = out[:, 2] - out[:, 0]
    hv = out[:, 3] - out[:, 1]
    keep = (wv > 1.0) & (hv > 1.0)

    out_boxes_list: List[torch.Tensor] = []
    keep_list: List[torch.Tensor] = []
    start = 0
    for i in range(B):
        n = counts[i]
        if n == 0:
            out_boxes_list.append(torch.zeros((0, 4), device=device, dtype=torch.float32))
            keep_list.append(torch.zeros((0, ), device=device, dtype=torch.bool))
        else:
            out_boxes_list.append(out[start:start + n])
            keep_list.append(keep[start:start + n])
        start += n
    return out_boxes_list, keep_list

@torch.no_grad()
def random_affine_gpu_batch(
    imgs: torch.Tensor,
    boxes_list: List[torch.Tensor],
    imgsz: int,
    degrees,
    translate,
    scale,
    shear,
    pad_value: int,
    hyp: Dict[str, Any] | None = None,
):
    device = imgs.device
    B, C, H, W = imgs.shape

    s_min, s_max = _parse_aug_param(scale)
    d_min, d_max = _parse_aug_param(degrees)
    sh_min, sh_max = _parse_aug_param(shear)
    t_min, t_max = _parse_aug_param(translate)

    s_rand = (1.0 + torch.empty((B, ), device=device).uniform_(s_min, s_max))
    ang = torch.empty((B, ), device=device).uniform_(d_min, d_max)
    shx = torch.empty((B, ), device=device).uniform_(sh_min, sh_max)
    shy = torch.empty((B, ), device=device).uniform_(sh_min, sh_max)
    tx = torch.empty((B, ), device=device).uniform_(t_min, t_max) * float(W)
    ty = torch.empty((B, ), device=device).uniform_(t_min, t_max) * float(H)

    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    Cm = torch.tensor([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], device=device, dtype=torch.float32)
    Cinv = torch.tensor([[1, 0, cx], [0, 1, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    Cm_b = Cm.unsqueeze(0).expand(B, 3, 3)
    Cinv_b = Cinv.unsqueeze(0).expand(B, 3, 3)

    a = torch.deg2rad(ang.to(torch.float32))
    cs, sn = torch.cos(a), torch.sin(a)
    S_mat = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    S_mat[:, 0, 0] = s_rand.to(torch.float32)
    S_mat[:, 1, 1] = s_rand.to(torch.float32)
    S_mat[:, 2, 2] = 1.0

    R_mat = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    R_mat[:, 0, 0] = cs
    R_mat[:, 0, 1] = -sn
    R_mat[:, 1, 0] = sn
    R_mat[:, 1, 1] = cs
    R_mat[:, 2, 2] = 1.0

    shx_t = torch.tan(torch.deg2rad(shx.to(torch.float32)))
    shy_t = torch.tan(torch.deg2rad(shy.to(torch.float32)))
    Sh_mat = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    Sh_mat[:, 0, 0] = 1.0
    Sh_mat[:, 1, 1] = 1.0
    Sh_mat[:, 2, 2] = 1.0
    Sh_mat[:, 0, 1] = shx_t
    Sh_mat[:, 1, 0] = shy_t

    M_affine = Cinv_b @ Sh_mat @ R_mat @ S_mat @ Cm_b
    M_core = M_affine

    src = torch.tensor([[0.0, 0.0, 1.0], [W - 1.0, 0.0, 1.0], [W - 1.0, H - 1.0, 1.0],
                        [0.0, H - 1.0, 1.0]],
                       device=device,
                       dtype=torch.float32).unsqueeze(0)
    tc = src @ M_core.transpose(1, 2)
    zc = tc[:, :, 2:3].clamp_min(1e-6)
    xy = tc[:, :, :2] / zc
    min_xy = xy.min(dim=1).values
    max_xy = xy.max(dim=1).values
    aabb_w = (max_xy[:, 0] - min_xy[:, 0])
    aabb_h = (max_xy[:, 1] - min_xy[:, 1])
    dx_shift = -min_xy[:, 0]
    dy_shift = -min_xy[:, 1]
    dx_center = 0.5 * (float(W) - aabb_w)
    dy_center = 0.5 * (float(H) - aabb_h)
    T_center = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    T_center[:, 0, 0] = 1.0
    T_center[:, 1, 1] = 1.0
    T_center[:, 2, 2] = 1.0
    T_center[:, 0, 2] = dx_shift + dx_center
    T_center[:, 1, 2] = dy_shift + dy_center
    M_centered = T_center @ M_core

    T_last = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
    T_last[:, 0, 0] = 1.0
    T_last[:, 1, 1] = 1.0
    T_last[:, 2, 2] = 1.0
    T_last[:, 0, 2] = tx.to(torch.float32)
    T_last[:, 1, 2] = ty.to(torch.float32)
    M = T_last @ M_centered

    Tn2p = torch.tensor([[float(W) * 0.5, 0.0, float(W) * 0.5 - 0.5],
                         [0.0, float(H) * 0.5, float(H) * 0.5 - 0.5], [0.0, 0.0, 1.0]],
                        device=device,
                        dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
    Tp2n = torch.tensor([[2.0 / float(W), 0.0, -1.0 + 1.0 / float(W)],
                         [0.0, 2.0 / float(H), -1.0 + 1.0 / float(H)], [0.0, 0.0, 1.0]],
                        device=device,
                        dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
    Minv = torch.linalg.inv(M)
    A = Tp2n @ Minv @ Tn2p
    theta = A[:, :2, :]

    grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)
    warped = F.grid_sample(imgs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    pad = float(pad_value) / 255.0
    ones = torch.ones((B, 1, H, W), device=device, dtype=imgs.dtype)
    mask = F.grid_sample(ones, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    imgs_out = warped + pad * (1.0 - mask)

    out_boxes_list, keep_list = _transform_boxes_xyxy_by_H_batched(boxes_list, M, W, H)
    out_boxes_list_filtered = []
    for b, k in zip(out_boxes_list, keep_list):
        out_boxes_list_filtered.append(b[k] if (b is not None and b.numel()) else b)
    return imgs_out, out_boxes_list_filtered, keep_list

@torch.no_grad()
def _grid_from_homography_batch(
    M: torch.Tensor, out_h: int, out_w: int, in_h: int, in_w: int, device: torch.device
):
    B = M.shape[0]
    try:
        Minv = torch.linalg.inv(M)
        if not torch.isfinite(Minv).all():
            raise RuntimeError("Minv contains non-finite values")
    except Exception:
        Minv = torch.linalg.pinv(M)
    ys = torch.arange(out_h, device=device, dtype=torch.float32)
    xs = torch.arange(out_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    ones = torch.ones_like(xx)
    tgt = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3)  # [H*W,3]
    tgt_b = tgt.unsqueeze(0).expand(B, -1, -1)
    src = torch.bmm(tgt_b, Minv.transpose(1, 2))
    z = src[..., 2].clamp_min_(1e-6)
    x = src[..., 0] / z
    y = src[..., 1] / z
    valid = (x >= 0.0) & (x <= float(in_w - 1)) & (y >= 0.0) & (y <= float(in_h - 1))
    x_n = (x + 0.5) / float(in_w) * 2.0 - 1.0
    y_n = (y + 0.5) / float(in_h) * 2.0 - 1.0
    grid = torch.stack([x_n, y_n], dim=-1).reshape(B, out_h, out_w, 2)
    return grid, valid.reshape(B, out_h, out_w)

@torch.no_grad()
def random_perspective_gpu_batch(
    imgs: torch.Tensor,
    boxes_list: List[torch.Tensor],
    perspective: float,
    pad_value: int,
):
    device = imgs.device
    B, C, H, W = imgs.shape
    p_min, p_max = _parse_aug_param(perspective)
    p = torch.empty((B, ), device=device).uniform_(p_min, p_max)
    src = torch.tensor([[0.0, 0.0], [W - 1.0, 0.0], [W - 1.0, H - 1.0], [0.0, H - 1.0]],
                       device=device,
                       dtype=torch.float32)  # [4,2]
    amp = p.abs().view(B, 1, 1)
    jitter = torch.empty((B, 4, 2), device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
    jitter = jitter * amp * torch.tensor([W, H], device=device, dtype=torch.float32)
    dst0 = src.unsqueeze(0) + jitter  # [B,4,2]
    min_xy = dst0.min(dim=1).values
    max_xy = dst0.max(dim=1).values
    aabb = (max_xy - min_xy).clamp_min(1e-6)
    ctr = 0.5 * (min_xy + max_xy)
    s_fit_w = (float(W) - 2.0) / aabb[:, 0]
    s_fit_h = (float(H) - 2.0) / aabb[:, 1]
    s = torch.minimum(torch.minimum(s_fit_w, s_fit_h), torch.ones_like(s_fit_w)).clamp_min(1e-6)
    s = s.view(B, 1, 1)
    dst1 = ctr.view(B, 1, 2) + s * (dst0 - ctr.view(B, 1, 2))
    min_xy1 = dst1.min(dim=1).values
    max_xy1 = dst1.max(dim=1).values
    ctr1 = 0.5 * (min_xy1 + max_xy1)
    canvas_ctr = torch.tensor([(W - 1.0) * 0.5, (H - 1.0) * 0.5],
                              device=device,
                              dtype=torch.float32)
    delta = canvas_ctr.view(1, 2) - ctr1
    dst = dst1 + delta.view(B, 1, 2)
    A = torch.zeros((B, 8, 9), device=device, dtype=torch.float32)
    x = src[:, 0]
    y = src[:, 1]
    for k in range(4):
        u = dst[:, k, 0]
        v = dst[:, k, 1]
        xx = x[k]
        yy = y[k]
        A[:, 2 * k, 0] = xx
        A[:, 2 * k, 1] = yy
        A[:, 2 * k, 2] = 1.0
        A[:, 2 * k, 6] = -u * xx
        A[:, 2 * k, 7] = -u * yy
        A[:, 2 * k, 8] = -u
        A[:, 2 * k + 1, 3] = xx
        A[:, 2 * k + 1, 4] = yy
        A[:, 2 * k + 1, 5] = 1.0
        A[:, 2 * k + 1, 6] = -v * xx
        A[:, 2 * k + 1, 7] = -v * yy
        A[:, 2 * k + 1, 8] = -v
    try:
        _, _, Vh = torch.linalg.svd(A)
        h = Vh[:, -1, :]
    except Exception:
        _, _, Vh = torch.linalg.svd(A.to(torch.float64))
        h = Vh[:, -1, :].to(torch.float32)
    Hm = h.view(B, 3, 3)
    Hm = Hm / Hm[:, 2:3, 2:3].clamp_min(1e-12)
    grid, valid = _grid_from_homography_batch(Hm, H, W, H, W, device)
    warped = F.grid_sample(imgs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    pad = float(pad_value) / 255.0
    img_out = torch.full((B, C, H, W), pad, device=device, dtype=imgs.dtype)
    img_out = torch.where(valid.unsqueeze(1), warped, img_out)
    out_boxes_list, keep_list = _transform_boxes_xyxy_by_H_batched(boxes_list, Hm, W, H)
    out_boxes_list_filtered = []
    for b, k in zip(out_boxes_list, keep_list):
        out_boxes_list_filtered.append(b[k] if (b is not None and b.numel()) else b)
    return img_out, out_boxes_list_filtered, keep_list

@torch.no_grad()
def apply_gpu_affine_on_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    hyp: Dict[str, Any],
    pad_value: int,
):
    """Apply batched GPU ops (HSV, flips, perspective, affine) and rebuild YOLO targets.
    No Python loops over images.
    """
    device = images.device
    B, _, H, W = images.shape
    bi = targets[:, 0].to(torch.long)
    cls_list: List[torch.Tensor] = []
    boxes_list: List[torch.Tensor] = []
    if targets.numel():
        for i in range(B):
            ti = targets[bi == i]
            if ti.numel():
                cls_i = ti[:, 1].to(torch.int64)
                cx, cy, ww, hh = ti[:, 2] * W, ti[:, 3] * H, ti[:, 4] * W, ti[:, 5] * H
                x1 = (cx - ww * 0.5).clamp_(0, W)
                y1 = (cy - hh * 0.5).clamp_(0, H)
                x2 = (cx + ww * 0.5).clamp_(0, W)
                y2 = (cy + hh * 0.5).clamp_(0, H)
                boxes_i = torch.stack([x1, y1, x2, y2], dim=1)
            else:
                cls_i = torch.zeros((0, ), device=device, dtype=torch.int64)
                boxes_i = torch.zeros((0, 4), device=device, dtype=torch.float32)
            cls_list.append(cls_i)
            boxes_list.append(boxes_i)
    else:
        for _ in range(B):
            cls_list.append(torch.zeros((0, ), device=device, dtype=torch.int64))
            boxes_list.append(torch.zeros((0, 4), device=device, dtype=torch.float32))

    def _parse_aug_param(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return float(v[0]), float(v[1])
        v = float(v or 0.0)
        return (-abs(v), abs(v))

    h_hyp = hyp.get('hsv_h', 0.0)
    s_hyp = hyp.get('hsv_s', 0.0)
    v_hyp = hyp.get('hsv_v', 0.0)
    if h_hyp or s_hyp or v_hyp:
        h_min, h_max = _parse_aug_param(h_hyp)
        s_min, s_max = _parse_aug_param(s_hyp)
        v_min, v_max = _parse_aug_param(v_hyp)
        dH = torch.empty((B, 1, 1), device=device).uniform_(h_min, h_max)
        dS = torch.empty((B, 1, 1), device=device).uniform_(s_min, s_max)
        dV = torch.empty((B, 1, 1), device=device).uniform_(v_min, v_max)
        r, g, b = images[:, 0], images[:, 1], images[:, 2]
        maxc, _ = torch.max(images, dim=1)
        minc, _ = torch.min(images, dim=1)
        v = maxc
        s = (maxc - minc) / (maxc + 1e-6)
        rc = (maxc - r) / (maxc - minc + 1e-6)
        gc = (maxc - g) / (maxc - minc + 1e-6)
        bc = (maxc - b) / (maxc - minc + 1e-6)
        h = torch.zeros_like(maxc)
        h = torch.where((maxc == r) & (maxc != minc), (bc - gc) % 6.0, h)
        h = torch.where((maxc == g) & (maxc != minc), (2.0 + rc - bc), h)
        h = torch.where((maxc == b) & (maxc != minc), (4.0 + gc - rc), h)
        h = h / 6.0
        Hh = (h * 180.0)
        Ss = (s * 255.0)
        Vv = (v * 255.0)
        Hh = (Hh + dH * 180.0) % 180.0
        Ss = (Ss * (1.0 + dS)).clamp(0.0, 255.0)
        Vv = (Vv * (1.0 + dV)).clamp(0.0, 255.0)
        h = (Hh / 180.0).clamp(0, 1)
        s = (Ss / 255.0).clamp(0, 1)
        v = (Vv / 255.0).clamp(0, 1)
        i = torch.floor(h * 6.0).to(torch.int64)
        f = h * 6.0 - i.to(h.dtype)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6
        out = torch.zeros_like(images)
        conds = [
            (i == 0, torch.stack([v, t, p], dim=1)),
            (i == 1, torch.stack([q, v, p], dim=1)),
            (i == 2, torch.stack([p, v, t], dim=1)),
            (i == 3, torch.stack([p, q, v], dim=1)),
            (i == 4, torch.stack([t, p, v], dim=1)),
            (i == 5, torch.stack([v, p, q], dim=1)),
        ]
        for cond, rgb in conds:
            out = torch.where(cond.unsqueeze(1), rgb, out)
        images = out.clamp(0.0, 1.0)

    p_lr = float(hyp.get('fliplr', 0.0) or 0.0)
    p_ud = float(hyp.get('flipud', 0.0) or 0.0)
    if p_lr > 0.0:
        m = torch.rand((B, ), device=device) < p_lr
        images[m] = torch.flip(images[m], dims=[3])
        for i in torch.nonzero(m, as_tuple=False).squeeze(1).tolist():
            b = boxes_list[i]
            if b is not None and b.numel():
                x1 = b[:, 0].clone()
                x2 = b[:, 2].clone()
                b[:, 0] = W - x2
                b[:, 2] = W - x1
                boxes_list[i] = b
    if p_ud > 0.0:
        m = torch.rand((B, ), device=device) < p_ud
        images[m] = torch.flip(images[m], dims=[2])
        for i in torch.nonzero(m, as_tuple=False).squeeze(1).tolist():
            b = boxes_list[i]
            if b is not None and b.numel():
                y1 = b[:, 1].clone()
                y2 = b[:, 3].clone()
                b[:, 1] = H - y2
                b[:, 3] = H - y1
                boxes_list[i] = b

    persp = hyp.get('perspective', 0.0)
    if float(persp or 0.0) != 0.0:
        images, boxes_list, keep_list = random_perspective_gpu_batch(
            images, boxes_list, persp, pad_value
        )
        cls_list = [
            c[k] if (c is not None and c.numel()) else c for c, k in zip(cls_list, keep_list)
        ]

    imgs_ai, boxes_ai, keep_list = random_affine_gpu_batch(
        images,
        boxes_list,
        int(H),
        hyp.get('degrees', 0.0),
        hyp.get('translate', 0.0),
        hyp.get('scale', 0.0),
        hyp.get('shear', 0.0),
        int(pad_value),
        hyp=hyp,
    )

    new_t_list = []
    for i in range(B):
        b = boxes_ai[i]
        if b is None or b.numel() == 0:
            continue
        k = keep_list[i]
        if (
            cls_list[i] is not None and cls_list[i].numel() and k is not None and
            hasattr(k, 'numel') and k.numel() == cls_list[i].shape[0]
        ):
            cls_i = cls_list[i][k]
        else:
            cls_i = cls_list[i]
        if cls_i.shape[0] != b.shape[0]:
            n = min(cls_i.shape[0], b.shape[0])
            if n == 0:
                continue
            b = b[:n]
            cls_i = cls_i[:n]
        wv = (b[:, 2] - b[:, 0]).clamp_min(1.0)
        hv = (b[:, 3] - b[:, 1]).clamp_min(1.0)
        cx = b[:, 0] + 0.5 * wv
        cy = b[:, 1] + 0.5 * hv
        yolo = torch.stack([cls_i.to(images.dtype), cx / W, cy / H, wv / W, hv / H], dim=1)
        bi_full = torch.full((yolo.shape[0], 1), i, device=device, dtype=images.dtype)
        new_t_list.append(torch.cat([bi_full, yolo], 1))
    new_targets = torch.cat(new_t_list, 0) if new_t_list else torch.zeros((0, 6),
                                                                          device=device,
                                                                          dtype=images.dtype)
    return imgs_ai, new_targets

def _transform_boxes_homography(boxes_xyxy: np.ndarray, M: np.ndarray, out_w: int,
                                out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform boxes by homography M and clip/filter.
    This version calculates a bounding box by blending between the simple AABB
    of the transformed corners and the largest possible AABB inscribed within the
    oriented bounding box (OBB) of the transformed corners. This provides a

    balance, creating tighter boxes than a simple AABB without being overly
    aggressive and clipping too much of the object.
    Returns (boxes_xyxy_out, keep_mask).
    """
    if boxes_xyxy is None or boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, ), dtype=np.bool_)
    b = boxes_xyxy.astype(np.float32)
    corners = np.array(
        [
            [b[:, 0], b[:, 1]],
            [b[:, 2], b[:, 1]],
            [b[:, 2], b[:, 3]],
            [b[:, 0], b[:, 3]],
        ],
        dtype=np.float32,
    ).transpose(2, 0, 1)  # [N,4,2]
    ones = np.ones((corners.shape[0], 4, 1), dtype=np.float32)
    corners_h = np.concatenate([corners, ones], axis=2)  # [N,4,3]
    tc = corners_h @ M.T  # [N,4,3]
    z = np.clip(tc[:, :, 2:3], 1e-6, None)
    xy = tc[:, :, :2] / z  # [N, 4, 2] transformed corners

    out_boxes = []
    for quad_corners in xy:
        quad_corners_f32 = quad_corners.astype(np.float32)

        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(quad_corners_f32)

        obb_corners = cv2.boxPoints(((cx, cy), (w, h), angle_deg))
        x1_outer, y1_outer = obb_corners.min(axis=0)
        x2_outer, y2_outer = obb_corners.max(axis=0)
        box_outer = np.array([x1_outer, y1_outer, x2_outer, y2_outer])

        angle_rad = np.deg2rad(angle_deg)
        c = np.abs(np.cos(angle_rad))
        s = np.abs(np.sin(angle_rad))

        denom1 = h * c + w * s
        denom2 = w * c + h * s

        if denom1 > 1e-6 and denom2 > 1e-6:
            w_inscribed = (w * h) / denom1
            h_inscribed = (w * h) / denom2
            x1_inner = cx - w_inscribed / 2.0
            y1_inner = cy - h_inscribed / 2.0
            x2_inner = cx + w_inscribed / 2.0
            y2_inner = cy + h_inscribed / 2.0
            box_inner = np.array([x1_inner, y1_inner, x2_inner, y2_inner])
        else:
            box_inner = box_outer.copy()

        alpha = 0.35
        blended_box = box_outer * (1 - alpha) + box_inner * alpha
        out_boxes.append(blended_box)

    if not out_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, ), dtype=np.bool_)

    out = np.array(out_boxes, dtype=np.float32)

    out[:, [0, 2]] = out[:, [0, 2]].clip(0, out_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, out_h)
    wv = out[:, 2] - out[:, 0]
    hv = out[:, 3] - out[:, 1]
    keep = (wv > 1) & (hv > 1)
    return out.astype(np.float32), keep.astype(np.bool_)

def apply_hsv_inplace(img: np.ndarray, h_hyp: Any, s_hyp: Any, v_hyp: Any) -> np.ndarray:
    """
    Applies HSV color jitter.
    h_hyp, s_hyp, v_hyp are parsed by _parse_aug_param to get sampling ranges.

    vulture: ignore[unused-function] — public augmentation API (used in GPU/CPU pipelines and tests).
    """
    if not (h_hyp or s_hyp or v_hyp) or img.size == 0:
        return img

    h_min, h_max = _parse_aug_param(h_hyp)
    s_min, s_max = _parse_aug_param(s_hyp)
    v_min, v_max = _parse_aug_param(v_hyp)

    dh = np.random.uniform(h_min, h_max)
    ds = np.random.uniform(s_min, s_max)
    dv = np.random.uniform(v_min, v_max)

    if all(abs(x) < 1e-6 for x in (dh, ds, dv)):
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    h = (h + dh * 180) % 180.0
    s = np.clip(s * (1 + ds), 0, 255)
    v = np.clip(v * (1 + dv), 0, 255)

    hsv_aug = cv2.merge((h, s, v)).astype(np.uint8)
    out = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2RGB)
    return out

def _parse_aug_param(value: Any) -> Tuple[float, float]:
    """
    Parse augmentation parameter to a (min, max) tuple.
    - List/Tuple: `[-0.5, 0.5]` -> `(-0.5, 0.5)`
    - String: `"+0.5"` -> `(0, 0.5)`; `"-0.5"` -> `(-0.5, 0)`; `"[-0.5, 0.5]"` -> `(-0.5, 0.5)`
    - Number: `0.5` -> `(-0.5, 0.5)`; `-0.5` -> `(-0.5, 0)`
    """
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('+'):
            abs_val = float(value[1:])
            return 0.0, abs_val
        if value.startswith('-'):
            abs_val = float(value[1:])
            return -abs_val, 0.0
        if value.startswith('[') and value.endswith(']'):
            try:
                import ast
                lst = ast.literal_eval(value)
                if isinstance(lst, (list, tuple)) and len(lst) == 2:
                    return float(lst[0]), float(lst[1])
            except (ValueError, SyntaxError):
                pass  # Fallback to float conversion

    val = float(value)
    if val < 0:
        return val, 0.0
    return -val, val

class ExactLetterboxTransform:
    def __init__(
        self,
        img_size: int | Tuple[int, int],
        center: bool = True,
        pad_value: int = 114,
        scaleup: bool = False,
        pad_to_stride: Optional[int] = None,
    ):
        self.img_size = img_size
        self.center = bool(center)
        self.pad_value = int(pad_value)
        self.scaleup = bool(scaleup)
        self.pad_to_stride = pad_to_stride

    def __call__(self, img: np.ndarray, boxes_xyxy_abs: np.ndarray = None, cls: np.ndarray = None):
        (h0, w0) = img.shape[:2]
        meta = compute_letterbox(
            h0,
            w0,
            self.img_size,
            center=self.center,
            scaleup=self.scaleup,
            pad_to_stride=self.pad_to_stride
        )

        r = meta.r
        new_h, new_w = meta.new_hw
        Hc, Wc = meta.canvas_hw
        left, top, right, bottom = meta.pad

        if (new_w, new_h) != (w0, h0):
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        canvas = np.full((Hc, Wc, 3), self.pad_value, dtype=img.dtype)
        canvas[top:top + new_h, left:left + new_w] = resized

        if boxes_xyxy_abs is not None and cls is not None and boxes_xyxy_abs.size > 0:
            xyxy_scaled = boxes_xyxy_abs.copy().astype(np.float32)
            xyxy_scaled[:, [0, 2]] = xyxy_scaled[:, [0, 2]] * r + left
            xyxy_scaled[:, [1, 3]] = xyxy_scaled[:, [1, 3]] * r + top

            x1, y1, x2, y2 = np.split(xyxy_scaled, 4, axis=1)
            w = (x2 - x1)
            h = (y2 - y1)
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5

            yolo = np.concatenate([
                cls.reshape(-1, 1).astype(np.float32),
                (cx / Wc).astype(np.float32),
                (cy / Hc).astype(np.float32),
                (w / Wc).astype(np.float32),
                (h / Hc).astype(np.float32),
            ],
                                  axis=1)

            return canvas, yolo, (h0, w0), r, (left, top, right, bottom)

        return (canvas, np.zeros((0, 5), dtype=np.float32), (h0, w0), r, (left, top, right, bottom))

class GPUExactLetterboxTransform:
    """
    Torch-only variant of ExactLetterboxTransform for optional GPU preprocessing.
    Expects CHW float tensor in [0,1]. Returns identical semantics to CPU version.

    vulture: ignore[unused-class] — part of the optional GPU dataloader path.
    """
    def __init__(
        self,
        img_size: int | Tuple[int, int],
        center: bool = True,
        pad_value: int = 114,
        scaleup: bool = False,
        pad_to_stride: Optional[int] = None,
    ):
        self.img_size = img_size
        self.center = bool(center)
        self.pad_value = float(pad_value) / 255.0
        self.scaleup = bool(scaleup)
        self.pad_to_stride = pad_to_stride

    @torch.no_grad()
    def __call__(
        self,
        img: torch.Tensor,
        boxes_xyxy_abs: Optional[torch.Tensor] = None,
        cls: Optional[torch.Tensor] = None
    ):
        """
        img: [3,H,W] float in [0,1] on device
        boxes_xyxy_abs: [N,4] in original pixels
        cls: [N]
        """
        device = img.device
        _, h0, w0 = img.shape
        meta = compute_letterbox(
            h0,
            w0,
            self.img_size,
            center=self.center,
            scaleup=self.scaleup,
            pad_to_stride=self.pad_to_stride
        )
        r = meta.r
        new_h, new_w = meta.new_hw
        Hc, Wc = meta.canvas_hw
        left, top, right, bottom = meta.pad
        if (new_w, new_h) != (w0, h0):
            resized = F.interpolate(
                img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            resized = img
        canvas = torch.ones((3, Hc, Wc), device=device, dtype=img.dtype) * self.pad_value
        canvas[:, top:top + new_h, left:left + new_w] = resized
        if boxes_xyxy_abs is not None and cls is not None and boxes_xyxy_abs.numel():
            b = boxes_xyxy_abs.to(torch.float32).clone()
            b[:, [0, 2]] = b[:, [0, 2]] * r + left
            b[:, [1, 3]] = b[:, [1, 3]] * r + top
            w = (b[:, 2] - b[:, 0]).clamp_min_(1e-6)
            h = (b[:, 3] - b[:, 1]).clamp_min_(1e-6)
            cx = b[:, 0] + 0.5 * w
            cy = b[:, 1] + 0.5 * h
            yolo = torch.stack([
                cls.to(torch.float32), cx / float(Wc), cy / float(Hc), w / float(Wc), h / float(Hc)
            ],
                               dim=1)
            return canvas, yolo, (h0, w0), float(r), (
                float(left), float(top), float(right), float(bottom)
            )
        else:
            return canvas, torch.zeros(
                (0, 5), dtype=torch.float32, device=device
            ), (h0, w0), float(r), (float(left), float(top), float(right), float(bottom))

def smart_crop_to_size(
    img: np.ndarray,
    boxes: np.ndarray,
    size: int,
    cls: Optional[np.ndarray] = None,
    hyp: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Crop size×size region maximizing the count of box centers inside; clamp boxes; drop degenerate boxes.
    If cls is provided, it is filtered to match the kept boxes.
    """
    if cls is not None and cls.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"[smart_crop_to_size] boxes ({boxes.shape[0]}) and cls ({cls.shape[0]}) length mismatch"
        )
    h, w = img.shape[:2]
    topk = int(hyp.get("smart_crop_topk", 1)) if isinstance(hyp, dict) else 1
    jitter_frac = float(hyp.get("smart_crop_jitter", 0.0)) if isinstance(hyp, dict) else 0.0
    unbiased_p = float(hyp.get("smart_crop_unbiased_p", 0.0)) if isinstance(hyp, dict) else 0.0
    smart_p = float(hyp.get("smart_crop_prob", 1.0)) if isinstance(hyp, dict) else 1.0
    if h <= size and w <= size:
        return (img, boxes) if cls is None else (img, boxes, cls)
    max_x0 = max(0, w - size)
    max_y0 = max(0, h - size)

    def _uniform_crop() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        ux0 = 0 if w <= size else np.random.randint(0, max_x0 + 1)
        uy0 = 0 if h <= size else np.random.randint(0, max_y0 + 1)
        ux1, uy1 = ux0 + size, uy0 + size
        crop_u = img[uy0:uy1, ux0:ux1]
        if boxes.size:
            b = boxes.copy()
            b[:, [0, 2]] -= ux0
            b[:, [1, 3]] -= uy0
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, size)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, size)
            wv = (b[:, 2] - b[:, 0])
            hv = (b[:, 3] - b[:, 1])
            keep = (wv > 1) & (hv > 1)
            if cls is None:
                return crop_u, b[keep].astype(np.float32)
            kept_cls = cls[keep] if cls.size else np.zeros((0, ), dtype=np.int64)
            return crop_u, b[keep].astype(np.float32), kept_cls.astype(np.int64)
        else:
            empty = np.zeros((0, 4), dtype=np.float32)
            return (crop_u,
                    empty) if cls is None else (crop_u, empty, np.zeros((0, ), dtype=np.int64))

    if boxes.size == 0:
        return _uniform_crop()
    cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).clip(0, w - 1)
    cy = ((boxes[:, 1] + boxes[:, 3]) * 0.5).clip(0, h - 1)

    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    box_areas = box_w * box_h

    cand_x0 = np.clip((cx - size * 0.5).astype(np.int32), 0, max_x0)
    cand_y0 = np.clip((cy - size * 0.5).astype(np.int32), 0, max_y0)

    if np.random.rand() > smart_p:
        return _uniform_crop()

    if unbiased_p > 0.0 and np.random.rand() < unbiased_p:
        return _uniform_crop()

    num_eval = min(64, len(cand_x0))
    if num_eval <= 0:
        return _uniform_crop()
    idxs = np.arange(len(cand_x0), dtype=np.int32)
    if len(idxs) > num_eval:
        idxs = np.random.choice(idxs, size=num_eval, replace=False)
    scores = np.empty((len(idxs), ), dtype=np.float64)
    x0s = np.empty_like(idxs, dtype=np.int32)
    y0s = np.empty_like(idxs, dtype=np.int32)
    for j, i in enumerate(idxs):
        x0 = int(cand_x0[i])
        y0 = int(cand_y0[i])
        x1, y1 = x0 + size, y0 + size
        c_in = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        scores[j] = float((c_in * box_areas).sum())
        x0s[j] = x0
        y0s[j] = y0
    order = np.argsort(-scores)  # descending
    k = int(max(1, min(topk, len(order))))
    pick = np.random.randint(0, k) if k > 1 else 0
    sel = order[pick]
    x0 = int(x0s[sel])
    y0 = int(y0s[sel])

    if jitter_frac > 0.0 and (max_x0 > 0 or max_y0 > 0):
        jx = int(round(jitter_frac * max(1, max_x0)))
        jy = int(round(jitter_frac * max(1, max_y0)))
        if jx > 0:
            x0 = int(np.clip(x0 + np.random.randint(-jx, jx + 1), 0, max_x0))
        if jy > 0:
            y0 = int(np.clip(y0 + np.random.randint(-jy, jy + 1), 0, max_y0))
    x1, y1 = x0 + size, y0 + size
    crop = img[y0:y1, x0:x1]
    if boxes.size:
        b = boxes.copy()
        b[:, [0, 2]] -= x0
        b[:, [1, 3]] -= y0
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, size)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, size)
        wv = (b[:, 2] - b[:, 0])
        hv = (b[:, 3] - b[:, 1])
        keep = (wv > 1) & (hv > 1)
        if cls is None:
            return crop, b[keep].astype(np.float32)
        kept_cls = cls[keep] if cls.size else np.zeros((0, ), dtype=np.int64)
        return crop, b[keep].astype(np.float32), kept_cls.astype(np.int64)
    else:
        b = boxes
    return (crop, b.astype(
        np.float32
    )) if cls is None else (crop, b.astype(np.float32), np.zeros((0, ), dtype=np.int64))

def random_affine_raw(
    img: np.ndarray,
    boxes: np.ndarray,
    cls: Optional[np.ndarray],
    imgsz: int,
    degrees: Any,
    translate: Any,
    scale: Any,
    shear: Any,
    perspective: Any,
    pad_value: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Performs a comprehensive, center-based geometric transformation.
    The transformation order is: perspective, scale, rotation, shear, then translation.
    The affine component (scale, rotate, shear) is centered, and the canvas is
    expanded to fit the transformed image. Translation is applied last within this
    fixed canvas.

    vulture: ignore[unused-function] — public augmentation API (CPU path); used by tests.
    """
    h, w = img.shape[:2]

    s_min, s_max = _parse_aug_param(scale)
    s_rand = 1.0 + np.random.uniform(s_min, s_max)
    d_min, d_max = _parse_aug_param(degrees)
    ang = np.random.uniform(d_min, d_max)
    sh_min, sh_max = _parse_aug_param(shear)
    shx = float(np.random.uniform(sh_min, sh_max))
    shy = float(np.random.uniform(sh_min, sh_max))
    t_min, t_max = _parse_aug_param(translate)
    tx = np.random.uniform(t_min, t_max) * w
    ty = np.random.uniform(t_min, t_max) * h
    p_min, p_max = _parse_aug_param(perspective)
    p = float(np.random.uniform(p_min, p_max))

    if all(abs(v) < 1e-9 for v in (s_rand - 1.0, ang, shx, shy, tx, ty, p)):
        if cls is None:
            return img, boxes.astype(np.float32)
        return img, boxes.astype(np.float32), cls

    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    C_inv = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)

    a = np.deg2rad(ang)
    cs, sn = np.cos(a), np.sin(a)
    S_mat = np.array([[s_rand, 0, 0], [0, s_rand, 0], [0, 0, 1]], dtype=np.float32)
    R_mat = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], dtype=np.float32)
    Sh_mat = np.array([[1, np.tan(np.deg2rad(shx)), 0], [np.tan(np.deg2rad(shy)), 1, 0], [0, 0, 1]],
                      dtype=np.float32)
    M_affine = C_inv @ Sh_mat @ R_mat @ S_mat @ C

    H = np.eye(3, dtype=np.float32)
    if abs(p) > 1e-9:
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        jitter = np.random.uniform(-abs(p), abs(p), size=(4, 2)).astype(np.float32)
        dst = src + jitter * np.array([w, h], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

    M_core = M_affine @ H  # Perspective first, then affine

    src_corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]],
                           dtype=np.float32)
    tc_core = (src_corners @ M_core.T)
    z_core = np.clip(tc_core[:, 2:3], 1e-6, None)
    xy_core = tc_core[:, :2] / z_core
    min_xy_core, max_xy_core = xy_core.min(axis=0), xy_core.max(axis=0)

    aabb_w_core = max_xy_core[0] - min_xy_core[0]
    aabb_h_core = max_xy_core[1] - min_xy_core[1]
    out_w = int(np.ceil(max(w, aabb_w_core)))
    out_h = int(np.ceil(max(h, aabb_h_core)))

    dx_shift_core = -min_xy_core[0]
    dy_shift_core = -min_xy_core[1]
    dx_center = 0.5 * (out_w - aabb_w_core)
    dy_center = 0.5 * (out_h - aabb_h_core)
    T_center = np.array([[1, 0, dx_shift_core + dx_center], [0, 1, dy_shift_core + dy_center],
                         [0, 0, 1]])
    M_centered_core = T_center @ M_core

    T_mat = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    M = T_mat @ M_centered_core

    img_out = cv2.warpPerspective(
        img,
        M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(pad_value, pad_value, pad_value)
    )
    b_trans, keep = _transform_boxes_homography(boxes, M, out_w, out_h)

    if b_trans.size:
        b_final = b_trans[keep]
        cls_final = cls[keep] if cls is not None and cls.size > 0 else np.zeros((0, ),
                                                                                dtype=np.int64)
    else:
        b_final = b_trans
        cls_final = np.zeros((0, ), dtype=np.int64) if cls is not None else None

    if cls is None:
        return img_out, b_final
    return img_out, b_final, cls_final

def copy_paste_raw(
    dataset: 'MultiFormatDataset',
    img: np.ndarray,
    boxes: np.ndarray,
    cls: np.ndarray,
    occlusion_threshold: float = 0.7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Copy-Paste with donor letterbox+scaleup prior to selecting objects, as requested.
    Donor is letterboxed to base HxW with scaleup=True; donor boxes are scaled accordingly.
    Then we select patches and paste while avoiding heavy overlap.
    Handles occlusion by removing original boxes that are heavily covered by pasted objects.
    """
    if len(dataset) < 2 or boxes.size == 0:
        return img, boxes, cls

    idx2 = np.random.randint(0, len(dataset))
    donor_raw, b2, c2, (h2, w2) = dataset._load_raw(idx2)
    base_h, base_w = img.shape[:2]

    if not b2.size:
        return img, boxes, cls

    meta = compute_letterbox(
        h2, w2, (base_h, base_w), center=True, scaleup=True, pad_to_stride=None
    )
    r, (new_h, new_w), (Hc, Wc), (left, top, _, _) = meta.r, meta.new_hw, meta.canvas_hw, meta.pad
    if (new_w, new_h) != (w2, h2):
        donor_resized = cv2.resize(donor_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        donor_resized = donor_raw
    donor_canvas = np.full((Hc, Wc, 3), dataset.pad_value, dtype=np.uint8)
    donor_canvas[top:top + new_h, left:left + new_w] = donor_resized
    b2r = b2.astype(np.float32).copy()
    b2r[:, [0, 2]] = b2r[:, [0, 2]] * r + left
    b2r[:, [1, 3]] = b2r[:, [1, 3]] * r + top
    keep = (b2r[:, 2] - b2r[:, 0] > 1) & (b2r[:, 3] - b2r[:, 1] > 1)
    b2r, c2 = b2r[keep], c2[keep]

    if not b2r.size:
        return img, boxes, cls

    n = min(len(b2r), np.random.randint(1, len(b2r) + 1))
    sel = np.random.choice(len(b2r), n, replace=False)
    donor_boxes = b2r[sel]
    donor_cls = c2[sel]

    pasted_boxes = []
    pasted_cls = []

    grid_n = int(np.ceil(np.sqrt(len(donor_boxes))))
    grid_n = max(2, min(grid_n, 6))  # 2..6 grid
    cell_w = base_w // grid_n
    cell_h = base_h // grid_n
    cells = [(gy * cell_h, gx * cell_w) for gy in range(grid_n) for gx in range(grid_n)]
    np.random.shuffle(cells)

    base_area = float(base_w * base_h)

    for idx, (pbox, pcl) in enumerate(zip(donor_boxes, donor_cls)):
        if idx >= len(cells):
            break
        cy0, cx0 = cells[idx]
        cx_mid = cx0 + cell_w // 2
        cy_mid = cy0 + cell_h // 2

        x1, y1, x2, y2 = pbox.astype(np.int32)
        x1c = int(np.clip(x1, 0, Wc - 1))
        y1c = int(np.clip(y1, 0, Hc - 1))
        x2c = int(np.clip(x2, 0, Wc))
        y2c = int(np.clip(y2, 0, Hc))
        if x2c <= x1c or y2c <= y1c:
            continue
        patch = donor_canvas[y1c:y2c, x1c:x2c]
        ph, pw = patch.shape[:2]
        if ph < 2 or pw < 2:
            continue

        s_init = float(np.random.uniform(0.8, 1.2))
        s_cell_max = min(cell_w / pw, cell_h / ph)
        s_area_min = np.sqrt(0.05 * base_area / (pw * ph + 1e-7))
        s_area_max = np.sqrt(0.40 * base_area / (pw * ph + 1e-7))
        s_final = np.clip(s_init, s_area_min, min(s_area_max, s_cell_max))
        if s_final <= 0:
            continue
        new_w = max(1, int(round(pw * s_final)))
        new_h = max(1, int(round(ph * s_final)))
        if (new_w, new_h) != (pw, ph):
            patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            patch_resized = patch

        x0 = int(np.clip(cx_mid - new_w // 2, 0, base_w - new_w))
        y0 = int(np.clip(cy_mid - new_h // 2, 0, base_h - new_h))
        x1b, y1b = x0 + new_w, y0 + new_h

        img[y0:y1b, x0:x1b] = patch_resized[:(y1b - y0), :(x1b - x0)]
        pasted_boxes.append([x0, y0, x1b, y1b])
        pasted_cls.append(int(pcl))

    pasted_boxes = np.array(pasted_boxes, dtype=np.float32)
    pasted_cls = np.array(pasted_cls, dtype=np.int64)

    if boxes.size and pasted_boxes.size:
        x1_orig, y1_orig, x2_orig, y2_orig = np.split(boxes, 4, axis=1)
        x1_p, y1_p, x2_p, y2_p = np.split(pasted_boxes, 4, axis=1)

        inter_x1 = np.maximum(x1_orig, x1_p.T)
        inter_y1 = np.maximum(y1_orig, y1_p.T)
        inter_x2 = np.minimum(x2_orig, x2_p.T)
        inter_y2 = np.minimum(y2_orig, y2_p.T)

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        intersection_area = inter_w * inter_h

        total_intersection_per_original = intersection_area.sum(axis=1)

        original_areas = (x2_orig - x1_orig) * (y2_orig - y1_orig)
        occlusion_ratio = total_intersection_per_original / (original_areas.flatten() + 1e-7)

        keep_original = occlusion_ratio < occlusion_threshold
        boxes = boxes[keep_original]
        cls = cls[keep_original]

    if pasted_boxes.size and boxes.size:
        final_boxes = np.concatenate([boxes, pasted_boxes], 0)
        final_cls = np.concatenate([cls, pasted_cls], 0)
    elif pasted_boxes.size:
        final_boxes = pasted_boxes
        final_cls = pasted_cls
    else:
        final_boxes = boxes
        final_cls = cls

    return img, final_boxes, final_cls

def mosaic4_raw(dataset: 'MultiFormatDataset',
                idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tile 4 RAW images to a fixed s×s canvas.
    For each image: smart crop to q×q. If a single box dominates the crop,
    attempt to re-sample up to 2 times. Finally, tile quadrants and apply a
    final smart crop to the mosaic for safety.
    """
    s = int(dataset.imgsz)
    q = s // 2
    pad_value = dataset.pad_value
    indices = [idx] + np.random.choice(len(dataset), 3, replace=False).tolist()
    patches, labels_list, cls_list = [], [], []

    used_indices = set(indices)

    for i in range(len(indices)):
        current_idx = indices[i]

        for attempt in range(3):  # 1 initial try + 2 retries
            im, b, c, (h0, w0) = dataset._load_raw(current_idx)

            imc, bc_cropped, cc_cropped = smart_crop_to_size(
                im, b, q, cls=c, hyp=getattr(dataset, 'hyp', None)
            )

            is_dominant = False
            if bc_cropped.size > 0:
                crop_area = float(imc.shape[0] * imc.shape[1])
                box_w = bc_cropped[:, 2] - bc_cropped[:, 0]
                box_h = bc_cropped[:, 3] - bc_cropped[:, 1]
                box_areas = box_w * box_h
                if np.any(box_areas / (crop_area + 1e-7) >= 0.90):
                    is_dominant = True

            if not is_dominant:
                break  # Found a suitable image, exit retry loop

            if attempt < 2:  # If not the last attempt, find a new image
                while True:
                    new_idx = np.random.randint(0, len(dataset))
                    if new_idx not in used_indices:
                        current_idx = new_idx
                        used_indices.add(new_idx)
                        break

        H, W = imc.shape[:2]
        if H < q or W < q:
            canvas_q = np.full((q, q, 3), pad_value, dtype=np.uint8)
            canvas_q[:H, :W] = imc
            imc = canvas_q

        patches.append(imc)
        labels_list.append(bc_cropped)
        cls_list.append(cc_cropped)

    canvas = np.full((s, s, 3), pad_value, dtype=np.uint8)
    offsets = [(0, 0), (0, q), (q, 0), (q, q)]
    labels4, cls4 = [], []

    for i, p in enumerate(patches):
        oy, ox = offsets[i]
        canvas[oy:oy + q, ox:ox + q] = p[:q, :q]
        if i < len(labels_list) and len(labels_list[i]):
            b = labels_list[i].copy().astype(np.float32)
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, s)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, s)
            wv = b[:, 2] - b[:, 0]
            hv = b[:, 3] - b[:, 1]
            keep = (wv > 1) & (hv > 1)
            if np.any(keep):
                labels4.append(b[keep])
                cls4.append(cls_list[i][keep])

    if labels4:
        labels4 = np.concatenate(labels4, 0)
        cls4 = np.concatenate(cls4, 0)
    else:
        labels4 = np.zeros((0, 4), dtype=np.float32)
        cls4 = np.zeros((0, ), dtype=np.int64)

    final_img, final_boxes, final_cls = smart_crop_to_size(
        canvas, labels4, s, cls=cls4, hyp=getattr(dataset, 'hyp', None)
    )

    return final_img, final_boxes, final_cls

def mixup_raw(dataset: 'MultiFormatDataset', img: np.ndarray, boxes: np.ndarray,
              cls: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(dataset) < 2:
        return img, boxes, cls
    idx2 = np.random.randint(0, len(dataset))
    img2, b2, c2, (h2, w2) = dataset._load_raw(idx2)
    h1, w1 = img.shape[:2]

    meta = compute_letterbox(h2, w2, (h1, w1), center=True, scaleup=True)
    r = meta.r
    new_h, new_w = meta.new_hw
    left, top, _, _ = meta.pad

    if (new_w, new_h) != (w2, h2):
        img2_resized = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img2_resized = img2

    canvas2 = np.full((h1, w1, 3), dataset.pad_value, dtype=img2.dtype)
    canvas2[top:top + new_h, left:left + new_w] = img2_resized

    if b2.size:
        b2r = b2.copy().astype(np.float32)
        b2r[:, [0, 2]] = b2r[:, [0, 2]] * r + left
        b2r[:, [1, 3]] = b2r[:, [1, 3]] * r + top
    else:
        b2r = b2

    lam = float(np.random.beta(32.0, 32.0))
    lam = float(np.clip(lam, 0.35, 0.65))
    img_aug = (lam * img + (1.0 - lam) * canvas2).astype(np.uint8)
    if b2r.size:
        boxes_aug = np.concatenate([boxes, b2r], 0) if boxes.size else b2r
        cls_aug = np.concatenate([cls, c2], 0) if cls.size else c2
    else:
        boxes_aug, cls_aug = boxes, cls
    return img_aug, boxes_aug, cls_aug

@torch.no_grad()
def apply_gpu_batch_aug(imgs, targets, hyp):
    """
    Applies augmentations to a batch of images and targets on the GPU.
    - Operates on float32 tensors in [0,1] range.
    - Modifies imgs and targets in-place.

    vulture: ignore[unused-function] — used by optional GPU dataloader collate path.
    """
    device = imgs.device
    B, C, H, W = imgs.shape
    p_fliplr = float(hyp.get("fliplr", 0.5))
    if p_fliplr > 0.0:
        flip_mask = torch.rand(B, device=device) < p_fliplr
        if flip_mask.any():
            imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[3])
            if targets.numel():
                bi = targets[:, 0].long()
                target_flip_mask = torch.isin(bi, torch.where(flip_mask)[0])
                if target_flip_mask.any():
                    targets[target_flip_mask, 2] = 1.0 - targets[target_flip_mask, 2]
    p_flipud = float(hyp.get("flipud", 0.0))
    if p_flipud > 0.0:
        flip_mask = torch.rand(B, device=device) < p_flipud
        if flip_mask.any():
            imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[2])
            if targets.numel():
                bi = targets[:, 0].long()
                target_flip_mask = torch.isin(bi, torch.where(flip_mask)[0])
                if target_flip_mask.any():
                    targets[target_flip_mask, 3] = 1.0 - targets[target_flip_mask, 3]
    hsv_s = hyp.get('hsv_s', 0.7)
    hsv_v = hyp.get('hsv_v', 0.4)
    if hsv_v > 0:
        delta = (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * hsv_v
        imgs = (imgs + delta).clamp_(0, 1)
    if hsv_s > 0:
        fac = 1.0 + (torch.rand(B, 1, 1, 1, device=device) * 2 - 1) * (0.5 * hsv_s)
        mean = imgs.mean(dim=(2, 3), keepdim=True)
        imgs = (imgs - mean) * fac + mean
        imgs.clamp_(0, 1)
    p_mixup = float(hyp.get("mixup", 0.0))
    if p_mixup > 0 and B >= 2:
        if torch.rand(1, device=device).item() < p_mixup:
            perm = torch.randperm(B, device=device)
            lam = torch.distributions.Beta(32.0, 32.0).sample().item()
            imgs = (lam * imgs + (1.0 - lam) * imgs[perm]).clamp_(0, 1)
            if targets.numel():
                targets2 = targets.clone()
                b2 = targets2[:, 0].long()
                targets2[:, 0] = perm[b2].to(targets2.dtype)
                targets = torch.cat([targets, targets2], dim=0)
    return imgs, targets
