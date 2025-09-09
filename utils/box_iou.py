import math
import torch

__all__ = ("bbox_iou_aligned", "pairwise_box_iou")  # extend export

def pairwise_box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    NxM pairwise IoU for xyxy boxes on device.
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)  # NxM

def _pairwise_iou_xyxy(b1, b2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2.unbind(-1)
    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    raw_union = w1 * h1 + w2 * h2 - inter
    iou = inter / (raw_union + eps)
    return iou, (w1, h1, w2, h2), raw_union

def _apply_variants(
    iou, b1, b2, w1, h1, w2, h2, union, GIoU, DIoU, CIoU, SIoU, MPDIoU, eps=1e-7, img_wh=None
):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2.unbind(-1)
    cw = (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp_min(eps)
    ch = (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp_min(eps)
    w1s, h1s = w1.clamp_min(eps), h1.clamp_min(eps)
    w2s, h2s = w2.clamp_min(eps), h2.clamp_min(eps)

    if not (CIoU or DIoU or SIoU or MPDIoU or GIoU):
        return iou.clamp(0.0, 1.0)

    if CIoU or DIoU or SIoU or MPDIoU:
        c2 = cw.pow(2) + ch.pow(2) + eps
        b1_cx = (b1_x1 + b1_x2) * 0.5
        b1_cy = (b1_y1 + b1_y2) * 0.5
        b2_cx = (b2_x1 + b2_x2) * 0.5
        b2_cy = (b2_y1 + b2_y2) * 0.5
        dx = (b2_cx - b1_cx)
        dy = (b2_cy - b1_cy)
        rho2 = dx.pow(2) + dy.pow(2)

        if MPDIoU:
            bx1 = torch.min(b1[..., 0], b1[..., 2])
            by1 = torch.min(b1[..., 1], b1[..., 3])
            bx2 = torch.max(b1[..., 0], b1[..., 2])
            by2 = torch.max(b1[..., 1], b1[..., 3])
            gx1 = torch.min(b2[..., 0], b2[..., 2])
            gy1 = torch.min(b2[..., 1], b2[..., 3])
            gx2 = torch.max(b2[..., 0], b2[..., 2])
            gy2 = torch.max(b2[..., 1], b2[..., 3])

            iw = (torch.min(bx2, gx2) - torch.max(bx1, gx1)).clamp(min=0)
            ih = (torch.min(by2, gy2) - torch.max(by1, gy1)).clamp(min=0)
            inter = iw * ih
            a1 = (bx2 - bx1) * (by2 - by1)
            a2 = (gx2 - gx1) * (gy2 - gy1)
            union_exact = a1 + a2 - inter
            if torch.any(union_exact <= 0):
                raise ValueError(
                    "Invalid boxes for MPDIoU: zero/negative area leads to undefined IoU."
                )
            iou_exact = inter / union_exact

            d1 = (bx1 - gx1).pow(2) + (by1 - gy1).pow(2)
            d2 = (bx2 - gx2).pow(2) + (by2 - gy2).pow(2)
            if img_wh is None:
                raise ValueError("MPDIoU requires img_wh=(W,H) to match the paper exactly.")
            W, H = img_wh
            denom = float(W)**2 + float(H)**2  # no epsilon (paper-exact)
            return iou_exact - (d1 + d2) / denom

        if SIoU:
            wx = (w1s + w2s) * 0.5
            hy = (h1s + h2s) * 0.5

            theta = torch.atan2(dy.abs(), dx.abs() + eps)
            angle_cost = torch.sin(2.0 * theta)  # 0 at best (axis-aligned), 1 at worst (diagonal)

            rho_x = (dx / (wx + eps)).pow(2)
            rho_y = (dy / (hy + eps)).pow(2)
            gamma = 2.0 - (1.0 - angle_cost)  # smaller when aligned
            distance_cost = 2.0 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

            t = 4.0
            sc_w = (1 - torch.exp(-(w1s - w2s).abs() / (torch.max(w1s, w2s) + eps))).pow(t)
            sc_h = (1 - torch.exp(-(h1s - h2s).abs() / (torch.max(h1s, h2s) + eps))).pow(t)
            sc = sc_w + sc_h

            return iou - 0.5 * (distance_cost + sc)

        if CIoU:
            v = (4.0 / (math.pi**2)) * (torch.atan(w2s / (h2s)) - torch.atan(w1s / (h1s))).pow(2)
            alpha = (v / (1.0 - iou + v + eps)).detach()
            return iou - (rho2 / c2 + v * alpha)

        return iou - rho2 / c2

    if GIoU:
        c_area = (cw * ch) + eps
        return iou - (c_area - union.clamp_min(eps)) / c_area
    return iou

def bbox_iou_aligned(b1, b2, iou_type="CIoU", eps=1e-7, img_wh=None):
    """
    Calculates the aligned intersection over union (IoU) of two sets of bounding boxes.

    Args:
        b1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in xyxy format.
        b2 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in xyxy format.
        iou_type (str, optional): The type of IoU to calculate. One of "CIoU", "DIoU", "GIoU",
                                  "SIoU", "MPDIoU", or "IoU". Defaults to "CIoU".
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N,) representing the IoU of each pair of boxes.
    """
    iou_type = iou_type.upper().replace("IOU", "")
    assert iou_type in ("", "C", "D", "G", "S", "MPD"), f"Unsupported IoU type: {iou_type}IoU"
    kwargs = {
        "GIoU": "G" in iou_type,
        "DIoU": "D" in iou_type,
        "CIoU": "C" in iou_type,
        "SIoU": "S" in iou_type,
        "MPDIoU": "MPD" in iou_type,
    }

    iou, (w1, h1, w2, h2), union = _pairwise_iou_xyxy(b1, b2, eps)
    return _apply_variants(iou, b1, b2, w1, h1, w2, h2, union, img_wh=img_wh, **kwargs)
