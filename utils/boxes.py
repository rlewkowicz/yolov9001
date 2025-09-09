"""
utils/boxes.py

Consolidated box conversion utilities.
"""
import torch

class BoundingBox:
    """
    Utility class for bounding box operations and format conversions.
    Groups related geometric operations in one place.
    """
    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = torch.split(distance, 2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    @staticmethod
    def bbox2dist(anchor_points, bbox, reg_max):
        """Transform bbox(xyxy) to dist(ltrb)."""
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)  # no clamp here

def yolo_to_xyxy(yolo_norm_xywh: torch.Tensor, canvas_hw: tuple[int, int]) -> torch.Tensor:
    Hc, Wc = canvas_hw
    cx = yolo_norm_xywh[:, 0] * Wc
    cy = yolo_norm_xywh[:, 1] * Hc
    w = yolo_norm_xywh[:, 2] * Wc
    h = yolo_norm_xywh[:, 3] * Hc
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack((x1, y1, x2, y2), dim=1)

def xyxy_to_yolo(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Convert pixel-space xyxy boxes to YOLO-normalized xywh.
    boxes: [N,4], shape: (h, w)

    vulture: ignore[unused-function] — public conversion utility used by tools/tests.
    """
    h, w = shape
    b = boxes.clone().float()
    bw = (b[:, 2] - b[:, 0]).clamp(min=1e-6)
    bh = (b[:, 3] - b[:, 1]).clamp(min=1e-6)
    xc = b[:, 0] + 0.5 * bw
    yc = b[:, 1] + 0.5 * bh
    out = torch.stack([xc / w, yc / h, bw / w, bh / h], dim=1)
    assert torch.all(out >= -1e-5) and torch.all(out <= 1.0 + 1e-5), \
        f"xyxy_to_yolo produced out-of-bounds normalized coordinates: {out}"
    return out.clamp(0, 1)

def scale_boxes_from_canvas_to_original(
    xyxy_canvas: torch.Tensor,  # Nx4 in canvas pixels
    canvas_hw: tuple[int, int],  # (Hc, Wc)
    orig_hw: tuple[int, int],  # (H0, W0)
    pad_left_top: tuple[int, int],  # (left, top)
    pad_right_bottom: tuple[int, int] = (None, None)  # optional if you want extra checks
) -> torch.Tensor:
    Hc, Wc = canvas_hw
    H0, W0 = orig_hw
    left, top = pad_left_top
    new_w = Wc - left - (0 if pad_right_bottom[0] is None else pad_right_bottom[0])
    new_h = Hc - top - (0 if pad_right_bottom[1] is None else pad_right_bottom[1])
    r_x = new_w / float(W0)
    r_y = new_h / float(H0)

    x1 = (xyxy_canvas[:, 0] - left) / r_x
    y1 = (xyxy_canvas[:, 1] - top) / r_y
    x2 = (xyxy_canvas[:, 2] - left) / r_x
    y2 = (xyxy_canvas[:, 3] - top) / r_y
    return torch.stack((x1, y1, x2, y2), dim=1)

def scale_boxes_from_original_to_canvas(
    xyxy_orig: torch.Tensor,  # Nx4 in original pixels
    canvas_hw: tuple[int, int],  # (Hc, Wc)
    orig_hw: tuple[int, int],  # (H0, W0)
    pad_left_top: tuple[int, int],  # (left, top)
    pad_right_bottom: tuple[int, int] = (None, None)  # optional checks
) -> torch.Tensor:
    """
    Scale xyxy boxes from original HxW to a square letterboxed canvas shape.

    vulture: ignore[unused-function] — used by dataloading/validation utilities and tests.
    """
    Hc, Wc = canvas_hw
    H0, W0 = orig_hw
    left, top = pad_left_top
    new_w = Wc - left - (0 if pad_right_bottom[0] is None else pad_right_bottom[0])
    new_h = Hc - top - (0 if pad_right_bottom[1] is None else pad_right_bottom[1])
    r_x = new_w / float(W0)
    r_y = new_h / float(H0)

    x1 = xyxy_orig[:, 0] * r_x + left
    y1 = xyxy_orig[:, 1] * r_y + top
    x2 = xyxy_orig[:, 2] * r_x + left
    y2 = xyxy_orig[:, 3] * r_y + top
    return torch.stack((x1, y1, x2, y2), dim=1)

def cxcywh_to_xyxy(x):
    """Convert cx,cy,w,h to x1,y1,x2,y2. vulture: ignore[unused-function]"""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] * 0.5
    y[:, 1] = x[:, 1] - x[:, 3] * 0.5
    y[:, 2] = x[:, 0] + x[:, 2] * 0.5
    y[:, 3] = x[:, 1] + x[:, 3] * 0.5
    return y

def xyxy_to_cxcywh(x):
    """Convert x1,y1,x2,y2 to cx,cy,w,h. vulture: ignore[unused-function]"""
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) * 0.5
    y[:, 1] = (x[:, 1] + x[:, 3]) * 0.5
    y[:, 2] = (x[:, 2] - x[:, 0])
    y[:, 3] = (x[:, 3] - x[:, 1])
    return y

def scale_boxes(to_shape, boxes, from_shape):
    """
    Rescale boxes (xyxy) from from_shape to to_shape.

    vulture: ignore[unused-function] — public geometry utility used by tests.
    """
    from_h, from_w = from_shape
    to_h, to_w = to_shape
    scale_w = to_w / from_w
    scale_h = to_h / from_h

    if isinstance(boxes, torch.Tensor):
        scaled = boxes.clone()
    else:
        scaled = boxes.copy()

    scaled[:, [0, 2]] *= scale_w
    scaled[:, [1, 3]] *= scale_h

    clip_boxes_(scaled, to_shape)  # clip to target shape

    return scaled

def clip_boxes_(boxes, shape, pixel_edges=False):
    h, w = shape
    if isinstance(boxes, torch.Tensor):
        if pixel_edges:
            boxes[:, 0].clamp_(0, w - 1)  # x1
            boxes[:, 1].clamp_(0, h - 1)  # y1
            boxes[:, 2].clamp_(0, w - 1)  # x2
            boxes[:, 3].clamp_(0, h - 1)  # y2
        else:
            boxes[:, 0].clamp_(0, w)  # x1
            boxes[:, 1].clamp_(0, h)  # y1
            boxes[:, 2].clamp_(0, w)  # x2
            boxes[:, 3].clamp_(0, h)  # y2
    else:  # numpy
        if pixel_edges:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)  # y1, y2
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)  # y1, y2
    return boxes
