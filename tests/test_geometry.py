import torch

from utils.boxes import (
    yolo_to_xyxy,
    xyxy_to_yolo,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    scale_boxes,
    scale_boxes_from_canvas_to_original,
)
from utils.geometry import _assert_ltrb_order_is_consistent, DFLDecoder
from utils.augment import ExactLetterboxTransform

def test_yolo_and_xyxy_roundtrip():
    shape = (640, 640)
    yolo = torch.rand(10, 4)
    xyxy = yolo_to_xyxy(yolo, shape)
    yolo2 = xyxy_to_yolo(xyxy, shape)
    assert torch.allclose(yolo, yolo2, atol=1e-6)

def test_cxcywh_and_xyxy_roundtrip():
    xyxy = torch.rand(10, 4) * 640
    xyxy[:, 2:] += xyxy[:, :2]
    cxcywh = xyxy_to_cxcywh(xyxy)
    xyxy2 = cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(xyxy, xyxy2, atol=1e-5)

def test_scale_boxes():
    shape1 = (640, 640)
    shape2 = (320, 320)
    boxes = torch.rand(10, 4) * 640
    boxes[:, 2:] += boxes[:, :2]
    scaled_boxes = scale_boxes(shape2, boxes, shape1)
    assert (scaled_boxes <= 320).all()

def test_canvas_xy_formats_roundtrip():
    imgsz = 640
    canvas_shape = (imgsz, imgsz)
    boxes_xyxy = torch.tensor([[10, 20, 110, 220], [0, 0, 1, 1], [100, 100, 639, 639]],
                              dtype=torch.float32)
    yolo = xyxy_to_yolo(boxes_xyxy, canvas_shape)
    back = yolo_to_xyxy(yolo, canvas_shape)
    assert torch.allclose(back, boxes_xyxy, atol=1e-6)

def test_box_iou_properties():
    from utils.box_iou import pairwise_box_iou, bbox_iou_aligned
    a = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    b = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    P = pairwise_box_iou(a, b)
    assert torch.isfinite(P).all()
    assert abs(P[0, 0].item() - 1.0) < 1e-6
    assert abs(P[1, 1].item() - 0.0) < 1e-6
    I = bbox_iou_aligned(a, a, iou_type="IoU")
    assert torch.allclose(I, torch.ones_like(I))

def test_ltrb_channel_order_sanity():
    _assert_ltrb_order_is_consistent()

def test_dfl_decoder_init():
    decoder = DFLDecoder(reg_max=16)
    assert decoder.reg_max == 16

def test_dfl_decoder_get_anchors():
    decoder = DFLDecoder(reg_max=16, strides=[8, 16, 32])
    feats = [torch.randn(1, 3, 80, 80), torch.randn(1, 3, 40, 40), torch.randn(1, 3, 20, 20)]
    anchors, strides = decoder.get_anchors(feats)
    assert anchors.shape[0] == 8400
    assert strides.shape[0] == 8400

def test_dfl_decoder_bins_device_and_regmax():
    """Tests DFLDecoder's internal cache handling for device and reg_max changes."""
    d = DFLDecoder(reg_max=16, strides=[8, 16], device='cpu')
    b1 = d._bins.clone()

    if torch.cuda.is_available():
        d.to(torch.device('cuda'))
        assert b1.device != d._bins.device
    else:
        d.to(torch.device('cpu'))
        assert b1.device == d._bins.device

    d.reg_max = 20
    b2 = d._bins
    assert b2.numel() == 20
    assert b2.dtype == torch.float32
    assert d.reg_max == 20

def test_validator_gt_canvas_conversion():
    """
    Ensures the ground truth box conversion from YOLO format to canvas pixel space is correct.
    The original bug was using the wrong source dimensions for this conversion.
    """
    canvas_shape = (640, 640)
    gt_yolo = torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)  # [cls, x, y, w, h]

    gt_canvas_xyxy = yolo_to_xyxy(gt_yolo[:, 1:], canvas_shape)

    expected_xyxy = torch.tensor([[256., 256., 384., 384.]])

    assert torch.allclose(gt_canvas_xyxy, expected_xyxy, atol=1.0)

def direct_canvas_xyxy(xyxy_abs, r, left, top):
    xy = xyxy_abs.copy()
    xy[:, [0, 2]] = xy[:, [0, 2]] * r + left
    xy[:, [1, 3]] = xy[:, [1, 3]] * r + top
    return xy

def test_letterbox_forward_canvas_normalization():
    import numpy as np, torch
    h0, w0 = 720, 540
    img = np.zeros((h0, w0, 3), np.uint8)
    xyxy_abs = np.array([[w0 * 0.1, h0 * 0.2, w0 * 0.9, h0 * 0.8]], np.float32)
    cls = np.array([0], np.int64)
    transform = ExactLetterboxTransform(
        img_size=640, center=True, pad_value=114, scaleup=False, pad_to_stride=32
    )

    canvas, yolo_norm, (oh, ow), r, pad = transform(img, xyxy_abs, cls)
    left, top, right, bottom = pad
    xyxy_canvas = yolo_to_xyxy(torch.from_numpy(yolo_norm[:, 1:]), canvas.shape[:2]).numpy()
    xyxy_exp = direct_canvas_xyxy(xyxy_abs, r, left, top)
    assert np.allclose(xyxy_canvas, xyxy_exp, atol=1e-4), (xyxy_canvas, xyxy_exp)

def test_letterbox_roundtrip_to_original():
    import numpy as np, torch
    h0, w0 = 217, 349
    img = np.zeros((h0, w0, 3), np.uint8)
    xyxy_abs = np.array([[0, 0, 10, 10], [w0 - 11, h0 - 11, w0 - 1, h0 - 1],
                         [w0 / 2 - 0.5, h0 / 2 - 0.5, w0 / 2 + 0.5, h0 / 2 + 0.5]], np.float32)
    cls = np.array([0, 1, 2], np.int64)
    transform = ExactLetterboxTransform(
        img_size=256, center=True, pad_value=114, scaleup=False, pad_to_stride=32
    )
    canvas, yolo_norm, (oh, ow), r, pad = transform(img, xyxy_abs, cls)
    left, top, right, bottom = pad

    xyxy_canvas = yolo_to_xyxy(torch.from_numpy(yolo_norm[:, 1:]), canvas.shape[:2]).float()
    xyxy_orig = scale_boxes_from_canvas_to_original(
        xyxy_canvas.clone(), canvas.shape[:2], (oh, ow), (left, top), (right, bottom)
    )
    assert torch.allclose(xyxy_orig, torch.from_numpy(xyxy_abs), atol=1.0), "Round-trip mismatch"

def test_decoder_stride_cache_invalidation():
    """
    Creates a decoder, calls get_anchors, changes decoder.strides, calls get_anchors again,
    and asserts that the stride_tensor changed accordingly.
    """
    shapes = [(80, 80), (40, 40)]
    strides1 = [8, 16]
    strides2 = [10, 20]

    decoder = DFLDecoder(strides=strides1, device='cpu')

    _, stride_tensor1 = decoder.get_anchors(shapes)

    decoder.strides = strides2

    _, stride_tensor2 = decoder.get_anchors(shapes)

    assert not torch.equal(stride_tensor1, stride_tensor2)
    assert torch.all(torch.unique(stride_tensor2) == torch.tensor(strides2, dtype=torch.float32))

def test_dfl_decoder_anchors_dtype_is_fp32_even_with_fp16_feats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = [
        torch.zeros(1, 32, 8, 8, device=device, dtype=torch.float16),
        torch.zeros(1, 64, 4, 4, device=device, dtype=torch.float16),
    ]
    decoder = DFLDecoder(reg_max=16, strides=[8, 16], device=device)
    anchors, strides = decoder.get_anchors(feats)
    assert anchors.dtype == torch.float32, "anchors should be float32 for numerical stability"
    assert strides.dtype == torch.float32, "stride tensor should be float32 for numerical stability"
