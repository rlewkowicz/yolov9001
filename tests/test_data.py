"""
Tests for dataloaders and data transformations.
"""
import os
import psutil
from pathlib import Path
import numpy as np
import cv2
import torch
import pytest
from utils.dataloaders import create_dataloader
from utils.augment import ExactLetterboxTransform
from utils.boxes import scale_boxes_from_canvas_to_original
from utils.metrics import box_iou

def _write_img(dst, w, h, color=(128, 64, 32)):
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    cv2.imwrite(str(dst), arr)

def _write_yolo_txt(dst, boxes_xywh_norm):
    lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes_xywh_norm]
    Path(dst).write_text("".join(lines))

def _make_tiny_yolo_dataset(root: Path):
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("nc: 3\nnames: ['class_0', 'class_1', 'class_2']\n")
    train_imgs = [
        ("a.jpg", 320, 240, [(0, 0.5, 0.5, 0.25, 0.25)]),
        ("b.jpg", 480, 360, [(1, 0.3, 0.3, 0.2, 0.2)]),
        ("c.jpg", 640, 480, []),
    ]
    for name, w, h, boxes in train_imgs:
        _write_img(root / "images" / "train" / name, w, h)
        _write_yolo_txt(root / "labels" / "train" / (Path(name).stem + ".txt"), boxes)

def test_create_dataloader_shapes_and_types(tmp_path):
    _make_tiny_yolo_dataset(tmp_path)
    loader, info = create_dataloader(
        data_path=tmp_path, split="train", img_size=256, batch_size=2, num_workers=0, augment=False
    )
    assert info["nc"] >= 1 and info["format"] in {"yolo", "coco"}
    imgs, labels, _, _, _ = next(iter(loader))
    assert imgs.dtype == torch.uint8 and imgs.ndim == 4 and imgs.shape[2:] == (256, 256)
    assert (imgs >= 0).all() and (imgs <= 255).all()
    assert isinstance(labels, torch.Tensor)
    if labels.numel():
        assert labels.shape[1] == 6
        assert (labels[:, 2:6] >= 0).all() and (labels[:, 2:6] <= 1).all()

def test_augment_scale_preserves_valid_targets(tmp_path):
    _make_tiny_yolo_dataset(tmp_path)
    hyp = {"scale": 0.5}
    loader, _ = create_dataloader(
        data_path=tmp_path,
        split="train",
        img_size=256,
        batch_size=2,
        num_workers=0,
        augment=True,
        hyp=hyp
    )
    imgs, labels, _, _, _ = next(iter(loader))
    if labels.numel():
        assert (labels[:, 2:6] >= 0).all() and (labels[:, 2:6]
                                                <= 1).all(), "labels must be clipped to [0,1]"

def test_letterbox_to_square_basic():
    img = np.zeros((150, 300, 3), dtype=np.uint8)
    transform = ExactLetterboxTransform(img_size=256, center=True, pad_to_stride=32)
    out, _, _, ratio, pad = transform(img)
    assert out.shape[:2] == (256, 256)
    assert ratio > 0 and isinstance(pad, tuple)

@pytest.mark.optional
def test_memory_usage():
    """Test dataloader memory with different cache settings."""
    data_path = os.environ.get('DATA_PATH')
    if not data_path or not Path(data_path).exists():
        pytest.skip(f"Dataset not found at {data_path}. Set DATA_PATH to test.")

    def get_mem():
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    mem_before_nocache = get_mem()
    loader_no_cache, _ = create_dataloader(data_path, 'train', 640, 8, 0, cache=None, augment=False)
    mem_after_nocache_init = get_mem()
    del loader_no_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mem_before_cache = get_mem()
    loader_ram_cache, _ = create_dataloader(
        data_path, 'train', 640, 8, 0, cache='ram', augment=False
    )
    mem_after_cache_init = get_mem()
    del loader_ram_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    delta_cache = mem_after_cache_init - mem_before_cache
    delta_nocache = mem_after_nocache_init - mem_before_nocache
    assert delta_cache > delta_nocache

def test_edge_case():
    import torch
    h0, w0 = 720, 1280
    canvas_shape = (640, 640)

    ratio = min(canvas_shape[0] / h0, canvas_shape[1] / w0)  # 0.5
    padw = (canvas_shape[1] - w0 * ratio) / 2  # 0
    padh = (canvas_shape[0] - h0 * ratio) / 2  # 140
    pad_left_top = (padw, padh)
    pad_right_bottom = (padw, padh)

    pred_canvas = torch.tensor([[635.0, 150.0, 645.0, 180.0]])  # xyxy
    gt_canvas = torch.tensor([[630.0, 150.0, 640.0, 180.0]])

    pred_orig = scale_boxes_from_canvas_to_original(
        pred_canvas, canvas_shape, (h0, w0), pad_left_top, pad_right_bottom
    )
    gt_orig = scale_boxes_from_canvas_to_original(
        gt_canvas, canvas_shape, (h0, w0), pad_left_top, pad_right_bottom
    )

    iou_canvas = box_iou(pred_canvas, gt_canvas)[0, 0].item()
    iou_orig = box_iou(pred_orig, gt_orig)[0, 0].item()

    assert abs(iou_canvas - 0.33333) < 1e-4, f"Expected canvas IoU around 0.333, got {iou_canvas}"
    assert abs(
        iou_orig - iou_canvas
    ) < 1e-6, f"Expected original space IoU to be {iou_canvas}, got {iou_orig}"

@pytest.mark.parametrize("pad_w_ratio", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("pad_h_ratio", [0.0, 0.1, 0.2])
def test_iou_invariance_after_descaling(pad_w_ratio, pad_h_ratio):
    h0, w0 = 720, 1280
    canvas_shape = (640, 640)

    ratio = min(canvas_shape[0] / h0, canvas_shape[1] / w0)
    padw = (canvas_shape[1] - w0 * ratio) * pad_w_ratio
    padh = (canvas_shape[0] - h0 * ratio) * pad_h_ratio
    pad_left_top = (padw, padh)
    pad_right_bottom = (padw, padh)

    pred_canvas = torch.tensor([10, 10, 110, 110], dtype=torch.float32).unsqueeze(0)
    gt_canvas = torch.tensor([60, 60, 160, 160], dtype=torch.float32).unsqueeze(0)

    pred_orig = scale_boxes_from_canvas_to_original(
        pred_canvas.clone(), canvas_shape, (h0, w0), pad_left_top, pad_right_bottom
    )
    gt_orig = scale_boxes_from_canvas_to_original(
        gt_canvas.clone(), canvas_shape, (h0, w0), pad_left_top, pad_right_bottom
    )

    iou_canvas = box_iou(pred_canvas, gt_canvas)[0, 0].item()
    iou_orig = box_iou(pred_orig, gt_orig)[0, 0].item()

    if padw == 0 and padh == 0:
        assert abs(iou_canvas - iou_orig) < 1e-6, "IoU should be nearly identical for pure scaling"
    else:
        assert not torch.allclose(
            pred_canvas, pred_orig
        ), "Original box should be different from canvas box after padding"
        assert not torch.allclose(
            gt_canvas, gt_orig
        ), "Original box should be different from canvas box after padding"
