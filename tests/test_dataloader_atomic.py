import json
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import pytest

from utils.dataloaders import create_dataloader, collate_fn, infer_splits
from utils.boxes import yolo_to_xyxy, scale_boxes_from_canvas_to_original, scale_boxes_from_original_to_canvas
from utils.augment import apply_gpu_batch_aug
from utils.dataloaders import IMG_EXTS  # if exported; otherwise hardcode

RNG = np.random.default_rng(0)

def _rand_hw(max_w=1920, max_h=1080) -> Tuple[int, int]:
    w = int(RNG.integers(32, max_w))
    h = int(RNG.integers(32, max_h))
    return w, h

def _write_solid(dst: Path, w: int, h: int, color=(114, 114, 114)):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    k = RNG.integers(1, 4)
    for _ in range(int(k)):
        x1, y1 = RNG.integers(0, max(1, w // 2)), RNG.integers(0, max(1, h // 2))
        x2, y2 = RNG.integers(max(1, w // 2), w), RNG.integers(max(1, h // 2), h)
        color2 = tuple(int(c) for c in RNG.integers(0, 255, size=3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color2, thickness=-1)
    cv2.imwrite(str(dst), img)

def _rand_yolo_box(w, h):
    cx = float(RNG.random())
    cy = float(RNG.random())
    ww = float(RNG.uniform(0.05, 0.6))
    hh = float(RNG.uniform(0.05, 0.6))
    return (cx, cy, ww, hh)

def _write_yolo_labels(dst: Path, n: int, nc: int = 3):
    lines = []
    for _ in range(n):
        c = int(RNG.integers(0, nc))
        cx, cy, ww, hh = _rand_yolo_box(1, 1)
        lines.append(f"{c} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")
    dst.write_text("".join(lines))

def _make_big_yolo_dataset(root: Path, n=1024, nc=3, label_prob=0.7):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    names = [f"class_{i}" for i in range(nc)]
    (root / "data.yaml").write_text(json.dumps({"nc": nc, "names": names}))
    for i in range(n):
        split = "train" if i % 2 == 0 else "val"
        w, h = _rand_hw()
        p = root / "images" / split / f"im_{i:05d}.jpg"
        _write_solid(p, w, h)
        lbl_path = root / "labels" / split / f"im_{i:05d}.txt"
        if RNG.random() < label_prob:
            nbox = int(RNG.integers(1, 5))
            _write_yolo_labels(lbl_path, nbox, nc)
        else:
            lbl_path.write_text("")  # empty allowed

@pytest.fixture(scope="module")
def big_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("big_yolo")
    _make_big_yolo_dataset(d, n=1024, nc=5, label_prob=0.75)
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_infer_splits_and_yaml_names(big_dataset):
    splits = infer_splits(big_dataset)
    assert "train" in splits and "val" in splits

def test_create_dataloader_and_types(big_dataset):
    loader, info = create_dataloader(
        big_dataset, "train", img_size=640, batch_size=8, num_workers=0, augment=False
    )
    assert info["nc"] >= 1 and info["format"] in {"yolo", "coco"}
    imgs, labels, orig_shapes, ratios, pads = next(iter(loader))
    assert imgs.dtype == torch.uint8 and imgs.ndim == 4 and imgs.shape[-2:] == (640, 640)
    assert (imgs >= 0).all() and (imgs <= 255).all()
    assert isinstance(labels, torch.Tensor) and labels.shape[1] == 6

def test_letterbox_forward_and_inverse_on_loader(big_dataset):
    loader, _ = create_dataloader(
        big_dataset, "train", img_size=640, batch_size=4, num_workers=0, augment=False
    )
    imgs, labels, orig_shapes, ratios, pads = next(iter(loader))
    if labels.numel() == 0:
        pytest.skip("no labels in this batch; re-run")
    bidx = int(labels[0, 0].item())
    yolo_box = labels[labels[:, 0] == bidx][:, 2:6]
    canvas_shape = imgs.shape[-2:]  # (Hc, Wc)
    xyxy_canvas = yolo_to_xyxy(yolo_box, canvas_shape)
    left, top, right, bottom = pads[bidx]
    oh, ow = orig_shapes[bidx]
    xyxy_orig = scale_boxes_from_canvas_to_original(
        xyxy_canvas, canvas_shape, (oh, ow), (left, top), (right, bottom)
    )
    xyxy_canvas2 = scale_boxes_from_original_to_canvas(
        xyxy_orig, canvas_shape, (oh, ow), (left, top), (right, bottom)
    )
    assert torch.allclose(xyxy_canvas, xyxy_canvas2, atol=2.0)

def test_gpu_batch_aug_flips_and_hsv():
    B, C, H, W = 4, 3, 320, 480
    imgs = torch.rand(B, C, H, W)
    tlist = []
    for i in range(B):
        tlist.append(torch.tensor([i, 0, 0.2 + 0.1 * i, 0.3, 0.25, 0.25], dtype=torch.float32))
    targets = torch.stack(tlist, 0)
    hyp = {"fliplr": 1.0, "flipud": 1.0, "hsv_s": 0.5, "hsv_v": 0.5}
    imgs2, targets2 = apply_gpu_batch_aug(imgs.clone(), targets.clone(), hyp)
    assert torch.allclose(targets2[:, 2], 1.0 - targets[:, 2], atol=1e-6)
    assert torch.allclose(targets2[:, 3], 1.0 - targets[:, 3], atol=1e-6)
    assert (imgs2 >= 0).all() and (imgs2 <= 1).all()

def test_dataset_level_mixup_copy_paste(big_dataset):
    loader, _ = create_dataloader(
        big_dataset,
        "train",
        img_size=320,
        batch_size=2,
        num_workers=0,
        augment=True,
        hyp={'mixup': 1.0, 'copy_paste': 1.0}
    )
    dataset = loader.dataset
    img, labels, _, _, _ = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert labels.ndim == 2 and labels.shape[1] == 5
    assert (labels[:, 1:] >= 0).all() and (labels[:, 1:] <= 1).all()

def test_collate_fn_dtype_and_shape(big_dataset):
    loader, _ = create_dataloader(
        big_dataset, "train", img_size=320, batch_size=3, num_workers=0, augment=False
    )
    batch = [loader.dataset[i] for i in range(3)]
    imgs, targets, orig_shapes, ratios, pads = collate_fn(batch)
    assert imgs.shape[0] == 3 and imgs.dtype == torch.uint8
    assert targets.ndim == 2 and targets.shape[1] == 6

def test_scale_augmentation_does_not_break_labels(big_dataset):
    hyp = {"scale": 0.5}
    loader, _ = create_dataloader(
        big_dataset, "train", img_size=320, batch_size=4, num_workers=0, augment=True, hyp=hyp
    )
    imgs, labels, *_ = next(iter(loader))
    if labels.numel():
        assert (labels[:, 2:6] >= 0).all() and (labels[:, 2:6]
                                                <= 1).all(), "labels must be clipped to [0,1]"

def test_letterbox_forward_inverse_pair_consistency():
    H0, W0 = 375, 500
    Hc, Wc = 640, 640
    left, top, right, bottom = 34, 10, 34, 10  # example pads
    xyxy_orig = torch.tensor([[50.0, 40.0, 200.0, 180.0]], dtype=torch.float32)

    xyxy_canvas = scale_boxes_from_original_to_canvas(
        xyxy_orig, (Hc, Wc), (H0, W0), (left, top), (right, bottom)
    )
    xyxy_orig2 = scale_boxes_from_canvas_to_original(
        xyxy_canvas, (Hc, Wc), (H0, W0), (left, top), (right, bottom)
    )
    assert torch.allclose(xyxy_orig, xyxy_orig2, atol=1e-4)
