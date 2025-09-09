import numpy as np
import torch
import pytest

from utils.augment import ExactLetterboxTransform, GPUExactLetterboxTransform

@pytest.mark.parametrize("imgsz", [640, 672])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scaleup", [False])  # training/val setting: never scale up
def test_cpu_gpu_letterbox_parity(imgsz, center, scaleup):
    rng = np.random.default_rng(123)
    h0, w0 = 511, 799
    img = (rng.uniform(0, 255, size=(h0, w0, 3)).astype(np.uint8))
    n = 5
    xyxy_abs = np.stack([
        rng.integers(0, w0 // 2, size=n),
        rng.integers(0, h0 // 2, size=n),
        rng.integers(w0 // 2, w0, size=n),
        rng.integers(h0 // 2, h0, size=n),
    ], 1).astype(np.float32)
    cls = rng.integers(0, 80, size=(n, ), dtype=np.int64)

    pad_value = 114
    lb = ExactLetterboxTransform(
        img_size=imgsz, center=center, pad_value=pad_value, scaleup=scaleup, pad_to_stride=32
    )
    canvas_cpu, yolo_cpu, (oh, ow), r_cpu, pad_cpu = lb(img, xyxy_abs, cls)

    timg = torch.from_numpy(img.transpose(2, 0, 1)).float().cuda() / 255.0
    xyxy_t = torch.from_numpy(xyxy_abs).cuda()
    cls_t = torch.from_numpy(cls).cuda()
    glb = GPUExactLetterboxTransform(
        img_size=imgsz, center=center, pad_value=pad_value, scaleup=scaleup, pad_to_stride=32
    )
    canvas_gpu, yolo_gpu, (ohg, owg), r_gpu, pad_gpu = glb(timg, xyxy_t, cls_t)

    assert (oh, ow) == (ohg, owg)
    assert abs(r_cpu - r_gpu) < 1e-6
    assert all(abs(float(a) - float(b)) < 1e-5 for a, b in zip(pad_cpu, pad_gpu))

    cpu_t = torch.from_numpy(canvas_cpu.transpose(2, 0, 1)).float().cuda() / 255.0
    mae = (cpu_t - canvas_gpu).abs().mean().item()
    assert mae < 1e-3

    assert yolo_cpu.shape == yolo_gpu.shape
    if yolo_cpu.size:
        y_cpu = torch.from_numpy(yolo_cpu).cuda()
        diff = (y_cpu - yolo_gpu).abs().max().item()
        assert diff < 1e-3

def test_letterbox_no_scale_up_train():
    h0, w0 = 300, 500
    img = np.full((h0, w0, 3), 127, dtype=np.uint8)
    lb = ExactLetterboxTransform(
        img_size=640, center=True, pad_value=114, scaleup=False, pad_to_stride=32
    )
    canvas, _, (oh, ow), r, pad = lb(img)
    assert r == 1.0
    assert canvas.shape[0] % 32 == 0 and canvas.shape[1] % 32 == 0
