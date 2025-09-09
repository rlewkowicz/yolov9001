import numpy as np
from utils.augment import mosaic4_raw, random_affine_raw, apply_hsv_inplace

class _DummyDataset:
    def __init__(self, imgsz=640, pad_value=114, k=200):
        self.imgsz = imgsz
        self.pad_value = pad_value
        self._items = []
        rng = np.random.default_rng(42)
        for i in range(k):
            h = rng.integers(400, 900)
            w = rng.integers(400, 1000)
            img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            n = rng.integers(1, 6)
            boxes = np.stack([
                rng.integers(0, w // 2, size=n),
                rng.integers(0, h // 2, size=n),
                rng.integers(w // 2, w, size=n),
                rng.integers(h // 2, h, size=n),
            ], 1).astype(np.float32)
            cls = rng.integers(0, 10, size=(n, ), dtype=np.int64)
            self._items.append((img, boxes, cls, (h, w)))

    def __len__(self):
        return len(self._items)

    def _load_raw(self, idx):
        return self._items[idx]

def test_mosaic_fixed_size_and_bounds():
    ds = _DummyDataset(imgsz=640)
    img, boxes, cls = mosaic4_raw(ds, 0)
    h, w = img.shape[:2]
    assert (h, w) == (640, 640)
    if boxes.size:
        assert boxes.min() >= 0 - 1e-5
        assert boxes[:, 0].max() <= w + 1e-5 and boxes[:, 2].max() <= w + 1e-5
        assert boxes[:, 1].max() <= h + 1e-5 and boxes[:, 3].max() <= h + 1e-5

def test_affine_with_perspective_and_hsv():
    rng = np.random.default_rng(0)
    h0, w0 = 500, 700
    img = rng.integers(0, 255, size=(h0, w0, 3), dtype=np.uint8)
    boxes = np.array([[50, 60, 200, 220], [300, 100, 600, 450]], dtype=np.float32)
    cls = np.array([1, 2], dtype=np.int64)
    out = random_affine_raw(
        img,
        boxes,
        cls,
        640,  # imgsz parameter
        degrees=5.0,
        translate=0.05,
        scale=0.10,
        shear=2.0,
        perspective=0.001,
        pad_value=114
    )
    img2, b2, c2 = out
    assert len(img2.shape) == 3 and img2.shape[2] == 3  # Still RGB
    assert b2.shape[1] == 4 and c2.shape[0] == b2.shape[0]
    img3 = apply_hsv_inplace(img2.copy(), h_hyp=0.015, s_hyp=0.7, v_hyp=0.4)
    assert img3.shape == img2.shape