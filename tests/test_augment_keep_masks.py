import numpy as np
from utils.augment import random_affine_raw, smart_crop_to_size

def _make_img_wh(w, h, color=114):
    return np.full((h, w, 3), color, dtype=np.uint8)

def test_smart_crop_cls_sync_drops_boxes():
    w0, h0 = 800, 600
    img = _make_img_wh(w0, h0)
    boxes = np.array(
        [
            [50, 50, 150, 150],  # top-left
            [700, 450, 790, 590],  # bottom-right (likely to be dropped)
            [300, 200, 500, 400],  # center-ish
        ],
        dtype=np.float32
    )
    cls = np.array([1, 2, 3], dtype=np.int64)
    crop, b2, c2 = smart_crop_to_size(img, boxes, size=512, cls=cls)
    assert isinstance(c2, np.ndarray)
    assert b2.shape[0] == c2.shape[0], "boxes and cls must remain aligned after crop"
    assert b2.shape[1] == 4
    assert (b2[:, 2] > b2[:, 0]).all() and (b2[:, 3] > b2[:, 1]).all()

def test_random_affine_cls_sync_with_drop_attempts():
    w0, h0 = 700, 500
    img = _make_img_wh(w0, h0)
    boxes = np.array([[50, 60, 200, 220], [300, 100, 600, 450], [10, 10, 30, 30]], dtype=np.float32)
    cls = np.array([5, 7, 9], dtype=np.int64)
    ok = False
    for _ in range(30):
        out = random_affine_raw(
            img,
            boxes,
            cls,
            imgsz=512,
            degrees=10.0,
            translate=0.10,
            scale=0.10,
            shear=3.0,
            perspective=0.005,
            pad_value=114,
        )
        img2, b2, c2 = out
        assert img2.ndim == 3 and img2.shape[2] == 3, "Should be 3-channel image"
        assert b2.shape[1] == 4
        assert c2.shape[0] == b2.shape[0], "boxes and cls must remain aligned after affine"
        if b2.shape[0] < boxes.shape[0]:
            ok = True
            break
    assert ok, "Could not induce a drop; re-run test if flakiness occurs"
