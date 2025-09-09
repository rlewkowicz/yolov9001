"""
End-to-end and integration tests for the training and validation pipelines.
"""
import numpy as np
from core.config import get_config
from core.detector import Detector
from core.runtime import attach_runtime

def test_letterbox_and_pad_value_consistency(mock_model_factory):
    cfg = get_config({'pad_value': 77, 'letterbox_center': True})
    mock_model = mock_model_factory(pad_value=77, letterbox_center=True)
    det = Detector(mock_model, device='cpu', cfg=cfg)
    state = mock_model.get_detection_state()
    assert state['pad_value'] == 77 and state['letterbox_center'] is True

def test_runtime_letterbox_and_pad_consistency(mock_model_factory):
    hyp = {
        "pad_value": 77, "letterbox_center": False, "conf_thresh": 0.05, "iou_thresh": 0.5,
        "pre_nms_topk": 50, "post_nms_topk": 50, "class_agnostic_nms": True
    }
    cfg = get_config(hyp=hyp)
    model = mock_model_factory(device='cpu')
    model.hyp = cfg.to_dict()
    attach_runtime(model, imgsz=320)
    det = Detector(model=model, device='cpu', cfg=cfg)
    assert det.pad_value == 77
    assert det.letterbox_center is False
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    _ = det.detect(img, conf_thresh=0.05, iou_thresh=0.5, img_size=320)
