import torch, pytest
from utils.postprocess import Postprocessor, postprocess_detections
from utils.geometry import DFLDecoder
from core.config import get_config
from tests.config import DEVICE

def test_stride_drift_detection():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detect_layer = type(
                "DL", (), {"last_shapes": [(80, 80), (40, 40)], "strides": [8, 16]}
            )()
            self.strides = self.detect_layer.strides
            self.reg_max = 16
            self.nc = 1
            self.hyp = {}

    m = M()
    cfg = type(
        "C", (), {
            "postprocess_config": {
                "conf_thresh": 0.3, "iou_thresh": 0.5, "pre_nms_topk": 1000, "post_nms_topk": 300,
                "class_agnostic_nms": False
            }
        }
    )()
    pp = Postprocessor(
        cfg,
        nc=1,
        device=torch.device("cpu"),
        model=m,
        decoder=DFLDecoder(reg_max=16, strides=[8, 16], device="cpu")
    )
    logits = torch.randn(1, 1 + 4 * 16, 80 * 80 + 40 * 39)  # Wrong W on level 2 to trip the check
    with pytest.raises(AssertionError):
        pp(logits, img_size=640, feat_shapes=[(80, 80), (40, 39)])

def test_postprocessor_deterministic_channel_split(mock_model_factory):
    """Test that the postprocessor deterministically splits channels into [reg, cls]."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nc, reg_max, strides = 5, 16, [8, 16, 32]
    model = mock_model_factory(nc=nc, reg_max=reg_max, strides=strides, device=device)
    pp = Postprocessor(get_config(), nc=nc, device=device, model=model, decoder=model.dfl_decoder)
    num_anchors = sum(h * w for h, w in model.detect_layer.last_shapes)
    outputs = torch.randn(1, nc + 4 * reg_max, num_anchors, device=device)
    dets = pp(outputs, img_size=640, feat_shapes=model.detect_layer.last_shapes)
    assert isinstance(dets, list)

def test_postprocessor_raises_on_mismatched_anchor_count(mock_model_factory):
    """Test that Postprocessor raises an error for mismatched anchor counts."""
    cfg = get_config()
    model = mock_model_factory(nc=80, reg_max=16, strides=[16, 128], device='cpu')
    pp = Postprocessor(cfg, nc=80, device='cpu', model=model, decoder=model.dfl_decoder)
    outputs = torch.randn(1, 80 + 16 * 4, 500)
    wrong_shapes = [(40, 40), (5, 5)]
    with pytest.raises(AssertionError, match="Anchor count.*!= predictions"):
        pp(outputs, img_size=640, feat_shapes=wrong_shapes)

def test_non_square_level_rejected(mock_model_factory):
    mock_model = mock_model_factory()
    cfg = get_config()
    pp = Postprocessor(
        cfg, nc=mock_model.nc, device='cpu', model=mock_model, decoder=mock_model.dfl_decoder
    )
    wrong_shapes = [(40, 41), *mock_model.detect_layer.last_shapes[1:]]
    with pytest.raises(AssertionError, match="Non-square stride"):
        pp(
            torch.randn(
                1, mock_model.nc + 4 * mock_model.reg_max, sum(h * w for h, w in wrong_shapes)
            ),
            img_size=640,
            feat_shapes=wrong_shapes
        )

def test_class_aware_vs_agnostic_nms(mock_model_factory):
    mock_model = mock_model_factory(device='cpu')
    cfg = get_config({
        'conf_thresh': 0.01, 'iou_thresh': 0.5, 'pre_nms_topk': 50, 'post_nms_topk': 20,
        'class_agnostic_nms': False
    })
    pp = Postprocessor(
        cfg, nc=mock_model.nc, device='cpu', model=mock_model, decoder=mock_model.dfl_decoder
    )
    num_anchors = sum(h * w for h, w in mock_model.detect_layer.last_shapes)
    outputs = torch.randn(1, mock_model.nc + 4 * mock_model.reg_max, num_anchors)
    out1 = pp(outputs, img_size=640, feat_shapes=mock_model.detect_layer.last_shapes)
    cfg.hyp.update({'class_agnostic_nms': True})
    pp2 = Postprocessor(
        cfg, nc=mock_model.nc, device='cpu', model=mock_model, decoder=mock_model.dfl_decoder
    )
    out2 = pp2(outputs, img_size=640, feat_shapes=mock_model.detect_layer.last_shapes)
    assert isinstance(out1, list) and isinstance(out2, list)

def test_postprocess_tau_alignment(mock_model_factory):
    """Check that postprocessor's DFL decoder tau is updated by the model's tau hyp."""
    model = mock_model_factory(device='cpu')
    pp = model.get_postprocessor(device='cpu')
    assert pp.decoder.tau == 1.0
    hyp = {'dfl_tau': 0.5}
    cfg = get_config(hyp=hyp)
    model2 = mock_model_factory(device='cpu', cfg=cfg)
    pp.model = model2
    assert pp.decoder.tau == 0.5

def test_postprocessor_no_pre_nms_topk():
    """
    Verify that the postprocessor's keep_mask path works correctly when
    pre_nms_topk is None, which was previously an incomplete code path.
    """
    B, N, reg_max, nc = 2, 128, 16, 3
    outputs = torch.zeros(B, reg_max * 4 + nc, N, device=DEVICE)
    outputs[:, reg_max * 4 + 0, 0:5] = 5.0
    outputs[:, 0:reg_max * 4:reg_max, :] = 3.0

    feat_shapes = [(8, 16)]  # 8*16=128 anchors
    decoder = DFLDecoder(reg_max=reg_max, strides=[8], centered=True, device=DEVICE)

    dets = postprocess_detections(
        outputs,
        img_size=640,
        nc=nc,
        device=DEVICE,
        conf_thresh=0.25,
        iou_thresh=0.7,
        reg_max=reg_max,
        decoder=decoder,
        feat_shapes=feat_shapes,
        pre_nms_topk=None  # Explicitly test the previously broken path
    )

    assert isinstance(dets, list)
    assert len(dets) == B
    assert all(isinstance(d, dict) for d in dets)
    assert all('boxes' in d and d['boxes'].ndim == 2 for d in dets)

def test_nms_class_separation():
    """
    Builds two overlapping boxes, different classes; asserts class-agnostic NMS suppresses one,
    while class-aware NMS keeps both.
    """
    B, N, reg_max, nc = 1, 2, 16, 2
    iou_thresh = 0.5
    boxes = torch.tensor([[[100, 100, 200, 200], [110, 110, 210, 210]]],
                         device=DEVICE,
                         dtype=torch.float32)
    scores = torch.tensor([[0.9, 0.8]], device=DEVICE, dtype=torch.float32)
    classes = torch.tensor([[0, 1]], device=DEVICE)

    flat_boxes = boxes.reshape(-1, 4)
    flat_scores = scores.reshape(-1)
    flat_classes = classes.reshape(-1)
    flat_batch = torch.zeros_like(flat_scores, dtype=torch.long)

    try:
        from torchvision.ops import batched_nms
    except ImportError:
        batched_nms = None

    assert batched_nms is not None, "torchvision.ops.batched_nms not available"

    idxs_agnostic = torch.zeros_like(flat_classes)
    keep_agnostic = batched_nms(flat_boxes, flat_scores, idxs_agnostic, iou_thresh)
    assert keep_agnostic.shape[0] == 1, "Class-agnostic NMS should suppress one box"

    idxs_aware = flat_classes
    keep_aware = batched_nms(flat_boxes, flat_scores, idxs_aware, iou_thresh)
    assert keep_aware.shape[0] == 2, "Class-aware NMS should keep both boxes"

import builtins

def _force_no_torchvision(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torchvision.ops"):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

@pytest.mark.cuda(optional=True)
def test_postprocess_nms_fallback_no_torchvision(monkeypatch):
    _force_no_torchvision(monkeypatch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 1
    reg_max = 4
    nc = 2
    img_size = 64

    decoder = DFLDecoder(reg_max=reg_max, strides=[8], device=device)
    feat_shapes = [(img_size // 8, img_size // 8)]  # (8,8) grid => N = 64
    anchor_points, stride_tensor = decoder.get_anchors(feat_shapes)
    N = anchor_points.size(0)

    outputs = torch.zeros(B, nc + reg_max * 4, N, device=device)
    cls_logits = torch.full((B, nc, N), -10.0, device=device)
    hot_idx = torch.tensor([0, 1], device=device)  # two anchors
    cls_logits[0, 0, hot_idx] = 10.0  # class 0, two anchors high
    outputs[:, reg_max * 4:, :] = cls_logits

    pred_dist = torch.full((B, N, 4 * reg_max), -6.0, device=device)
    for side in range(4):
        pred_dist[0, :, side * reg_max + 1] = 6.0
    outputs[:, :reg_max * 4, :] = pred_dist.permute(0, 2, 1).contiguous()

    dets = postprocess_detections(
        outputs=outputs,
        img_size=img_size,
        nc=nc,
        device=device,
        conf_thresh=0.2,
        iou_thresh=0.5,
        reg_max=reg_max,
        pre_nms_topk=None,
        post_nms_topk=None,
        class_agnostic_nms=True,  # simpler code path
        decoder=decoder,
        feat_shapes=feat_shapes
    )

    assert isinstance(dets, list) and len(dets) == B
    assert all(k in dets[0] for k in ("boxes", "scores", "class_ids"))
    assert dets[0]["scores"].numel() >= 1
