import torch

from utils.dataloaders import create_dummy_dataloader
from core.validator import Validator
from utils.boxes import scale_boxes_from_canvas_to_original, clip_boxes_

def dummy_model(nc=3):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nc = nc
            self.reg_max = 16
            self.names = {i: f"c{i}" for i in range(nc)}
            self.detect_layer = type(
                "dl", (), {
                    "last_shapes": [(80, 80), (40, 40),
                                    (20, 20)], "reg_max": 16, "strides": torch.tensor([8, 16, 32])
                }
            )()
            self.strides = [8, 16, 32]

        def get_detection_state(self):
            return {"strides": [8, 16, 32], "letterbox_center": True, "pad_value": 114}

        def get_postprocessor(self, device):
            from utils.postprocess import Postprocessor
            from utils.geometry import DFLDecoder
            dec = DFLDecoder(reg_max=16, strides=[8, 16, 32], centered=True, device=str(device))
            cfg = type(
                "Cfg", (), {
                    "postprocess_config": {
                        "conf_thresh": 0.01, "iou_thresh": 0.6, "pre_nms_topk": 500,
                        "post_nms_topk": 100, "class_agnostic_nms": False
                    }
                }
            )
            return Postprocessor(cfg(), self.nc, device, model=self, decoder=dec)

        def forward(self, x):
            B, C, H, W = x.shape
            N = (H // 8) * (W // 8) + (H // 16) * (W // 16) + (H // 32) * (W // 32)
            no = self.nc + 4 * self.reg_max
            out = torch.zeros(B, no, N, device=x.device)
            return out, self.detect_layer.last_shapes

    return M()

def test_validator_logs_original_images(tmp_path):
    dl, _ = create_dummy_dataloader(img_size=640, batch_size=2, num_classes=3, length=2)
    model = dummy_model()
    v = Validator(model, device="cpu", cfg={"log_val_images_original": True})
    v.validate(dl)
    assert hasattr(v, "sample_images_orig"), "original images not captured (should after patch)"
    assert isinstance(v.sample_images_orig, list) and len(v.sample_images_orig) > 0
    imgs_orig = v.sample_images_orig[0]
    assert imgs_orig.ndim == 3  # C, H, W
    assert imgs_orig.max() <= 1.0 and imgs_orig.min() >= 0.0

def test_metrics_canvas_vs_original_equivalence(tmp_path):
    Hc = Wc = 640
    pred = torch.tensor([[100.0, 120.0, 200.0, 240.0]])
    gt = torch.tensor([[110.0, 130.0, 210.0, 250.0]])
    H0, W0 = 360, 480
    left, top, right, bottom = 80, 40, 80, 240
    pred0 = scale_boxes_from_canvas_to_original(
        pred, (Hc, Wc), (H0, W0), (left, top), (right, bottom)
    )
    gt0 = scale_boxes_from_canvas_to_original(gt, (Hc, Wc), (H0, W0), (left, top), (right, bottom))
    clip_boxes_(pred0, (H0, W0), pixel_edges=True)
    clip_boxes_(gt0, (H0, W0), pixel_edges=True)
    from utils.metrics import box_iou
    iou_canvas = box_iou(pred, gt)
    iou_orig = box_iou(pred0, gt0)
    assert torch.allclose(
        iou_canvas, iou_orig, atol=1e-3
    ), "IoU mismatch between canvas and original (should pass after patch)"

def test_canvas_clipping_prevents_union_inflation():
    Hc, Wc = 640, 640
    gt = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    pred = torch.tensor([[-10.0, -10.0, 210.0, 210.0]])
    from utils.metrics import box_iou
    iou_no_clip = box_iou(pred, gt)
    from utils.boxes import clip_boxes_
    pred_clip = pred.clone()
    clip_boxes_(pred_clip, (Hc, Wc), pixel_edges=False)
    iou_clip = box_iou(pred_clip, gt)
    assert (iou_clip
            >= iou_no_clip).all(), "Clipping should not worsen IoU here (should pass after patch)"
