import numpy as np
import torch
from core.detector import Detector
from utils.augment import ExactLetterboxTransform

class DummyPost:
    def __init__(self, box_canvas, score=0.9, cls=0):
        self.conf_thresh = 0.0
        self.iou_thresh = 0.5
        self.decoder = None
        self.box_canvas = torch.tensor(box_canvas, dtype=torch.float32).unsqueeze(0)  # [1,4]

    def __call__(self, outputs, img_size, feat_shapes=None):
        return [{
            "boxes": self.box_canvas, "scores": torch.tensor([0.9]), "class_ids":
                torch.tensor([0], dtype=torch.long)
        }]

class DummyModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 1
        self.names = {0: "obj"}
        self.dummy_param = torch.nn.Parameter(torch.empty(0))
        self.reg_max = 16
        self.strides = [8, 16, 32]
        self.hyp = {}
        self.detect_layer = type(
            "DL", (),
            {"last_shapes": [(80, 80), (40, 40), (20, 20)], "strides": torch.tensor(self.strides)}
        )()

    def get_detection_state(self):
        return {"strides": self.strides, "letterbox_center": True, "pad_value": 114}

    def get_postprocessor(self, device=None):
        from utils.postprocess import Postprocessor
        from utils.geometry import DFLDecoder
        cfg = type(
            "C", (), {
                "postprocess_config": {
                    "conf_thresh": 0.01, "iou_thresh": 0.5, "pre_nms_topk": 1000, "post_nms_topk":
                        300, "class_agnostic_nms": False
                }
            }
        )()
        return Postprocessor(
            cfg,
            self.nc,
            device,
            model=self,
            decoder=DFLDecoder(reg_max=16, strides=self.strides, device=device.type)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        self.detect_layer.last_shapes = [(H // s, W // s) for s in self.strides]
        N = sum(h * w for h, w in self.detect_layer.last_shapes)
        logits = torch.randn(B, self.nc + 4 * 16, N, device=x.device)  # random predictions
        return logits, self.detect_layer.last_shapes

def test_detector_rescale_identity(tmp_path):
    h0, w0, imgsz = 240, 320, 256
    original_box = torch.tensor([[10, 20, 110, 220]], dtype=torch.float32)

    img = np.zeros((h0, w0, 3), dtype=np.uint8)
    transform = ExactLetterboxTransform(img_size=imgsz, center=True, pad_to_stride=32)
    canvas, _, _, r, pad = transform(img)

    canvas_box = original_box.clone()
    canvas_box[:, [0, 2]] = original_box[:, [0, 2]] * r + pad[0]
    canvas_box[:, [1, 3]] = original_box[:, [1, 3]] * r + pad[1]

    model = DummyModel2()
    model.get_postprocessor = lambda device=None: DummyPost(canvas_box.squeeze().tolist())

    det = Detector(model, device="cpu")

    result = det.detect(img, img_size=imgsz)

    assert torch.allclose(result['boxes'], original_box, atol=1e-4)
