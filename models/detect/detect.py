import torch
import inspect
import torch.nn as nn
from pathlib import Path
from utils.helpers import load_yaml
from utils.logging import get_logger

class DetectModelBase(nn.Module):
    def __init__(self, nc=80, cfg=None, hyp=None):
        super().__init__()
        self.nc = nc
        if isinstance(hyp, (str, Path)):
            self.hyp = load_yaml(Path(hyp))
        elif isinstance(hyp, dict):
            self.hyp = hyp
        else:
            self.hyp = {}

        if cfg is None:
            subclass_file = Path(inspect.getfile(self.__class__))
            cfg = subclass_file.parent / "model.yaml"

        self.yaml_path = Path(cfg)
        self.yaml = load_yaml(self.yaml_path)

        from core.config import get_config
        config = get_config(hyp=self.hyp)
        self.config_obj = config
        self.hyp = config.to_dict()
        self.strides = config.get(
            'strides'
        )  # sourced from YOLOConfig; attach_runtime will finalize

        self.layers, self.save = self.parse_model(self.yaml)

        self.reg_max = int(self.config_obj.get("reg_max", 16))

        detect_layer = next((m for m in self.layers if m.__class__.__name__ == "Detect"), None)
        if detect_layer:
            detect_layer.reg_max = self.reg_max

        self.dfl_decoder = None
        self._postprocessor = None

        import os
        if os.environ.get("YOLO_DEBUG"):
            try:
                from utils.geometry import _assert_ltrb_order_is_consistent
                _assert_ltrb_order_is_consistent()
                get_logger().info("model/init", "LTRB order consistency check passed.")
            except Exception as e:
                get_logger().warning("model/init", f"LTRB order consistency check failed: {e}")

    def parse_model(self, yaml_cfg):
        """
        Parse a loaded YAML model definition into layers and save indices.

        Note:
            The `yaml_cfg` parameter is intentionally part of the public API and
            is used by subclasses to customize parsing.
            vulture: ignore[unused-variable]
        """
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        if hasattr(self, "dfl_decoder") and self.dfl_decoder is not None:
            self.dfl_decoder.to(device)
        if hasattr(self, "postprocessor") and self.postprocessor is not None:
            try:
                self.postprocessor.device = device
                self.postprocessor.model = self
            except Exception:
                pass
        return self

    def get_postprocessor(self, device=None):
        if not hasattr(self, 'postprocessor') or self.postprocessor is None:
            raise RuntimeError(
                "Model postprocessor not attached. Call attach_runtime(model) first."
            )
        if device and self.postprocessor.device != device:
            self.postprocessor.device = device
        return self.postprocessor

    @torch.no_grad()
    def detect(self, x):
        self.eval()
        return self.forward(x)

    def get_detection_state(self):
        """Returns dict of detection head state needed for postprocessing."""
        if not hasattr(self, "detect_layer"):
            raise AttributeError(
                "Missing detect_layer; call core.runtime.attach_runtime(model) first."
            )

        return {
            "names": getattr(self, "names", None),
            "nc": getattr(self, "nc", None),
            "strides": self.strides,
            "reg_max": self.reg_max,
            "pad_value": int(self.config_obj.get("pad_value", 114)),
            "letterbox_center": bool(self.config_obj.get("letterbox_center", True)),
            "feat_shapes": getattr(self.detect_layer, "last_shapes", None),
        }

    def set_class_names(self, names):
        self.names = [str(n) for n in names]
        self.nc = len(self.names)
        return self

    def get_feature_by_stride(self, target_stride: int = 16):
        """
        Return (index, feature) for the FPN level closest to target_stride.

        Requires Detect head to cache `last_feats` and `strides` during forward.
        Returns None if unavailable (e.g., before first forward).
        """
        det = getattr(self, "detect_layer", None)
        feats = getattr(det, "last_feats", None)
        strides = getattr(det, "strides", None)
        if feats is None or strides is None:
            return None

    def get_pyramid(self):
        """
        Return a list of feature maps from the Detect head if available, else [].
        Intended for tasks like distillation which need intermediate features.
        """
        det = getattr(self, "detect_layer", None)
        feats = getattr(det, "last_feats", None)
        if feats is None:
            return []
        try:
            return list(feats)
        except Exception:
            return feats

    def select_distill_layers(self, teacher_hw: tuple[int, int], k: int = 1):
        """
        Heuristically choose student feature map(s) whose spatial grid best matches the
        teacher token grid. Returns list of (index, feature) pairs.

        Strategy:
        - Compare each student feature map shape (H,W) with teacher (Ht,Wt)
        - Score by L1 grid difference: |H-Ht| + |W-Wt|
        - Return top-k with smallest scores
        Fallbacks: returns [] if no features cached yet.
        """
        try:
            Ht, Wt = int(teacher_hw[0]), int(teacher_hw[1])
        except Exception:
            return []
        det = getattr(self, "detect_layer", None)
        feats = getattr(det, "last_feats", None)
        if feats is None:
            return []
        shapes = []
        for i, f in enumerate(feats):
            try:
                h, w = int(f.shape[-2]), int(f.shape[-1])
                score = abs(h - Ht) + abs(w - Wt)
                shapes.append((score, i, f))
            except Exception:
                continue
        if not shapes:
            return []
        shapes.sort(key=lambda t: (t[0], t[1]))
        out = [(i, f) for (_, i, f) in shapes[:max(1, int(k))]]
        return out
        try:
            if isinstance(strides, torch.Tensor):
                s = strides.detach().float().cpu().tolist()
            else:
                s = list(strides)
            best_i = min(range(len(s)), key=lambda i: abs(float(s[i]) - float(target_stride)))
            return best_i, feats[best_i]
        except Exception:
            return None
