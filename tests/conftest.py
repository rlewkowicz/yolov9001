"""
Pytest configuration and shared fixtures.

- Force conservative defaults for dataloading during tests
- Provide common fixtures: mock_model_factory, model, loss_fn, dummy_input, dummy_targets
"""
import torch
import pytest
from tests.config import DEVICE, NUM_CLASSES, IMG_SIZE, REG_MAX
from core.config import YOLOConfig
from models.detect.hyper.model import HyperModel
from utils.loss import DetectionLoss

def pytest_sessionstart(session):
    YOLOConfig.DEFAULTS["gpu_collate"] = False
    YOLOConfig.DEFAULTS["cuda_prefetch"] = False
    YOLOConfig.DEFAULTS["stats_enabled"] = False

@pytest.fixture(scope="module")
def mock_model_factory():
    """Factory to build a minimal HyperModel for tests with configurable params."""
    def _create(
        nc=NUM_CLASSES,
        reg_max=REG_MAX,
        strides=(8, 16, 32),
        device=DEVICE,
        cfg=None,
        **hyp_overrides
    ):
        if cfg is not None:
            try:
                hyp_dict = cfg.to_dict() if hasattr(cfg, 'to_dict') else dict(cfg)
            except Exception:
                hyp_dict = YOLOConfig().to_dict()
        else:
            hyp_dict = YOLOConfig().to_dict()
        hyp_dict.update(hyp_overrides)
        hyp_dict['reg_max'] = int(reg_max)
        m = HyperModel(nc=nc, ch=3, hyp=hyp_dict).to(device)
        from core.runtime import attach_runtime
        attach_runtime(m, imgsz=IMG_SIZE)
        try:
            if hasattr(m, 'detect_layer') and m.detect_layer is not None and strides is not None:
                s = torch.as_tensor(list(strides), dtype=torch.int64, device=device)
                m.detect_layer.strides = s
                m.strides = s
        except Exception:
            pass
        return m

    return _create

@pytest.fixture(scope="module")
def model(mock_model_factory):
    return mock_model_factory(nc=NUM_CLASSES, reg_max=REG_MAX, strides=(8, 16, 32), device=DEVICE)

@pytest.fixture(scope="module")
def loss_fn(model):
    return DetectionLoss(model, imgsz=IMG_SIZE)

@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

@pytest.fixture(scope="module")
def dummy_targets():
    t0 = torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=DEVICE)
    t1 = torch.zeros((0, 5), dtype=torch.float32, device=DEVICE)
    return [t0, t1]
