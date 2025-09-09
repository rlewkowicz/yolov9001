"""
Tests for Exponential Moving Average (EMA).
"""
import torch
import torch.nn as nn

from utils.ema import ModelEMA
from tests.config import DEVICE

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.nc = 1

        class Detect(nn.Module):
            def __init__(self):
                super().__init__()
                self.reg_max = 16
                self.strides = [8, 16, 32]

        self.detect_layer = Detect()
        self.layers = nn.ModuleList([self.detect_layer])

    def forward(self, x):
        return self.linear(x)

def test_ema_kept_in_fp32_and_eval():
    """Test that EMA model is kept in FP32 and in eval mode."""
    m = SimpleModel().to(DEVICE)
    if DEVICE == 'cuda':
        m.half()
    ema = ModelEMA(m)
    for v in ema.ema.state_dict().values():
        if v.dtype.is_floating_point:
            assert v.dtype == torch.float32
    assert not ema.ema.training

def test_ema_decay_zero_exact_copy():
    """Test that EMA with zero decay is an exact copy of the model."""
    m = SimpleModel()
    ema = ModelEMA(m, decay=0.0)
    with torch.no_grad():
        m.linear.weight += 1.0
    ema.update(m)
    for v_ema, v_model in zip(ema.ema.state_dict().values(), m.state_dict().values()):
        assert torch.allclose(v_ema, v_model)
