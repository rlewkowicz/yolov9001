"""
Tests for optimizers and learning rate schedulers.
"""
import torch
import pytest

from utils.optimizer import Lion, cosine_scheduler, build_optimizer
from tests.config import DEVICE, NUM_CLASSES

@pytest.fixture(scope="module")
def yolo_model():
    from models.detect.hyper.model import HyperModel
    return HyperModel(nc=NUM_CLASSES).to(DEVICE)

def test_lion_optimizer_steps(yolo_model):
    """Test Lion optimizer basic functionality."""
    optimizer = Lion(yolo_model.parameters(), lr=0.001)
    x = torch.randn(2, 3, 64, 64, device=DEVICE)
    yolo_model.train()
    y = yolo_model(x)
    loss = torch.cat([o.view(yolo_model.nc + 4 * yolo_model.reg_max, -1) for o in y], 1).mean()
    optimizer.zero_grad()
    loss.backward()
    assert all(p.grad is not None for p in yolo_model.parameters() if p.requires_grad)
    optimizer.step()

def test_cosine_scheduler_monotonic_nonincreasing():
    """Test that the cosine scheduler is non-increasing."""
    epochs, lrf = 100, 0.01
    vals = [cosine_scheduler(e, epochs, lrf) for e in range(epochs)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

def test_build_optimizer_and_scheduler(yolo_model):
    """Test the build_optimizer factory function."""
    optimizer, scheduler = build_optimizer(yolo_model, epochs=100)
    assert isinstance(optimizer, Lion)
    assert scheduler is not None
    assert len(optimizer.param_groups) >= 3
    for group in optimizer.param_groups:
        if 'bias' in group.get('name', '') or 'bn' in group.get('name', ''):
            assert group.get('weight_decay', 0) == 0

def test_param_group_norm_bias_exemptions(yolo_model):
    """Test that norm and bias parameters are exempt from weight decay."""
    optimizer, _ = build_optimizer(yolo_model, epochs=5)
    names = {g.get('name', ''): g for g in optimizer.param_groups}
    for key in ['backbone_norm', 'head_norm', 'backbone_biases', 'head_biases']:
        if key in names:
            assert names[key].get('weight_decay', 0.0) == 0.0
