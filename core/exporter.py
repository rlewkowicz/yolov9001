import torch.nn as nn
from .base import BaseRunner

class Exporter(BaseRunner):
    """Export YOLO models to various formats."""
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.model.eval()
