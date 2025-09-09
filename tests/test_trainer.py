"""
Tests for the training pipeline.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from core.trainer import Trainer
from core.callbacks import CheckpointCallback
from utils.dataloaders import collate_fn

class MockDataset(Dataset):
    """A mock dataset for testing."""
    def __init__(self, nc=80, size=2, imgsz=640):
        self.nc = nc
        self.size = size
        self.imgsz = imgsz

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.randn(3, self.imgsz, self.imgsz)
        target = torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]], dtype=torch.float32)
        orig_shape = (self.imgsz, self.imgsz)
        ratio = 1.0
        pad = (0.0, 0.0)
        return img, target, orig_shape, ratio, pad

def test_trainer_single_step_updates(mock_model_factory):
    """Test that a single training step updates model parameters."""
    cfg = {"epochs": 1, "img_size": 32, "use_ema": False}
    dataset = MockDataset(nc=80, size=2, imgsz=32)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = mock_model_factory(nc=80, device="cpu")
    trainer = Trainer(model, device="cpu", cfg=cfg, amp=False)
    before = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    trainer.train_epoch(loader, trainer.criterion)
    after = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    changed = any((a - b).abs().sum().item() > 1e-8 for a, b in zip(after, before))
    assert changed, "Parameters should change after a training step"

def test_checkpoint_save_and_resume(tmp_path, mock_model_factory):
    """Test that checkpoints are saved and can be resumed correctly."""
    model = mock_model_factory(nc=2, reg_max=16, strides=[8, 16, 32], device="cpu")
    cfg = {"epochs": 2, "ckpt_dir": str(tmp_path), "use_ema": False}
    trainer = Trainer(model, device="cpu", cfg=cfg, amp=False)
    cb = CheckpointCallback(Path(cfg["ckpt_dir"]))
    cb.on_epoch_end(trainer, epoch=0, metrics={"fitness": 0.42})
    assert (Path(cfg["ckpt_dir"]) / "best.pt").is_file()
    new_model = mock_model_factory(nc=2, reg_max=16, strides=[8, 16, 32], device="cpu")
    new_trainer = Trainer(new_model, device="cpu", cfg=cfg, amp=False)
    ckpt = new_trainer.resume_training(Path(cfg["ckpt_dir"]) / "best.pt")
    assert new_trainer.epoch == 1 and ckpt["fitness"] == 0.42
