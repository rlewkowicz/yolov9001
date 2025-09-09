import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from utils.logging import get_logger

@dataclass
class TrainingState:
    epoch: int = 0
    opt_step: int = 0  # increments on successful optimizer updates
    log_step: int = 0  # increments every batch for logging
    global_step: int = 0
    best_fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

class BaseRunner:
    def __init__(
        self, model: nn.Module, device: str = "cuda", cfg: Optional[Dict[str, Any]] = None
    ):
        device_str = str(device) if isinstance(device, torch.device) else device
        use_cuda = torch.cuda.is_available() and device_str.startswith("cuda")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = model.to(self.device)
        try:
            self.model.to(memory_format=torch.channels_last)
        except Exception:
            pass
        self.cfg = cfg or {}
        self.logger = get_logger()
        self.state = TrainingState()
        self.current_step = 0

    def preprocess(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self.preprocess(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.preprocess(x) for x in batch)
        return batch

    def _maybe_set_log_step(self, step: Optional[int] = None):
        if step is not None:
            self.current_step = step
            if hasattr(self, 'state') and isinstance(self.state, TrainingState):
                self.state.log_step = int(step)
                self.state.global_step = int(step)
            self.logger.set_step(step)
        return self.current_step

    def save_checkpoint(self, path: Path, **kwargs):
        """
        Save comprehensive checkpoint with model, optimizer, scheduler, and training state.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional state to save (optimizer, scheduler, scaler, ema, epoch, etc.)
        """
        try:
            base_model = getattr(self.model, "_orig_mod", self.model)
            if hasattr(base_model, "module"):
                base_model = base_model.module
            model_state = base_model.state_dict()
        except Exception:
            model_state = self.model.state_dict()

        ckpt = {
            "model": model_state,
            "cfg": self.cfg,
            "device": str(self.device),
        }

        if "optimizer" in kwargs and kwargs["optimizer"] is not None:
            ckpt["optimizer"] = kwargs["optimizer"].state_dict()

        if "scheduler" in kwargs and kwargs["scheduler"] is not None:
            ckpt["scheduler"] = kwargs["scheduler"].state_dict()

        if "scaler" in kwargs and kwargs["scaler"] is not None:
            ckpt["scaler"] = kwargs["scaler"].state_dict()

        if "ema" in kwargs and kwargs["ema"] is not None:
            ckpt["ema"] = kwargs["ema"].ema.state_dict()
            ckpt["ema_updates"] = kwargs["ema"].updates

        for key in ["epoch", "best_fitness", "train_loss", "fitness"]:
            if key in kwargs:
                ckpt[key] = kwargs[key]
        ckpt["opt_step"] = int(getattr(self.state, "opt_step", 0))
        ckpt["log_step"] = int(getattr(self.state, "log_step", 0))
        ckpt["global_step"] = int(getattr(self.state, "opt_step", 0))

        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path: Path, strict: bool = True):
        """
        Load comprehensive checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Dict containing checkpoint data
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        def _clean_keys(sd: dict) -> dict:
            if not isinstance(sd, dict):
                return sd
            keys = list(sd.keys())
            if not keys:
                return sd
            cleaned = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("_orig_mod."):
                    nk = nk[len("_orig_mod."):]
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                cleaned[nk] = v
            return cleaned

        model_sd = ckpt.get("model", ckpt)
        try:
            model_sd = _clean_keys(model_sd)
            self.model.load_state_dict(model_sd, strict=strict)
        except Exception:
            self.model.load_state_dict(model_sd, strict=False)

        if "cfg" in ckpt:
            self.cfg.update(ckpt["cfg"])

        if "opt_step" in ckpt:
            self.state.opt_step = int(ckpt["opt_step"])
        if "log_step" in ckpt:
            self.state.log_step = int(ckpt["log_step"])
        if "global_step" in ckpt and "opt_step" not in ckpt:
            self.state.opt_step = int(ckpt["global_step"])
        self.state.global_step = int(getattr(self.state, 'log_step', 0))

        return ckpt
