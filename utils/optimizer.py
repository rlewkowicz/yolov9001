import math
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
from torch.optim import SGD as _TorchSGD

from utils.logging import get_logger

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.99),
        weight_decay: float = 0.01,
        alpha: float = 1.0,
        use_bias_correction: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            alpha=alpha,
            use_bias_correction=use_bias_correction,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            alpha = group["alpha"]
            bc = group["use_bias_correction"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avg = state["exp_avg"]

                c_t = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)

                c_hat = c_t.div(1 - beta1**state["step"]) if bc else c_t

                update = (2.0 / math.pi) * torch.atan(alpha * c_hat)

                if wd != 0:
                    update = update.add(p, alpha=wd)

                p.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

def cosine_scheduler(epoch: int, epochs: int, lrf: float = 0.01) -> float:
    """
    Pure cosine schedule (NO WARMUP).
    Guarantees:
      - epoch == 0         -> 1.0
      - epoch == epochs-1  -> lrf
    """
    if epochs <= 1:
        return lrf
    t = epoch / (epochs - 1)
    return lrf + (1 - lrf) * (1 + math.cos(math.pi * t)) / 2

def _norm_types():
    """Get all normalization layer types from PyTorch."""
    mods = []
    for name in (
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "GroupNorm",
        "LayerNorm",
        "LocalResponseNorm",
        "SyncBatchNorm",
    ):
        if hasattr(nn, name):
            mods.append(getattr(nn, name))
    return tuple(mods)

def _build_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_len: int,
    lr_scale_backbone: float,
    lr_scale_head: float,
):
    """
    Build parameter groups with proper weight decay settings.
    Bias and BatchNorm parameters have no weight decay.
    """
    groups = {
        "backbone_weights": [],
        "backbone_biases": [],
        "backbone_norm": [],
        "head_weights": [],
        "head_biases": [],
        "head_norm": [],
    }

    idx_map = {}
    if hasattr(model, "layers"):
        for m in model.layers.modules():
            if hasattr(m, "i"):
                for name, p in m.named_parameters(recurse=False):
                    idx_map[id(p)] = m.i

    norm_t = _norm_types()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        midx = idx_map.get(id(p), None)
        is_backbone = False
        if midx is not None:
            is_backbone = midx < backbone_len
        else:
            try:
                if "layers." in name:
                    parts = name.split(".")
                    li = int(parts[1])
                    is_backbone = li < backbone_len
            except Exception:
                is_backbone = False

        is_bias = name.endswith(".bias")

        owner_is_norm = False
        try:
            mod_name = name.rsplit(".", 1)[0]
            owner = dict(model.named_modules()).get(mod_name, None)
            if owner is not None and isinstance(owner, norm_t):
                owner_is_norm = True
        except Exception:
            owner_is_norm = False

        key = None
        if is_backbone:
            if owner_is_norm:
                key = "backbone_norm"
            elif is_bias:
                key = "backbone_biases"
            else:
                key = "backbone_weights"
        else:
            if owner_is_norm:
                key = "head_norm"
            elif is_bias:
                key = "head_biases"
            else:
                key = "head_weights"

        groups[key].append(p)

    param_groups = []

    if groups["backbone_weights"]:
        param_groups.append({
            "params": groups["backbone_weights"],
            "lr": base_lr * lr_scale_backbone,
            "weight_decay": weight_decay,
            "name": "backbone_weights",
        })

    if groups["backbone_biases"]:
        param_groups.append({
            "params": groups["backbone_biases"],
            "lr": base_lr * lr_scale_backbone,
            "weight_decay": 0.0,  # No weight decay for biases
            "name": "backbone_biases",
        })

    if groups["backbone_norm"]:
        param_groups.append({
            "params": groups["backbone_norm"],
            "lr": base_lr * lr_scale_backbone,
            "weight_decay": 0.0,  # No weight decay for BatchNorm
            "name": "backbone_norm",
        })

    if groups["head_weights"]:
        param_groups.append({
            "params": groups["head_weights"],
            "lr": base_lr * lr_scale_head,
            "weight_decay": weight_decay,
            "name": "head_weights",
        })

    if groups["head_biases"]:
        param_groups.append({
            "params": groups["head_biases"],
            "lr": base_lr * lr_scale_head,
            "weight_decay": 0.0,  # No weight decay for biases
            "name": "head_biases",
        })

    if groups["head_norm"]:
        param_groups.append({
            "params": groups["head_norm"],
            "lr": base_lr * lr_scale_head,
            "weight_decay": 0.0,  # No weight decay for BatchNorm
            "name": "head_norm",
        })

    if not param_groups:
        param_groups = [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "all",
        }]

    return param_groups

def build_optimizer(
    model: torch.nn.Module,
    epochs: int = 100,
    hyp: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
):
    """
    Build Lion optimizer + cosine LR (no warmup).
    Accepts either a YOLOConfig object via cfg or a dict via hyp.
    """
    logger = get_logger()

    if cfg is not None:
        if hasattr(cfg, "to_dict"):
            hyp = cfg.to_dict()
        else:
            hyp = dict(cfg)
    elif hyp is None:
        hyp = getattr(model, "hyp", None)

    if hyp is None:
        raise ValueError(
            "No hyperparameters provided. Pass `cfg=`, `hyp=` or set `model.hyp` before calling build_optimizer()."
        )

    opt_cfg = hyp.get("optimizer", None)
    if opt_cfg is None or "LION" not in opt_cfg:
        raise ValueError(
            "Hyperparameters missing 'optimizer.LION' section required to build Lion optimizer."
        )

    opt_settings = opt_cfg["LION"]

    base_lr = float(opt_settings.get("lr0", 0.003))
    lrf = float(opt_settings.get("lrf", 0.01))
    weight_decay = float(opt_settings.get("weight_decay", 0.004))
    b1 = float(opt_settings.get("b1", 0.9))
    b2 = float(opt_settings.get("b2", 0.999))
    alpha = float(opt_settings.get("alpha", 30))
    bias_correction = bool(opt_settings.get("bias_correction", False))

    decoupled = opt_settings.get("decoupled_lr", {})
    lr_scale_backbone = float(decoupled.get("backbone", {}).get("lr_scale", 1.0))
    lr_scale_head = float(decoupled.get("head", {}).get("lr_scale", 1.0))

    logger.log_text(
        "config/optimizer",
        json.dumps({
            "type": "Lion",
            "base_lr": base_lr,
            "final_lr_factor": lrf,
            "weight_decay": weight_decay,
            "lr_scale_backbone": lr_scale_backbone,
            "lr_scale_head": lr_scale_head,
            "alpha": alpha,
            "betas": (b1, b2),
            "bias_correction": bias_correction,
        },
                   indent=2)
    )

    backbone_len = 10
    if hasattr(model,
               "yaml") and isinstance(getattr(model, "yaml"), dict) and "backbone" in model.yaml:
        try:
            backbone_len = len(model.yaml["backbone"])
        except Exception:
            backbone_len = 10

    param_groups = _build_param_groups(
        model, base_lr, weight_decay, backbone_len, lr_scale_backbone, lr_scale_head
    )

    for group in param_groups:
        num_params = sum(p.numel() for p in group["params"])
        logger.debug(
            f"optimizer/group/{group.get('name','group')}",
            {
                "params": num_params, "lr": group["lr"], "weight_decay":
                    group.get("weight_decay", 0.0)
            },
        )

    optimizer = Lion(
        param_groups,
        lr=base_lr,
        betas=(b1, b2),
        weight_decay=weight_decay,
        alpha=alpha,
        use_bias_correction=bias_correction,
    )

    def lf(e: int):
        return cosine_scheduler(e, epochs, lrf)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    logger.info("optimizer/scheduler", "cosine")
    mid_e = (epochs - 1) // 2 if epochs > 1 else 0
    last_e = max(0, epochs - 1)
    logger.debug(
        "optimizer/schedule_test",
        {"epoch_0": lf(0), "epoch_mid": lf(mid_e), "epoch_final": lf(last_e)},
    )

    return optimizer, scheduler

class WarmupCosineLRMomentum(lr_scheduler._LRScheduler):
    """
    SGD-friendly scheduler with linear warmup (LR + momentum) then cosine LR decay.

    - Warmup phase (0..warmup_epochs-1):
        lr_i(e) = warmup_lr[i] + (base_lr[i] - warmup_lr[i]) * (e / max(1, warmup_epochs))
        momentum(e) = m0 + (mT - m0) * (e / max(1, warmup_epochs))
    - Cosine phase (warmup_epochs..epochs-1):
        lr_i(e) = eta_min + (base_lr[i] - eta_min) * 0.5 * (1 + cos(pi * t)),
        where t = (e - warmup_epochs) / max(1, epochs - warmup_epochs - 1)

    Notes:
      - Momentum warmup is applied only if optimizer param groups have 'momentum'.
      - Designed to step per-epoch (like Ultralytics default). If you step per-iteration,
        pass epochs/ warmup in iterations.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        epochs: int,
        warmup_epochs: float = 3.0,
        lrf: float = 0.01,
        target_momentum: float = 0.937,
        warmup_momentum: float = 0.8,
        warmup_bias_lr: float = 0.05,
        last_epoch: int = -1,
    ) -> None:
        self.total_epochs = max(1, int(epochs))
        self.warmup_epochs = max(0.0, float(warmup_epochs))
        self.lrf = float(lrf)
        self.m_end = float(target_momentum)
        self.m_start = float(warmup_momentum)
        self.warmup_bias_lr = float(warmup_bias_lr)
        super().__init__(optimizer, last_epoch)
        self.base_lrs = [g.get('lr', 0.0) for g in optimizer.param_groups]
        warm = []
        for g in optimizer.param_groups:
            name = str(g.get('name', ''))
            if 'bias' in name:
                warm.append(self.warmup_bias_lr)
            else:
                warm.append(0.0)
        self.warmup_lrs = warm

    def get_lr(self):
        e = self.last_epoch
        if self.warmup_epochs > 0 and e < self.warmup_epochs:
            frac = float(max(0.0, e)) / max(1.0, self.warmup_epochs)
            return [w + (b - w) * frac for w, b in zip(self.warmup_lrs, self.base_lrs)]

        start = int(self.warmup_epochs)
        span = max(1, self.total_epochs - start - 1)
        t = float(max(0, e - start)) / float(span)
        return [
            b *
            (self.lrf + (1.0 - self.lrf) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t)))))
            for b in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):  # type: ignore[override]
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            frac = float(max(0.0, epoch)) / max(1.0, self.warmup_epochs)
            m = self.m_start + (self.m_end - self.m_start) * frac
            for g in self.optimizer.param_groups:
                if 'momentum' in g:
                    g['momentum'] = float(m)

def build_sgd_optimizer(
    model: torch.nn.Module,
    epochs: int = 100,
    hyp: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
):
    """
    Build SGD optimizer with Nesterov, YOLOv8-style warmup + cosine schedule.

    Expects a hyp/cfg structure with an 'optimizer.SGD' section, e.g.:
      optimizer:
        SGD:
          lr0: 0.01
          lrf: 0.01
          momentum: 0.937
          warmup_momentum: 0.8
          warmup_epochs: 3.0
          warmup_bias_lr: 0.05
          nesterov: true
          weight_decay: 0.0005
          decoupled_lr:
            backbone: { lr_scale: 1.0 }
            head: { lr_scale: 1.0 }
    """
    logger = get_logger()

    if cfg is not None:
        if hasattr(cfg, "to_dict"):
            hyp = cfg.to_dict()
        else:
            hyp = dict(cfg)
    elif hyp is None:
        hyp = getattr(model, "hyp", None)

    if hyp is None:
        raise ValueError(
            "No hyperparameters provided. Pass `cfg=`, `hyp=` or set `model.hyp` before calling build_sgd_optimizer()."
        )

    opt_cfg = hyp.get("optimizer", None)
    if opt_cfg is None or "SGD" not in opt_cfg:
        raise ValueError(
            "Hyperparameters missing 'optimizer.SGD' section required to build SGD optimizer."
        )

    s = opt_cfg["SGD"]
    base_lr = float(s.get("lr0", 0.01))
    lrf = float(s.get("lrf", 0.01))
    momentum = float(s.get("momentum", 0.937))
    warm_m = float(s.get("warmup_momentum", 0.8))
    warm_ep = float(s.get("warmup_epochs", 3.0))
    warm_bias_lr = float(s.get("warmup_bias_lr", 0.05))
    nesterov = bool(s.get("nesterov", True))
    weight_decay = float(s.get("weight_decay", 0.0005))

    decoupled = s.get("decoupled_lr", {})
    lr_scale_backbone = float(decoupled.get("backbone", {}).get("lr_scale", 1.0))
    lr_scale_head = float(decoupled.get("head", {}).get("lr_scale", 1.0))

    logger.log_text(
        "config/optimizer",
        json.dumps({
            "type": "SGD",
            "base_lr": base_lr,
            "final_lr_factor": lrf,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "warmup_epochs": warm_ep,
            "warmup_momentum": warm_m,
            "warmup_bias_lr": warm_bias_lr,
            "lr_scale_backbone": lr_scale_backbone,
            "lr_scale_head": lr_scale_head,
        },
                   indent=2)
    )

    backbone_len = 10
    if hasattr(model,
               "yaml") and isinstance(getattr(model, "yaml"), dict) and "backbone" in model.yaml:
        try:
            backbone_len = len(model.yaml["backbone"])
        except Exception:
            backbone_len = 10

    param_groups = _build_param_groups(
        model, base_lr, weight_decay, backbone_len, lr_scale_backbone, lr_scale_head
    )

    optimizer = _TorchSGD(
        param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov
    )

    for g in optimizer.param_groups:
        g['momentum'] = momentum

    scheduler = WarmupCosineLRMomentum(
        optimizer,
        epochs=epochs,
        warmup_epochs=warm_ep,
        lrf=lrf,
        target_momentum=momentum,
        warmup_momentum=warm_m,
        warmup_bias_lr=warm_bias_lr,
    )

    logger.info("optimizer/scheduler", "warmup+cosine (SGD)")
    return optimizer, scheduler

def print_optimizer_info(
    optimizer: Optimizer, scheduler: Optional[lr_scheduler._LRScheduler] = None
):
    logger = get_logger()
    info = {}
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group["params"])
        name = group.get("name", f"group_{i}")
        info[name] = {
            "params": num_params,
            "lr": group["lr"],
            "weight_decay": group.get("weight_decay", 0.0),
        }
    logger.heavy("optimizer/summary", info)
