import contextlib
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

def _num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_macs_and_params(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (3, 640, 640),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Rough MACs (multiply-accumulates) counter using forward hooks for common layers.
    Returns (total_macs, total_params, breakdown).
    Notes:
      - Counts Conv2d, Linear, BatchNorm, Pooling, Activation, Upsample.
      - Custom modules are not counted (included in breakdown as 0); total is a lower bound.
      - MACs are reported (not FLOPs). FLOPs ~= 2 * MACs for conv/linear.
    """
    was_training = model.training
    model.eval()

    total: Dict[str, float] = {
        "conv2d": 0.0,
        "linear": 0.0,
        "bn": 0.0,
        "pool": 0.0,
        "act": 0.0,
        "upsample": 0.0,
        "other": 0.0,
    }

    handles = []

    def register(m: nn.Module):
        m.__class__.__name__.lower()

        def hook(module, inp, out):
            try:
                if isinstance(out, (list, tuple)):
                    o = out[0]
                else:
                    o = out
                if not isinstance(o, torch.Tensor):
                    return
                macs = 0.0
                if isinstance(module, nn.Conv2d):
                    out_h, out_w = o.shape[-2], o.shape[-1]
                    k_h, k_w = module.kernel_size
                    cin = module.in_channels
                    cout = module.out_channels
                    groups = max(1, module.groups)
                    macs = float(cout) * float(out_h * out_w) * float((cin / groups) * k_h * k_w)
                    total["conv2d"] += macs
                elif isinstance(module, nn.Linear):
                    in_f = module.in_features
                    out_f = module.out_features
                    macs = float(in_f) * float(out_f)
                    total["linear"] += macs
                elif isinstance(
                    module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)
                ):
                    nelem = o.numel()
                    macs = float(nelem) * 2.0
                    total["bn"] += macs
                elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                    out_h, out_w = o.shape[-2], o.shape[-1]
                    c = o.shape[-3]
                    k = module.kernel_size if isinstance(module.kernel_size,
                                                         int) else module.kernel_size[0]
                    macs = float(c) * float(out_h * out_w) * float(k * k)
                    total["pool"] += macs
                elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.Hardsigmoid)):
                    macs = float(o.numel())
                    total["act"] += macs
                elif isinstance(module, nn.Upsample):
                    macs = float(o.numel())
                    total["upsample"] += macs
                else:
                    total["other"] += 0.0
            except Exception:
                pass

        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))

    model.apply(register)

    dummy = torch.zeros((1, *input_shape), device=device, dtype=dtype)
    with torch.no_grad():
        with torch.autocast(
            device_type=str(device).split(":")[0] if torch.cuda.is_available() else "cpu",
            enabled=False
        ):
            _ = model(dummy)

    for h in handles:
        with contextlib.suppress(Exception):
            h.remove()

    macs_total = sum(total.values())
    params_total = _num_params(model)
    breakdown = {k: float(v) for k, v in total.items()}

    if was_training:
        model.train()
    return macs_total, params_total, breakdown
