import torch
import torch.nn as nn
import inspect
from utils.downloads import attempt_download
from models.common import GatedSPPF, GatedPool, Requant, QARepNBottleneck

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        outs = []
        for module in self:
            y = module(x, augment=augment, profile=profile, visualize=visualize)
            if isinstance(y, (list, tuple)):
                y = y[0]
            outs.append(y)
        y = torch.cat(outs, 1)
        return (y, None)

def _filter_forward_kwargs(m: nn.Module) -> nn.Module:
    f = m.forward
    sig = inspect.signature(f)
    allowed = tuple(k for k in sig.parameters.keys() if k != "self")

    def fw(*args, **kwargs):
        fk = {k: v for k, v in kwargs.items() if k in allowed}
        return f(*args, **fk)

    m.forward = fw
    return m

def attempt_load(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt_path = attempt_download(w)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        obj = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        m = obj.to(device).float() if hasattr(obj, "to") else obj

        if not hasattr(m, "stride"):
            m.stride = torch.tensor([32.0])
        if hasattr(m, "names") and isinstance(m.names, (list, tuple)):
            m.names = dict(enumerate(m.names))

        for mod in m.modules():
            if isinstance(mod, GatedSPPF):
                if hasattr(mod, "m") and not hasattr(mod, "m1"):
                    k = mod.m.max_pool.kernel_size
                    s = mod.m.max_pool.stride
                    mod.m1 = GatedPool(kernel_size=k, stride=s)
                    mod.m2 = GatedPool(kernel_size=k, stride=s)
                    mod.m3 = GatedPool(kernel_size=k, stride=s)
                    mod.m1.load_state_dict(mod.m.state_dict())
                    mod.m2.load_state_dict(mod.m.state_dict())
                    mod.m3.load_state_dict(mod.m.state_dict())
                    delattr(mod, "m")
                if not hasattr(mod, "requant_cat"):
                    mod.requant_cat = Requant()
            elif isinstance(mod, GatedPool):
                if not hasattr(mod, "requant_out"):
                    mod.requant_gate_in = Requant()
                    mod.requant_mp_in = Requant()
                    mod.requant_ap_in = Requant()
                    mod.requant_out = Requant()
            elif isinstance(mod, QARepNBottleneck):
                if not hasattr(mod, "requant") and getattr(mod, "add", False):
                    mod.requant = Requant()

        m = m.fuse().eval() if fuse and hasattr(m, "fuse") else m.eval()
        m = _filter_forward_kwargs(m)
        model.append(m)

    for mod in model.modules():
        t = type(mod)
        if t.__name__ == "Model" or t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            mod.inplace = inplace
        elif t is nn.Upsample and (not hasattr(mod, "recompute_scale_factor")):
            mod.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]

    print(f"Ensemble created with {weights}\n")
    for k in ("names", "nc", "yaml"):
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    assert all((model[0].nc == m.nc for m in model)), f"Models have different class counts: {[m.nc for m in model]}"
    return model
