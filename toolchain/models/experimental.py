import torch
import torch.nn as nn
from utils.downloads import attempt_download
from models.common import GatedSPPF, GatedPool, Requant, QARepNBottleneck

class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 1)
        return (y, None)

def attempt_load(weights, device=None, inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt_path = attempt_download(w)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()

        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))

        for m in ckpt.modules():
            if isinstance(m, GatedSPPF):
                if hasattr(m, 'm') and not hasattr(m, 'm1'):
                    k = m.m.max_pool.kernel_size
                    s = m.m.max_pool.stride
                    m.m1 = GatedPool(kernel_size=k, stride=s)
                    m.m2 = GatedPool(kernel_size=k, stride=s)
                    m.m3 = GatedPool(kernel_size=k, stride=s)
                    m.m1.load_state_dict(m.m.state_dict())
                    m.m2.load_state_dict(m.m.state_dict())
                    m.m3.load_state_dict(m.m.state_dict())
                    del m.m
                if not hasattr(m, 'requant_cat'):
                    m.requant_cat = Requant()

            elif isinstance(m, GatedPool):
                if not hasattr(m, 'requant_out'):
                    m.requant_gate_in = Requant()
                    m.requant_mp_in = Requant()
                    m.requant_ap_in = Requant()
                    m.requant_out = Requant()

            elif isinstance(m, QARepNBottleneck):
                if not hasattr(m, 'requant') and getattr(m, 'add', False):
                    m.requant = Requant()

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())

    for m in model.modules():
        t = type(m)
        if t.__name__ == 'Model' or t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            m.inplace = inplace
        elif t is nn.Upsample and (not hasattr(m, "recompute_scale_factor")):
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]

    print(f"Ensemble created with {weights}\n")
    for k in ("names", "nc", "yaml"):
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
    assert all((model[0].nc == m.nc
                for m in model)), (f"Models have different class counts: {[m.nc for m in model]}")
    return model
