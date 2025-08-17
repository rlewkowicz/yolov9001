import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import (
    Conv, DFL, Requant, RepBlockAdd, QARepVGGBlockV2, AConv, GatedSPPF, QARepNCSPELAN, QuantAdd
)
from utils.general import LOGGER, make_divisible
from utils.tal.anchor_generator import make_anchors, dist2bbox
from utils.torch_utils import initialize_weights, model_info

class RN_DualDDetect(nn.Module):
    """
    Dual-head YOLO detect head (aux + main), with regression normalization.
    No math changes vs your version — only stride bookkeeping made explicit.
    """
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)

    def __init__(self, nc=80, ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch) // 2  # number of P-levels per branch
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.inplace = inplace
        self.register_buffer('stride', torch.tensor([], dtype=torch.float32))  # main head strides
        self.register_buffer('strides', torch.tensor([], dtype=torch.float32))
        self.export_logits = True

        c2 = make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4)
        c3 = max(ch[0], min(self.nc * 2, 128))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3, g=4), nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4)
            ) for x in ch[:self.nl]
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch[:self.nl]
        )
        self.dfl = DFL(self.reg_max)

        c4 = make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4)
        c5 = max(ch[self.nl], min(self.nc * 2, 128))
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3), Conv(c4, c4, 3, g=4), nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4)
            ) for x in ch[self.nl:]
        )
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1))
            for x in ch[self.nl:]
        )
        self.dfl2 = DFL(self.reg_max)

        self.requant1 = Requant()
        self.requant2 = Requant()

    def set_strides(self, stride_main: torch.Tensor, stride_all: torch.Tensor | None = None):
        if stride_all is None:
            stride_all = stride_main
        self.stride = stride_main.detach().float().cpu()
        self.strides = stride_all.detach().float().cpu()

    def forward(self, x):
        shape = x[0].shape  # (B,C,H,W) of the smallest-level feature
        nl = self.nl

        if self.training:
            rq1 = getattr(self, "requant1", nn.Identity())
            rq2 = getattr(self, "requant2", nn.Identity())
            d1, d2 = [], []

            for i in range(nl):
                box_aux = self.cv2[i](x[i])
                cls_aux = self.cv3[i](x[i])

                if hasattr(self.dfl, '_update_ema'):
                    b, c, h, w = box_aux.shape
                    self.dfl._update_ema(box_aux.view(b, c, -1))

                d1.append(rq1(torch.cat((box_aux, cls_aux), 1)))

            for i in range(nl):
                box_main = self.cv4[i](x[nl + i])
                cls_main = self.cv5[i](x[nl + i])

                if hasattr(self.dfl2, '_update_ema'):
                    b, c, h, w = box_main.shape
                    self.dfl2._update_ema(box_main.view(b, c, -1))

                d2.append(rq2(torch.cat((box_main, cls_main), 1)))

            return [d1, d2]

        rq2 = getattr(self, "requant2", nn.Identity())
        d2 = []
        input_offset = nl if len(x) >= 2 * nl else 0
        for i in range(nl):
            d2.append(
                rq2(
                    torch.cat((self.cv4[i](x[input_offset + i]), self.cv5[i](x[input_offset + i])),
                              1)
                )
            )

        dev, dtype = d2[0].device, d2[0].dtype
        strides_local = self.stride.to(device=dev, dtype=dtype)

        anc, strd = (t.transpose(0, 1) for t in make_anchors(d2, strides_local, 0.5))
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2],
                               2).split((self.reg_max * 4, self.nc), 1)

        pixel_anchor_points2 = anc.unsqueeze(0) * strd
        pixel_pred_dist2 = self.dfl2(box2) * strd
        dbox2 = dist2bbox(pixel_pred_dist2, pixel_anchor_points2, xywh=True, dim=1)

        cls_out_main = cls2 if self.export_logits else F.hardsigmoid(cls2)
        y_main = torch.cat((dbox2, cls_out_main), 1)

        return y_main if self.export else (y_main, d2)

    def bias_init(self):
        if hasattr(self, "cv2"):
            for a, b, s in zip(self.cv2, self.cv3, self.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s)**2)
        for a, b, s in zip(self.cv4, self.cv5, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s)**2)

class Model(nn.Module):
    """
    Builds the model from yaml and wires strides properly for dual-head.
    """
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, hyp=None, anchors=None):
        super().__init__()
        self.hyp = hyp
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        if anchors:
            LOGGER.info("Overriding model.yaml anchors with passed anchors")
            self.yaml['anchors'] = anchors

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)

        m = self.model[-1]
        if isinstance(m, RN_DualDDetect):
            s = 256
            m.inplace = self.inplace
            with torch.no_grad():
                dummy_input = torch.zeros(1, ch, s, s)
                dummy_feats = self._forward_once(
                    dummy_input, get_feats=True
                )  # list of aux+main inputs
            strides_all = torch.tensor([s / x.shape[-2] for x in dummy_feats], dtype=torch.float32)
            nl = m.nl
            stride_main = strides_all[-nl:] if strides_all.numel() >= nl else strides_all
            m.set_strides(stride_main=stride_main, stride_all=strides_all)
            self.stride = m.stride.clone()
            m.bias_init()

        initialize_weights(self)
        self.info()

    def _forward_once(self, x, get_feats=False):
        y = []
        head_inputs = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if isinstance(m, RN_DualDDetect):
                head_inputs = [y[j] for j in m.f]
        if get_feats:
            return head_inputs
        return x

    def forward(self, x, **kwargs):
        return self._forward_once(x)

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

def parse_model(d, ch):
    """Parses a model dictionary and builds the layers."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    act = d.get('activation')
    if act:
        for module_class in (Conv, RepBlockAdd, QARepVGGBlockV2):
            if hasattr(module_class, 'default_act'):
                module_class.default_act = eval(act)
        LOGGER.info(f"Activation: {act}")

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        n = n_ = max(round(n * gd), 1) if n > 1 else n

        if m is RepBlockAdd:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)
            block_class = args[2] if len(args) > 2 else QARepVGGBlockV2
            m_ = m(c1, c2, n=n_, block=block_class)
            args = [c1, c2, n_, block_class.__name__]
        else:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            if m in {Conv, AConv, nn.ConvTranspose2d, QARepVGGBlockV2, GatedSPPF, QARepNCSPELAN}:
                c2 = args[0]
                c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is nn.BatchNorm2d:
                args = [c1]
                c2 = c1
            elif m is QuantAdd:
                c2 = ch[f[1]]
                args = [c1, c2]
            elif m is RN_DualDDetect:
                args.append([ch[x] for x in f])  # pass channels of all inputs to head
                c2 = c1
            else:
                c2 = c1
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
