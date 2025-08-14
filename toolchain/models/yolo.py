# yolo.py
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
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from models.common import *
from utils.general import LOGGER, make_divisible
from utils.tal.anchor_generator import make_anchors, dist2bbox
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    scale_img,
)

try:
    import thop
except ImportError:
    thop = None

class RN_DualDDetect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.nl = len(ch) // 2
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.inplace = inplace
        self.stride = torch.zeros(self.nl)
        self.export_logits = True

        (c2, c3) = (
            make_divisible(max((ch[0] // 4, self.reg_max * 4, 16)), 4),
            max((ch[0], min((self.nc * 2, 128)))),
        )
        self.cv2 = nn.ModuleList((
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3, g=4),
                nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4),
            ) for x in ch[:self.nl]
        ))
        self.cv3 = nn.ModuleList((
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch[:self.nl]
        ))
        self.dfl = DFL(self.reg_max)

        (c4, c5) = (
            make_divisible(max((ch[self.nl] // 4, self.reg_max * 4, 16)), 4),
            max((ch[self.nl], min((self.nc * 2, 128)))),
        )
        self.cv4 = nn.ModuleList((
            nn.Sequential(
                Conv(x, c4, 3),
                Conv(c4, c4, 3, g=4),
                nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4),
            ) for x in ch[self.nl:]
        ))
        self.cv5 = nn.ModuleList((
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nc, 1))
            for x in ch[self.nl:]
        ))
        self.dfl2 = DFL(self.reg_max)

        self.requant1 = Requant()
        self.requant2 = Requant()

    def forward(self, x):
        shape = x[0].shape
        has_aux_branch = (
            hasattr(self, "cv2") and hasattr(self, "cv3") and
            isinstance(getattr(self, "cv2"), nn.ModuleList) and
            isinstance(getattr(self, "cv3"), nn.ModuleList)
        )

        rq1 = getattr(self, "requant1", nn.Identity())
        rq2 = getattr(self, "requant2", nn.Identity())
        d1, d2 = [], []

        if has_aux_branch:
            for i in range(self.nl):
                d1.append(rq1(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)))
                d2.append(
                    rq2(torch.cat((self.cv4[i](x[self.nl + i]), self.cv5[i](x[self.nl + i])), 1))
                )
        else:
            for i in range(self.nl):
                d2.append(rq2(torch.cat((self.cv4[i](x[i]), self.cv5[i](x[i])), 1)))

        if self.training:
            return [d1, d2]

        anc, strd = (t.transpose(0, 1) for t in make_anchors(d2, self.stride, 0.5))
        box2, cls2 = torch.cat([di.view(shape[0], self.no, -1) for di in d2],
                               2).split((self.reg_max * 4, self.nc), 1)
        pixel_anchor_points2 = anc.unsqueeze(0) * strd
        pixel_pred_dist2 = self.dfl2(box2) * strd
        dbox2 = dist2bbox(pixel_pred_dist2, pixel_anchor_points2, xywh=True, dim=1)
        cls_out_main = cls2 if self.export_logits else F.hardsigmoid(cls2)
        y_main = torch.cat((dbox2, cls_out_main), 1)

        if not has_aux_branch:
            return y_main if self.export else (y_main, d2)

        box, cls = torch.cat([di.view(shape[0], self.no, -1) for di in d1],
                             2).split((self.reg_max * 4, self.nc), 1)
        aux_anchors, aux_strides = (t.transpose(0, 1) for t in make_anchors(d1, self.stride, 0.5))
        pixel_anchor_points1 = aux_anchors.unsqueeze(0) * aux_strides
        pixel_pred_dist1 = self.dfl(box) * aux_strides
        dbox = dist2bbox(pixel_pred_dist1, pixel_anchor_points1, xywh=True, dim=1)
        cls_out = cls if self.export_logits else F.hardsigmoid(cls)
        y_aux = torch.cat((dbox, cls_out), 1)

        return [y_aux, y_main] if self.export else ([y_aux, y_main], [d1, d2])

    def bias_init(self):
        if hasattr(self, "cv2"):
            for a, b, s in zip(self.cv2, self.cv3, self.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s)**2)
        for a, b, s in zip(self.cv4, self.cv5, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s)**2)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = None
        self.dequant = None
        self.qconfig = None

    def forward(self, x, profile=False):
        if self.quant is not None and self.dequant is not None:
            x = self.quant(x)
            x = self._forward_once(x, profile)
            x = self.dequant(x)
            return x
        return self._forward_once(x, profile)

    def _forward_once(self, x, profile=False):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def fuse(self):
        LOGGER.info("Fusing layers... ")
        detect_head = self.model[-1]
        if isinstance(detect_head, RN_DualDDetect) and hasattr(detect_head, "cv2"):
            backbone_len = 0
            if hasattr(self, "yaml") and "backbone" in self.yaml:
                backbone_len = len(self.yaml["backbone"])
            main_head_input_indices = detect_head.f[detect_head.nl:]
            layers_to_keep = set(range(backbone_len))
            q = list(main_head_input_indices)
            visited = set(q)
            while q:
                idx = q.pop(0)
                layers_to_keep.add(idx)
                from_indices = self.model[idx].f
                if isinstance(from_indices, int):
                    from_indices = [from_indices]
                for from_idx in from_indices:
                    actual_idx = from_idx if from_idx != -1 else idx - 1
                    if actual_idx not in visited:
                        q.append(actual_idx)
                        visited.add(actual_idx)
            for i in range(backbone_len, len(self.model) - 1):
                if i not in layers_to_keep:
                    silent_module = Silence()
                    silent_module.i = self.model[i].i
                    silent_module.f = self.model[i].f
                    silent_module.type = "models.common.Silence"
                    silent_module.np = 0
                    self.model[i] = silent_module
            if hasattr(detect_head, "cv2"):
                delattr(detect_head, "cv2")
            if hasattr(detect_head, "cv3"):
                delattr(detect_head, "cv3")
            if hasattr(detect_head, "dfl"):
                delattr(detect_head, "dfl")
            detect_head.f = main_head_input_indices

        for m in self.model.modules():
            if isinstance(m, RepBlockAdd) and not hasattr(m, "requant"):
                m.requant = Requant() if getattr(m, "add", False) else nn.Identity()
            elif isinstance(m, (QuantAdd, QARepNCSPELAN, DFL)) and not hasattr(m, "requant"):
                m.requant = Requant()
        detect_head = self.model[-1]
        if isinstance(detect_head, RN_DualDDetect):
            if not hasattr(detect_head, "requant1"):
                detect_head.requant1 = Requant()
            if not hasattr(detect_head, "requant2"):
                detect_head.requant2 = Requant()

        for m in self.model.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()
            if isinstance(m, RepConvN) and hasattr(m, "fuse_convs"):
                rq = getattr(m, "requant", None)
                m.fuse_convs()
                m.forward = m.forward_fuse
                if rq is not None:
                    m.requant = rq
            elif isinstance(m, Conv) and hasattr(m, "bn"):
                rq = getattr(m, "requant", None)
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
                if rq is not None:
                    m.requant = rq

        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]
        if isinstance(m, RN_DualDDetect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def _fuse_for_qat(self):
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn') and m.bn is not None:
                try:
                    tq.fuse_modules(m, ['conv', 'bn'], inplace=True)
                except Exception:
                    pass

    def enable_qat(self, backend='fbgemm', per_channel=True):
        torch.backends.quantized.engine = backend
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        if per_channel:
            self.qconfig = tq.get_default_qat_qconfig(backend)
        else:
            self.qconfig = tq.qconfig.QConfig(
                activation=tq.default_fake_quant, weight=tq.default_weight_fake_quant
            )
        self.train()
        self._fuse_for_qat()
        tq.prepare_qat(self, inplace=True)

class DetectionModel(BaseModel):
    def __init__(
        self, cfg="yolo.yaml", ch=3, nc=None, anchors=None, qat=False, qat_per_channel=True
    ):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)
        (self.model, self.save) = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)
        m = self.model[-1]
        if isinstance(m, RN_DualDDetect):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self._forward_once(x)[1]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        initialize_weights(self)
        self.info()
        if qat:
            self.enable_qat(backend='fbgemm', per_channel=bool(qat_per_channel))
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            output = self._forward_once(xi)
            yi = (
                output[0][1]
                if isinstance(output, tuple) and isinstance(output[0], list) else output
            )
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return (torch.cat(y, 1), None)

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]
        else:
            (x, y, wh) = (p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale)
            if flips == 2:
                y = img_size[0] - y
            elif flips == 3:
                x = img_size[1] - x
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        nl = self.model[-1].nl
        g = sum((4**x for x in range(nl)))
        e = 1
        i = y[0].shape[1] // g * sum((4**x for x in range(e)))
        y[0] = y[0][:, :-i]
        i = y[-1].shape[1] // g * sum((4**(nl - 1 - x) for x in range(e)))
        y[-1] = y[-1][:, i:]
        return y

Model = DetectionModel

def parse_model(d, ch):
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    (anchors, nc, gd, gw, act) = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
    )
    if act:
        act_module = eval(act)
        for module_class in (Conv, RepConvN, QARepVGGBlockV2):
            if hasattr(module_class, 'default_act'):
                module_class.default_act = act_module
        LOGGER.info(f"activation: {act}")
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    (layers, save, c2) = ([], [], ch[-1])

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        n = n_ = max(round(n * gd), 1) if n > 1 else n

        if m is RepBlockAdd:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            block_class = args[2] if len(args) > 2 else QARepVGGBlockV2
            m_ = m(c1, c2, n=n_, block=block_class)
            args = [c1, c2, n_, block_class.__name__]
        else:
            c1 = ch[f] if isinstance(f, int) else ch[f[0]]
            if m in {Conv, AConv, nn.ConvTranspose2d, QARepVGGBlockV2, GatedSPPF, QARepNCSPELAN}:
                c2 = args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            elif m is nn.BatchNorm2d:
                args = [c1]
                c2 = c1
            elif m is QuantAdd:
                c2_target = ch[f[1]]
                args = [c1, c2_target]
                c2 = c2_target
            elif m is RN_DualDDetect:
                args.append([ch[x] for x in f])
                c2 = c1
            else:
                c2 = c1
            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        (m_.i, m_.f, m_.type, m_.np) = (i, f, t, np)
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return (nn.Sequential(*layers), sorted(save))
