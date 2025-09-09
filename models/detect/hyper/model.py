import torch.nn as nn
import contextlib
from pathlib import Path
import ast

from ..detect import DetectModelBase
from .layers import *

class HyperModel(DetectModelBase):
    def __init__(self, nc=80, ch=3, hyp=None):
        self.input_ch = ch
        cfg = Path(__file__).with_name("model.yaml")
        super().__init__(nc=nc, cfg=cfg, hyp=hyp)

        for m in self.layers:
            if m.__class__.__name__ == "Detect":
                m.init_from_config(self.config_obj)

        self.detect_layer = None
        for layer in self.layers:
            if hasattr(layer, '__class__') and layer.__class__.__name__ == 'Detect':
                self.detect_layer = layer
                self.reg_max = layer.reg_max
                self.strides = getattr(layer, 'strides', None)
                break

        if self.detect_layer is None:
            raise RuntimeError("No Detect layer found; check model.yaml for a Detect head.")

    def _compute_layer_args(self, m, f_m, args, ch, depth, width, max_channels, threshold):
        c1 = ch[f_m] if isinstance(f_m, int) else sum(ch[x] for x in f_m)
        if m in (Conv, C2f, MANet, SPPF):
            c2 = min(int(args[0] * width), max_channels)
            return [c1, c2, *args[1:]], c2, None
        if m is HyperComputeModule:
            c2 = min(int(args[0] * width), max_channels)
            return [c1, c2, threshold], c2, None
        if m is Concat:
            return args, sum(ch[x] for x in f_m), None
        if m is Detect:
            adj = [(args_i if not (isinstance(args_i, str) and args_i == "nc") else self.nc)
                   for args_i in args]
            adj.append([ch[x] for x in f_m])
            rmax = int(self.config_obj.get("reg_max", 16))
            adj.append(rmax)
            return adj, None, None
        if m is nn.Upsample:
            size = args[0] if len(args) >= 1 and args[0] is not None else None
            scale = args[1] if len(args) >= 2 and args[1] is not None else None
            mode = args[2] if len(args) >= 3 else "nearest"
            return [], ch[f_m], nn.Upsample(size=size, scale_factor=scale, mode=mode)
        return args, ch[f_m], None

    def parse_model(self, cfg):
        scales = cfg.get('scales')
        if scales:
            scale_key = cfg.get('scale') or next(iter(scales))
            depth, width, max_channels, threshold = scales[scale_key]
        else:
            depth, width, max_channels, threshold = 1.0, 1.0, float('inf'), cfg.get(
                'threshold', 0.01
            )

        def map_idx(i_cur, idx):
            return i_cur + idx if isinstance(idx, int) and idx < 0 else idx

        ch = [self.input_ch]
        layers, save = [], []
        for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
            m = getattr(nn, m[3:]) if isinstance(m, str) and m.startswith('nn.') else globals()[m]
            for j, a in enumerate(args):
                if isinstance(a, str):
                    with contextlib.suppress(Exception):
                        args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

            n = max(round(n * depth), 1) if n > 1 else n

            if isinstance(f, int):
                f_m = map_idx(i, f)
            else:
                f_m = [map_idx(i, x) for x in f]

            args, c2, maybe_inst = self._compute_layer_args(
                m, f_m, args, ch, depth, width, max_channels, threshold
            )
            m_ = maybe_inst if maybe_inst is not None else (
                nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            )

            m_.i, m_.f = i, f
            layers.append(m_)
            if isinstance(f, list):
                save.extend(x % i for x in f if x != -1)
            elif f != -1:
                save.append(f % i)

            if i == 0:
                ch = []
            ch.append(c2)

        return nn.Sequential(*layers), sorted(save)

    def forward(self, x):
        img_size = x.shape[-1]
        y = []
        for i, layer in enumerate(self.layers):
            f = layer.f
            if f == -1:
                input_val = x
            elif isinstance(f, int):
                input_val = y[f]
            elif isinstance(f, list):
                input_val = [y[j] if j != -1 else y[-1] for j in f]

            if isinstance(layer, Detect):
                x = layer(input_val, img_size)
            else:
                x = layer(input_val)
            y.append(x)

        if not self.training and isinstance(x, tuple):
            return x[0], x[1]
        return x
