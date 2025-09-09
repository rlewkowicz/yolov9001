import math
import torch
import torch.nn as nn
from utils.helpers import autopad

class Requant(nn.Module):
    def __init__(self, tag: str | None = None):
        super().__init__()
        self.act = nn.Identity()
        self.tag = tag

    def forward(self, x):
        return self.act(x)

class Conv(nn.Module):
    default_act = nn.SiLU()  # Changed from ReLU6 to SiLU for better accuracy

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module
                                                                         ) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class MANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1), GroupConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(block(y[-1]) for block in self.m)

        return self.cv_final(torch.cat(y, 1))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))

class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            return norm_out * X
        elif self.agg_method == "sum":
            return X
        else:
            raise ValueError(f"Unknown agg_method={self.agg_method}")

class HypConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        x = self.e2v(E, H)

        return x

class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HypConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, c, -1).transpose(1, 2).contiguous()
        feature = x.clone()
        distance = torch.cdist(feature, feature)
        hg = distance < self.threshold
        hg = hg.float().to(x.device).to(x.dtype)
        x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.act(self.bn(x))

        return x

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class GroupConv(Conv):
    """Grouped convolution."""
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        d=1,
        act=True
    ):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class GatedPool(nn.Module):
    def __init__(self, kernel_size=5, stride=1):
        super(GatedPool, self).__init__()
        padding = (kernel_size - 1) // 2
        self.gate_conv = nn.Conv2d(1, 1, 1, bias=True)
        self.gate_act = nn.Hardsigmoid()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.requant_gate_in = Requant(tag="gated_gate_in")
        self.requant_mp_in = Requant(tag="gated_mp_in")
        self.requant_ap_in = Requant(tag="gated_ap_in")
        self.requant_out = Requant(tag="gated_out")

    def forward(self, x):
        gate_input = torch.mean(x, dim=1, keepdim=True)
        g = self.gate_act(self.gate_conv(self.requant_gate_in(gate_input)))
        mp = self.max_pool(self.requant_mp_in(x))
        ap = self.avg_pool(self.requant_ap_in(x))
        out = ap + g * (mp - ap)
        return self.requant_out(out)

class Detect(nn.Module):
    """
    YOLO detection head that outputs predictions for bounding boxes and classes.
    
    Output format:
    - Training mode: Returns list of tensors, one per scale level
      Each tensor shape: [B, (reg_max*4 + nc), H, W]
    - Eval mode: Returns single concatenated tensor
      Shape: [B, (reg_max*4 + nc), N] where N = sum of all H*W across scales
    """
    def __init__(self, nc: int = 80, ch: tuple = (), reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max
        self.strides = None
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(x, x, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(x, 4 * reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, x, 3, 1, 1), nn.SiLU(inplace=True), nn.Conv2d(x, nc, 1))
            for x in ch
        )
        self.initialize_biases()

    def init_from_config(self, cfg):
        cls_prior = float(cfg.get('detect_init', {}).get('cls_prior', 1e-2))
        reg_first_bin = float(cfg.get('detect_init', {}).get('reg_bias_first_bin', 2.0))
        reg_other_bins = float(cfg.get('detect_init', {}).get('reg_bias_other_bins', -2.0))
        self.initialize_biases(
            prior_prob=cls_prior, first_bin=reg_first_bin, other_bins=reg_other_bins
        )
        try:
            self.pad_value = int(cfg.get('pad_value', 114))
        except Exception:
            self.pad_value = 114

    def initialize_biases(self, prior_prob=1e-2, first_bin=2.0, other_bins=-2.0):
        try:
            prior_prob = float(prior_prob)
        except Exception:
            prior_prob = 1e-2
        prior_prob = min(max(prior_prob, 1e-6), 1.0 - 1e-6)
        for conv in self.cv2:
            b = conv[-1].bias
            b.data.zero_()
            reg_bias = torch.full((self.reg_max, ), other_bins)
            reg_bias[0] = first_bin
            reg_bias = reg_bias.repeat(4)  # L, T, R, B
            b.data.copy_(reg_bias.to(b.device))

        bias_val = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.cv3:
            conv[-1].bias.data.fill_(bias_val)  # In-place update; do not re-wrap Parameter

    def forward(self, feats, img_size=None):
        """
        Forward pass of detection head.
        
        Args:
            feats: List of feature maps from FPN, each shape [B, C, H, W]
            img_size: Optional input image size for stride calculation on first pass.
        
        Returns:
            Training: List of predictions per scale, each [B, no, H, W]
            Eval: Concatenated predictions [B, no, N] where N = total anchors
        """
        outputs = []
        self.last_shapes = []  # Cache feature map shapes for anchor generation
        try:
            self.last_feats = feats
        except Exception:
            pass

        assert len(
            feats
        ) == self.nl, f"Detect layer expected {self.nl} feature maps, but got {len(feats)}"

        for i in range(self.nl):
            h, w = feats[i].shape[-2], feats[i].shape[-1]
            self.last_shapes.append((h, w))  # Store H, W for each scale level
            box = self.cv2[i](feats[i])
            cls = self.cv3[i](feats[i])
            out = torch.cat([box, cls], 1)

            assert out.shape[1] == 4 * self.reg_max + self.nc, "Detect head channel count mismatch"

            outputs.append(out)

        if (self.strides is None) and (img_size is not None):
            inferred = []
            for (h, w) in self.last_shapes:
                s_h = float(img_size) / float(h)
                s_w = float(img_size) / float(w)
                assert abs(s_h - s_w) <= 1e-3, \
                    f"Non-square stride implied by H={h}, W={w} at img_size={img_size}: s_h={s_h}, s_w={s_w}"
                inferred.append(int(round(s_h)))
            dev = feats[0].device
            self.strides = torch.as_tensor(inferred, dtype=torch.float32, device=dev)

        return outputs if self.training else (
            torch.cat([o.flatten(2) for o in outputs], dim=2), tuple(self.last_shapes)
        )
