import math
import torch
import torch.nn as nn
from utils.helpers import autopad

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

class DWConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=2, p=None, d=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module
                                                                         ) else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))

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

class HypConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)

    def forward(self, x, H):
        """
        Batched HyperConv with explicit degree normalization:
          X' = X + Dv^{-1} H De^{-1} H^T (X Θ)
        Where Θ is a linear projection (self.fc).
        Shapes:
          x: [B, N, C_in], H: [B, N, E] (E may equal N under epsilon-ball construction)
        """
        B, N, _ = x.shape
        X_lin = self.fc(x)  # [B, N, C2]
        eps = 1e-9
        Dv = H.sum(dim=2).clamp_min(eps)  # [B, N]
        De = H.sum(dim=1).clamp_min(eps)  # [B, E]
        H_De_inv = H / De.unsqueeze(1)
        S = (H_De_inv @ H.transpose(1, 2)) / Dv.unsqueeze(2)
        out = S @ X_lin  # [B, N, C2]
        return out

class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HypConv(c1, c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.dist_embed = nn.Conv2d(c1, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.k = 24

    def forward(self, x):
        import torch.nn.functional as F
        B, C, H, W = x.shape
        ds = 2 if (H % 40 == 0 and H // 40 == 1) or (H == 40) else 1
        if ds > 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        Hd, Wd = x.shape[-2], x.shape[-1]

        emb = self.dist_embed(x)  # [B,64,Hd,Wd]
        N = Hd * Wd
        emb_flat = emb.view(B, 64, N).transpose(1, 2).contiguous()  # [B,N,64]

        dist = torch.cdist(emb_flat, emb_flat)  # [B,N,N]

        k = min(self.k, N)
        vals, idx = torch.topk(-dist, k=k, dim=-1)  # negative for smallest distances
        knn_mask = torch.zeros_like(dist, dtype=torch.bool)
        knn_mask.scatter_(-1, idx, True)
        if self.threshold is not None and float(self.threshold) > 0:
            thr_mask = (dist <= float(self.threshold))
            adj = (knn_mask & thr_mask)
        else:
            adj = knn_mask
        eye = torch.eye(N, device=dist.device, dtype=torch.bool).unsqueeze(0)
        adj = adj | eye

        x_flat = x.view(B, C, N).transpose(1, 2).contiguous()  # [B,N,C]
        adj_f = adj.to(x_flat.dtype)
        out_flat = self.hgconv(x_flat, adj_f) + x_flat  # residual in node space
        out = out_flat.transpose(1, 2).contiguous().view(B, C, Hd, Wd)

        if ds > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        out = self.act(self.bn(out))
        return out

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
        self.no = nc + 4 * reg_max  # vulture: ignore[unused-attribute] (public head attribute)
        self.strides = None
        self.stems = nn.ModuleList(
            nn.Sequential(nn.Conv2d(x, x, 3, 1, 1), nn.SiLU(inplace=True)) for x in ch
        )
        self.head_reg = nn.ModuleList(nn.Conv2d(x, 4 * reg_max, 1) for x in ch)
        self.head_cls = nn.ModuleList(nn.Conv2d(x, nc, 1) for x in ch)
        self.head_obj = nn.ModuleList(nn.Conv2d(x, 1, 1) for x in ch)
        self.initialize_biases()

    def init_from_config(self, cfg):
        cls_prior = float(cfg.get('detect_init', {}).get('cls_prior', 1e-2))
        reg_first_bin = float(cfg.get('detect_init', {}).get('reg_bias_first_bin', 2.0))
        reg_other_bins = float(cfg.get('detect_init', {}).get('reg_bias_other_bins', -2.0))
        self.initialize_biases(
            prior_prob=cls_prior, first_bin=reg_first_bin, other_bins=reg_other_bins
        )

    def initialize_biases(self, prior_prob=1e-2, first_bin=2.0, other_bins=-2.0):
        try:
            prior_prob = float(prior_prob)
        except Exception:
            prior_prob = 1e-2
        prior_prob = min(max(prior_prob, 1e-6), 1.0 - 1e-6)
        for conv in self.head_reg:
            b = conv.bias
            b.data.zero_()
            reg_bias = torch.full((self.reg_max, ), other_bins)
            reg_bias[0] = first_bin
            reg_bias = reg_bias.repeat(4)  # L, T, R, B
            b.data.copy_(reg_bias.to(b.device))

        bias_val = -math.log((1 - prior_prob) / prior_prob)
        for conv in self.head_cls:
            conv.bias.data.fill_(bias_val)
        for conv in self.head_obj:
            conv.bias.data.fill_(bias_val)

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
        obj_logits_flat = []
        self.last_shapes = []  # Cache feature map shapes for anchor generation

        assert len(
            feats
        ) == self.nl, f"Detect layer expected {self.nl} feature maps, but got {len(feats)}"

        try:
            self.last_feats = feats  # list of [B,C,H,W]
        except Exception:
            self.last_feats = None
        for i in range(self.nl):
            h, w = feats[i].shape[-2], feats[i].shape[-1]
            self.last_shapes.append((h, w))  # Store H, W for each scale level
            stem = self.stems[i](feats[i])
            box = self.head_reg[i](stem)
            cls = self.head_cls[i](stem)
            obj = self.head_obj[i](stem)
            out = torch.cat([box, cls], 1)

            assert out.shape[1] == 4 * self.reg_max + self.nc, "Detect head channel count mismatch"

            outputs.append(out)
            obj_logits_flat.append(obj.flatten(2))  # [B,1,N_l]

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

        try:
            self.last_obj_logits_flat = torch.cat(obj_logits_flat, dim=2).squeeze(1)  # [B, N]
        except Exception:
            self.last_obj_logits_flat = None

        return outputs if self.training else (
            torch.cat([o.flatten(2) for o in outputs], dim=2), tuple(self.last_shapes)
        )

class WFPB(nn.Module):
    """
    Window Feature Propagation Block (WFPB).
    Captures and shares local features via KxK conv and pooling, with 1x1 reduce/restore and residual.
    Args:
      c: input/output channels (preserved)
      k: kernel size for conv/pools (default 3)
      reduce: optional explicit reduced channels
      reduce_ratio: if reduce is None, reduced channels = max(1, int(round(c*reduce_ratio)))
    """
    def __init__(
        self, c: int, k: int = 3, reduce: int | None = None, reduce_ratio: float | None = 0.5
    ):
        super().__init__()
        if isinstance(reduce, float) and (reduce_ratio is None or reduce_ratio == 0.5):
            reduce_ratio = float(reduce)
            reduce = None
        if reduce is None:
            rr = 0.5 if reduce_ratio is None else float(reduce_ratio)
            reduce = max(1, int(round(c * rr)))
        p = k // 2
        self.reduce = nn.Conv2d(c, reduce, 1, 1, 0)
        self.conv_k = nn.Conv2d(reduce, reduce, k, 1, p)
        self.pool_max = nn.MaxPool2d(kernel_size=k, stride=1, padding=p)
        self.pool_avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=p)
        self.restore = nn.Conv2d(reduce, c, 1, 1, 0)
        self.act = nn.SiLU(inplace=True)
        try:
            nn.init.zeros_(self.restore.weight)
            if self.restore.bias is not None:
                nn.init.zeros_(self.restore.bias)
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = x
        y = self.act(self.reduce(x))
        y_conv = self.conv_k(y)
        y_pool = self.pool_max(y) + self.pool_avg(y)
        y = self.restore(self.act(y_conv + y_pool))
        return idn + y

class CISBA(nn.Module):
    """
    Convolutional Integrated Spatial–Channel Attention (CISBA).
    Lightweight residual attention: y = x + gamma * (x * Att), with Att = sigmoid(CA) * sigmoid(SA).
    CA: channel MLP with reduction r; SA: 7x7 conv over [avg,max] channel maps.
    """
    def __init__(self, c: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        r = max(1, int(reduction))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ca_reduce = nn.Conv2d(c, c // r, 1, 1, 0, bias=True)
        self.ca_expand = nn.Conv2d(c // r, c, 1, 1, 0, bias=True)
        k = spatial_kernel
        p = k // 2
        self.sa = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=p, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.sa.weight)
        if self.sa.bias is not None:
            nn.init.zeros_(self.sa.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = self.pool(x)
        ca = torch.sigmoid(self.ca_expand(self.act(self.ca_reduce(z))))  # [B,C,1,1]
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa(torch.cat([avg, mx], dim=1)))  # [B,1,H,W]
        att = ca * sa  # broadcast to [B,C,H,W]
        return x + self.gamma * (x * att)
