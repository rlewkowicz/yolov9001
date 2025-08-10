from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from utils import TryExcept
from utils.general import LOGGER, colorstr, increment_path, is_notebook, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box
from torch.nn import Parameter
from utils import TryExcept
from utils.general import LOGGER, colorstr, increment_path, is_notebook, xyxy2xywh
from utils.plots import Annotator, colors, save_one_box

class Requant(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.Hardswish()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module
                                                                         ) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class AConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)

class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.Hardswish()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module
                                                                         ) else nn.Identity()
        self.bn = None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=p - k // 2, g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        (kernel3x3, bias3x3) = self._fuse_bn_tensor(self.conv1)
        (kernel1x1, bias1x1) = self._fuse_bn_tensor(self.conv2)
        (kernelid, biasid) = self._fuse_bn_tensor(self.bn)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid
        )

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.c1
        groups = self.g
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size**2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return (0, 0)
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return (kernel * t, beta - running_mean * gamma / std)

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        (kernel, bias) = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class DFL(nn.Module):
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1
        self.requant = Requant()

    def forward(self, x):
        b, c, a = x.shape
        rq = getattr(self, "requant", nn.Identity())
        x = x.view(b, 4, self.c1, a).transpose(2, 1)
        x = rq(x.softmax(1))
        return self.conv(x).view(b, 4, a)

class Silence(nn.Module):
    def __init__(self):
        super(Silence, self).__init__()

    def forward(self, x):
        return x

class DetectMultiBackend(nn.Module):
    def __init__(
        self,
        weights='yolo.pt',
        device=torch.device('cpu'),
        dnn=False,
        data=None,
        fp16=False,
        fuse=True
    ):
        from models.experimental import attempt_download, attempt_load
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        if not (w.endswith('.pt') or w.endswith('.pth')):
            w = attempt_download(w)
        model = attempt_load(
            weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse
        )
        stride = max(int(model.stride.max()), 32)
        names = model.module.names if hasattr(model, 'module') else model.names
        model.half() if fp16 else model.float()
        self.model = model
        self.stride = stride
        self.names = names
        self.device = device
        self.fp16 = fp16

    def forward(self, im, augment=False, visualize=False):
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im, augment=augment, visualize=visualize)

    def from_numpy(self, x):
        import numpy as np
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        if self.device.type != 'cpu':
            im = torch.empty(
                *imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device
            )
            self.forward(im)

class Detections:
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]
        self.ims = ims
        self.pred = pred
        self.names = names
        self.files = files
        self.times = times
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.n = len(self.pred)
        self.t = tuple((x.t / self.n * 1000.0 for x in times))
        self.s = tuple(shape)

    def _run(
        self,
        pprint=False,
        show=False,
        save=False,
        crop=False,
        render=False,
        labels=True,
        save_dir=Path('')
    ):
        (s, crops) = ('', [])
        for (i, (im, pred)) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for (*box, conf, cls) in reversed(pred):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[
                                int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box, 'conf': conf, 'cls': cls, 'label': label, 'im':
                                    save_one_box(box, im, file=file, save=save)
                            })
                        else:
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'
            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im
            if show:
                display(im) if is_notebook() else im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)
                if i == self.n - 1:
                    LOGGER.info(
                        f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}"
                    )
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)
        self._run(save=True, labels=labels, save_dir=save_dir)

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)

    def render(self, labels=True):
        self._run(render=True, labels=labels)
        return self.ims

    def pandas(self):
        new = copy(self)
        ca = ('xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name')
        cb = ('xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name')
        for (k, c) in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()]
                 for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        r = range(self.n)
        x = [
            Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names,
                       self.s) for i in r
        ]
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):
        return self.n

    def __str__(self):
        return self._run(pprint=True)

    def __repr__(self):
        return f'YOLO {self.__class__} instance\n' + self.__str__()

class ConvModule(nn.Module):
    """A combination of Conv + BN + Activation"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation_type,
        padding=None,
        groups=1,
        bias=False
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation_type is not None:
            self.act = nn.Hardswish()
        self.activation_type = activation_type

    def forward(self, x):
        if self.activation_type is None:
            return self.bn(self.conv(x))
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        if self.activation_type is None:
            return self.conv(x)
        return self.act(self.conv(x))

class QARepVGGBlockV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode='zeros',
        deploy=False,
        use_se=False
    ):
        super(QARepVGGBlockV2, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = nn.Hardswish()
        assert kernel_size == 3 and padding == 1
        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            self.rbr_dense = ConvModule(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                activation_type=None,
                padding=padding,
                groups=groups
            )
            self.rbr_1x1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False
            )
            self.rbr_identity = nn.Identity(
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_avg = nn.AvgPool2d(
                kernel_size=kernel_size, stride=stride, padding=padding
            ) if out_channels == in_channels and stride == 1 else None
            self.bn = nn.BatchNorm2d(out_channels)
            self._id_tensor = None

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        id_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0
        avg_out = self.rbr_avg(inputs) if self.rbr_avg is not None else 0
        return self.nonlinearity(
            self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out + avg_out))
        )

    def get_equivalent_kernel_bias(self):
        (kernel3x3, bias3x3) = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_avg is not None:
            kernelavg = self._avg_to_3x3_tensor(self.rbr_avg)
            kernel = kernel + kernelavg.to(self.rbr_1x1.weight.device)
        bias = bias3x3
        if self.rbr_identity is not None:
            if not hasattr(self, '_id_tensor') or self._id_tensor is None:
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self._id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            id_tensor = self._id_tensor
            kernel = kernel + id_tensor
        return (kernel, bias)

    def _avg_to_3x3_tensor(self, avgp):
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        input_dim = channels // groups
        k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size**2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return (0, 0)
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return (kernel * t, beta - running_mean * gamma / std)
        return (0, 0)

    def switch_to_deploy(self):
        if self.deploy:
            return
        (kernel, bias) = self.get_equivalent_kernel_bias()
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = beta + (bias - running_mean) * gamma / std
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.rbr_dense.conv.stride,
            padding=1,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.groups,
            bias=True
        )
        self.rbr_reparam.weight.data = fused_kernel
        self.rbr_reparam.bias.data = fused_bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.__delattr__('bn')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'rbr_avg'):
            self.__delattr__('rbr_avg')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, '_id_tensor'):
            self.__delattr__('_id_tensor')
        self.deploy = True

class RepBlockAdd(nn.Module):
    def __init__(
        self, in_channels, out_channels, n=1, block=QARepVGGBlockV2, basic_block=QARepVGGBlockV2
    ):
        super().__init__()
        self.add = in_channels == out_channels
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(
            *(block(out_channels, out_channels) for _ in range(n - 1))
        ) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                *(
                    BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True)
                    for _ in range(n - 1)
                )
            ) if n > 1 else None
        self.requant = Requant() if self.add else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.block is not None:
            out = self.block(out)
        out = out + identity if self.add else out
        rq = getattr(self, "requant", nn.Identity())
        return rq(out)

class BottleRep(nn.Module):
    def __init__(self, in_channels, out_channels, basic_block=QARepVGGBlockV2, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class GatedPool(nn.Module):
    def __init__(self, kernel_size=5, stride=1):
        super(GatedPool, self).__init__()
        padding = (kernel_size - 1) // 2
        self.gate_conv = nn.Conv2d(1, 1, 1, bias=True)
        self.gate_act = nn.Hardsigmoid()
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        gate_input = torch.mean(x, dim=1, keepdim=True)
        g = self.gate_act(self.gate_conv(gate_input))
        mp = self.max_pool(x)
        ap = self.avg_pool(x)
        return ap + g * (mp - ap)

class GatedSPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = GatedPool(kernel_size=k, stride=1)
        self.norm_x = Conv(c_, c_, 1, 1)
        self.norm_y1 = Conv(c_, c_, 1, 1)
        self.norm_y2 = Conv(c_, c_, 1, 1)
        self.norm_y3 = Conv(c_, c_, 1, 1)

    def forward(self, x):
        x_hidden = self.cv1(x)
        y1 = self.m(x_hidden)
        y2 = self.m(y1)
        y3 = self.m(y2)
        norm_x = self.norm_x(x_hidden)
        norm_y1 = self.norm_y1(y1)
        norm_y2 = self.norm_y2(y2)
        norm_y3 = self.norm_y3(y3)
        return self.cv2(torch.cat([norm_x, norm_y1, norm_y2, norm_y3], 1))

class QuantAdd(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 1, 1)
        self.requant = Requant()

    def forward(self, x):
        rq = getattr(self, "requant", nn.Identity())
        return rq(self.cv1(x[0]) + self.cv2(x[1]))

class QARepNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = QARepVGGBlockV2(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class QARepNCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(QARepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class QARepNCSPELAN(nn.Module):
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c_split = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(QARepNCSP(self.c_split, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(QARepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.proj1 = Conv(self.c_split, c2, 1, 1)
        self.proj2 = Conv(self.c_split, c2, 1, 1)
        self.proj3 = Conv(c4, c2, 1, 1)
        self.proj4 = Conv(c4, c2, 1, 1)
        self.requant = Requant()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1]) for m in [self.cv2, self.cv3]))
        p1 = self.proj1(y[0])
        p2 = self.proj2(y[1])
        p3 = self.proj3(y[2])
        p4 = self.proj4(y[3])
        rq = getattr(self, "requant", nn.Identity())
        return rq(p1 + p2 + p3 + p4)
