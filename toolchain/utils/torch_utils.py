import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe
from utils.lion import Lion

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

try:
    import thop
except ImportError:
    thop = None

warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')
warnings.filterwarnings('ignore', category=UserWarning)

def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate

def smartCrossEntropyLoss(label_smoothing=0.0):
    if check_version(torch.__version__, '1.10.0'):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f'WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0')
    return nn.CrossEntropyLoss()

def smart_DDP(model):
    assert not check_version(torch.__version__, '1.12.0', pinned=True), \
        'torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. ' \
        'Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395'
    if check_version(torch.__version__, '1.11.0'):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

def reshape_classifier_output(model, n=1000):
    from models.common import Classify
    name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]
    if isinstance(m, Classify):
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def device_count():
    assert platform.system() in ('Linux', 'Windows'), 'device_count() only supported on Linux or Windows'
    try:
        cmd = 'nvidia-smi -L | wc -l' if platform.system() == 'Linux' else 'nvidia-smi -L | find /c /v ""'
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0

def select_device(device='', batch_size=0, newline=True):
    s = f'YOLO 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} '
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')
    cpu = device == 'cpu'
    mps = device == 'mps'
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():
        devices = device.split(',') if device else '0'
        n = len(devices)
        if n > 1 and batch_size > 0:
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        s += 'MPS\n'
        arg = 'mps'
    else:
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def profile(input, ops, n=10, device=None):
    """ YOLOv5 speed/memory/FLOPs profiler
    Usage:
        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list' for x in (x, y))
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    return model.module if is_parallel(model) else model

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def find_modules(model, mclass=nn.Conv2d):
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]

def sparsity(model):
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def prune(model, amount=0.3):
    import torch.nn.utils.prune as prune
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)
            prune.remove(m, 'weight')
    LOGGER.info(f'Model pruned to {sparsity(model):.3g} global sparsity')

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def model_info(model, verbose=False, imgsz=640):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
        fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'
    except Exception:
        fs = ''

    name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))
    img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
    if not same_shape:
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)

    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, bn):
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)

        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                g[1].append(v.im.implicit)
            else:
                for iv in v.im:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                g[1].append(v.ia.implicit)
            else:
                for iv in v.ia:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im2'):
            if hasattr(v.im2, 'implicit'):
                g[1].append(v.im2.implicit)
            else:
                for iv in v.im2:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia2'):
            if hasattr(v.ia2, 'implicit'):
                g[1].append(v.ia2.implicit)
            else:
                for iv in v.ia2:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im3'):
            if hasattr(v.im3, 'implicit'):
                g[1].append(v.im3.implicit)
            else:
                for iv in v.im3:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia3'):
            if hasattr(v.ia3, 'implicit'):
                g[1].append(v.ia3.implicit)
            else:
                for iv in v.ia3:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im4'):
            if hasattr(v.im4, 'implicit'):
                g[1].append(v.im4.implicit)
            else:
                for iv in v.im4:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia4'):
            if hasattr(v.ia4, 'implicit'):
                g[1].append(v.ia4.implicit)
            else:
                for iv in v.ia4:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im5'):
            if hasattr(v.im5, 'implicit'):
                g[1].append(v.im5.implicit)
            else:
                for iv in v.im5:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia5'):
            if hasattr(v.ia5, 'implicit'):
                g[1].append(v.ia5.implicit)
            else:
                for iv in v.ia5:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im6'):
            if hasattr(v.im6, 'implicit'):
                g[1].append(v.im6.implicit)
            else:
                for iv in v.im6:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia6'):
            if hasattr(v.ia6, 'implicit'):
                g[1].append(v.ia6.implicit)
            else:
                for iv in v.ia6:
                    g[1].append(iv.implicit)

        if hasattr(v, 'im7'):
            if hasattr(v.im7, 'implicit'):
                g[1].append(v.im7.implicit)
            else:
                for iv in v.im7:
                    g[1].append(iv.implicit)
        if hasattr(v, 'ia7'):
            if hasattr(v.ia7, 'implicit'):
                g[1].append(v.ia7.implicit)
            else:
                for iv in v.ia7:
                    g[1].append(iv.implicit)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0, amsgrad=True)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name == 'LION':
        optimizer = Lion(g[2], lr=lr, betas=(momentum, 0.99), weight_decay=0.0)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer

def smart_hub_load(repo='ultralytics/yolov5', model='yolov5s', **kwargs):
    if check_version(torch.__version__, '1.9.1'):
        kwargs['skip_validation'] = True
    if check_version(torch.__version__, '1.12.0'):
        kwargs['trust_repo'] = True
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)

def smart_resume(ckpt, optimizer, ema=None, weights='yolov5s.pt', epochs=300, resume=True):
    best_fitness = 0.0
    start_epoch = ckpt['epoch'] + 1
    if ckpt['optimizer'] is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
        best_fitness = ckpt['best_fitness']
    if ema and ckpt.get('ema'):
        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']
    if resume:
        assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.\n' \
                                f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        LOGGER.info(f'Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs')
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt['epoch']
    return best_fitness, start_epoch, epochs

class EarlyStopping:
    def __init__(self, patience=30):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.patience = patience or float('inf')
        self.possible_stop = False

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        stop = delta >= self.patience
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop

class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)
