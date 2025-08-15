# --------------------------- ENV + IMPORTS (order matters) ---------------------------
import os

# Disable Inductor CUDA Graphs globally and tune allocator to reduce fragmentation.
# Must be set BEFORE importing torch.
os.environ.setdefault("TORCHINDUCTOR_USE_CUDA_GRAPHS", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "0") 
os.environ.setdefault("TORCH_LOGS", "-aot,-dynamo")   
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pow_by_natural.*",
    module="torch.utils._sympy.interp",
)

import argparse
import gc
import math
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import contextlib

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

# --------------------------- PROJECT IMPORTS ---------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.torch_utils import de_parallel, EarlyStopping, ModelEMA, select_device, smart_DDP, smart_resume, torch_distributed_zero_first  # noqa: E402
from models.common import Requant  # noqa: E402
import val as validate  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.autobatch import check_train_batch_size  # noqa: E402
from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.downloads import attempt_download, is_url  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER, TQDM_BAR_FORMAT, check_dataset, check_file, check_img_size, check_suffix, check_yaml,
    colorstr, get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
    labels_to_image_weights, methods, one_cycle, one_flat_cycle, print_args, print_mutation,
    strip_optimizer, yaml_save
)
from utils.loggers import Loggers  # noqa: E402
from utils.loggers.comet.comet_utils import check_comet_resume  # noqa: E402
from utils.loss_tal import RN_ComputeLoss  # noqa: E402
from utils.metrics import fitness  # noqa: E402
from utils.plots import plot_evolve  # noqa: E402
from utils.lion import Lion  # noqa: E402

# Inductor runtime switches (extra safety; complements env var above)
try:
    from torch._inductor import config as inductor_config
    inductor_config.use_cuda_graphs = False
    if hasattr(inductor_config, "triton"):
        inductor_config.triton.cudagraphs = False
except Exception:
    pass

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# --------------------------- UTILS ---------------------------
@contextlib.contextmanager
def SuppressLogs():
    fd_null = os.open(os.devnull, os.O_RDWR)
    fd_stdout_original = os.dup(1)
    fd_stderr_original = os.dup(2)
    os.dup2(fd_null, 1)
    os.dup2(fd_null, 2)
    try:
        yield
    finally:
        os.dup2(fd_stdout_original, 1)
        os.dup2(fd_stderr_original, 2)
        os.close(fd_null)
        os.close(fd_stdout_original)
        os.close(fd_stderr_original)

from torch._inductor import config as inductor_config

inductor_config.max_autotune = False
inductor_config.coordinate_descent_tuning = False

if hasattr(inductor_config, "triton"):
    inductor_config.triton.cudagraphs = False

def _mark_cudagraph_step():
    """Mark iteration boundary to prevent graph-replay hazards."""
    try:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        # Best-effort; safe to ignore if not available
        pass


def _clone_for_backward(obj):
    """
    Clone tensors that will be consumed by backward() to avoid
    'CUDAGraphs output overwritten by subsequent run' errors when compiled.
    """
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, (list, tuple)):
        out = [_clone_for_backward(x) for x in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    if isinstance(obj, dict):
        return {k: _clone_for_backward(v) for k, v in obj.items()}
    return obj


def tie_requant_observers_fx(model: torch.nn.Module) -> torch.nn.Module:
    try:
        import torch.fx as fx
        from torch.ao.quantization.fake_quantize import FakeQuantize

        if not isinstance(model, fx.GraphModule):
            return model

        last_obs = None
        for node in model.graph.nodes:
            if node.op == "call_module":
                mod = dict(model.named_modules()).get(node.target, None)
                if mod is None:
                    continue
                if hasattr(mod, "activation_post_process") and isinstance(
                    getattr(mod, "activation_post_process"), FakeQuantize
                ):
                    last_obs = mod.activation_post_process
                if isinstance(mod, Requant):
                    if hasattr(mod, "activation_post_process") and isinstance(
                        getattr(mod, "activation_post_process"), FakeQuantize
                    ):
                        if last_obs is not None:
                            mod.activation_post_process = last_obs
        return model
    except Exception:
        return model


# --------------------------- DDP / RANK ---------------------------
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = None


# --------------------------- TRAIN ---------------------------
def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    callbacks.run("on_pretrain_routine_start")

    w = save_dir / "weights"
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)
    last, best = w / "last.pt", w / "best.pt"

    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)

    optimizer_settings = hyp["optimizer"][opt.optimizer]
    log_hyp = hyp.copy()
    log_hyp.update(optimizer_settings)
    if "optimizer" in log_hyp:
        del log_hyp["optimizer"]
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in log_hyp.items()))
    opt.hyp = hyp.copy()

    if not evolve:
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))

    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
        data_dict = loggers.remote_dataset
        if resume:
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size
            optimizer_settings = hyp["optimizer"][opt.optimizer]

    plots = not evolve and not opt.noplots
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=False)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)

    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')

    check_suffix(weights, ".pt")
    pretrained = weights.endswith(".pt")

    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        model = Model(
            cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors"), qat=opt.qat
        ).to(device)
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
        model_in_ckpt = ckpt['model']
        csd = model_in_ckpt.float().state_dict() if hasattr(model_in_ckpt, 'state_dict') else model_in_ckpt
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors"), qat=opt.qat).to(device)

    _YOLO_ATTRS_TO_SAVE = ("stride", "names", "nc", "hyp", "class_weights", "yaml", "inplace")
    _yolo_attr_backup = {k: getattr(model, k) for k in _YOLO_ATTRS_TO_SAVE if hasattr(model, k)}

    model.nc = nc
    model.hyp = hyp
    model.names = names

    amp = bool(opt.amp and not opt.qat)

    # Freeze
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    try:
        gs = max(int(model.stride.max()), 32)
    except Exception:
        _stride = _yolo_attr_backup.get("stride", torch.tensor([32], device=device))
        gs = max(int(_stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Autobatch
    if RANK == -1 and batch_size == -1:
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    compute_loss = RN_ComputeLoss(model)

    # QAT
    if opt.qat:
        from torch.ao.quantization.quantize_fx import prepare_qat_fx
        from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
        import torch.ao.nn.intrinsic as nni

        torch.backends.quantized.engine = "fbgemm"
        qconfig = get_default_qat_qconfig("fbgemm")

        qconfig_mapping = (
            QConfigMapping()
            .set_global(qconfig)
            .set_object_type(nn.Conv2d, qconfig)
            .set_object_type(nn.BatchNorm2d, qconfig)
            .set_object_type(nn.Linear, qconfig)
            .set_object_type(Requant, qconfig)
        )
        for fused_t in (getattr(nni, "ConvBn2d", None),
                        getattr(nni, "ConvReLU2d", None),
                        getattr(nni, "ConvBnReLU2d", None)):
            if fused_t is not None:
                qconfig_mapping = qconfig_mapping.set_object_type(fused_t, qconfig)

        model.train()
        example_inputs = (torch.randn(1, 3, int(opt.imgsz), int(opt.imgsz), device=device), )
        model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
        for k, v in _yolo_attr_backup.items():
            try:
                setattr(model, k, v)
            except Exception:
                pass
        model = tie_requant_observers_fx(model)

    # Backend knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    # Optimizer / param groups
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    lambda_target = optimizer_settings.get("weight_decay", 0.0)
    scaled_weight_decay = lambda_target
    decoupled_settings = optimizer_settings.get("decoupled_lr", {})
    lr_scale_backbone = decoupled_settings.get("backbone", {}).get("lr_scale", 1.0)
    lr_scale_head = decoupled_settings.get("head", {}).get("lr_scale", 1.0)
    lr_scale_sppf = decoupled_settings.get("sppf", {}).get("lr_scale", (lr_scale_backbone + lr_scale_head) / 2.0)

    if hasattr(model, 'yaml') and 'backbone' in model.yaml:
        backbone_full_len = len(model.yaml['backbone'])
        sppf_index = backbone_full_len - 1
        backbone_len = backbone_full_len - 1
        LOGGER.info(f"Dynamically determined backbone length: {backbone_len} layers.")
    else:
        backbone_len = 10
        sppf_index = backbone_len
        LOGGER.warning("Could not find model.yaml['backbone']. Falling back to default backbone length of %d.", backbone_len)

    backbone_indices = range(backbone_len)
    param_groups = {
        "backbone_weights": {"params": [], "lr_scale": lr_scale_backbone, "weight_decay": scaled_weight_decay},
        "backbone_others": {"params": [], "lr_scale": lr_scale_backbone, "weight_decay": 0.0},
        "sppf_weights": {"params": [], "lr_scale": lr_scale_sppf, "weight_decay": scaled_weight_decay},
        "sppf_others": {"params": [], "lr_scale": lr_scale_sppf, "weight_decay": 0.0},
        "head_weights": {"params": [], "lr_scale": lr_scale_head, "weight_decay": scaled_weight_decay},
        "head_others": {"params": [], "lr_scale": lr_scale_head, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        module_index = -1
        try:
            module_index = int(name.split('.')[1])
        except (ValueError, IndexError):
            pass
        if module_index == sppf_index:
            target_group_prefix = "sppf"
        else:
            is_backbone = module_index in backbone_indices
            target_group_prefix = "backbone" if is_backbone else "head"
        target_group_suffix = "weights" if p.ndim > 1 and '.bias' not in name else "others"
        param_groups[f"{target_group_prefix}_{target_group_suffix}"]["params"].append(p)

    optimizer_param_groups = []
    base_lr = optimizer_settings["lr0"]
    for group_name, group_data in param_groups.items():
        if group_data["params"]:
            group_lr = base_lr * group_data["lr_scale"]
            optimizer_param_groups.append({
                "params": group_data["params"],
                "lr": group_lr,
                "initial_lr": group_lr,
                "weight_decay": group_data["weight_decay"],
                "name": group_name
            })

    if opt.optimizer == "SGD":
        optimizer = torch.optim.SGD(optimizer_param_groups, momentum=optimizer_settings["momentum"], nesterov=True)
    elif opt.optimizer == "LION":
        optimizer = Lion(
            optimizer_param_groups,
            betas=(optimizer_settings["b1"], optimizer_settings["b2"]),
            alpha=optimizer_settings.get("alpha", 30),
            use_bias_correction=optimizer_settings.get("bias_correction", False),
        )
    elif opt.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            betas=(optimizer_settings["b1"], optimizer_settings["b2"]),
            eps=optimizer_settings.get('eps', 1e-8),
        )
    else:
        raise NotImplementedError(f"Unknown optimizer {opt.optimizer}")

    # LR schedule
    lr_scheduler_type = optimizer_settings.get("lr_scheduler_type", "cosine")
    lrf = optimizer_settings["lrf"]
    if lr_scheduler_type == "cosine":
        lf_original = one_cycle(1, lrf, epochs)
    elif lr_scheduler_type == "flat_cosine":
        lf_original = one_flat_cycle(1, lrf, epochs)
    elif lr_scheduler_type == "fixed":
        lf_original = lambda x: 1.0
    else:
        lf_original = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf

    def lf(epoch):
        factor = lf_original(epoch)
        ddp_warmup_epochs = 0
        min_lr_mul = 0.6
        if RANK != -1 and epoch < ddp_warmup_epochs:
            cosine = 0.5 * (1 + math.cos(math.pi * epoch / ddp_warmup_epochs))
            damp = min_lr_mul + (1.0 - min_lr_mul) * (1.0 - cosine)
            factor *= damp
        return factor

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # SyncBN if DDP
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Compile (train graph)
    if hasattr(torch, "compile") and not opt.qat:
        model.to(memory_format=torch.channels_last)
        LOGGER.info("Compiling model with torch.compile(mode='reduce-overhead')...")
        with SuppressLogs():
            model = torch.compile(model, backend="inductor", mode="reduce-overhead", fullgraph=False, dynamic=False)

    # EMA
    ema = None
    if RANK in {-1, 0}:
        ema = ModelEMA(model)

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained and resume:
        best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DataParallel if single-process multi-GPU
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Dataloaders
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == 'val' else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        close_mosaic=opt.close_mosaic != 0,
        quad=opt.quad,
        prefix=colorstr('train: '),
        shuffle=True,
        min_items=opt.min_items
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max()) if len(dataset.labels) else 0
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}.'

    if RANK in {-1, 0}:
        # Validation batch size: double train batch size per requirements
        val_bs = batch_size // WORLD_SIZE * 2
        val_loader = create_dataloader(
            val_path,
            imgsz,
            val_bs,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr('val: ')
        )[0]
        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            callbacks.run("on_pretrain_routine_end", labels, names)
    # DDP wrap
    if cuda and RANK != -1:
        model = smart_DDP(model)

    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    t0 = time.time()
    nb = len(train_loader)
    nw = max(round(optimizer_settings["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False

    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\nUsing {train_loader.num_workers * WORLD_SIZE} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\nStarting training for {epochs} epochs..."
    )

    # Validation precision: FP16 unless QAT is enabled
    half_val = (not opt.qat)

    for epoch in range(start_epoch, epochs):
        callbacks.run("on_train_epoch_start")
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)

        if epoch == (epochs - opt.close_mosaic):
            if RANK in {-1, 0}:
                LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        mloss = torch.zeros(3, device=device)

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)

        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size"))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch

            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            if amp:
                imgs = imgs.half()
            imgs = imgs / 255

            # warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(
                        ni, xi,
                        [optimizer_settings["warmup_bias_lr"] if x.get('name', '').endswith('others') else 0.0,
                         x["initial_lr"] * lf(epoch)]
                    )
                    if 'momentum' in optimizer_settings:
                        x['momentum'] = np.interp(
                            ni, xi, [optimizer_settings['warmup_momentum'], optimizer_settings['momentum']]
                        )
                    if 'betas' in x:
                        new_beta1 = np.interp(ni, xi, [optimizer_settings['warmup_momentum'], optimizer_settings['b1']])
                        x['betas'] = (new_beta1, x['betas'][1])

            will_step = (ni - last_opt_step) >= accumulate

            if hasattr(model, "no_sync") and not will_step:
                with model.no_sync():
                    with torch.cuda.amp.autocast(enabled=amp):
                        _mark_cudagraph_step()
                        pred = model(imgs)
                        pred = _clone_for_backward(pred)
                        loss, loss_items = compute_loss(pred, targets.to(device))
                        if opt.quad:
                            loss *= 4.0
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=amp):
                    _mark_cudagraph_step()
                    pred = model(imgs)
                    pred = _clone_for_backward(pred)
                    loss, loss_items = compute_loss(pred, targets.to(device))
                    if opt.quad:
                        loss *= 4.0
                scaler.scale(loss).backward()

            if will_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5) %
                    (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return

        lr = [x["lr"] for x in optimizer.param_groups]
        scheduler.step()

        if RANK in {-1, 0}:
            callbacks.run("on_train_epoch_end", epoch=epoch)
            if ema:
                ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])

            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

            if not noval or final_epoch:
                prev = compute_loss.ddp_reduce
                compute_loss.ddp_reduce = False

                # mark boundary before switching to eval/val run
                _mark_cudagraph_step()
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=val_bs,
                    imgsz=imgsz,
                    half=half_val,  # FP16 for val unless QAT
                    model=ema.ema if ema else model,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss
                )
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                compute_loss.ddp_reduce = prev

            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi

            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            if (not nosave) or (final_epoch and not evolve):

                def _clone_for_save(mm):
                    m2 = deepcopy(de_parallel(mm)).float()
                    for _m in m2.modules():
                        if hasattr(_m, "qconfig"):
                            try:
                                _m.qconfig = None
                            except Exception:
                                pass
                    return m2

                ckpt_model = _clone_for_save(model)
                ckpt_ema = _clone_for_save(ema.ema) if ema else None
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": ckpt_model,
                    "ema": ckpt_ema,
                    "updates": ema.updates if ema else None,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,
                    "date": datetime.now().isoformat(),
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        if RANK != -1:
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break

    if RANK in {-1, 0}:
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in (last, best):
            if f.exists():
                strip_optimizer(f)
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    prev = compute_loss.ddp_reduce
                    compute_loss.ddp_reduce = False

                    _mark_cudagraph_step()
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=val_bs,
                        imgsz=imgsz,
                        model=attempt_load(f, device).float(),
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        half=half_val  # keep same val precision policy
                    )
                    compute_loss.ddp_reduce = prev
        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()
    gc.collect()
    return results


# --------------------------- ARGS / MAIN ---------------------------
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=-1, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "LION"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local-rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--min-items", type=int, default=0, help="Experimental")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Experimental")
    parser.add_argument("--entity", default=None, help="W&B: Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="W&B: Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="W&B: Version of dataset artifact to use")
    parser.add_argument("--amp", default=True, action="store_true", help="Enable mixed precision (fp16 amp)")
    parser.add_argument("--qat", action="store_true", help="Enable quantization-aware training")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # If QAT requested, disable AMP for training (fake-quant needs FP32 compute path)
    if opt.qat and opt.amp:
        opt.amp = False

    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"
        opt_data = opt.data
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu", weights_only=False)["opt"]
        opt = argparse.Namespace(**d)
        opt.cfg, opt.weights, opt.resume = "", str(last), True
        if is_url(opt_data):
            opt.data = check_file(opt_data)
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == "exp":
            opt.name = Path(opt.cfg).stem
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    device = select_device(opt.device, batch_size=opt.batch_size)

    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLO Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        meta = {
            'lr0': (1, 1e-5, 1e-1), 'lrf': (1, 0.01, 1.0), 'momentum': (0.3, 0.6, 0.98),
            'weight_decay': (1, 0.0, 0.001), 'warmup_epochs': (1, 0.0, 5.0), 'warmup_momentum': (1, 0.0, 0.95),
            'warmup_bias_lr': (1, 0.0, 0.2), 'box': (1, 0.02, 0.2), 'cls': (1, 0.2, 4.0),
            'cls_pw': (1, 0.5, 2.0), 'obj': (1, 0.2, 4.0), 'obj_pw': (1, 0.5, 2.0), 'iou_t': (0, 0.1, 0.7),
            'fl_gamma': (0, 0.0, 2.0), 'hsv_h': (1, 0.0, 0.1), 'hsv_s': (1, 0.0, 0.9), 'hsv_v': (1, 0.0, 0.9),
            'degrees': (1, 0.0, 45.0), 'translate': (1, 0.0, 0.9), 'scale': (1, 0.0, 0.9), 'shear': (1, 0.0, 10.0),
            'perspective': (0, 0.0, 0.001), 'flipud': (1, 0.0, 1.0), 'fliplr': (0, 0.0, 1.0),
            'mosaic': (1, 0.0, 1.0), 'mixup': (1, 0.0, 1.0), 'copy_paste': (1, 0.0, 1.0)
        }
        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)
        optimizer_settings = hyp['optimizer'][opt.optimizer]
        flat_hyp = {k: v for k, v in hyp.items() if k != 'optimizer'}
        flat_hyp.update(optimizer_settings)
        evolve_keys = sorted([k for k in meta if k in flat_hyp])
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            os.system(f"gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}")
        for gen in range(opt.evolve):
            if evolve_csv.exists():
                parent = 'single'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min() + 1e-6
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()
                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time() + gen))
                g = np.array([meta[k][0] for k in evolve_keys])
                ng = len(evolve_keys)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                parent_values = x[7:]
                for i, k in enumerate(evolve_keys):
                    flat_hyp[k] = parent_values[i] * v[i]
            for k, v_meta in meta.items():
                if k in flat_hyp:
                    flat_hyp[k] = max(flat_hyp[k], v_meta[1])
                    flat_hyp[k] = min(flat_hyp[k], v_meta[2])
                    flat_hyp[k] = round(flat_hyp[k], 5)
            hyp_for_this_gen = deepcopy(hyp)
            for k, v_flat in flat_hyp.items():
                if k in hyp_for_this_gen['optimizer'][opt.optimizer]:
                    hyp_for_this_gen['optimizer'][opt.optimizer][k] = v_flat
                elif k in hyp_for_this_gen:
                    hyp_for_this_gen[k] = v_flat
            results = train(hyp_for_this_gen, opt, device, callbacks)
            callbacks = Callbacks()
            keys = (
                'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                'val/box_loss', 'val/obj_loss', 'val/cls_loss'
            )
            print_mutation(keys, results, flat_hyp, save_dir, opt.bucket)
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
