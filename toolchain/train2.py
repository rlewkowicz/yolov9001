import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import lr_scheduler, AdamW, SGD
from tqdm import tqdm

from models.common import DFL
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size, colorstr, get_latest_run,
    increment_path, one_cycle, one_flat_cycle, print_args, yaml_save, fitness
)
from utils.loss_tal import RN_ComputeLoss
from utils.metrics import fitness as _fitness  # keep alias if needed elsewhere
from utils.torch_utils import select_device
from utils.lion import Lion  # Lion optimizer
import val as validate

# -----------------------------------------------------------------------------
# Fast paths
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP rank (for optional scheduler dampening parity with old trainer)
RANK = int(os.getenv("RANK", -1))

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _toggle_dfl_int8(model: torch.nn.Module, enabled: bool):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    for m in base.modules():
        if isinstance(m, DFL):
            if enabled:
                m.update_lut_from_ema()
            m.enable_int8_lut(enabled)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer:
    """
    Modern training class updated to mirror the legacy YOLO optimizer behavior:
      - Per-group LR scaling (backbone/head/SPPF)
      - Per-step warmup (LR, momentum/beta1)
      - Configurable schedulers (cosine / flat_cosine / fixed / linear)
      - Grad clipping before optimizer.step()
    """
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device

        self.save_dir = Path(opt.save_dir)
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last_pt = self.weights_dir / 'last.pt'
        self.best_pt = self.weights_dir / 'best.pt'
        self.last_graph = self.weights_dir / 'last_graph.pt2'

        with open(opt.hyp, errors="ignore") as f:
            self.hyp = yaml.safe_load(f)

        self.data_dict = check_dataset(opt.data)
        self.train_path, self.val_path = self.data_dict['train'], self.data_dict['val']
        self.nc = int(self.data_dict['nc'])
        self.names = self.data_dict['names']

        # Build / load model
        if opt.use_graph and self.last_graph.exists():
            LOGGER.info(f"Loading exported graph from {self.last_graph}...")
            self.model = torch.export.load(str(self.last_graph)).to(self.device)
        else:
            LOGGER.info(f"Creating model from {opt.cfg}...")
            from models.yolo import Model  # local import after sys.path
            self.model = Model(
                opt.cfg, ch=3, nc=self.nc, hyp=self.hyp, anchors=self.hyp.get('anchors')
            ).to(self.device)

            if not opt.no_opt and hasattr(torch, "compile"):
                LOGGER.info("Compiling model with torch.compile() and custom options...")
                opt_dict = {
                    "max_autotune": True,
                    "coordinate_descent_tuning": True,
                    "shape_padding": True,
                    "use_fast_math": True,
                    "triton.cudagraphs": True,
                }
                self.model = torch.compile(
                    self.model, backend="inductor", fullgraph=True, dynamic=False, options=opt_dict
                )
            else:
                LOGGER.info("Skipping model compilation for fast startup.")

        # Stride / image size
        self.gs = 32
        if hasattr(self.model, 'stride'):
            self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        # Optimizer + scheduler + loss
        self.optimizer = self._setup_optimizer()
        self.scheduler, self._lf_epoch = self._setup_scheduler()  # keep epoch factor fn
        self.loss_fn = RN_ComputeLoss(self.model)

        # AMP scaler
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=True)

        # Checkpoint resume
        self.start_epoch = 0
        self.best_fitness = 0.0
        if opt.resume:
            self.load_checkpoint(opt.weights)

        # Data
        self.train_loader, self.dataset, self.val_loader = self._setup_dataloaders()

        # Live logging
        self.global_step = 0
        self.tb = None
        self._log_header_printed = False
        if getattr(opt, "tb", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.save_dir / "tb"))
                LOGGER.info(f"TensorBoard logging → {self.save_dir / 'tb'} (run: tensorboard --logdir {self.save_dir.parent})")
            except Exception as e:
                LOGGER.warning(f"TensorBoard not available: {e}")

        # Accumulation (parity with YOLO classic behavior)
        self._nbs = 64  # nominal batch size
        # set once; during warmup we may re-interp accumulate dynamically per-step
        self.accumulate = max(round(self._nbs / max(1, self.opt.batch_size)), 1)

    # ------- optimizer / scheduler builders ---------------------------------
    def _setup_optimizer(self):
        """Param groups with per-part LR scales + group metadata (name, initial_lr)."""
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]

        # Figure out layer indices for grouping
        uncompiled = self.model
        if hasattr(uncompiled, '_orig_mod'):
            uncompiled = uncompiled._orig_mod
        elif hasattr(uncompiled, 'module'):
            uncompiled = uncompiled.module

        if hasattr(uncompiled, 'yaml') and 'backbone' in uncompiled.yaml:
            backbone_full_len = len(uncompiled.yaml['backbone'])
            sppf_index = backbone_full_len - 1
            backbone_len = backbone_full_len - 1
        else:
            backbone_len = 10
            sppf_index = backbone_len

        # Decoupled LR scaling (default 1.0)
        dec = optimizer_settings.get("decoupled_lr", {})
        lr_scale_backbone = dec.get("backbone", {}).get("lr_scale", 1.0)
        lr_scale_head = dec.get("head", {}).get("lr_scale", 1.0)
        lr_scale_sppf = dec.get("sppf", {}).get("lr_scale", (lr_scale_backbone + lr_scale_head) / 2.0)

        # Build groups
        param_groups = {
            "backbone_weights": {"params": [], "lr_scale": lr_scale_backbone, "weight_decay": optimizer_settings.get("weight_decay", 0.0)},
            "backbone_others":  {"params": [], "lr_scale": lr_scale_backbone, "weight_decay": 0.0},
            "sppf_weights":     {"params": [], "lr_scale": lr_scale_sppf,   "weight_decay": optimizer_settings.get("weight_decay", 0.0)},
            "sppf_others":      {"params": [], "lr_scale": lr_scale_sppf,   "weight_decay": 0.0},
            "head_weights":     {"params": [], "lr_scale": lr_scale_head,   "weight_decay": optimizer_settings.get("weight_decay", 0.0)},
            "head_others":      {"params": [], "lr_scale": lr_scale_head,   "weight_decay": 0.0},
        }

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            # try to infer module index from name
            module_index = -1
            try:
                name_parts = name.split('.')
                if name_parts[0] in ('_orig_mod', 'module'):
                    module_index = int(name_parts[2])
                else:
                    module_index = int(name_parts[1])
            except (ValueError, IndexError):
                pass

            if module_index == sppf_index:
                prefix = "sppf"
            else:
                prefix = "backbone" if module_index in range(backbone_len) else "head"

            suffix = "weights" if p.ndim > 1 and '.bias' not in name else "others"
            param_groups[f"{prefix}_{suffix}"]["params"].append(p)

        # Materialize torch optimizer groups with metadata
        base_lr = optimizer_settings["lr0"]
        optimizer_param_groups = []
        for group_name, g in param_groups.items():
            if not g["params"]:
                continue
            group_lr = base_lr * g["lr_scale"]
            optimizer_param_groups.append({
                "params": g["params"],
                "lr": group_lr,
                "initial_lr": group_lr,         # used by warmup target
                "weight_decay": g["weight_decay"],
                "name": group_name              # used to detect "others" for warmup_bias_lr
            })

        # Construct optimizer
        if self.opt.optimizer == 'AdamW':
            return AdamW(
                optimizer_param_groups,
                betas=(optimizer_settings['b1'], optimizer_settings['b2']),
                eps=optimizer_settings.get('eps', 1e-8)
            )
        elif self.opt.optimizer == 'SGD':
            return SGD(
                optimizer_param_groups,
                momentum=optimizer_settings['momentum'],
                nesterov=True
            )
        elif self.opt.optimizer == 'LION':
            return Lion(
                optimizer_param_groups,
                betas=(optimizer_settings['b1'], optimizer_settings['b2']),
                alpha=optimizer_settings.get("alpha", 30),
                use_bias_correction=optimizer_settings.get("bias_correction", False),
            )
        else:
            raise NotImplementedError(f"Optimizer {self.opt.optimizer} not implemented.")

    def _setup_scheduler(self):
        """Return (scheduler, epoch_factor_fn) with selectable policy."""
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]
        lr_scheduler_type = optimizer_settings.get("lr_scheduler_type", "cosine")
        lrf = optimizer_settings['lrf']

        if lr_scheduler_type == "cosine":
            lf_original = one_cycle(1, lrf, self.opt.epochs)
        elif lr_scheduler_type == "flat_cosine":
            lf_original = one_flat_cycle(1, lrf, self.opt.epochs)
        elif lr_scheduler_type == "fixed":
            lf_original = lambda x: 1.0
        else:  # linear decay fallback
            lf_original = lambda x: (1 - x / self.opt.epochs) * (1.0 - lrf) + lrf

        # Optional DDP-style dampening wrapper (keeps parity with old train loop)
        def lf(epoch):
            factor = lf_original(epoch)
            ddp_warmup_epochs = 0
            min_lr_mul = 0.6
            if RANK != -1 and epoch < ddp_warmup_epochs:
                cosine = 0.5 * (1 + math.cos(math.pi * epoch / ddp_warmup_epochs))
                damp = min_lr_mul + (1.0 - min_lr_mul) * (1.0 - cosine)
                factor *= damp
            return factor

        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        return scheduler, lf

    # ------- data ------------------------------------------------------------
    def _setup_dataloaders(self):
        LOGGER.info("Setting up dataloaders...")
        train_loader, dataset = create_dataloader(
            self.train_path,
            self.imgsz,
            self.opt.batch_size,
            self.gs,
            single_cls=False,
            hyp=self.hyp,
            augment=True,
            rect=self.opt.rect,  # respect flag
            workers=self.opt.workers,
            prefix=colorstr('train: '),
            cache=self.opt.cache,
            shuffle=True
        )
        val_loader = create_dataloader(
            self.val_path,
            self.imgsz,
            self.opt.batch_size,
            self.gs,
            single_cls=False,
            hyp=self.hyp,
            workers=self.opt.workers,
            prefix=colorstr('val: '),
            cache=self.opt.cache
        )[0]
        return train_loader, dataset, val_loader

    # ------- training --------------------------------------------------------
    def train(self):
        LOGGER.info(f"Starting training from epoch {self.start_epoch} for {self.opt.epochs} epochs...")

        # Warmup bookkeeping (parity with YOLO classic)
        optim_set = self.hyp['optimizer'][self.opt.optimizer]
        nb = len(self.train_loader)
        nw = max(round(optim_set.get("warmup_epochs", 0.0) * nb), 100)  # number of warmup steps
        last_opt_step = -1

        for epoch in range(self.start_epoch, self.opt.epochs):
            mloss = self._train_epoch(epoch + 1, nw, last_opt_step)
            self.scheduler.step()

            # validate with LUT int8 toggled on
            results, _ = self._run_validation()
            mp, mr, map50, map_ = results[:4]
            LOGGER.info(
                f"Epoch {epoch+1:>3}/{self.opt.epochs} | "
                f"loss(box/cls/dfl): {mloss[0]:.4f}/{mloss[1]:.4f}/{mloss[2]:.4f} | "
                f"mP {mp:.4f} mR {mr:.4f} mAP50 {map50:.4f} mAP50-95 {map_:.4f}"
            )

            if self.tb is not None:
                self.tb.add_scalar("val/mP", mp, epoch + 1)
                self.tb.add_scalar("val/mR", mr, epoch + 1)
                self.tb.add_scalar("val/mAP50", map50, epoch + 1)
                self.tb.add_scalar("val/mAP", map_, epoch + 1)

            current_fitness = fitness(np.array(results).reshape(1, -1))[0]
            is_best = current_fitness > self.best_fitness
            if is_best:
                self.best_fitness = current_fitness
            self.save_checkpoint(epoch, is_best)

        if self.opt.save_graph:
            self.export_graph()
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()

    def _train_epoch(self, epoch, nw, last_opt_step):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT)
        pbar.set_description(f'Epoch {epoch}/{self.opt.epochs}')

        mloss = torch.zeros(3, device=self.device)
        nb = len(self.train_loader)
        optim_set = self.hyp['optimizer'][self.opt.optimizer]

        for i, (imgs, targets, paths, _) in enumerate(pbar):
            ni = i + nb * (epoch - 1)  # number of seen iterations overall

            # Warmup per-step interpolation (LR / momentum / beta1) + dynamic accumulate
            if ni <= nw:
                xi = [0, nw]
                # During warmup, ramp accumulate from 1 -> nbs/batch_size
                self.accumulate = int(max(1, np.interp(ni, xi, [1, self._nbs / max(1, self.opt.batch_size)]).round()))
                for pg in self.optimizer.param_groups:
                    # Bias groups ("others") get warmup_bias_lr start; weights start at 0.0
                    start_lr = optim_set.get("warmup_bias_lr", 0.0) if str(pg.get("name", "")).endswith("others") else 0.0
                    target_lr = pg["initial_lr"] * self._lf_epoch(epoch - 1)  # use epoch-1 factor like old path
                    pg["lr"] = float(np.interp(ni, xi, [start_lr, target_lr]))
                    # Momentum for SGD
                    if 'momentum' in pg:
                        pg['momentum'] = float(np.interp(
                            ni, xi, [optim_set.get('warmup_momentum', 0.0), optim_set.get('momentum', pg['momentum'])]
                        ))
                    # Beta1 warmup for AdamW/Lion
                    if 'betas' in pg:
                        new_beta1 = float(np.interp(
                            ni, xi, [optim_set.get('warmup_momentum', 0.0), optim_set.get('b1', pg['betas'][0])]
                        ))
                        pg['betas'] = (new_beta1, pg['betas'][1])

            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=True):
                preds = self.model(imgs)
                loss, loss_items = self.loss_fn(preds, targets)

            self.scaler.scale(loss).backward()

            # Gradient accumulation: only step when enough micro-batches have been seen
            will_step = (ni - last_opt_step) >= self.accumulate
            if will_step:
                # Unscale, clip, step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                last_opt_step = ni

            # Running mean for bar
            mloss = (mloss * i + loss_items) / (i + 1)

            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            lr0 = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(mem=mem, lr=f"{lr0:.3e}", box=mloss[0].item(), cls=mloss[1].item(), dfl=mloss[2].item())

            if self.tb is not None:
                self.tb.add_scalar("train/loss_box", loss_items[0].item(), self.global_step)
                self.tb.add_scalar("train/loss_cls", loss_items[1].item(), self.global_step)
                self.tb.add_scalar("train/loss_dfl", loss_items[2].item(), self.global_step)
                self.tb.add_scalar("train/lr", lr0, self.global_step)

            self.global_step += 1

        return mloss.detach()

    # ------- validation / ckpt ----------------------------------------------
    def _run_validation(self):
        _toggle_dfl_int8(self.model, True)
        LOGGER.info("Running validation...")
        try:
            results, maps, _ = validate.run(
                data=self.data_dict,
                batch_size=self.opt.batch_size,
                imgsz=self.imgsz,
                model=self.model,
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                compute_loss=self.loss_fn,
                half=True,
                plots=False
            )
        finally:
            _toggle_dfl_int8(self.model, False)
        return results, maps

    def save_checkpoint(self, epoch, is_best):
        checkpoint = {
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'model_state_dict': deepcopy(self.model.state_dict()),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'opt': vars(self.opt),
        }
        torch.save(checkpoint, self.last_pt)
        if is_best:
            LOGGER.info(f"New best model saved to {self.best_pt} (fitness: {self.best_fitness:.5f})")
            torch.save(checkpoint, self.best_pt)

    def load_checkpoint(self, path):
        if not Path(path).exists():
            LOGGER.warning(f"Checkpoint not found at {path}. Starting from scratch.")
            return
        LOGGER.info(f"Resuming training from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_fitness = ckpt.get('best_fitness', 0.0)
        LOGGER.info(f"Resumed from epoch {self.start_epoch}. Best fitness: {self.best_fitness:.5f}")

    def export_graph(self):
        try:
            LOGGER.info("Exporting model graph...")
            uncompiled_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
            dummy_input = torch.randn(self.opt.batch_size, 3, self.imgsz, self.imgsz).to(self.device)
            exported_program = torch.export.export(uncompiled_model, (dummy_input,))
            torch.export.save(exported_program, str(self.last_graph))
            LOGGER.info(f"Successfully saved exported graph to {self.last_graph}")
        except Exception as e:
            LOGGER.error(f"Failed to export model graph: {e}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path or checkpoint to resume')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/detect/yolov9001-np.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=-1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--rect', action='store_true', default=True, help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW', 'LION'], default='LION', help='optimizer')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='max dataloader workers')
    parser.add_argument('--project', default=ROOT / '../runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cache', type=str, default=None, help='image --cache ram/disk/hybrid')
    parser.add_argument('--qat', action='store_true', help='Enable quantization-aware training (placeholder)')
    parser.add_argument('--save-graph', action='store_true', help='Save an exported model graph for faster loading')
    parser.add_argument('--use-graph', action='store_true', help='Use a pre-exported model graph for faster startup')
    parser.add_argument('--no-opt', action='store_true', help='Disable performance optimizations for fast startup')
    parser.add_argument('--tb', action='store_true', help='Enable TensorBoard logging')
    return parser if known else parser.parse_args()


def main():
    parser = parse_opt(known=True)
    opt, _ = parser.parse_known_args()

    if opt.no_opt:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        LOGGER.info("Performance optimizations disabled.")

    assert opt.cfg or opt.weights, 'Either --cfg or --weights must be specified for a run.'
    device = select_device(opt.device)

    if opt.batch_size == -1:
        if device.type != 'cpu':
            world_size = 1
            num_gpus = torch.cuda.device_count()
            reference_batch_size = 15 if opt.qat else 40
            gpu_mem_bytes = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]
            min_mem_gib = min(gpu_mem_bytes) / (1024**3)
            reference_memory_gib = 23.55
            per_gpu_batch = max(1, math.ceil((min_mem_gib / reference_memory_gib) * reference_batch_size))
            opt.batch_size = per_gpu_batch * world_size
            LOGGER.info(f"Auto-batch size calculation: using batch size {opt.batch_size}")
        else:
            opt.batch_size = 64
            LOGGER.info(f"Running on CPU, setting batch size to {opt.batch_size}")

    if opt.resume:
        last_run_dir = get_latest_run()
        if last_run_dir:
            last_pt = Path(last_run_dir) / 'weights' / 'last.pt'
            if last_pt.exists():
                LOGGER.info(f"Resuming from latest run: {last_pt}")
                opt.weights = str(last_pt)
                opt_yaml = last_pt.parent.parent / 'opt.yaml'
                if opt_yaml.is_file():
                    with open(opt_yaml, errors='ignore') as f:
                        previous_opt = yaml.safe_load(f)
                        opt.cfg = previous_opt.get('cfg', opt.cfg)
                        opt.hyp = previous_opt.get('hyp', opt.hyp)
                        opt.data = previous_opt.get('data', opt.data)
                        opt.project = Path(previous_opt.get('save_dir', opt.project)).parent
                        opt.name = Path(previous_opt.get('save_dir', opt.name)).name

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.resume))
    save_dir.mkdir(parents=True, exist_ok=True)
    opt.save_dir = str(save_dir)

    yaml_save(save_dir / 'opt.yaml', vars(opt))
    print_args(vars(opt))

    trainer = Trainer(opt, select_device(opt.device))
    trainer.train()


if __name__ == "__main__":
    main()
