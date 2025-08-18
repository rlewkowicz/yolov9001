import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
import torch._dynamo

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
from utils.metrics import fitness as _fitness
from utils.torch_utils import select_device, ModelEMA
from utils.lion import Lion
import val as validate

# -----------------------------------------------------------------------------
# Fast paths
# -----------------------------------------------------------------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP rank
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
    """Toggles the DFL INT8 LUT on or off."""
    # Use the EMA model for toggling if it exists, otherwise use the base model
    model_to_toggle = getattr(model, 'module', model)
    base = model_to_toggle._orig_mod if hasattr(model_to_toggle, "_orig_mod") else model_to_toggle
    for m in base.modules():
        if isinstance(m, DFL):
            if enabled:
                m.update_lut_from_ema()
            m.enable_int8_lut(enabled)

def _prepare_eval_model(m: torch.nn.Module) -> torch.nn.Module:
    """Return a float32, eval, no-grad copy/reference suitable for validation."""
    m.eval()
    m.float()
    for p in m.parameters():
        p.requires_grad_(False)
    return m

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer:
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

        if opt.use_graph and self.last_graph.exists():
            LOGGER.info(f"Loading exported graph from {self.last_graph}...")
            self.model = torch.export.load(str(self.last_graph)).to(self.device)
        else:
            LOGGER.info(f"Creating model from {opt.cfg}...")
            from models.yolo import Model
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
                    "triton.cudagraphs": False,
                }
                self.model = torch.compile(
                    self.model, backend="inductor", fullgraph=True, dynamic=False, options=opt_dict
                )
            else:
                LOGGER.info("Skipping model compilation for fast startup.")

        self.gs = 32
        if hasattr(self.model, 'stride'):
            self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        self.optimizer = self._setup_optimizer()
        self.scheduler, self._lf_epoch = self._setup_scheduler()
        self.loss_fn = RN_ComputeLoss(self.model)

        base_for_ema = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        self.ema = ModelEMA(base_for_ema)

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=True)

        self.start_epoch = 0
        self.best_fitness = 0.0
        if opt.resume:
            self.load_checkpoint(opt.weights)

        self.train_loader, self.dataset, self.val_loader = self._setup_dataloaders()

        self.global_step = 0
        self.tb = None
        if getattr(opt, "tb", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=str(self.save_dir / "tb"))
                LOGGER.info(f"TensorBoard logging → {self.save_dir / 'tb'} (run: tensorboard --logdir {self.save_dir.parent})")
            except Exception as e:
                LOGGER.warning(f"TensorBoard not available: {e}")

        self._nbs = 64
        self.accumulate = max(round(self._nbs / max(1, self.opt.batch_size)), 1)

    def _setup_optimizer(self):
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]
        uncompiled = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model

        if hasattr(uncompiled, 'yaml') and 'backbone' in uncompiled.yaml:
            backbone_len = len(uncompiled.yaml['backbone']) - 1
            sppf_index = backbone_len
        else:
            backbone_len, sppf_index = 10, 10

        dec = optimizer_settings.get("decoupled_lr", {})
        lr_scale_backbone = dec.get("backbone", {}).get("lr_scale", 1.0)
        lr_scale_head = dec.get("head", {}).get("lr_scale", 1.0)
        lr_scale_sppf = dec.get("sppf", {}).get("lr_scale", (lr_scale_backbone + lr_scale_head) / 2.0)

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
            try:
                name_parts = name.split('.')
                prefix_map = {'_orig_mod': 2, 'module': 2}
                module_index = int(name_parts[prefix_map.get(name_parts[0], 1)])
            except (ValueError, IndexError):
                module_index = -1

            prefix = "sppf" if module_index == sppf_index else ("backbone" if module_index < backbone_len else "head")
            suffix = "weights" if p.ndim > 1 and '.bias' not in name else "others"
            param_groups[f"{prefix}_{suffix}"]["params"].append(p)

        base_lr = optimizer_settings["lr0"]
        optimizer_param_groups = [
            {
                "params": g["params"],
                "lr": base_lr * g["lr_scale"],
                "initial_lr": base_lr * g["lr_scale"],
                "weight_decay": g["weight_decay"],
                "name": name
            }
            for name, g in param_groups.items() if g["params"]
        ]

        if self.opt.optimizer == 'AdamW':
            return AdamW(optimizer_param_groups, betas=(optimizer_settings['b1'], optimizer_settings['b2']), eps=optimizer_settings.get('eps', 1e-8))
        elif self.opt.optimizer == 'SGD':
            return SGD(optimizer_param_groups, momentum=optimizer_settings['momentum'], nesterov=True)
        elif self.opt.optimizer == 'LION':
            return Lion(optimizer_param_groups, betas=(optimizer_settings['b1'], optimizer_settings['b2']), **optimizer_settings.get("lion_extras", {}))
        else:
            raise NotImplementedError(f"Optimizer {self.opt.optimizer} not implemented.")

    def _setup_scheduler(self):
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]
        lr_scheduler_type = optimizer_settings.get("lr_scheduler_type", "cosine")
        lrf = optimizer_settings['lrf']
        lf_map = {"cosine": one_cycle, "flat_cosine": one_flat_cycle}
        lf_original = lf_map.get(lr_scheduler_type, lambda *args: (lambda x: (1 - x / self.opt.epochs) * (1.0 - lrf) + lrf))(1, lrf, self.opt.epochs)

        def lf(epoch):
            factor = lf_original(epoch)
            if RANK != -1 and epoch < 0:
                damp = 0.6 + 0.4 * (1 - 0.5 * (1 + math.cos(math.pi * epoch / 2)))
                factor *= damp
            return factor

        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf), lf

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
            rect=False,
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
            rect=False,
            hyp=self.hyp,
            workers=self.opt.workers,
            # pad=0.5,
            prefix=colorstr('val: '),
            cache=self.opt.cache,
            close_mosaic=True,
            shuffle=False
        )[0]
        return train_loader, dataset, val_loader

    # -------------------- EMA-aware validation model selection --------------------
    def _pick_val_model(self) -> torch.nn.Module:
        """Use EMA if it has at least one update; otherwise fall back to the raw model."""
        use_ema = self.ema is not None and getattr(self.ema, "updates", 0) > 0
        model_to_validate = self.ema.ema if use_ema else (self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model)
        return _prepare_eval_model(model_to_validate)

    def _get_raw_eval_model(self) -> torch.nn.Module:
        """Return the non-EMA model prepared for eval/validation."""
        m = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        return _prepare_eval_model(m)

    def train(self):
        LOGGER.info(f"Starting training from epoch {self.start_epoch} for {self.opt.epochs} epochs...")
        optim_set = self.hyp['optimizer'][self.opt.optimizer]
        nb = len(self.train_loader)
        # Warmup steps (batches)
        nw = max(round(optim_set.get("warmup_epochs", 0.0) * nb), 100)
        last_opt_step = -1

        for epoch in range(self.start_epoch, self.opt.epochs):
            mloss, last_opt_step = self._train_epoch(epoch, nw, last_opt_step)
            self.scheduler.step()
            # Keep EMA attrs synced so validation doesn't break early
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride'])

            # -------------------- Validation (EMA-aware) --------------------
            results, _ = self._run_validation()  # picks EMA if ready, otherwise raw
            mp, mr, map50, map_ = results[:4]
            LOGGER.info(f"Epoch {epoch+1:>3}/{self.opt.epochs} | "
                        f"loss(box/cls/dfl): {mloss[0]:.4f}/{mloss[1]:.4f}/{mloss[2]:.4f} | "
                        f"mP {mp:.4f} mR {mr:.4f} mAP50 {map50:.4f} mAP {map_:.4f}")

            # Optional side-by-side: compare EMA vs RAW when EMA is initialized
            if getattr(self.ema, "updates", 0) > 0:
                raw_results, _ = self._run_validation(model=self._get_raw_eval_model())
                LOGGER.info(f"Validation compare → mAP50 EMA={map50:.4f} | RAW={raw_results[2]:.4f}")

            if self.tb:
                self.tb.add_scalar("val/mP", mp, epoch + 1)
                self.tb.add_scalar("val/mR", mr, epoch + 1)
                self.tb.add_scalar("val/mAP50", map50, epoch + 1)
                self.tb.add_scalar("val/mAP", map_, epoch + 1)

            current_fitness = fitness(np.array(results).reshape(1, -1))[0]
            if current_fitness > self.best_fitness:
                self.best_fitness = current_fitness
            self.save_checkpoint(epoch, current_fitness == self.best_fitness)

            torch._dynamo.reset()

        if self.opt.save_graph: 
            self.export_graph()
        if self.tb: 
            self.tb.close()

    def _train_epoch(self, epoch, nw, last_opt_step):
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT, desc=f'Epoch {epoch}/{self.opt.epochs}')
        mloss = torch.zeros(3, device=self.device)
        optim_set = self.hyp['optimizer'][self.opt.optimizer]
        nb = len(self.train_loader)

        # Safety net: accumulation should not exceed number of batches (prevents zero-step epoch)
        self.accumulate = min(self.accumulate, nb)

        for i, (imgs, targets, _, _) in enumerate(pbar):
            # Use canonical definition of ni so warmup and step logic are stable
            ni = i + nb * epoch

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                self.accumulate = int(max(1, np.interp(ni, xi, [1, self._nbs / max(1, self.opt.batch_size)]).round()))
                self.accumulate = min(self.accumulate, nb)  # never exceed number of batches
                for pg in self.optimizer.param_groups:
                    start_lr = optim_set.get("warmup_bias_lr", 0.0) if "others" in pg.get("name", "") else 0.0
                    pg["lr"] = float(np.interp(ni, xi, [start_lr, pg["initial_lr"] * self._lf_epoch(epoch)]))
                    if 'momentum' in pg:
                        pg['momentum'] = float(np.interp(ni, xi, [optim_set.get('warmup_momentum', 0.0), optim_set['momentum']]))
                    if 'betas' in pg:
                        beta1 = float(np.interp(ni, xi, [optim_set.get('warmup_momentum', 0.0), optim_set['b1']]))
                        pg['betas'] = (beta1, pg['betas'][1])

            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            with torch.amp.autocast(self.device.type, enabled=True):
                preds = self.model(imgs)
                loss, loss_items = self.loss_fn(preds, targets.to(self.device))
            self.scaler.scale(loss).backward()

            # Guarantee at least one optimizer step per epoch: step on last batch too
            will_step = (ni - last_opt_step) >= self.accumulate or (i == nb - 1)

            if will_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)
                last_opt_step = ni

            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix(
                mem=f'{torch.cuda.memory_reserved()/1E9:.3g}G',
                lr=f'{self.optimizer.param_groups[0]["lr"]:.3e}',
                box=mloss[0].item(),
                cls=mloss[1].item(),
                dfl=mloss[2].item()
            )
            if self.tb:
                self.tb.add_scalar("train/loss_box", loss_items[0].item(), self.global_step)
                self.tb.add_scalar("train/loss_cls", loss_items[1].item(), self.global_step)
                self.tb.add_scalar("train/loss_dfl", loss_items[2].item(), self.global_step)
                self.tb.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.global_step += 1

        return mloss.detach(), last_opt_step

    def _run_validation(self, model: torch.nn.Module = None):
        # Pick EMA if ready, else raw model, unless a specific model is provided
        model_to_validate = model if model is not None else self._pick_val_model()
        _toggle_dfl_int8(model_to_validate, True)
        LOGGER.info("Running validation...")
        try:
            # Validate in float32 for stability with compiled graphs and EMA
            model_to_validate = _prepare_eval_model(model_to_validate)
            results, maps, _ = validate.run(
                data=self.data_dict,
                batch_size=self.opt.batch_size,
                imgsz=self.imgsz,
                model=model_to_validate,
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                compute_loss=self.loss_fn,
                half=True,          # keep validation in float32 to avoid early-epoch EMA issues
                plots=False
            )
        finally:
            _toggle_dfl_int8(model_to_validate, False)
        return results, maps

    def save_checkpoint(self, epoch, is_best):
        model_sd = (self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model).state_dict()
        ema_sd = self.ema.ema.state_dict()
        checkpoint = {
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'model_state_dict': model_sd,
            'ema_state_dict': ema_sd,
            'ema_updates': getattr(self.ema, 'updates', 0),
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
        if 'ema_state_dict' in ckpt:
            self.ema.ema.load_state_dict(ckpt['ema_state_dict'])
        if 'ema_updates' in ckpt:
            try:
                self.ema.updates = ckpt['ema_updates']
            except Exception:
                pass
        self.start_epoch = ckpt['epoch'] + 1
        self.best_fitness = ckpt.get('best_fitness', 0.0)
        LOGGER.info(f"Resumed from epoch {self.start_epoch}. Best fitness: {self.best_fitness:.5f}")

    def export_graph(self):
        try:
            LOGGER.info("Exporting model graph...")
            model_to_export = self.ema.ema._orig_mod if hasattr(self.ema.ema, '_orig_mod') else self.ema.ema
            dummy_input = torch.randn(self.opt.batch_size, 3, self.imgsz, self.imgsz).to(self.device)
            exported_program = torch.export.export(model_to_export, (dummy_input,))
            torch.export.save(exported_program, str(self.last_graph))
            LOGGER.info(f"Successfully saved exported graph to {self.last_graph}")
        except Exception as e:
            LOGGER.error(f"Failed to export model graph: {e}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_opt(known=False):
    """Parses command-line arguments."""
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
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main():
    """Main function to run the training process."""
    opt = parse_opt()

    if opt.no_opt:
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        LOGGER.info("Performance optimizations disabled.")

    assert opt.cfg or opt.weights, 'Either --cfg or --weights must be specified for a run.'
    device = select_device(opt.device)

    if opt.batch_size == -1 and device.type != 'cpu':
        world_size = 1
        num_gpus = torch.cuda.device_count()
        reference_batch_size = 15 if opt.qat else 40
        gpu_mem_gb = min(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)) / 1E9
        reference_mem_gb = 23.55
        per_gpu_batch = max(1, math.ceil((gpu_mem_gb / reference_mem_gb) * reference_batch_size))
        opt.batch_size = per_gpu_batch * world_size
        LOGGER.info(f"Auto-batch size: {opt.batch_size}")
    elif opt.batch_size == -1:
        opt.batch_size = 64
        LOGGER.info(f"CPU training, setting batch size to {opt.batch_size}")

    if opt.resume:
        last_run_dir = get_latest_run()
        if last_run_dir:
            opt.weights = str(Path(last_run_dir) / 'weights' / 'last.pt')
            LOGGER.info(f"Resuming from latest run: {opt.weights}")
            opt_yaml = Path(last_run_dir) / 'opt.yaml'
            if opt_yaml.is_file():
                with open(opt_yaml, errors='ignore') as f:
                    previous_opt = yaml.safe_load(f)
                opt.cfg, opt.hyp, opt.data, opt.project, opt.name = \
                    previous_opt.get('cfg', opt.cfg), previous_opt.get('hyp', opt.hyp), \
                    previous_opt.get('data', opt.data), Path(previous_opt.get('save_dir', opt.project)).parent, \
                    Path(previous_opt.get('save_dir', opt.name)).name

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.resume))
    save_dir.mkdir(parents=True, exist_ok=True)
    opt.save_dir = str(save_dir)
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    print_args(vars(opt))

    trainer = Trainer(opt, device)
    trainer.train()

if __name__ == "__main__":
    main()
