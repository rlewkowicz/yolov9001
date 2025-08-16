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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Or the appropriate parent directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import val as validate
from models.yolo import Model  # Use the refactored yolo.py
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size, check_yaml, colorstr, get_latest_run,
    increment_path, one_cycle, print_args, yaml_save
)
from utils.loss_tal import RN_ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import select_device
from utils.lion import Lion  # Import the Lion optimizer

class Trainer:
    """
    A modern training class for PyTorch models, designed with torch.compile first.

    This class encapsulates the entire training process, including data loading,
    model compilation, training loops, and checkpointing.
    """
    def __init__(self, opt, device):
        """
        Initializes the Trainer.

        Args:
            opt (argparse.Namespace): Configuration options.
            device (torch.device): The device to train on.
        """
        self.opt = opt
        self.device = device

        self.save_dir = Path(opt.save_dir)
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)  # Ensure weights dir exists
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
            self.model = Model(
                opt.cfg, ch=3, nc=self.nc, hyp=self.hyp, anchors=self.hyp.get('anchors')
            ).to(self.device)

            if not opt.no_opt:
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

        self.gs = 32  # Fallback, will be updated if model has stride
        if hasattr(self.model, 'stride'):
            self.gs = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(opt.imgsz, self.gs, floor=self.gs * 2)

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_fn = RN_ComputeLoss(self.model)

        self.scaler = torch.amp.GradScaler(self.device.type, enabled=True)

        self.start_epoch = 0
        self.best_fitness = 0.0
        if opt.resume:
            self.load_checkpoint(opt.weights)

        self.train_loader, self.dataset, self.val_loader = self._setup_dataloaders()

    def _setup_optimizer(self):
        """Configures and returns the optimizer with parameter groups."""
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]

        uncompiled_model = self.model
        if hasattr(self.model, '_orig_mod'):
            uncompiled_model = self.model._orig_mod
        elif hasattr(self.model, 'module'):
            uncompiled_model = self.model.module  # For exported graphs

        if hasattr(uncompiled_model, 'yaml') and 'backbone' in uncompiled_model.yaml:
            backbone_full_len = len(uncompiled_model.yaml['backbone'])
            sppf_index = backbone_full_len - 1
            backbone_len = backbone_full_len - 1
        else:
            backbone_len = 10
            sppf_index = backbone_len

        param_groups = {
            "backbone_weights": {
                "params": [], "weight_decay": optimizer_settings.get("weight_decay", 0.0)
            },
            "backbone_others": {"params": [], "weight_decay": 0.0},
            "sppf_weights": {
                "params": [], "weight_decay": optimizer_settings.get("weight_decay", 0.0)
            },
            "sppf_others": {"params": [], "weight_decay": 0.0},
            "head_weights": {
                "params": [], "weight_decay": optimizer_settings.get("weight_decay", 0.0)
            },
            "head_others": {"params": [], "weight_decay": 0.0},
        }

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue

            module_index = -1
            try:
                name_parts = name.split('.')
                if name_parts[0] in ('_orig_mod', 'module'):
                    module_index = int(name_parts[2])
                else:
                    module_index = int(name_parts[1])
            except (ValueError, IndexError):
                pass

            prefix = "sppf" if module_index == sppf_index else "backbone" if module_index in range(
                backbone_len
            ) else "head"
            suffix = "weights" if p.ndim > 1 and '.bias' not in name else "others"
            param_groups[f"{prefix}_{suffix}"]["params"].append(p)

        optimizer_param_groups = []
        base_lr = optimizer_settings["lr0"]
        for group_data in param_groups.values():
            if group_data["params"]:
                optimizer_param_groups.append({
                    "params": group_data["params"], "lr": base_lr, "weight_decay":
                        group_data["weight_decay"]
                })

        if self.opt.optimizer == 'AdamW':
            return AdamW(
                optimizer_param_groups,
                lr=base_lr,
                betas=(optimizer_settings['b1'], optimizer_settings['b2'])
            )
        elif self.opt.optimizer == 'SGD':
            return SGD(
                optimizer_param_groups,
                lr=base_lr,
                momentum=optimizer_settings['momentum'],
                nesterov=True
            )
        elif self.opt.optimizer == 'LION':
            return Lion(
                optimizer_param_groups,
                lr=base_lr,
                betas=(optimizer_settings['b1'], optimizer_settings['b2'])
            )
        else:
            raise NotImplementedError(f"Optimizer {self.opt.optimizer} not implemented.")

    def _setup_scheduler(self):
        """Configures and returns the learning rate scheduler."""
        optimizer_settings = self.hyp['optimizer'][self.opt.optimizer]
        lf = one_cycle(1, optimizer_settings['lrf'], self.opt.epochs)
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)

    def _setup_dataloaders(self):
        """Creates and returns the training and validation dataloaders."""
        LOGGER.info("Setting up dataloaders...")
        train_loader, dataset = create_dataloader(
            self.train_path,
            self.imgsz,
            self.opt.batch_size,
            self.gs,
            single_cls=False,
            hyp=self.hyp,
            augment=True,
            rect=self.opt.rect,
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
            rect=True,
            workers=self.opt.workers,
            pad=0.5,
            prefix=colorstr('val: '),
            cache=self.opt.cache
        )[0]
        return train_loader, dataset, val_loader

    def train(self):
        """The main training loop over epochs."""
        LOGGER.info(
            f"Starting training from epoch {self.start_epoch} for {self.opt.epochs} epochs..."
        )
        for epoch in range(self.start_epoch, self.opt.epochs):
            self._train_epoch(epoch + 1)
            self.scheduler.step()

            results, _ = self._run_validation()

            current_fitness = fitness(np.array(results).reshape(1, -1))[0]
            is_best = current_fitness > self.best_fitness
            if is_best:
                self.best_fitness = current_fitness

            self.save_checkpoint(epoch, is_best)

        if self.opt.save_graph:
            self.export_graph()

    def _train_epoch(self, epoch):
        """Trains the model for one epoch."""
        self.model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), bar_format=TQDM_BAR_FORMAT)
        pbar.set_description(f'Epoch {epoch}/{self.opt.epochs}')

        mloss = torch.zeros(3, device=self.device)

        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(self.device, non_blocking=True).float() / 255.0
            targets = targets.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=True):
                preds = self.model(imgs)
                loss, loss_items = self.loss_fn(preds, targets)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
            pbar.set_postfix(
                mem=mem,
                box_loss=mloss[0].item(),
                cls_loss=mloss[1].item(),
                dfl_loss=mloss[2].item()
            )

    def _run_validation(self):
        """Runs validation on the validation set."""
        LOGGER.info("Running validation...")
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
        return results, maps

    def save_checkpoint(self, epoch, is_best):
        """Saves a checkpoint with model state_dict."""
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
            LOGGER.info(
                f"New best model saved to {self.best_pt} (fitness: {self.best_fitness:.5f})"
            )
            torch.save(checkpoint, self.best_pt)

    def load_checkpoint(self, path):
        """Loads a checkpoint to resume training."""
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
        """Exports the model graph using torch.export for faster loading."""
        try:
            LOGGER.info("Exporting model graph...")
            uncompiled_model = self.model._orig_mod if hasattr(
                self.model, '_orig_mod'
            ) else self.model
            dummy_input = torch.randn(self.opt.batch_size, 3, self.imgsz,
                                      self.imgsz).to(self.device)
            exported_program = torch.export.export(uncompiled_model, (dummy_input, ))
            torch.export.save(exported_program, str(self.last_graph))
            LOGGER.info(f"Successfully saved exported graph to {self.last_graph}")
        except Exception as e:
            LOGGER.error(f"Failed to export model graph: {e}")

def parse_opt(known=False):
    """Parses command-line arguments for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', type=str, default='', help='initial weights path or checkpoint to resume'
    )
    parser.add_argument(
        '--cfg', type=str, default=ROOT / 'models/detect/yolov9001-np.yaml', help='model.yaml path'
    )
    parser.add_argument(
        '--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path'
    )
    parser.add_argument(
        '--hyp',
        type=str,
        default=ROOT / 'data/hyps/hyp.scratch-high.yaml',
        help='hyperparameters path'
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=-1,
        help='total batch size for all GPUs, -1 for autobatch'
    )
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--rect', action='store_true', default=True, help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--optimizer', type=str, choices=['SGD', 'AdamW', 'LION'], default='LION', help='optimizer'
    )
    parser.add_argument(
        '--workers', type=int, default=os.cpu_count(), help='max dataloader workers'
    )
    parser.add_argument('--project', default=ROOT / '../runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument(
        '--exist-ok', action='store_true', help='existing project/name ok, do not increment'
    )
    parser.add_argument('--cache', type=str, default=None, help='image --cache ram/disk')
    parser.add_argument(
        '--qat', action='store_true', help='Enable quantization-aware training (placeholder)'
    )
    parser.add_argument(
        '--save-graph', action='store_true', help='Save an exported model graph for faster loading'
    )
    parser.add_argument(
        '--use-graph',
        action='store_true',
        help='Use a pre-exported model graph for faster startup'
    )
    parser.add_argument(
        '--no-opt', action='store_true', help='Disable performance optimizations for fast startup'
    )

    return parser if known else parser.parse_args()

def main():
    """Main function to run the training pipeline."""
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
            gpu_mem_bytes = [
                torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)
            ]
            min_mem_gib = min(gpu_mem_bytes) / (1024**3)
            reference_memory_gib = 23.55
            per_gpu_batch = max(
                1, math.ceil((min_mem_gib / reference_memory_gib) * reference_batch_size)
            )
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

    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.resume)
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    opt.save_dir = str(save_dir)

    yaml_save(save_dir / 'opt.yaml', vars(opt))

    print_args(vars(opt))

    trainer = Trainer(opt, device)
    trainer.train()

if __name__ == "__main__":
    main()
