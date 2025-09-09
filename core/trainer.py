from pathlib import Path
import os
from typing import Optional, Callable, Dict, Any
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseRunner
from .validator import Validator
from utils.optimizer import build_optimizer, print_optimizer_info
from utils.ema import ModelEMA
from utils.prefetcher import CUDAPrefetcher
from .config import get_config
from .callbacks import (
    CallbackManager,
    LoggingCallback,
    LRSchedulerCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsLoggerCallback,
    GradientLoggerCallback,
    ValidationLoggingCallback,
)
from utils.dino_teacher import DINOTeacher
from utils.assigner import TaskAlignedAssigner

def _jsonify_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert tensors in a config dict to JSON-serializable types."""
    json_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, torch.Tensor):
            json_cfg[k] = v.tolist()
        elif isinstance(v, dict):
            json_cfg[k] = _jsonify_cfg(v)
        else:
            try:
                json.dumps(v)
                json_cfg[k] = v
            except TypeError:
                json_cfg[k] = str(v)
    return json_cfg

class Trainer(BaseRunner):
    """
    Minimal, pragmatic trainer with:
      - AMP
      - grad accumulation + max-norm clipping
      - auto optimizer/scheduler (Lion + cosine) if not provided
      - optional auto dataloaders + auto-fit via cfg
      - checkpointing: last.pt, best.pt
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        amp: bool = True,
        grad_accum_steps: int = 1,
        max_norm: Optional[float] = None,
        cfg: Optional[Dict[str, Any]] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable[[Any, Any], torch.Tensor]] = None,
        auto_fit: bool = False,
        use_ema: bool = True,
    ):
        super().__init__(model, device=device, cfg=cfg or {})
        self.model.to(memory_format=torch.channels_last)
        self.cfg = cfg or {}

        model_hyp = getattr(model, 'hyp', None)
        config = get_config(cfg=self.cfg, hyp=model_hyp)

        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_norm = max_norm if max_norm is not None else config.get('grad_clip', 10.0)
        self.amp = bool(amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        self.epoch = 0
        self.best_fitness = float("-inf")
        self.stop_training = False  # For early stopping
        self.last_grad_norm = None  # Store for callback logging
        self._dino_log_interval = 100  # can be overridden by cfg['dino'].get('log_interval')
        self._dino_step_count = 0
        self._dino_epoch_stats = {
            "sum_patch": 0.0,
            "sum_sal": 0.0,
            "sum_global": 0.0,
            "sum_contrast": 0.0,
            "sum_total": 0.0,
            "count": 0,
        }

        ema_cfg = config.ema_config
        self.use_ema = use_ema and ema_cfg['enabled']
        self.ema = None
        if self.use_ema:
            self.ema = ModelEMA(model, decay=ema_cfg['decay'])

        self.train_loader: Optional[DataLoader] = train_loader
        self.val_loader: Optional[DataLoader] = val_loader
        self.validator = Validator(model, device=device, cfg=cfg)

        if self.train_loader is None:
            self._maybe_build_dataloaders_from_cfg()

        if criterion is None:
            from utils.loss import DetectionLoss
            if self.train_loader is not None and hasattr(self.train_loader, 'dataset'):
                dl_imgsz = getattr(
                    self.train_loader.dataset, "imgsz", self.cfg.get("img_size", 640)
                )
            else:
                dl_imgsz = self.cfg.get("img_size", 640)
            criterion = DetectionLoss(self.model, imgsz=int(dl_imgsz))
        self.criterion: Callable[[Any, Any], torch.Tensor] = criterion
        try:
            self.l1_ramp_step = int(get_config(cfg=self.cfg, hyp=model_hyp).get('l1_ramp_step', 0))
        except Exception:
            self.l1_ramp_step = 0

        if self.cfg.get("compile", False):
            try:
                self.logger.info("trainer/compile", "torch.compile enabled with custom options")
                base_options = {
                    "coordinate_descent_tuning": True,
                    "shape_padding": True,
                    "use_fast_math": True,
                }

                model_options = {**base_options, "max_autotune": True, "triton.cudagraphs": True}
                self.model = torch.compile(
                    self.model,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                    options=model_options
                )
                self.logger.info(
                    "trainer/compile_model", "Model compiled with fullgraph and max_autotune"
                )

            except Exception as e:
                self.logger.debug("trainer/compile_error", f"{e}")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = int(self.cfg.get("epochs", 100))
        self.per_iteration_scheduler = False  # Will be set to True if using warmup scheduler

        if self.optimizer is None or self.scheduler is None:
            try:
                self.optimizer, self.scheduler = build_optimizer(
                    self.model, epochs=self.epochs, cfg=config
                )

                self.per_iteration_scheduler = False

                print_optimizer_info(self.optimizer, self.scheduler)
            except Exception as e:
                self.logger.debug("trainer/optimizer_auto_build_error", f"{e}")
                raise

        self.callbacks = CallbackManager([
            LoggingCallback(),
            LRSchedulerCallback(),
            CheckpointCallback(Path(self.logger.log_dir) / "checkpoints"),
            MetricsLoggerCallback(),  # Add metrics logger by default
            GradientLoggerCallback(log_interval=int(self.cfg.get('grad_log_interval', 200))
                                  ),  # from config
            ValidationLoggingCallback(),  # Centralize validation logging
        ])

        if self.cfg.get('early_stop', False):
            patience = self.cfg.get('early_stop_patience', 10)
            self.callbacks.add_callback(EarlyStoppingCallback(patience=patience))

        try:
            if self.logger.writer and self.logger.is_main_process:
                bs = max(1, int(self.cfg.get("batch_size", 1)))
                img_size = int(self.cfg.get("img_size", 640))
                dummy = torch.randn(bs, 3, img_size, img_size,
                                    device=self.device).to(memory_format=torch.channels_last)
                self.logger.writer.add_graph(self.model, dummy)
        except Exception as e:
            self.logger.debug("tb/graph_error", str(e))

        try:
            from thop import profile  # type: ignore
            from copy import deepcopy
            img_size = int(self.cfg.get("img_size", 640))
            try:
                base_model = getattr(self.model, "_orig_mod", self.model)
                prof_model = deepcopy(base_model).to("cpu").eval()
            except Exception:
                prof_model = deepcopy(self.model).to("cpu").eval()

            dummy = torch.randn(1, 3, img_size, img_size,
                                device="cpu").to(memory_format=torch.channels_last)
            flops, params = profile(prof_model, inputs=(dummy, ), verbose=False)
            del prof_model
            info = {
                "gflops": round(flops / 1e9, 3),
                "params_m": round(params / 1e6, 3),
            }
            self.logger.info("model/compute", info)
        except Exception as e:
            self.logger.warning(
                "compute/thop_missing",
                f"Install thop to enable FLOPs reporting (pip install thop): {e}"
            )

        if auto_fit:
            ckpt_dir = Path(self.cfg.get("ckpt_dir", "runs/checkpoints"))
            self.fit(
                epochs=self.epochs,
                criterion=self.criterion,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                ckpt_dir=ckpt_dir,
            )

        self._init_dino_cfg()
        try:
            self.logger.info("dino/config", self.dino_cfg)
        except Exception:
            pass
        self.dino_teacher = None
        self.dino_teacher_proj = None
        self.dino_student_proj = None  # legacy single-level (unused when multi-level)
        self.dino_aux_obj = None  # legacy single-level (unused when multi-level)
        self.dino_global_proj = None  # legacy single-level (unused when multi-level)
        self.dino_student_proj_list = None
        self.dino_aux_obj_list = None
        self.dino_global_proj_list = None
        self._dino_ready = False
        self._dino_levels = 0
        if self.dino_cfg["enabled"]:
            try:
                quant_mode = str(
                    self.dino_cfg.get(
                        "quant",
                        self.dino_cfg.get("use_int4", False) and "int4" or "bf16"
                    )
                ).lower()
                if quant_mode == "fp16":
                    dtype_pref = torch.float16
                elif quant_mode == "fp32":
                    dtype_pref = torch.float32
                elif quant_mode == "bf16":
                    dtype_pref = torch.bfloat16
                else:
                    dtype_pref = torch.bfloat16 if self.amp else torch.float32
                self.dino_teacher = DINOTeacher(
                    model_name=self.dino_cfg["model_name"],
                    resolution=self.dino_cfg["resolution"],
                    quant=quant_mode,
                    dtype=dtype_pref,
                    device=self.device,
                    hf_token=(self.dino_cfg.get("hf_token") or os.getenv("HF_TOKEN")),
                    sal_from=str(self.dino_cfg.get("sal_from", "auto")),
                )
                try:
                    self._dino_log_interval = int(self.dino_cfg.get("log_interval", 100))
                except Exception:
                    self._dino_log_interval = 100
                if bool(self.dino_cfg.get("prewarm", False)):
                    try:
                        self.dino_teacher._lazy_init()
                        if not getattr(self.dino_teacher, "_init_ok", False):
                            err = getattr(self.dino_teacher, "_last_error", None)
                            raise RuntimeError(
                                f"DINO prewarm failed to initialize the teacher model. Last error: {err}"
                            )
                        self.logger.info(
                            "dino/prewarm", {
                                "model": self.dino_cfg["model_name"],
                                "resolution": int(self.dino_cfg["resolution"]),
                            }
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"DINO teacher load failed: {e}. If the model repo is gated, set HF_TOKEN or pass dino.hf_token in hyp, or run 'huggingface-cli login'."
                        )
                self.logger.info(
                    "dino/enabled", {
                        "resolution": int(self.dino_cfg["resolution"]),
                        "every_n": int(self.dino_cfg["every_n"]),
                        "quant": quant_mode,
                    }
                )
            except Exception:
                raise
        else:
            try:
                self.logger.info("dino/disabled", "DINO distillation not enabled by config")
            except Exception:
                pass

        try:
            opt_cfg = config.get("optimizer", {}).get("LION", {})
        except Exception:
            opt_cfg = {}
        dec = opt_cfg.get("decoupled_lr", {}) if isinstance(opt_cfg, dict) else {}
        self._bb_scale_cfg = float(dec.get("backbone", {}).get("lr_scale", 1.0))
        self._head_scale_cfg = float(dec.get("head", {}).get("lr_scale", 1.0))
        gate_cfg = self.dino_cfg.get("bb_lr_gate", {}) if isinstance(self.dino_cfg, dict) else {}
        self._bb_gate_enabled = bool(gate_cfg.get("enabled", True))
        if bool(self.dino_cfg.get("objfor02", False)):
            self._bb_gate_enabled = False
        self._bb_gate_min = float(gate_cfg.get("min_scale", 0.1))
        self._bb_gate_ema = float(gate_cfg.get("ema", 0.9))
        self._bb_gate_relax = float(gate_cfg.get("relax", 0.1))
        self._bb_gate_complement_head = bool(gate_cfg.get("complement_head", False))
        self._bb_gate_head_max_scale = float(gate_cfg.get("head_max_scale", 2.0))
        self._bb_group_idx: list[int] = []
        self._bb_nominal_lr_epoch: list[float] = []
        self._head_group_idx: list[int] = []
        self._head_nominal_lr_epoch: list[float] = []
        self._dino_loss_ema: float = 0.0
        self._dino_loss_peak: float = 0.0

    def _disable_dino(self, reason: str = ""):
        """Disable DINO distillation for the rest of training and unload the teacher."""
        try:
            if hasattr(self, 'dino_cfg') and isinstance(self.dino_cfg, dict):
                self.dino_cfg['enabled'] = False
        except Exception:
            pass
        try:
            hyp = getattr(self.model, 'hyp', None)
            if isinstance(hyp, dict):
                mode = str(hyp.get('assign_mode', 'ult')).lower()
                if mode in ('simota', 'mixed'):
                    hyp['assign_mode'] = 'ult'
            crit = getattr(self, 'criterion', None)
            if crit is not None and hasattr(crit, 'assign_mode') and hasattr(crit, 'assigner'):
                mode_now = str(getattr(crit, 'assign_mode', 'ult')).lower()
                if mode_now in ('simota', 'mixed'):
                    num_classes = int(getattr(crit, 'nc', getattr(self.model, 'nc', 80)))
                    topk = int(getattr(crit, 'assign_topk', 10))
                    alpha = float(getattr(crit, 'assign_alpha', 0.5))
                    beta = float(getattr(crit, 'assign_beta', 6.0))
                    cfg = getattr(crit, 'hyp', {}) if hasattr(crit, 'hyp') else {}
                    lambda_iou = float(cfg.get('assign_lambda_iou', 3.0))
                    topq = int(cfg.get('assign_topq', 10))
                    center_radius = float(getattr(crit, 'assign_radius', 2.5))
                    min_stride_alpha = float(cfg.get('min_stride_box_alpha', 0.0))
                    filter_tiny_gt = bool(cfg.get('filter_tiny_gt', True))
                    objless_target_iou = float(cfg.get('objless_target_iou', 0.0))
                    objless_weight_floor = float(cfg.get('objless_weight_floor', 0.05))
                    crit.assigner = TaskAlignedAssigner(
                        mode='ult',
                        num_classes=num_classes,
                        topk=topk,
                        alpha=alpha,
                        beta=beta,
                        lambda_iou=lambda_iou,
                        topq=topq,
                        center_radius=center_radius,
                        input_is_logits=True,
                        min_stride_alpha=min_stride_alpha,
                        filter_tiny_gt=filter_tiny_gt,
                        objless_target_iou=objless_target_iou,
                        objless_weight_floor=objless_weight_floor,
                    )
                    crit.assign_mode = 'ult'
                    try:
                        self.logger.info('assigner/switch', 'DINO off: switched assign_mode to ULT')
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if getattr(self, 'dino_teacher', None) is not None:
                try:
                    self.dino_teacher.unload()
                except Exception:
                    pass
        finally:
            self.dino_teacher = None
        self._dino_ready = False
        self._dino_levels = 0
        try:
            msg = {"epoch": int(self.epoch), "reason": reason or "disabled"}
            self.logger.info("dino/disabled", msg)
        except Exception:
            pass

    def train_epoch(
        self, dataloader: DataLoader, criterion: Callable[[Any, Any], torch.Tensor]
    ) -> float:
        self.model.train()
        try:
            setattr(self.model, 'epoch', int(self.epoch))
        except Exception:
            pass
        total_loss = 0.0
        nb = len(dataloader)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            if getattr(self, 'dino_cfg', {}).get("enabled", False):
                max_ep = int(self.dino_cfg.get("max_epochs", 0))
                if max_ep > 0 and int(self.epoch) >= max_ep:
                    self._disable_dino("hard_stop")
        except Exception:
            pass
        self._dino_epoch_stats = {
            "sum_patch": 0.0,
            "sum_sal": 0.0,
            "sum_global": 0.0,
            "sum_contrast": 0.0,
            "sum_total": 0.0,
            "count": 0,
        }
        self._dino_step_count = 0
        if getattr(self, 'dino_cfg', {}).get("enabled", False) and (self.dino_teacher is None):
            try:
                self.logger.debug("dino/status", "enabled but teacher not initialized")
            except Exception:
                pass

        use_prefetch = (self.device.type == "cuda") and bool(self.cfg.get("cuda_prefetch", True))

        if use_prefetch:
            max_prefetch = int(self.cfg.get("prefetch_max_batches", 3))
            mem_frac = float(self.cfg.get("prefetch_mem_fraction", 0.80))
            iter_src = CUDAPrefetcher(
                dataloader,
                self.device,
                amp=self.amp,
                max_prefetch_batches=max_prefetch,
                mem_fraction=mem_frac
            )
        else:
            iter_src = dataloader

        pbar = tqdm(enumerate(iter_src), total=nb, desc=f"epoch {self.epoch}")
        try:
            for i, batch in pbar:
                self.callbacks.on_batch_start(self, i)

                if use_prefetch:
                    images, targets, _, _, _ = batch  # already on device and channels_last
                else:
                    images, targets, _, _, _ = batch
                    images = self.preprocess(images)
                    if images.dtype == torch.uint8:
                        images = images.to(torch.float32).div_(255.0)
                    images = images.to(memory_format=torch.channels_last)
                    targets = self.preprocess(targets)

                with torch.amp.autocast('cuda', enabled=self.amp):
                    if hasattr(criterion,
                               'l1_weight') and float(getattr(criterion, 'l1_weight', 0.0)) > 0.0:
                        if int(getattr(self, 'l1_ramp_step', 0)) > 0:
                            scale = min(
                                1.0,
                                float(getattr(self.state, 'opt_step', 0)) /
                                float(max(1, self.l1_ramp_step))
                            )
                            try:
                                setattr(criterion, 'l1_ramp_scale', float(scale))
                            except Exception:
                                pass
                    outputs = self.model(images)
                    loss, loss_items = criterion(outputs, targets)
                    base_loss_val = float(loss.detach().item())
                    if getattr(self, 'dino_cfg', {}).get("enabled", False):
                        add_loss, add_items = self._compute_dino_losses(images, targets)
                        if add_loss is not None:
                            try:
                                clamp_cfg = self.dino_cfg.get("share_clamp", {}) if isinstance(
                                    self.dino_cfg, dict
                                ) else {}
                                if bool(clamp_cfg.get("enabled", True)):
                                    smax = float(clamp_cfg.get("max", 0.2))
                                    dtot_val = float(
                                        add_items.get(
                                            "dino_total", float(add_loss.detach().item())
                                        )
                                    )
                                    denom = max(1e-8, base_loss_val + dtot_val)
                                    share = dtot_val / denom
                                    if share > smax and dtot_val > 0.0:
                                        scale = smax / share
                                        add_loss = add_loss * scale
                                        for k in (
                                            "dino_patch", "dino_sal", "dino_global", "dino_contrast"
                                        ):
                                            if k in add_items:
                                                try:
                                                    add_items[k] = float(add_items[k]
                                                                        ) * float(scale)
                                                except Exception:
                                                    pass
                                        add_items["dino_total"] = float(dtot_val * scale)
                                        add_items["dino_clamp_scale"] = float(scale)
                            except Exception as e:
                                self.logger.debug("dino/share_clamp_error", str(e))

                            loss = loss + add_loss
                            try:
                                loss_items.update(add_items)
                            except Exception:
                                pass
                            try:
                                if i == 0 and (not self._bb_group_idx):
                                    self._bb_group_idx = []
                                    self._bb_nominal_lr_epoch = []
                                    self._head_group_idx = []
                                    self._head_nominal_lr_epoch = []
                                    for gi, g in enumerate(self.optimizer.param_groups):
                                        name = str(g.get('name', ''))
                                        if name.startswith('backbone_'):
                                            self._bb_group_idx.append(gi)
                                            self._bb_nominal_lr_epoch.append(
                                                float(g.get('lr', 0.0))
                                            )
                                        elif self._bb_gate_complement_head and name.startswith(
                                            'head_'
                                        ):
                                            self._head_group_idx.append(gi)
                                            self._head_nominal_lr_epoch.append(
                                                float(g.get('lr', 0.0))
                                            )
                                dl = float(add_loss.detach().item())
                                alpha = float(self._bb_gate_ema)
                                self._dino_loss_ema = alpha * self._dino_loss_ema + (
                                    1.0 - alpha
                                ) * dl if self._dino_loss_ema > 0 else dl
                                if self._dino_loss_ema > self._dino_loss_peak:
                                    self._dino_loss_peak = self._dino_loss_ema
                                peak = max(self._dino_loss_peak, 1e-6)
                                ratio = max(0.0, min(1.0, self._dino_loss_ema / peak))
                                s_min = float(self._bb_gate_min)
                                s_max = float(self._bb_scale_cfg) if self._bb_scale_cfg > 0 else 1.0
                                s = s_min + (s_max - s_min) * (1.0 - ratio)
                                mul = s / max(1e-6, float(self._bb_scale_cfg))
                                for idx, gi in enumerate(self._bb_group_idx):
                                    base_lr = self._bb_nominal_lr_epoch[idx] if idx < len(
                                        self._bb_nominal_lr_epoch
                                    ) else float(self.optimizer.param_groups[gi].get('lr', 0.0))
                                    self.optimizer.param_groups[gi]['lr'] = base_lr * mul
                                if self._bb_gate_complement_head and self._head_group_idx:
                                    head_mul = 1.0 / max(1e-6, mul)
                                    head_mul = min(
                                        float(self._bb_gate_head_max_scale), max(0.5, head_mul)
                                    )
                                    for idx, gi in enumerate(self._head_group_idx):
                                        base_lr = self._head_nominal_lr_epoch[idx] if idx < len(
                                            self._head_nominal_lr_epoch
                                        ) else float(
                                            self.optimizer.param_groups[gi].get('lr', 0.0)
                                        )
                                        self.optimizer.param_groups[gi]['lr'] = base_lr * head_mul
                            except Exception as e:
                                self.logger.debug("train/bb_lr_gate_error", str(e))

                            try:
                                self._dino_step_count += 1
                                dp = float(add_items.get("dino_patch", 0.0))
                                ds = float(add_items.get("dino_sal", 0.0))
                                dg = float(add_items.get("dino_global", 0.0))
                                dcont = float(add_items.get("dino_contrast", 0.0))
                                dcls = float(loss_items.get("dino_clsproto", 0.0))
                                dtot = float(add_items.get("dino_total", dp + ds + dg + dcont))
                                denom = max(1e-8, base_loss_val + dtot)
                                dshare = float(dtot / denom)
                                self._dino_epoch_stats["sum_patch"] += dp
                                self._dino_epoch_stats["sum_sal"] += ds
                                self._dino_epoch_stats["sum_global"] += dg
                                self._dino_epoch_stats["sum_contrast"] += dcont
                                self._dino_epoch_stats["sum_total"] += dtot
                                self._dino_epoch_stats.setdefault("sum_share", 0.0)
                                self._dino_epoch_stats["sum_share"] += dshare
                                self._dino_epoch_stats["count"] += 1
                                if (self._dino_step_count <= 3) or \
                                   (self._dino_step_count % max(1, self._dino_log_interval) == 0):
                                    self.logger.basic(
                                        "dino/total", dtot, step=getattr(self.state, 'log_step', 0)
                                    )
                                    self.logger.basic(
                                        "dino/patch", dp, step=getattr(self.state, 'log_step', 0)
                                    )
                                    self.logger.basic(
                                        "dino/sal", ds, step=getattr(self.state, 'log_step', 0)
                                    )
                                    self.logger.basic(
                                        "dino/global", dg, step=getattr(self.state, 'log_step', 0)
                                    )
                                    if dcont:
                                        self.logger.basic(
                                            "dino/contrast",
                                            dcont,
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if dcls:
                                        self.logger.basic(
                                            "dino/clsproto",
                                            dcls,
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    try:
                                        self.logger.basic(
                                            "dino/share",
                                            dshare,
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    except Exception:
                                        pass
                                    try:
                                        if "dino_clamp_scale" in add_items:
                                            self.logger.basic(
                                                "dino/clamp_scale",
                                                float(add_items.get("dino_clamp_scale", 1.0)),
                                                step=getattr(self.state, 'log_step', 0)
                                            )
                                    except Exception:
                                        pass
                                try:
                                    if "dino_sal_kl_raw" in add_items:
                                        self.logger.basic(
                                            "dino/sal_kl_raw",
                                            float(add_items.get("dino_sal_kl_raw", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if "dino_sal_valid_tokens" in add_items:
                                        self.logger.basic(
                                            "dino/sal_valid_tokens",
                                            float(add_items.get("dino_sal_valid_tokens", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if "dino_sal_weighted" in add_items:
                                        self.logger.basic(
                                            "dino/sal_weighted",
                                            float(add_items.get("dino_sal_weighted", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if "dino_sal_beta" in add_items:
                                        self.logger.basic(
                                            "dino/sal_beta",
                                            float(add_items.get("dino_sal_beta", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if "dino_patch_alpha" in add_items:
                                        self.logger.basic(
                                            "dino/patch_alpha",
                                            float(add_items.get("dino_patch_alpha", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                    if "dino_global_gamma" in add_items:
                                        self.logger.basic(
                                            "dino/global_gamma",
                                            float(add_items.get("dino_global_gamma", 0.0)),
                                            step=getattr(self.state, 'log_step', 0)
                                        )
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        else:
                            try:
                                if self._bb_gate_enabled and self._bb_group_idx and self._bb_nominal_lr_epoch:
                                    relax = float(self._bb_gate_relax)
                                    for idx, gi in enumerate(self._bb_group_idx):
                                        base_lr = self._bb_nominal_lr_epoch[idx]
                                        cur = float(
                                            self.optimizer.param_groups[gi].get('lr', base_lr)
                                        )
                                        self.optimizer.param_groups[gi][
                                            'lr'] = cur * (1.0 - relax) + base_lr * relax
                                    if self._bb_gate_complement_head and self._head_group_idx and self._head_nominal_lr_epoch:
                                        for idx, gi in enumerate(self._head_group_idx):
                                            base_lr = self._head_nominal_lr_epoch[idx]
                                            cur = float(
                                                self.optimizer.param_groups[gi].get('lr', base_lr)
                                            )
                                            self.optimizer.param_groups[gi][
                                                'lr'] = cur * (1.0 - relax) + base_lr * relax
                            except Exception:
                                pass
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                do_step = (i + 1) % self.grad_accum_steps == 0 or (i + 1) == nb
                if do_step:
                    self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_norm
                    )
                    self.last_grad_norm = grad_norm.item(
                    ) if torch.is_tensor(grad_norm) else grad_norm

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.logger.warning(
                            "train/grad_anomaly",
                            f"NaN/Inf gradients detected at step {i}, skipping update"
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        try:
                            self.state.log_step += 1
                            self.state.global_step = self.state.log_step
                        except Exception:
                            pass
                    else:
                        self.scaler.step(self.optimizer)
                        try:
                            self.state.opt_step += 1
                            self.state.log_step += 1
                            self.state.global_step = self.state.log_step
                        except Exception:
                            pass
                        if self.use_ema and self.ema is not None:
                            self.ema.update(
                                self.model
                            )  # Now handles both parameters and buffers via state_dict

                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += loss.detach().item()
                cur = loss.detach().item() * self.grad_accum_steps
                pbar.set_postfix(loss=f"{cur:.4f}")
                self._maybe_set_log_step(getattr(self.state, 'log_step', 0))

                self.callbacks.on_batch_end(self, i, loss, loss_items)
        finally:
            if use_prefetch and isinstance(iter_src, CUDAPrefetcher):
                try:
                    iter_src.close()
                except Exception:
                    pass

            self._bb_group_idx = []
            self._bb_nominal_lr_epoch = []

        return total_loss / max(1, nb)

    def _init_dino_cfg(self):
        """Read DINO config once from centralized YOLOConfig without mutating it here."""
        try:
            model_hyp = getattr(self.model, 'hyp', None)
            config = get_config(cfg=self.cfg, hyp=model_hyp)
            d = config.get("dino", {})
            self.dino_cfg = dict(d) if isinstance(d, dict) else {}
        except Exception:
            self.dino_cfg = {}

    def _maybe_init_dino_heads(self, student_feats, teacher_tokens: torch.Tensor):
        """Initialize distillation heads. Accepts a single feature map or a list of maps.
        Multi-level mode is used when a list/tuple is provided.
        """
        feats_list = student_feats
        if not isinstance(feats_list, (list, tuple)):
            feats_list = [student_feats]

        if self._dino_ready and self._dino_levels == len(feats_list):
            return

        Ct = int(teacher_tokens.shape[-1])
        chs = [int(f.shape[1]) for f in feats_list]
        median_cs = sorted(chs)[len(chs) // 2]
        Cd = min(256, max(64, median_cs))

        self.dino_teacher_proj = torch.nn.Linear(Ct, Cd, bias=False).to(self.device)
        for p in self.dino_teacher_proj.parameters():
            p.requires_grad_(False)

        self.dino_student_proj_list = torch.nn.ModuleList([
            torch.nn.Conv2d(int(f.shape[1]), Cd, kernel_size=1) for f in feats_list
        ]).to(self.device)
        self.dino_aux_obj_list = torch.nn.ModuleList([
            torch.nn.Conv2d(int(f.shape[1]), 1, kernel_size=1) for f in feats_list
        ]).to(self.device)
        self.dino_global_proj_list = torch.nn.ModuleList([
            torch.nn.Linear(int(f.shape[1]), Ct, bias=True) for f in feats_list
        ]).to(self.device)

        if self.optimizer is not None:
            try:
                params = []
                params += list(self.dino_student_proj_list.parameters())
                params += list(self.dino_aux_obj_list.parameters())
                params += list(self.dino_global_proj_list.parameters())
                if len(params):
                    self.optimizer.add_param_group({"params": params})
            except Exception as e:
                self.logger.debug("dino/opt_add_param_group_error", str(e))

        self._dino_levels = len(feats_list)
        self._dino_ready = True

    def _dino_weights(self) -> tuple[float, float, float]:
        """Constant weights during active epochs; hard stop after max_epochs (if > 0).
        Uses defaults aligned with core/config.py DEFAULTS.dino if any key is missing.
        """
        e = float(self.epoch)
        max_ep = int(self.dino_cfg.get("max_epochs", 0))
        if max_ep > 0 and e >= max_ep:
            return 0.0, 0.0, 0.0
        a0 = float(self.dino_cfg.get("alpha", 0.5))
        b0 = float(self.dino_cfg.get("beta", 0.2))
        g0 = float(self.dino_cfg.get("gamma", 0.1))
        return a0, b0, g0

    def _compute_dino_losses(self, images: torch.Tensor, targets: torch.Tensor | None = None):
        try:
            step = int(getattr(self.state, 'log_step', 0))
        except Exception as e:
            try:
                self.logger.warning("dino/step_error", str(e))
            except Exception:
                pass
        if (not bool(self.dino_cfg.get("enabled", False))) or (self.dino_teacher is None):
            return None, {}
        try:
            max_ep = int(self.dino_cfg.get("max_epochs", 0))
            if max_ep > 0 and int(self.epoch) >= max_ep:
                self._disable_dino("hard_stop_step")
                return None, {}
        except Exception:
            pass
        every_n = 1 if bool(self.dino_cfg.get("objfor02", False)
                           ) else int(self.dino_cfg.get("every_n", 2))
        if every_n > 1 and (step % every_n) != 0:
            return None, {}

        a, b, g = self._dino_weights()
        objfor02 = bool(self.dino_cfg.get("objfor02", False))
        if (a + b + g) <= 1e-6 and (not objfor02):
            return None, {}

        teach = self.dino_teacher.forward(images)
        if teach is None:
            return None, {}
        tokens_t = teach["tokens"]  # [B,Ht,Wt,Ct]
        sal_t = teach["saliency"]  # [B,Ht,Wt]
        cls_t = teach["cls"]  # [B,Ct]

        det = getattr(self.model, "detect_layer", None)
        feats = getattr(det, "last_feats", None)
        if feats is None:
            pyr = getattr(self.model, "get_pyramid", None)
            feats = pyr() if callable(pyr) else None
        if feats is None:
            try:
                self.logger.debug("dino/no_feats", "Detect.last_feats not set")
            except Exception:
                pass
            return None, {}

        Ht, Wt = int(sal_t.shape[1]), int(sal_t.shape[2])

        if objfor02:
            try:
                obj_flat_list = []
                obj_levels: list[torch.Tensor] = []
                B = int(images.shape[0])
                for i, f in enumerate(feats):
                    Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
                    prior_lvl = torch.nn.functional.adaptive_avg_pool2d(
                        sal_t, (Hi, Wi)
                    )  # [B,Hi,Wi]
                    prior_lvl = prior_lvl.unsqueeze(1)  # [B,1,Hi,Wi]
                    vmin = prior_lvl.amin(dim=(2, 3), keepdim=True)
                    vmax = prior_lvl.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    prior_norm = (prior_lvl - vmin) / (vmax - vmin)
                    obj_flat_list.append(prior_norm.flatten(2))  # [B,1,N_l]
                    try:
                        obj_levels.append(prior_norm.squeeze(1).detach())  # [B,Hi,Wi]
                    except Exception:
                        pass
                obj_flat = torch.cat(obj_flat_list, dim=2).squeeze(1)  # [B,N]
                det.last_dino_obj_flat = obj_flat.detach()
                try:
                    det.last_dino_obj_levels = obj_levels  # list of [B,Hi,Wi]
                except Exception:
                    pass
                self.logger.basic("dino/obj_tokens", float(obj_flat.shape[1]))
            except Exception as e:
                try:
                    self.logger.debug("dino/obj_prior_error", str(e))
                except Exception:
                    pass
            return None, {}

        try:
            obj_levels: list[torch.Tensor] = []
            for f in feats:
                Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
                pl = torch.nn.functional.adaptive_avg_pool2d(sal_t, (Hi, Wi)).unsqueeze(1)
                vmin = pl.amin(dim=(2, 3), keepdim=True)
                vmax = pl.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                pn = (pl - vmin) / (vmax - vmin)
                obj_levels.append(pn.squeeze(1).detach())
            det.last_dino_obj_levels = obj_levels
        except Exception:
            pass

        try:
            pad_val = float(getattr(det, "pad_value", 114)) / 255.0
        except Exception:
            pad_val = 114.0 / 255.0
        eps = 1e-3
        valid_pix = (images - pad_val).abs().amax(dim=1) > eps  # [B,H,W]
        mask_hw = torch.nn.functional.adaptive_max_pool2d(valid_pix.float().unsqueeze(1),
                                                          (Ht, Wt)).squeeze(1) > 0.5  # [B,Ht,Wt]
        mask_flat = mask_hw.flatten(1).float()
        valid_counts = mask_flat.sum(dim=1).clamp_min(1.0)

        self._maybe_init_dino_heads(list(feats), tokens_t)

        t_proj = self.dino_teacher_proj(tokens_t.reshape(tokens_t.shape[0], -1, tokens_t.shape[-1]))
        t_proj = t_proj.reshape(tokens_t.shape[0], Ht, Wt, -1).permute(0, 3, 1, 2).contiguous()
        t_norm = torch.nn.functional.normalize(t_proj.flatten(2), dim=1)

        dists = []
        for f in feats:
            Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
            d = abs(Hi - Ht) + abs(Wi - Wt)
            dists.append(float(d) / float(max(1, Ht + Wt)))
        temp = float(self.dino_cfg.get("level_temp", 0.25))
        scores = torch.tensor([-d / max(1e-6, temp) for d in dists], device=self.device)
        w = torch.softmax(scores, dim=0)  # [L]

        loss_patch = torch.zeros((), device=self.device)
        loss_sal = torch.zeros((), device=self.device)
        kl_raw_sum = torch.zeros((), device=self.device)
        kl_raw_count = 0
        t_glb = torch.nn.functional.normalize(cls_t, dim=-1)
        s_glb_agg = None

        for i, f in enumerate(feats):
            wi = w[i]
            f_s = torch.nn.functional.adaptive_avg_pool2d(f, (Ht, Wt))

            if a > 0.0 and (not objfor02):
                s_proj = self.dino_student_proj_list[i](f_s)
                s_norm = torch.nn.functional.normalize(s_proj.flatten(2), dim=1)
                cos = (s_norm * t_norm).sum(dim=1)  # [B, Ht*Wt]
                lp = 1.0 - ((cos * mask_flat).sum(dim=1) / valid_counts).mean()
                loss_patch = loss_patch + wi * lp

            if (not objfor02):
                Ttau = float(self.dino_cfg.get("sal_temp", 0.7))
                sal_t_logits = (sal_t.to(torch.float32) / max(Ttau, 1e-6)).flatten(1)
                s_logits = self.dino_aux_obj_list[i](f_s).flatten(1).to(torch.float32)
                neg_inf = torch.finfo(s_logits.dtype).min
                inv_mask = (~mask_flat.bool())
                sal_t_logits = sal_t_logits.masked_fill(inv_mask, neg_inf)
                s_logits = s_logits.masked_fill(inv_mask, neg_inf)
                p = torch.nn.functional.softmax(sal_t_logits, dim=1)
                q = torch.nn.functional.log_softmax(s_logits, dim=1)
                ls = torch.nn.functional.kl_div(q, p, reduction='batchmean')
                loss_sal = loss_sal + wi * ls
                kl_raw_sum = kl_raw_sum + ls.detach()
                kl_raw_count += 1

            if g > 0.0 and (not objfor02):
                Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
                mask_pi = torch.nn.functional.adaptive_avg_pool2d(
                    valid_pix.float().unsqueeze(1), (Hi, Wi)
                ).clamp(0, 1)
                denom = mask_pi.sum(dim=(2, 3)).clamp_min(1e-6)
                s_sum = (f * mask_pi).sum(dim=(2, 3))
                s_gap = s_sum / denom
                s_glb_i = self.dino_global_proj_list[i](s_gap)
                s_glb_i = torch.nn.functional.normalize(s_glb_i, dim=-1)
                s_glb_agg = wi * s_glb_i if s_glb_agg is None else s_glb_agg + wi * s_glb_i

        if objfor02:
            try:
                obj_flat_list = []
                obj_levels: list[torch.Tensor] = []
                B = int(images.shape[0])
                for i, f in enumerate(feats):
                    Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
                    prior_lvl = torch.nn.functional.adaptive_avg_pool2d(
                        sal_t, (Hi, Wi)
                    )  # [B,Hi,Wi]
                    prior_lvl = prior_lvl.unsqueeze(1)  # [B,1,Hi,Wi]
                    vmin = prior_lvl.amin(dim=(2, 3), keepdim=True)
                    vmax = prior_lvl.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    prior_norm = (prior_lvl - vmin) / (vmax - vmin)
                    obj_flat_list.append(prior_norm.flatten(2))  # [B,1,N_l]
                obj_flat = torch.cat(obj_flat_list, dim=2).squeeze(1)  # [B,N]
                det.last_dino_obj_flat = obj_flat.detach()
                try:
                    obj_levels: list[torch.Tensor] = []
                    for f in feats:
                        Hi, Wi = int(f.shape[-2]), int(f.shape[-1])
                        pl = torch.nn.functional.adaptive_avg_pool2d(sal_t, (Hi, Wi)).unsqueeze(1)
                        vmin = pl.amin(dim=(2, 3), keepdim=True)
                        vmax = pl.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                        pn = (pl - vmin) / (vmax - vmin)
                        obj_levels.append(pn.squeeze(1).detach())
                    det.last_dino_obj_levels = obj_levels
                except Exception:
                    pass
                self.logger.basic("dino/obj_tokens", float(obj_flat.shape[1]))
            except Exception as e:
                try:
                    self.logger.debug("dino/obj_prior_error", str(e))
                except Exception:
                    pass

        if objfor02:
            return None, {}

        loss_glb = torch.zeros((), device=self.device)
        if g > 1e-8 and s_glb_agg is not None:
            s_glb = torch.nn.functional.normalize(s_glb_agg, dim=-1)
            loss_glb = 1.0 - (s_glb * t_glb).sum(dim=-1).mean()

        loss_contrast = torch.zeros((), device=self.device)
        contrast_cfg = self.dino_cfg.get("contrast", {}) if isinstance(self.dino_cfg, dict) else {}
        if bool(contrast_cfg.get("enabled", False)):
            try:
                s_fused = None
                for i, f in enumerate(feats):
                    f_s = torch.nn.functional.adaptive_avg_pool2d(f, (Ht, Wt))
                    s_i = self.dino_student_proj_list[i](f_s)
                    s_fused = (w[i] * s_i) if s_fused is None else (s_fused + w[i] * s_i)
                s_tok = s_fused.flatten(2)  # [B,Cd,N]
                s_tok = torch.nn.functional.normalize(s_tok, dim=1)

                t_tok = t_proj.flatten(2)  # [B,Cd,N]
                t_tok = torch.nn.functional.normalize(t_tok, dim=1)

                B, _, N = t_tok.shape
                int(contrast_cfg.get("pos_k", 1))
                neg_k = int(contrast_cfg.get("neg_k", 32))
                samp = int(contrast_cfg.get("samples_per_img", 32))
                sal_topk = int(contrast_cfg.get("sal_topk", 0))
                tau = float(contrast_cfg.get("temp", 0.2))

                total = 0.0
                count = 0
                for b in range(B):
                    if sal_topk and sal_topk > 0:
                        sflat = sal_t[b].flatten()
                        kA = min(int(sal_topk), int(sflat.numel()))
                        anchors = torch.topk(sflat, k=kA, dim=0).indices
                    else:
                        kA = min(samp, int(Ht * Wt))
                        anchors = torch.randperm(int(Ht * Wt), device=self.device)[:kA]

                    if anchors.numel() == 0:
                        continue
                    t_b = t_tok[b]  # [Cd,N]
                    s_b = s_tok[b]  # [Cd,N]
                    sim_rows = torch.matmul(t_b.transpose(0, 1)[anchors], t_b)  # [A,N]
                    sim_rows.scatter_(1, anchors.view(-1, 1), -1e9)
                    pos1_idx = torch.topk(sim_rows, k=1, dim=1).indices.squeeze(1)  # [A]

                    neg_idx = torch.randint(
                        0, N, (anchors.shape[0], max(1, neg_k)), device=self.device
                    )
                    neg_idx = neg_idx.clamp(0, N - 1)

                    q = s_b[:, anchors]  # [Cd,A]
                    q = torch.nn.functional.normalize(q, dim=0)
                    pos = s_b.transpose(0, 1)[pos1_idx]  # [A,Cd]
                    qT = q.transpose(0, 1)  # [A,Cd]
                    l_pos = (qT * pos).sum(dim=1)  # [A]
                    neg = s_b.transpose(0, 1)[neg_idx]  # [A,neg_k,Cd]
                    l_neg = torch.matmul(qT, neg.transpose(1, 2)).squeeze(-1)  # [A,neg_k]

                    l_pos = l_pos / max(1e-6, tau)
                    l_neg = l_neg / max(1e-6, tau)
                    ln_denom = torch.logsumexp(torch.cat([l_pos.unsqueeze(1), l_neg], dim=1), dim=1)
                    loss_b = (ln_denom - l_pos).mean()
                    total += loss_b
                    count += 1
                if count > 0:
                    loss_contrast = total / count
            except Exception as e:
                try:
                    self.logger.debug("dino/contrast_error", str(e))
                except Exception:
                    pass

        loss_clsproto = torch.zeros((), device=self.device)
        clsproto_cfg = self.dino_cfg.get("cls_proto", {}) if isinstance(self.dino_cfg, dict) else {}
        if bool(clsproto_cfg.get("enabled", False)) and targets is not None and targets.numel() > 0:
            try:
                proto_weight = float(clsproto_cfg.get("weight", 0.05))
                proto_temp = float(clsproto_cfg.get("temp", 0.3))
                use_momentum = bool(clsproto_cfg.get("use_momentum", True))
                mom = float(clsproto_cfg.get("momentum", 0.9))
                B = images.shape[0]
                Ct = int(tokens_t.shape[-1])
                C = int(getattr(self.model, "nc", 80))
                proto_batch = torch.zeros((C, Ct), device=tokens_t.device, dtype=tokens_t.dtype)
                counts = torch.zeros((C, ), device=tokens_t.device, dtype=torch.float32)
                Y = torch.arange(Ht, device=tokens_t.device).view(1, Ht, 1)
                X = torch.arange(Wt, device=tokens_t.device).view(1, 1, Wt)
                for b in range(B):
                    tb = tokens_t[b]  # [Ht,Wt,Ct]
                    t_b = targets[targets[:, 0].long() == b]
                    if t_b.numel() == 0:
                        continue
                    cl = t_b[:, 1].long().clamp_(0, C - 1)
                    xs = t_b[:, 2] * float(Wt)
                    ys = t_b[:, 3] * float(Ht)
                    ws = t_b[:, 4] * float(Wt)
                    hs = t_b[:, 5] * float(Ht)
                    x1 = (xs - ws * 0.5).clamp_(0, float(Wt - 1))
                    y1 = (ys - hs * 0.5).clamp_(0, float(Ht - 1))
                    x2 = (xs + ws * 0.5).clamp_(0, float(Wt - 1))
                    y2 = (ys + hs * 0.5).clamp_(0, float(Ht - 1))
                    G = t_b.shape[0]
                    if G == 0:
                        continue
                    x1v = x1.view(G, 1, 1)
                    y1v = y1.view(G, 1, 1)
                    x2v = x2.view(G, 1, 1)
                    y2v = y2.view(G, 1, 1)
                    mask = (X >= x1v) & (X <= x2v) & (Y >= y1v) & (Y <= y2v)
                    mask_flat = mask.view(G, -1).float()
                    t_flat = tb.view(-1, Ct)  # [Ht*Wt,Ct]
                    sums = mask_flat @ t_flat  # [G,Ct]
                    denom = mask_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
                    vecs = (sums / denom)  # [G,Ct]
                    proto_batch.index_add_(0, cl, vecs)
                    counts.index_add_(0, cl, torch.ones((G, ), device=tokens_t.device))
                nonzero = counts > 0
                if nonzero.any():
                    proto_batch[nonzero] = torch.nn.functional.normalize(
                        proto_batch[nonzero], dim=1
                    )
                if use_momentum:
                    if not hasattr(self, "_dino_proto_mem") or self._dino_proto_mem is None or \
                       self._dino_proto_mem.shape != proto_batch.shape:
                        self._dino_proto_mem = proto_batch.clone().detach()
                    else:
                        mem = self._dino_proto_mem
                        update = proto_batch.clone().detach()
                        upd_mask = nonzero.unsqueeze(1)
                        mem = torch.where(upd_mask, (mom * mem + (1.0 - mom) * update), mem)
                        mem_norm = torch.linalg.norm(mem, dim=1, keepdims=True).clamp_min(1e-6)
                        mem = mem / mem_norm
                        self._dino_proto_mem = mem
                    prototypes = self._dino_proto_mem.detach()
                else:
                    prototypes = proto_batch.detach()

                try:
                    self.criterion.dino_proto_enabled = True
                    self.criterion.dino_class_prototypes = prototypes.to(self.device)
                    self.criterion.dino_proto_temp = float(proto_temp)
                    self.criterion.dino_proto_weight = float(proto_weight)
                except Exception:
                    pass
            except Exception as e:
                try:
                    self.logger.debug("dino/clsproto_error", str(e))
                except Exception:
                    pass

        loss = a * loss_patch + b * loss_sal + g * loss_glb
        if loss_contrast is not None and torch.is_tensor(loss_contrast):
            loss = loss + float(contrast_cfg.get("weight", 0.05)) * loss_contrast
        items = {
            "dino_patch": float((a * loss_patch).detach().item()),
            "dino_sal": float((b * loss_sal).detach().item()),
            "dino_global": float((g * loss_glb).detach().item()),
        }
        try:
            items["dino_patch_alpha"] = float(a)
            items["dino_sal_beta"] = float(b)
            items["dino_global_gamma"] = float(g)
            items["dino_sal_weighted"] = float(loss_sal.detach().item())
        except Exception:
            pass
        try:
            if kl_raw_count > 0:
                items["dino_sal_kl_raw"] = float((kl_raw_sum / float(max(1, kl_raw_count))).item())
            items["dino_sal_valid_tokens"] = float(valid_counts.mean().item())
        except Exception:
            pass
        if bool(contrast_cfg.get("enabled", False)):
            try:
                items["dino_contrast"] = float(
                    loss_contrast.detach().item() * float(contrast_cfg.get("weight", 0.05))
                )
            except Exception:
                pass
        try:
            items["dino_total"] = float(loss.detach().item())
        except Exception:
            pass
        return loss, items

    @torch.no_grad()
    def validate(self, dataloader: DataLoader):
        """Run validation using the Validator and return its payload (for callbacks)."""
        try:
            self.validator.state.log_step = getattr(self.state, 'log_step', 0)
            self.validator.state.global_step = getattr(self.state, 'log_step', 0)
        except Exception:
            pass
        self.validator.logger.set_step(getattr(self.state, 'log_step', 0))
        payload = self.validator.validate(dataloader, model=self.ema.ema) if (
            self.use_ema and self.ema is not None
        ) else self.validator.validate(dataloader)

        if isinstance(payload, dict) and 'scalars' not in payload:
            keys = set(payload.keys())
            if any(k in keys for k in ('mAP50-95', 'mAP50', 'mAP75', 'fitness')):
                payload = {'scalars': payload}

        if getattr(self.validator, "sample_images", None) is not None:
            self.val_images = self.validator.sample_images
            self.val_boxes = getattr(self.validator, "sample_boxes", None)
        if getattr(self.validator, "sample_images_orig", None) is not None:
            self.val_images_orig = self.validator.sample_images_orig
            self.val_boxes_orig = getattr(self.validator, "sample_boxes_orig", None)

        return payload

    def fit(
        self,
        epochs: int,
        criterion: Optional[Callable[[Any, Any], torch.Tensor]] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        ckpt_dir: Path = Path("runs/checkpoints"),
    ):
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        assert self.train_loader is not None, "train_loader is required"

        if criterion is None:
            criterion = self.criterion
        assert criterion is not None, "Criterion must be provided to fit() or Trainer constructor"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs  # Store for callbacks

        self.callbacks.on_train_start(self)

        start_epoch = int(self.epoch)
        end_epoch_exclusive = start_epoch + int(epochs)
        for epoch in range(start_epoch, end_epoch_exclusive):
            self.epoch = epoch
            self.state.epoch = epoch

            self.callbacks.on_epoch_start(self, epoch)

            train_loss = self.train_epoch(self.train_loader, criterion)

            metrics = {"train_loss": train_loss}
            if getattr(self, 'dino_cfg', {}).get("enabled",
                                                 False) and self._dino_epoch_stats["count"] > 0:
                c = max(1, int(self._dino_epoch_stats["count"]))
                avg_patch = self._dino_epoch_stats["sum_patch"] / c
                avg_sal = self._dino_epoch_stats["sum_sal"] / c
                avg_global = self._dino_epoch_stats["sum_global"] / c
                avg_total = self._dino_epoch_stats.get("sum_total", 0.0) / c
                avg_share = self._dino_epoch_stats.get("sum_share", 0.0) / c
                metrics.update({
                    "dino_total": avg_total,
                    "dino_share": avg_share,
                    "dino_patch": avg_patch,
                    "dino_sal": avg_sal,
                    "dino_global": avg_global,
                })
                try:
                    self.logger.basic(
                        "dino/avg_total", float(avg_total), step=getattr(self.state, 'log_step', 0)
                    )
                    self.logger.basic(
                        "dino/avg_share", float(avg_share), step=getattr(self.state, 'log_step', 0)
                    )
                    self.logger.basic(
                        "dino/avg_patch", float(avg_patch), step=getattr(self.state, 'log_step', 0)
                    )
                    self.logger.basic(
                        "dino/avg_sal", float(avg_sal), step=getattr(self.state, 'log_step', 0)
                    )
                    self.logger.basic(
                        "dino/avg_global",
                        float(avg_global),
                        step=getattr(self.state, 'log_step', 0)
                    )
                except Exception:
                    pass

            if self.val_loader is not None:
                val_payload = self.validate(self.val_loader)
                if isinstance(val_payload, dict) and 'scalars' in val_payload:
                    val_metrics = val_payload['scalars']
                    metrics.update(val_metrics)
                    self.callbacks.on_val_end(self, val_payload)
                else:
                    val_metrics = val_payload if isinstance(val_payload, dict) else {}
                    metrics.update(val_metrics)
                    self.callbacks.on_val_end(self, val_payload)

            self.callbacks.on_epoch_end(self, epoch, metrics)

            if self.stop_training:
                self.logger.info("train/early_stop", f"Training stopped early at epoch {epoch}")
                break

        self.callbacks.on_train_end(self)

    def resume_training(self, checkpoint_path: Path):
        """
        Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file to resume from
        """
        self.logger.info("train/resume", f"Resuming from {checkpoint_path}")

        ckpt = self.load_checkpoint(checkpoint_path)

        if "optimizer" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        if "scaler" in ckpt and self.scaler is not None and self.amp:
            self.scaler.load_state_dict(ckpt["scaler"])

        if "ema" in ckpt and self.ema is not None:
            self.ema.ema.load_state_dict(ckpt["ema"])
            if "ema_updates" in ckpt:
                self.ema.updates = ckpt["ema_updates"]

        if "epoch" in ckpt:
            self.epoch = ckpt["epoch"] + 1  # Start from next epoch

        if "best_fitness" in ckpt:
            self.best_fitness = ckpt["best_fitness"]

        try:
            hyp_from_ckpt = None
            if isinstance(ckpt, dict) and isinstance(ckpt.get("cfg", None), dict):
                hyp_from_ckpt = ckpt["cfg"].get("hyp", None)
            if isinstance(hyp_from_ckpt, dict):
                self.model.hyp = hyp_from_ckpt
        except Exception:
            pass

        try:
            from core.runtime import attach_runtime
            attach_runtime(self.model, imgsz=int(self.cfg.get("img_size", 640)))
        except Exception as e:
            self.logger.debug("train/resume/runtime_attach_error", str(e))

        try:
            self.logger.set_step(getattr(self.state, 'log_step', 0))
        except Exception:
            pass

        self.logger.info("train/resumed", {"epoch": self.epoch, "best_fitness": self.best_fitness})

        return ckpt

    def _maybe_build_dataloaders_from_cfg(self):
        """
        If cfg contains a 'data' root, try to create train/val loaders.
        Falls back to dummy loader when cfg['allow_dummy'] is True.
        """
        data_root = self.cfg.get("data", None)
        if data_root is None:
            return  # nothing to build

        model_hyp = getattr(self.model, 'hyp', None)
        config = get_config(cfg=self.cfg, hyp=model_hyp)
        img_size = int(self.cfg.get("img_size", 640))
        batch_size = int(self.cfg.get("batch_size", 16))
        num_workers = int(self.cfg.get("num_workers", 4))
        pin_memory = bool(self.cfg.get("pin_memory", True))
        augment = bool(self.cfg.get("augment", config.get('augment', True)))
        cache = self.cfg.get("cache", None)
        use_dummy = bool(self.cfg.get("allow_dummy", False))
        cache_compression = bool(self.cfg.get("cache_compression", True))
        persistent_workers = self.cfg.get("persistent_workers", False)
        prefetch_factor = self.cfg.get("prefetch_factor", 2)

        try:
            from utils.dataloaders import create_dataloaders as create_dataloaders_mod
            self.logger.info("trainer/data_build", f"Building dataloaders from {data_root}")

            max_stride = int(
                max(self.model.strides)
            ) if hasattr(self.model, 'strides') and self.model.strides is not None else 32

            train_loader, val_loader, info = create_dataloaders_mod(
                data_path=data_root,
                img_size=img_size,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                augment=augment,
                cache=cache,
                model_nc=getattr(self.model, "nc", None),
                hyp=self.cfg.get('hyp', None),
                max_stride=max_stride,
                cache_compression=cache_compression,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            self.train_loader, self.val_loader = train_loader, val_loader

            state = self.model.get_detection_state()
            if self.train_loader and hasattr(self.train_loader.dataset, 'letterbox_center'):
                self.train_loader.dataset.letterbox_center = state['letterbox_center']
                self.train_loader.dataset.pad_value = state['pad_value']
            if self.val_loader and hasattr(self.val_loader.dataset, 'letterbox_center'):
                self.val_loader.dataset.letterbox_center = state['letterbox_center']
                self.val_loader.dataset.pad_value = state['pad_value']

            self.class_names = info.get("names", None)
            if self.class_names is not None:
                self.model.set_class_names(self.class_names)

            data_nc = int(info.get("nc", -1))
            model_nc = getattr(self.model, "nc", None)
            if model_nc is not None and data_nc > 0 and model_nc != data_nc:
                raise ValueError(f"Model nc={model_nc} but dataset nc={data_nc}")

            block = {
                "nc": int(info.get("nc", -1)),
                "names": info.get("names", []),
                "batch_size": batch_size,
                "img_size": img_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "augment": augment,
                "format": info.get("format", "yolo/coco"),
                "strides": getattr(self.model, "strides", None),
            }
            import json
            self.logger.log_text("run/config", json.dumps(_jsonify_cfg(block), indent=2))

        except Exception as e:
            import traceback
            self.logger.debug("trainer/data_build_error", f"{e}\n{traceback.format_exc()}")
            if use_dummy:
                self.logger.info("trainer/data_dummy", "Falling back to dummy dataloader")
                from utils.dataloaders import create_dummy_dataloader

                dummy, info = create_dummy_dataloader(
                    batch_size=batch_size,
                    img_size=img_size,
                    length=10,
                    num_classes=getattr(self.model, "nc", 80)
                )
                self.train_loader = dummy
                self.val_loader = dummy
            else:
                raise

    def _maybe_set_log_step(self, step: int) -> None:
        """Update logger step and return current step."""
        super()._maybe_set_log_step(step)
