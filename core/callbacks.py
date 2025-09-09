"""
core/callbacks.py

Callback/Hook system for training pipeline to reduce boilerplate and improve extensibility.
"""
from typing import Dict, Optional, List
from abc import ABC
import torch
from pathlib import Path
import numpy as np
import yaml

class Callback(ABC):
    """Base callback class for training hooks."""
    def on_train_start(self, trainer):
        """Called at the beginning of training."""

    def on_train_end(self, trainer):
        """Called at the end of training."""

    def on_epoch_start(self, trainer, epoch: int):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""

    def on_batch_start(self, trainer, batch_idx: int):
        """Called before processing each batch."""

    def on_batch_end(
        self,
        trainer,
        batch_idx: int,
        loss: torch.Tensor,
        loss_items: Optional[Dict[str, float]] = None
    ):
        """Called after processing each batch."""

    def on_val_end(self, trainer, metrics: Dict[str, float]):
        """Called after validation."""

class LoggingCallback(Callback):
    """Handles logging during training."""
    def on_train_start(self, trainer):
        import json
        trainer.logger.log_text(
            "run/info", json.dumps({"epochs": trainer.epochs, "epoch_start": 0}, indent=2)
        )

    def on_epoch_start(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        trainer.logger.log_losses({"train": metrics.get('train_loss', 0.0)},
                                  step=getattr(trainer.state, 'log_step', 0))

    def on_batch_end(self, trainer, batch_idx, loss, loss_items: Optional[Dict[str, float]] = None):
        try:
            total_loss_step = float(
                loss.detach().item() * max(1, int(getattr(trainer, 'grad_accum_steps', 1)))
            )
        except Exception:
            total_loss_step = float(loss.detach().item())
        trainer.logger.basic(
            "loss/total", total_loss_step, step=getattr(trainer.state, 'log_step', 0)
        )

        if isinstance(loss_items, dict):
            if 'box' in loss_items:
                trainer.logger.basic(
                    "loss/box",
                    float(loss_items['box']),
                    step=getattr(trainer.state, 'log_step', 0)
                )
            if 'cls' in loss_items:
                trainer.logger.basic(
                    "loss/cls",
                    float(loss_items['cls']),
                    step=getattr(trainer.state, 'log_step', 0)
                )
            if 'obj' in loss_items:
                trainer.logger.basic(
                    "loss/obj",
                    float(loss_items['obj']),
                    step=getattr(trainer.state, 'log_step', 0)
                )
            if 'dfl' in loss_items:
                trainer.logger.basic(
                    "loss/dfl",
                    float(loss_items['dfl']),
                    step=getattr(trainer.state, 'log_step', 0)
                )
            if 'dino_centroid' in loss_items:
                trainer.logger.basic(
                    "loss/dino_centroid",
                    float(loss_items['dino_centroid']),
                    step=getattr(trainer.state, 'log_step', 0)
                )

    def on_val_end(self, trainer, metrics):
        return

class ValidationLoggingCallback(Callback):
    """Logs validation payload (scalars, histograms, and figures) in BASIC; gates heavy visuals."""
    def on_val_end(self, trainer, payload):
        logger = trainer.logger
        if not isinstance(payload, dict):
            return
        names = payload.get('names') or []
        scalars = payload.get('scalars', {})
        if 'mAP50-95' in scalars:
            logger.basic(
                "metrics/mAP50-95",
                float(scalars['mAP50-95']),
                step=getattr(trainer.state, 'log_step', 0)
            )
        if 'mAP75' in scalars:
            logger.basic(
                "metrics/mAP75",
                float(scalars['mAP75']),
                step=getattr(trainer.state, 'log_step', 0)
            )
        if 'mAP50' in scalars:
            logger.basic(
                "metrics/mAP50",
                float(scalars['mAP50']),
                step=getattr(trainer.state, 'log_step', 0)
            )
        if 'precision_macro' in scalars:
            logger.basic(
                "val/precision_macro",
                float(scalars['precision_macro']),
                step=getattr(trainer.state, 'log_step', 0)
            )
        if 'recall_macro' in scalars:
            logger.basic(
                "val/recall_macro",
                float(scalars['recall_macro']),
                step=getattr(trainer.state, 'log_step', 0)
            )
        if 'fitness' in scalars:
            logger.basic(
                "val/fitness",
                float(scalars['fitness']),
                step=getattr(trainer.state, 'log_step', 0)
            )

        try:
            keys = ['mAP50-95', 'mAP50', 'mAP75', 'fitness']
            vals = {k: float(scalars[k]) for k in keys if k in scalars}
            for k in keys:
                if k in vals:
                    logger.info(f"metrics/{k}", vals[k], step=getattr(trainer.state, 'log_step', 0))
            if vals:
                parts = ", ".join(f"{k}:{vals[k]:.4f}" for k in keys if k in vals)
                logger.info(
                    "metrics/summary",
                    parts,
                    step=getattr(trainer.state, 'log_step', 0),
                    console_msg=parts
                )
        except Exception:
            pass

        stats = payload.get('stats') or {}
        try:
            tp = stats.get('tp')
            conf = stats.get('conf')
            tgt_cls = stats.get('target_cls')
            if tp is not None and conf is not None and tgt_cls is not None and tp.size:
                tp0 = tp[:, 0].astype(np.float64)
                order = np.argsort(-conf)
                tpc = tp0[order].cumsum()
                fpc = (1.0 - tp0[order]).cumsum()
                recall = tpc / max(len(tgt_cls), 1)
                precision = tpc / np.maximum(tpc + fpc, 1)
                f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
                best_idx = int(np.argmax(f1))
                best_f1 = float(f1[best_idx])
                best_p = float(precision[best_idx])
                best_r = float(recall[best_idx])
                thresh_sorted = np.sort(conf)[::-1]
                best_thr = float(thresh_sorted[best_idx]) if thresh_sorted.size else 0.0
                logger.basic("val/best_F1", best_f1, step=getattr(trainer.state, 'log_step', 0))
                logger.basic(
                    "val/best_F1_threshold", best_thr, step=getattr(trainer.state, 'log_step', 0)
                )
                logger.basic(
                    "val/precision@bestF1", best_p, step=getattr(trainer.state, 'log_step', 0)
                )
                logger.basic(
                    "val/recall@bestF1", best_r, step=getattr(trainer.state, 'log_step', 0)
                )
        except Exception as e:
            logger.debug("val/best_f1_error", str(e))

        dists = payload.get('distributions') or {}
        if isinstance(dists.get('confidence'), torch.Tensor) and dists['confidence'].numel():
            logger.basic(
                "val/confidence_hist",
                dists['confidence'],
                step=getattr(trainer.state, 'log_step', 0)
            )
        if isinstance(dists.get('iou'), torch.Tensor) and dists['iou'].numel():
            logger.basic("val/iou_hist", dists['iou'], step=getattr(trainer.state, 'log_step', 0))

        def _build_pr_fig():
            import matplotlib.pyplot as plt
            tp_np = stats.get('tp')
            conf_np = stats.get('conf')
            tgt_np = stats.get('target_cls')
            if tp_np is None or conf_np is None or tgt_np is None or tp_np.size == 0:
                return None
            tp0 = tp_np[:, 0].astype(np.float64)
            order = np.argsort(-conf_np)
            tpc = tp0[order].cumsum()
            fpc = (1.0 - tp0[order]).cumsum()
            recall = tpc / max(len(tgt_np), 1)
            precision = tpc / np.maximum(tpc + fpc, 1)
            fig = plt.figure()
            plt.plot(recall, precision)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR (IoU=0.5)")
            return fig

        def _build_f1_fig():
            import matplotlib.pyplot as plt
            tp_np = stats.get('tp')
            conf_np = stats.get('conf')
            tgt_np = stats.get('target_cls')
            if tp_np is None or conf_np is None or tgt_np is None or tp_np.size == 0:
                return None
            tp0 = tp_np[:, 0].astype(np.float64)
            order = np.argsort(-conf_np)
            tpc = tp0[order].cumsum()
            fpc = (1.0 - tp0[order]).cumsum()
            recall = tpc / max(len(tgt_np), 1)
            precision = tpc / np.maximum(tpc + fpc, 1)
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            fig = plt.figure()
            plt.plot(np.linspace(0, 1, len(f1)), f1)
            plt.xlabel("Recall proxy (index)")
            plt.ylabel("F1")
            plt.title("F1 (IoU=0.5)")
            return fig

        try:
            if hasattr(logger, 'log_figure_async'):
                logger.log_figure_async(
                    "val/PR_curve@0.5", _build_pr_fig, step=getattr(trainer.state, 'log_step', 0)
                )
                logger.log_figure_async(
                    "val/F1_curve@0.5", _build_f1_fig, step=getattr(trainer.state, 'log_step', 0)
                )
        except Exception as e:
            logger.debug("val/curve_build_error", str(e))

        try:
            from utils.metrics import plot_confusion_matrix_pretty, cm_stats
            cm = payload.get('cm')
            if isinstance(cm, torch.Tensor):
                if getattr(logger, 'tb_heavy_enabled', False):

                    def _build_cm_fig():
                        return plot_confusion_matrix_pretty(
                            cm,
                            names=list(names),
                            normalize="true",
                            drop_background=False,
                            topk=None,
                            title="Confusion Matrix (IoU=0.5)"
                        )

                    if hasattr(logger, 'log_figure_async'):
                        logger.log_figure_async(
                            "val/confusion_matrix",
                            _build_cm_fig,
                            step=getattr(trainer.state, 'log_step', 0)
                        )
                try:
                    p, r, _, _ = cm_stats(cm)
                    lines_pr = []
                    for ci in range(p.shape[0]):
                        name = str(names[ci]) if isinstance(names, (list, dict)) else str(ci)
                        lines_pr.append(
                            f"{ci:3d} {name:20s}  P: {float(p[ci]):.4f}  R: {float(r[ci]):.4f}"
                        )
                    if lines_pr:
                        logger.log_text(
                            "val/per_class_PR",
                            "\n".join(lines_pr),
                            step=getattr(trainer.state, 'log_step', 0)
                        )
                except Exception as e2:
                    logger.debug("val/confusion_matrix_pr_error", str(e2))
        except Exception as e:
            logger.debug("val/confusion_matrix_error", str(e))

        per_class = payload.get('per_class') or {}
        try:
            ap = per_class.get('ap')
            idx = per_class.get('idx')
            ap50 = per_class.get('ap50')
            if ap is not None and idx is not None and len(idx):
                order = np.argsort(-ap)
                classes = [int(idx[i]) for i in order]
                labels = [
                    str(names[c]) if isinstance(names, (list, dict)) else str(c) for c in classes
                ]
                ap_sorted = [float(ap[i]) for i in order]
                if hasattr(logger, 'log_figure_async'):

                    def _build_overall_ap_bar(labels_local=labels, values_local=ap_sorted):
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(
                            figsize=(max(6, min(16, 0.35 * len(labels_local))), 4.5)
                        )
                        ax.bar(range(len(labels_local)), values_local, color="#4c78a8")
                        ax.set_ylim(0.0, 1.0)
                        ax.set_ylabel("AP @0.5:0.95")
                        ax.set_title("Per-class AP")
                        ax.set_xticks(range(len(labels_local)))
                        ax.set_xticklabels(labels_local, rotation=90)
                        fig.tight_layout()
                        return fig

                    logger.log_figure_async(
                        "val/AP_per_class",
                        _build_overall_ap_bar,
                        step=getattr(trainer.state, 'log_step', 0)
                    )
                lines = []
                order = np.argsort(-ap)
                topk = order[:min(5, len(order))]
                lastk = order[-min(5, len(order)):]
                if len(topk):
                    lines.append("Top classes by AP:")
                    for i in topk:
                        c = int(idx[i])
                        name = str(names[c]) if isinstance(names, (list, dict)) else str(c)
                        lines.append(
                            f"{c:3d} {name:20s}  AP50: {float(ap50[i]):.4f}  AP: {float(ap[i]):.4f}"
                        )
                if len(lastk):
                    lines.append("")
                    lines.append("Worst classes by AP:")
                    for i in lastk:
                        c = int(idx[i])
                        name = str(names[c]) if isinstance(names, (list, dict)) else str(c)
                        lines.append(
                            f"{c:3d} {name:20s}  AP50: {float(ap50[i]):.4f}  AP: {float(ap[i]):.4f}"
                        )
                if lines:
                    logger.log_text(
                        "val/per_class_AP",
                        "\n".join(lines),
                        step=getattr(trainer.state, 'log_step', 0)
                    )
        except Exception as e:
            logger.debug("val/per_class_ap_bar_error", str(e))

        bins = payload.get('per_class_bins') or {}
        for name, data in (bins.items() if isinstance(bins, dict) else []):
            ap = data.get('ap')
            ap50 = data.get('ap50')
            idx = data.get('idx')
            if ap is None:
                continue
            try:
                ap_all = float(np.mean(ap)) if np.size(ap) else 0.0
                ap50_scalar = float(np.mean(ap50)) if np.size(ap50) else 0.0
                logger.basic(
                    f"val/AP_{name}_50-95", ap_all, step=getattr(trainer.state, 'log_step', 0)
                )
                logger.basic(
                    f"val/AP_{name}_50", ap50_scalar, step=getattr(trainer.state, 'log_step', 0)
                )
            except Exception:
                pass
            try:
                if ap is not None and idx is not None and len(idx):
                    ap_cls = ap if ap.ndim == 1 else ap.mean(axis=1)
                    order = np.argsort(-ap_cls)
                    classes = [int(idx[i]) for i in order]
                    labels = [
                        str(names[c]) if isinstance(names, (list, dict)) else str(c)
                        for c in classes
                    ]
                    ap_sorted = [float(ap_cls[i]) for i in order]
                    tag = f"val/AP_{name}_per_class"
                    if hasattr(logger, 'log_figure_async'):

                        def _build_bin_ap_bar(
                            labels_local=labels,
                            values_local=ap_sorted,
                            title=f"Per-class AP ({name})"
                        ):
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(
                                figsize=(max(6, min(16, 0.35 * len(labels_local))), 4.0)
                            )
                            ax.bar(range(len(labels_local)), values_local, color="#72b7b2")
                            ax.set_ylim(0.0, 1.0)
                            ax.set_ylabel("AP @0.5:0.95")
                            ax.set_title(title)
                            ax.set_xticks(range(len(labels_local)))
                            ax.set_xticklabels(labels_local, rotation=90)
                            fig.tight_layout()
                            return fig

                        logger.log_figure_async(
                            tag, _build_bin_ap_bar, step=getattr(trainer.state, 'log_step', 0)
                        )
            except Exception:
                pass

class LRSchedulerCallback(Callback):
    """Handles learning rate scheduling."""
    def on_batch_end(self, trainer, batch_idx, loss, loss_items: Optional[Dict[str, float]] = None):
        """Step scheduler per-iteration if configured for per-iteration stepping."""
        if trainer.scheduler is not None and hasattr(
            trainer, 'per_iteration_scheduler'
        ) and trainer.per_iteration_scheduler:
            try:
                trainer.scheduler.step()
                if getattr(trainer.state, 'opt_step', 0) % 100 == 0:
                    trainer.logger.log_lr(trainer.optimizer)
            except Exception as e:
                trainer.logger.debug("scheduler/error", str(e))

    def on_epoch_end(self, trainer, epoch, metrics):
        """Step scheduler per-epoch if not using per-iteration stepping."""
        if trainer.scheduler is not None and (
            not hasattr(trainer, 'per_iteration_scheduler') or not trainer.per_iteration_scheduler
        ):
            try:
                trainer.scheduler.step()
                trainer.logger.log_lr(trainer.optimizer)
            except Exception as e:
                trainer.logger.debug("scheduler/error", str(e))

class CheckpointCallback(Callback):
    """Handles model checkpointing and config persistence."""
    def __init__(self, ckpt_dir: Path = Path("runs/checkpoints")):
        self.ckpt_dir = ckpt_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _write_active_config_yaml(self, trainer):
        """Persist the active run configuration as YAML alongside checkpoints."""
        try:
            cfg_yaml_path = self.ckpt_dir / "config.yaml"
            cfg_dict = dict(getattr(trainer, 'cfg', {}) or {})
            try:
                from core.config import get_config
                model_hyp = getattr(trainer.model, 'hyp', None)
                active = get_config(cfg=cfg_dict, hyp=model_hyp)
                if isinstance(active, object) and hasattr(active, 'hyp'):
                    cfg_dict['hyp'] = dict(active.hyp)
            except Exception:
                pass
            with open(cfg_yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg_dict, f, sort_keys=False)
        except Exception as e:
            trainer.logger.debug("checkpoint/config_yaml_error", str(e))

    def on_train_start(self, trainer):
        self._write_active_config_yaml(trainer)

    def on_epoch_end(self, trainer, epoch, metrics):
        trainer.save_checkpoint(
            self.ckpt_dir / "last.pt",
            epoch=epoch,
            train_loss=metrics.get("train_loss", 0),
            best_fitness=trainer.best_fitness,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler if trainer.amp else None,
            ema=trainer.ema,
        )
        self._write_active_config_yaml(trainer)

        fitness = metrics.get("fitness", float("-inf"))
        if fitness > trainer.best_fitness:
            trainer.best_fitness = fitness
            trainer.save_checkpoint(
                self.ckpt_dir / "best.pt",
                epoch=epoch,
                fitness=trainer.best_fitness,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler if trainer.amp else None,
                ema=trainer.ema,
            )
            trainer.logger.info("checkpoint/best", {"epoch": epoch, "fitness": fitness})

class EarlyStoppingCallback(Callback):
    """Early stopping based on fitness metric (mAP50-95)."""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_fitness = float('-inf')
        self.counter = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch, metrics):
        fitness = metrics.get('fitness', 0.0)

        if fitness > self.best_fitness + self.min_delta:
            self.best_fitness = fitness
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            trainer.logger.info(
                "early_stopping/triggered",
                {"epoch": epoch, "patience": self.patience, "best_fitness": self.best_fitness}
            )
            self.stopped_epoch = epoch
            trainer.stop_training = True

    def on_train_end(self, trainer):
        if self.stopped_epoch > 0:
            trainer.logger.info("early_stopping/stopped_at", self.stopped_epoch)

class GradientLoggerCallback(Callback):
    """Logs gradient norms and per-parameter histograms."""
    def __init__(self, log_interval: int = 200):
        self.log_interval = log_interval

    def on_batch_end(self, trainer, batch_idx, loss, loss_items: Optional[Dict[str, float]] = None):
        do_step = (
            getattr(trainer.state, 'log_step', 0) > 0 and
            getattr(trainer.state, 'log_step', 0) % self.log_interval == 0
        )
        if not do_step:
            return
        if hasattr(trainer, 'last_grad_norm') and trainer.last_grad_norm is not None:
            trainer.logger.basic(
                "train/grad_norm",
                float(trainer.last_grad_norm),
                step=getattr(trainer.state, 'log_step', 0)
            )

        try:
            for name, p in trainer.model.named_parameters():
                if p.grad is not None and p.grad.numel() > 0:
                    trainer.logger.heavy(
                        f"grad/{name}",
                        p.grad.detach().float().cpu(),
                        step=getattr(trainer.state, 'log_step', 0)
                    )
                if p.data is not None and p.data.numel() > 0:
                    trainer.logger.heavy(
                        f"weight/{name}",
                        p.data.detach().float().cpu(),
                        step=getattr(trainer.state, 'log_step', 0)
                    )
        except Exception as e:
            trainer.logger.debug("grad_logger/error", str(e))

class MetricsLoggerCallback(Callback):
    """Advanced metrics logging with visualization support."""
    def __init__(self, log_images: bool = True, log_interval: int = 100):
        self.log_images = log_images
        self.log_interval = log_interval
        self.train_loss_ema = None
        self.ema_alpha = 0.1

    def on_batch_end(self, trainer, batch_idx, loss, loss_items: Optional[Dict[str, float]] = None):
        if self.train_loss_ema is None:
            self.train_loss_ema = loss.item()
        else:
            self.train_loss_ema = (1 - self.ema_alpha
                                  ) * self.train_loss_ema + self.ema_alpha * loss.item()

        if batch_idx % self.log_interval == 0:
            trainer.logger.info(
                "train/loss_ema", self.train_loss_ema, step=getattr(trainer.state, 'log_step', 0)
            )

    def on_epoch_end(self, trainer, epoch, metrics):
        trainer.logger.log_losses({
            "train": metrics.get('train_loss', 0.0), "train_ema": self.train_loss_ema or 0.0
        },
                                  step=getattr(trainer.state, 'log_step', 0))

        if self.log_images and (
            hasattr(trainer, 'val_images_orig') or hasattr(trainer, 'val_images')
        ):
            try:
                if hasattr(trainer, 'val_images_orig') and trainer.val_images_orig:
                    images = trainer.val_images_orig
                    boxes = trainer.val_boxes_orig if hasattr(trainer, 'val_boxes_orig'
                                                             ) else [None] * len(images)
                    tag = "val/predictions_orig"
                    for i, (img, box) in enumerate(zip(images, boxes)):
                        trainer.logger.log_images_with_boxes(
                            f"{tag}_{i}",
                            img.unsqueeze(0),
                            box.unsqueeze(0),
                            step=getattr(trainer.state, 'log_step', 0)
                        )

                elif hasattr(trainer, 'val_images'):
                    images = trainer.val_images[:4]
                    boxes = trainer.val_boxes[:4] if hasattr(trainer, 'val_boxes') else None
                    tag = "val/predictions"
                    trainer.logger.log_images_with_boxes(
                        tag, images, boxes, step=getattr(trainer.state, 'log_step', 0)
                    )
            except Exception as e:
                trainer.logger.debug("metrics_logger/image_log_failed", str(e))

class CallbackManager:
    """Manages multiple callbacks during training."""
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: Callback):
        """Add a callback to the manager."""
        self.callbacks.append(callback)

    def fire(self, event: str, *args, **kwargs):
        """Fire an event to all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is not None:
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    print(f"Error in callback {callback.__class__.__name__}.{event}: {e}")

    def on_train_start(self, trainer):
        self.fire("on_train_start", trainer)

    def on_train_end(self, trainer):
        self.fire("on_train_end", trainer)

    def on_epoch_start(self, trainer, epoch):
        self.fire("on_epoch_start", trainer, epoch)

    def on_epoch_end(self, trainer, epoch, metrics):
        self.fire("on_epoch_end", trainer, epoch, metrics)

    def on_batch_start(self, trainer, batch_idx):
        self.fire("on_batch_start", trainer, batch_idx)

    def on_batch_end(self, trainer, batch_idx, loss, loss_items: Optional[Dict[str, float]] = None):
        self.fire("on_batch_end", trainer, batch_idx, loss, loss_items)

    def on_val_end(self, trainer, metrics):
        self.fire("on_val_end", trainer, metrics)
