"""
core/config.py

Centralized configuration for hyperparameters to ensure consistency.
"""
from typing import Dict, Any, Optional
from pathlib import Path
from utils.helpers import load_yaml
import copy

def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively update mapping dst with src (in-place) and return dst."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

class YOLOConfig:
    """Central configuration management for YOLO training."""

    DEFAULTS = {
        # === Model Core ===
        "reg_max": 16,  # DFL regression max value
        "nc": 80,  # Number of classes
        "strides": [8, 16, 32],  # FPN strides [P3, P4, P5]
        "decode_centered": True,  # Use centered anchors (align train/infer)
        "detect_init": {  # Detect head init biases
            "cls_prior": 1e-2,  # Initial class prior
            "reg_bias_first_bin": 2.0,  # Bias for first DFL bin
            "reg_bias_other_bins": -2.0  # Bias for other DFL bins
        },

        # === Data Loading & Preprocessing ===
        "letterbox_center": True,  # Center letterbox padding (align train/infer)
        "pad_value": 114,  # Padding fill value for images
        "shuffle": True,  # Shuffle dataset each epoch
        "persistent_workers": True,  # Keep workers alive across epochs
        "prefetch_factor": 3,  # Samples prefetched per worker
        "cuda_prefetch": True,  # Enable CUDA prefetcher
        "prefetch_max_batches": 3,  # Max ready batches on GPU buffer
        "prefetch_mem_fraction": 0.90,  # Upper bound fraction of GPU memory to use
        "gpu_collate": False,  # Use GPU-side collate to assemble batches
        "stats_enabled": False,  # Enable runtime stats/timing (benchmarking only)

        # === Optimization & Training ===
        "use_ema": True,  # Enable EMA tracking of weights
        "ema_decay": 0.9999,  # EMA decay rate
        "grad_clip": 10.0,  # Gradient clipping max-norm
        "early_stop": False,  # Enable early stopping callback
        "early_stop_patience": 10,  # Patience (epochs) for early stopping
        "optimizer": {  # Optimizer defaults
            "LION": {
                "lr0": 0.003,  # Initial learning rate
                "lrf": 0.01,  # Final LR factor
                "b1": 0.9,  # Beta1
                "b2": 0.999,  # Beta2
                "alpha": 30,  # Alpha
                "bias_correction": False,  # Bias correction toggle
                "weight_decay": 0.004,  # Weight decay
                "decoupled_lr": {"backbone": {"lr_scale": 1.0}, "head": {"lr_scale": 1.0}}  # LR scales
            }
        },

        # === Augmentations ===
        "augment": True,  # Enable train-time augmentations (all magnitudes default to 0.0)
        "mosaic": 0.0,  # Mosaic probability
        "mixup": 0.0,  # Mixup probability
        "copy_paste": 0.0,  # Copy-paste probability
        # Smart crop controls (used by mosaic and other crops)
        "smart_crop_topk": 8,        # sample among top-k candidate windows
        "smart_crop_jitter": 0.15,   # fraction of slack to jitter chosen window
        "smart_crop_unbiased_p": 0.2,  # chance to take uniform random crop instead of smart
        "smart_crop_prob": 0.8,      # probability to apply smart crop (else uniform crop)
        "degrees": 0.0,  # Rotation magnitude (deg)
        "translate": 0.0,  # Translation magnitude (fraction)
        "scale": 0.0,  # Scale gain (fraction)
        "shear": 0.0,  # Shear magnitude (deg)
        "perspective": 0.0,  # Perspective magnitude (fraction)
        "flipud": 0.0,  # Vertical flip probability
        "fliplr": 0.0,  # Horizontal flip probability
        "hsv_h": 0.0,  # HSV hue jitter amplitude
        "hsv_s": 0.0,  # HSV saturation jitter amplitude
        "hsv_v": 0.0,  # HSV value jitter amplitude

        # === Evaluation & Logging ===
        "metrics_on_original": False,  # Compute metrics in original image space
        "log_val_images_original": True,  # Log de-letterboxed originals
        "clip_pred_to_canvas": True,  # Clamp predicted boxes to canvas
        "grad_log_interval": 200,  # Interval for GradientLoggerCallback

        # === Loss & IoU ===
        "box": 7.5,  # Box loss weight
        "cls": 0.5,  # Class loss weight
        "cls_pw": 1.0,  # Class positive weight
        "dfl": 1.5,  # DFL loss weight
        "iou_type": "CIoU",  # IoU type: IoU/GIoU/DIoU/CIoU/SIoU/MPDIoU
        "l1_weight": 0.0,  # Optional L1 loss weight
        "l1_beta": 3.0,   # Huber (Smooth L1) beta in pixels for optional L1 term
        "l1_ramp_step": 0,  # Steps to ramp L1 from 0 -> l1_weight (0 disables ramp)
        "l1_iou_gate": 0.5,  # Apply L1/Huber only when matched IoU > gate (0 disables gating)
        "dfl_eps": 1e-3,  # DFL boundary epsilon
        "dfl_label_smooth": 0.0,  # DFL label smoothing
        "dfl_tau": 1.0,  # DFL temperature for softmax
        "dfl_strict_targets": False,  # Assert on excessive out-of-range targets
        "dfl_clip_tolerance": 0.01,  # Allowed oob fraction before asserting
        "vfl_alpha": 0.75,  # VFL negative alpha
        "vfl_gamma": 2.0,  # VFL gamma
        "cls_type": "bce",  # Classification loss: vfl/qfl/bce
        "qfl_beta": 2.0,  # QFL beta for positives
        "assign_topk": 10,  # TaskAligned top-k per GT
        "assign_radius": 2.5,  # Center prior radius (in strides)
        "assign_alpha": 0.5,  # TaskAligned exponent on class prob
        "assign_beta": 6.0,  # TaskAligned exponent on IoU
        "assign_mode": "ult",  # Assigner mode: ult/simota/mixed
        "assign_lambda_iou": 3.0,  # Cost weight for -log(IoU)
        "assign_topq": 10,  # Dynamic-k from sum of top-q IoUs
        # Tiny GT filtering (train-time)
        "filter_tiny_gt": True,  # If True, ignore GTs smaller than alpha*stride per level
        "min_stride_box_alpha": 0.0,  # Alpha for per-stride min size 
        # MLA-specific weights
        "obj_weight": 1.0,  # Objectness/confidence loss weight (MLA mixed mode)
        # Obj-less configuration (single knob)
        "objless_target_iou": 0.0,  # Linear ramp target IoU for obj-less proxy (0 disables scaling)
        "objless_weight_floor": 0.05,  # Minimum scaling at IoU=0 (0..1)
        # Architectural toggles
        "wfpb_enabled": False,  # Enable Window Feature Propagation Block(s)

        # === Postprocess/NMS ===
        "conf_thresh": 0.001,  # Confidence threshold for NMS
        "iou_thresh": 0.7,  # IoU threshold for NMS
        "pre_nms_topk": 30000,  # Keep top-k before NMS
        "post_nms_topk": 300,  # Keep top-k after NMS
        "class_agnostic_nms": False,  # Class-agnostic NMS toggle
        "nms_free": False,  # If True, skip NMS (for NMS-free inference experiments)
        "use_objectness_eval": False,  # If True, use Detect.objectness for eval scoring

        # === DINO Distillation (Training-only) ===
        "dino": {
            "enabled": False,  # Off by default; opt-in via hyp/cfg
            "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",  # HF model id
            "hf_token": None,  # Optional HF token; else uses env or local login
            "resolution": 384,  # Conservative teacher resolution (24x24 tokens)
            "every_n": 3,  # Compute teacher every N steps to limit overhead
            "quant": "int4",  # Options: int8 | int4 | fp16 | fp32 | bf16
            "log_interval": 10,  # Log less frequently by default
            "prewarm": True,  # Avoid eager download/alloc unless explicitly enabled
            # Modest loss weights; only early epochs (see max_epochs)
            "alpha": 0.5,  # Patch-token cosine
            "beta": 0.3,  # Saliency KL (CLS-attn/energy -> aux obj) — slightly lowered
            "gamma": 0.2,  # Global cosine (CLS embedding)
            "decay_pct": 0.2,  # Short cosine decay (irrelevant with hard stop)
            "warmup_epochs": 3,  # Keep constant weights for first 3 epochs
            "max_epochs": 3,  # Apply distillation only for first N epochs
            "sal_temp": 0.4,  # Slightly higher temp to keep distribution conservative
            "sal_from": "auto",  # Use CLS-attn if concentrated; fallback to energy when too flat
            "level_temp": 0.25,  # Softmax temperature for level weighting by grid mismatch
            "share_clamp": {  # Adaptive clamp to prevent distill overtake
                "enabled": True,
                "max": 0.2
            },
            # Gate for using DINO teacher objectness for assignment cost bias only (no distill losses)
            "objfor02": False,
            # Saliency-weighted regression (boxes/DFL) — gated to positives only
            "reg_weight_with_saliency": False,
            "reg_weight_floor": 0.2,   # minimum weight when saliency is 0
            "reg_weight_power": 1.0,   # exponent for saliency -> weight mapping
            "bb_lr_gate": {  # Backbone LR gating inverse to distill loss
                "enabled": False,
                "min_scale": 0.5,   # start backbone scale at 0.1 of configured scale
                "ema": 0.9,         # EMA factor for distillation loss smoothing
                "relax": 0.2,       # relax rate toward nominal LR when not gating this step
                "complement_head": False,  # optionally counter-scale head LRs
                "head_max_scale": 1.4,     # clamp for head LR multiplier if complement enabled
            },
            "cls_proto": {  # Prototype-based class soft targets (labels required)
                "enabled": False,
                "weight": 0.05,
                "temp": 0.3,
                "momentum": 0.9,
                "use_momentum": True
            },
            "contrast": {  # Region-to-region InfoNCE on fused student tokens
                "enabled": True,
                "weight": 0.1,
                "temp": 0.2,
                "pos_k": 1,
                "neg_k": 32,
                "samples_per_img": 32,
                "sal_topk": 0  # if >0, select top-k by saliency; else uniform over valid tokens
            },
        },
    }

    def __init__(self, hyp: Optional[Dict[str, Any]] = None, hyp_path: Optional[Path] = None):
        """
        Initialize configuration with optional hyperparameters.
        Uses in-repo DEFAULTS as the only implicit base; optional hyp file or dict
        can be applied by callers (e.g., 9001.py defaults to high.yaml, not here).

        Args:
            hyp: Dictionary of hyperparameters (highest priority)
            hyp_path: Path to YAML file with hyperparameters
        """
        # Deep copy ensures nested dicts (e.g., 'dino') are not shared/mutated across instances
        self.hyp = copy.deepcopy(self.DEFAULTS)

        if hyp_path is not None:
            file_hyp = load_yaml(Path(hyp_path))
            _deep_update(self.hyp, file_hyp)

        if hyp is not None:
            _deep_update(self.hyp, hyp)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a hyperparameter value."""
        return self.hyp.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """Update hyperparameters with deep-merge semantics for nested dicts."""
        _deep_update(self.hyp, updates)

    @property
    def loss_weights(self) -> Dict[str, float]:
        """Get loss weights."""
        return {
            "box": self.hyp.get("box", self.DEFAULTS["box"]),
            "cls": self.hyp.get("cls", self.DEFAULTS["cls"]),
            "dfl": self.hyp.get("dfl", self.DEFAULTS["dfl"]),
            "cls_pw": self.hyp.get("cls_pw", self.DEFAULTS["cls_pw"]),
        }

    @property
    def ema_config(self) -> Dict[str, Any]:
        """Get EMA configuration."""
        return {
            "enabled": self.hyp.get("use_ema", self.DEFAULTS["use_ema"]),
            "decay": self.hyp.get("ema_decay", self.DEFAULTS["ema_decay"]),
        }

    @property
    def postprocess_config(self) -> Dict[str, Any]:
        """Get postprocessing configuration."""
        return {
            "conf_thresh":
                self.hyp.get("conf_thresh", self.DEFAULTS["conf_thresh"]),
            "iou_thresh":
                self.hyp.get("iou_thresh", self.DEFAULTS["iou_thresh"]),
            "pre_nms_topk":
                self.hyp.get("pre_nms_topk", self.DEFAULTS["pre_nms_topk"]),
            "post_nms_topk":
                self.hyp.get("post_nms_topk", self.DEFAULTS["post_nms_topk"]),
            "class_agnostic_nms":
                self.hyp.get("class_agnostic_nms", self.DEFAULTS["class_agnostic_nms"]),
            "nms_free":
                self.hyp.get("nms_free", self.DEFAULTS["nms_free"]),
            "use_objectness_eval":
                self.hyp.get("use_objectness_eval", self.DEFAULTS["use_objectness_eval"]),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.hyp.copy()

def get_config(
    cfg: Optional[Dict[str, Any]] = None,
    hyp: Optional[Dict[str, Any]] = None,
    hyp_path: Optional[Path] = None
) -> YOLOConfig:
    """
    Create a YOLOConfig instance from various sources.
    
    Priority order:
    1. hyp dict (highest)
    2. hyp_path file
    3. cfg['hyp'] if exists
    4. DEFAULTS (base)
    """
    config = YOLOConfig()

    if cfg and 'hyp' in cfg:
        if isinstance(cfg['hyp'], dict):
            config.update(cfg['hyp'])
        elif isinstance(cfg['hyp'], (str, Path)):
            file_hyp = load_yaml(Path(cfg['hyp']))
            config.update(file_hyp)

    if hyp_path:
        file_hyp = load_yaml(Path(hyp_path))
        config.update(file_hyp)

    if hyp:
        config.update(hyp)

    return config
