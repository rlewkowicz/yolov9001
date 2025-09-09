#!/usr/bin/env python3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # reduce TF verbose oneDNN notices
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["GLOG_minloglevel"] = "2"  # absl/glog-compatible
os.environ.pop('XLA_FLAGS', None)
os.environ['LD_LIBRARY_PATH'] = ''

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch._inductor.runtime.triton_helpers",
)

import argparse
import torch
from pathlib import Path

from models.detect.hyper.model import HyperModel
from core.trainer import Trainer
from core.validator import Validator
from core.benchmark import Benchmark
from core.detector import Detector
from core.runtime import attach_runtime
from utils.logging import get_logger, set_log_level
from utils.dataloaders import create_dataloader

import cv2

def run_benchmark(model, args, cfg):
    """Runs model benchmarking."""
    logger = get_logger()

    benchmark = Benchmark(model=model, device=args.device, cfg=cfg)

    benchmark.benchmark_speed(img_size=args.img_size, batch_size=args.batch_size)
    benchmark.benchmark_memory(img_size=args.img_size, batch_size=args.batch_size)

    if not args.data:
        logger.warning("benchmark/data", "No dataset path provided. Skipping accuracy benchmark.")
    else:
        try:
            test_loader, info = create_dataloader(
                data_path=args.data,
                split="test",
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.workers,
                cache=args.cache,
                pin_memory=True,
                hyp=cfg.get('hyp', None),
                persistent_workers=cfg.get('persistent_workers', False),
                prefetch_factor=cfg.get('prefetch_factor', 2),
            )
            model.set_class_names(info['names'])
            logger.info(
                "benchmark/dataloader",
                f"Loaded test dataloader with {len(test_loader.dataset)} images."
            )
            benchmark.benchmark_accuracy(dataloader=test_loader)
        except (ValueError, FileNotFoundError) as e:
            logger.error("benchmark/dataloader_error", str(e))

    benchmark.print_results()

def run_detect(model, args, cfg):
    """Runs detection on an image."""
    logger = get_logger()

    if not args.data:
        logger.error("detect/data", "Image path must be provided for detect mode.")
        return

    detector = Detector(model=model, device=args.device, cfg=cfg)

    image_path = Path(args.data)
    if not image_path.is_file():
        logger.error("detect/data", f"Invalid image path: {image_path}")
        return

    class_names = getattr(model, "names", None)
    if not class_names:
        class_names = [f"class_{i}" for i in range(getattr(model, "nc", 80))]

    annotated_image = detector.detect_and_annotate(
        image=str(image_path), class_names=class_names, img_size=args.img_size
    )

    output_dir = Path(args.log_dir) / "detect"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_path.name

    cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    logger.info("detect/result", f"Annotated image saved to {output_path}")

def main():
    try:
        if not os.environ.get("PYTORCH_KERNEL_CACHE_PATH"):
            cache_dir = Path.cwd() / ".torch_kernels"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["PYTORCH_KERNEL_CACHE_PATH"] = str(cache_dir)
    except Exception:
        pass
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    default_workers = max(1, os.cpu_count() - 1)

    parser = argparse.ArgumentParser(description="YOLOv9001")
    parser.add_argument("mode", choices=["train", "val", "detect", "benchmark", "export"])
    parser.add_argument("--model", type=str, default="hyper")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)  # directory root or image path
    parser.add_argument(
        "--hyp",
        type=str,
        default="./models/hyps/high.yaml",
        help="Path to hyperparameters file (required)."
    )
    parser.add_argument(
        "--hypo",
        type=str,
        default=None,
        help="Inline overrides k:v,k:v or YAML path. Ignored on --resume."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--cache", type=str, default=None, help="Cache type: 'ram' or 'disk'")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--gpu-aug", action="store_true", help="Enable GPU-side augmentation during training"
    )
    parser.add_argument(
        "--cache-compression", action="store_true", help="Enable compression for disk cache"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=
        "Resume training. Accepts a checkpoint file (.pt) or a run directory containing checkpoints/last.pt"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help=(
            "Logging controls (comma-separated). CLI levels: DEBUG, INFO, WARNING, ERROR. "
            "TensorBoard categories: BASIC (default), HEAVY (implies BASIC). "
            "Defaults to INFO,BASIC. Example: --log WARNING,HEAVY"
        ),
    )
    parser.add_argument(
        "--tb-only", action="store_true", help="TensorBoard-only logging (minimal CLI/file)"
    )
    parser.add_argument("--log-dir", type=str, default="runs/exp")
    args = parser.parse_args()

    from utils.paths import increment_path

    resume_ckpt: Path | None = None
    if args.mode == "train" and args.resume:
        cand = Path(args.resume)
        if cand.is_dir():
            last = cand / "checkpoints" / "last.pt"
            best = cand / "checkpoints" / "best.pt"
            resume_ckpt = last if last.is_file() else (best if best.is_file() else None)
            if resume_ckpt is None:
                raise SystemExit(
                    f"--resume directory provided but no checkpoints found under {cand}/checkpoints"
                )
        else:
            if cand.is_file():
                resume_ckpt = cand
            else:
                raise SystemExit(f"--resume checkpoint not found: {cand}")

    ckpt_meta = None
    ckpt_cfg = None
    ckpt_hyp = None
    if resume_ckpt is not None:
        try:
            ckpt_meta = torch.load(resume_ckpt, map_location=args.device, weights_only=False)
            if isinstance(ckpt_meta, dict):
                ckpt_cfg = ckpt_meta.get("cfg", None)
                if isinstance(ckpt_cfg, dict):
                    ckpt_hyp = ckpt_cfg.get("hyp", None)
        except Exception as e:
            raise SystemExit(f"Failed to load checkpoint metadata for resume: {e}")

    if resume_ckpt is not None:
        if resume_ckpt.parent.name == "checkpoints":
            run_dir = resume_ckpt.parent.parent
        else:
            run_dir = resume_ckpt.parent
    else:
        run_dir = increment_path(Path(args.log_dir))

    logger = get_logger(run_dir, tb_only=bool(args.tb_only))
    set_log_level(args.log)

    if not args.hyp and ckpt_hyp is None:
        raise SystemExit("Provide --hyp or resume from a checkpoint that includes cfg.hyp")

    if args.model != "hyper":
        raise ValueError(f"unknown model: {args.model}")

    def _parse_overrides(text: str) -> dict:
        from utils.helpers import load_yaml
        p = Path(text)
        if p.exists() and p.is_file():
            return load_yaml(p)
        result = {}
        parts = [x.strip() for x in text.split(',') if x.strip()]
        for item in parts:
            if ':' not in item:
                continue
            k, v = item.split(':', 1)
            k, v = k.strip(), v.strip()
            if v.lower() in ("true", "false"):
                result[k] = v.lower() == "true"
                continue
            try:
                if any(c in v for c in ['.', 'e', 'E']):
                    result[k] = float(v)
                else:
                    result[k] = int(v)
            except ValueError:
                result[k] = v
        return result

    if resume_ckpt is not None and ckpt_hyp is not None:
        base = ckpt_hyp if isinstance(ckpt_hyp, dict) else {}
        if args.hypo:
            try:
                overrides = _parse_overrides(args.hypo)
                base.update(overrides)
                logger.info("hyp/resume_overrides", "Applied --hypo overrides on checkpoint hyp")
            except Exception as e:
                logger.warning("hyp/override_parse_error", f"{e}")
        else:
            logger.info("hyp/resume", "Using checkpoint hyperparameters; --hyp ignored")
        hyp_for_model = base
    else:
        from utils.helpers import load_yaml
        base = load_yaml(Path(args.hyp)) if args.hyp else {}
        if args.hypo:
            try:
                overrides = _parse_overrides(args.hypo)
                base.update(overrides)
            except Exception as e:
                logger.warning("hyp/override_parse_error", f"{e}")
        hyp_for_model = base if isinstance(base, dict) and base else args.hyp
    model = HyperModel(hyp=hyp_for_model)
    nc_model = int(getattr(model, "nc", 80))
    logger.info("model/nc", nc_model)

    if args.weights:
        ckpt = torch.load(args.weights, map_location=args.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("_orig_mod."):
                    nk = nk[len("_orig_mod."):]
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                cleaned[nk] = v
            state = cleaned
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            model.load_state_dict(state, strict=False)

    attach_runtime(model, imgsz=args.img_size)

    base_cfg = {}
    if isinstance(ckpt_cfg, dict):
        base_cfg.update(ckpt_cfg)
    cfg = {
        "data":
            args.data if args.data is not None else base_cfg.get("data", None),
        "epochs":
            int(args.epochs) if args.epochs is not None else int(base_cfg.get("epochs", 100)),
        "img_size":
            int(args.img_size) if args.img_size is not None else int(base_cfg.get("img_size", 640)),
        "batch_size":
            int(args.batch_size)
            if args.batch_size is not None else int(base_cfg.get("batch_size", 16)),
        "num_workers":
            int(args.workers) if args.workers is not None else int(base_cfg.get("num_workers", 4)),
        "pin_memory":
            base_cfg.get("pin_memory", True),
        "augment":
            base_cfg.get("augment", True),
        "allow_dummy":
            base_cfg.get("allow_dummy", args.data is None),
        "compile":
            bool(args.compile) if
            (args.compile is not None) else bool(base_cfg.get("compile", False)),
        "ckpt_dir":
            str(run_dir / "checkpoints"),
        "hyp":
            model.hyp,
        "cuda_prefetch":
            True,
        "cache":
            args.cache if args.cache is not None else base_cfg.get("cache", None),
        "cache_compression":
            args.cache_compression or bool(base_cfg.get("cache_compression", False)),
        "persistent_workers":
            bool(getattr(model, "config_obj", None).get("persistent_workers", True)),
        "prefetch_factor":
            int(getattr(model, "config_obj", None).get("prefetch_factor", 3)),
    }

    if args.mode == "train":
        trainer = Trainer(
            model=model,
            device=args.device,
            cfg=cfg,
            auto_fit=False,
        )
        logger.info("trainer/ready", {"model_nc": nc_model})

        if resume_ckpt is not None:
            trainer.resume_training(resume_ckpt)
            logger.info("train/resumed", f"Resumed from {resume_ckpt}")
            target_epochs = int(cfg.get("epochs", 100))
            remaining_epochs = target_epochs - int(trainer.epoch)
            if remaining_epochs > 0:
                trainer.fit(
                    epochs=remaining_epochs,
                    criterion=None,  # let Trainer build the criterion
                    train_loader=None,
                    val_loader=None,
                )
            else:
                logger.info("train/complete", "Training already completed")
        else:
            trainer.fit(
                epochs=cfg.get("epochs", 100),
                criterion=None,  # let Trainer build the criterion
                train_loader=None,
                val_loader=None,
            )

    elif args.mode == "val":
        logger.info("validator/ready", {"model_nc": nc_model})
        if not args.data:
            logger.error("val/data", "Dataset path must be provided for val mode.")
            return
        val_loader, info = create_dataloader(
            args.data,
            "val",
            args.img_size,
            args.batch_size,
            args.workers,
            augment=False,
            cache=args.cache,
            cache_compression=cfg.get("cache_compression", True),
            persistent_workers=cfg.get('persistent_workers', False),
            prefetch_factor=cfg.get('prefetch_factor', 2),
        )
        try:
            state = model.get_detection_state()
            ds = getattr(val_loader, 'dataset', None)
            if ds is not None:
                if hasattr(ds, 'letterbox_center'):
                    ds.letterbox_center = state['letterbox_center']
                if hasattr(ds, 'pad_value'):
                    ds.pad_value = state['pad_value']
        except Exception as e:
            logger.warning("val/letterbox_sync_failed", f"{e}")
        model.set_class_names(info['names'])
        validator = Validator(model=model, device=args.device, cfg=cfg)
        validator.validate(val_loader)
        validator.print_results()

    elif args.mode == "detect":
        logger.info("detector/ready", {"model_nc": nc_model})
        run_detect(model, args, cfg)

    elif args.mode == "benchmark":
        logger.info("benchmark/ready", {"model_nc": nc_model})
        run_benchmark(model, args, cfg)

    elif args.mode == "export":
        logger.info("export/ready", {"model_nc": nc_model})

    logger.close()

if __name__ == "__main__":
    main()
