# YOLOv9001
This was based on yolov9 before the rewrite. Now it's completely distinct training platform for detection models. Similar concepts, very different architectures. The current core model is based on hyper model:

https://github.com/iMoonLab/Hyper-YOLO

- [YOLOv9001](#yolov9001)
  - [CLI Usage](#cli-usage)
  - [Examples](#examples)
  - [Runtime Attachment (core/runtime.py)](#runtime-attachment-coreruntimepy)
  - [Central Config (core/config.py)](#central-config-coreconfigpy)
  - [DINOv3 Integration](#dinov3-integration)
  - [Example Hyp: models/hyps/low.yaml](#example-hyp-modelshypslowyaml)
  - [Augmentation Validator](#augmentation-validator)
  - [Data Loading](#data-loading)
  - [Datasets](#datasets)
  - [Notes \& Tips](#notes--tips)


## CLI Usage
- Basic: `python 9001.py <mode> [options]`
- Modes: `train`, `val`, `detect`, `benchmark`, `export`
- Common options:
  - `--model`: currently `hyper` (default).
  - `--weights`: path to weights (`.pt`) to load before running.
  - `--data`: dataset root for `train`/`val`/`benchmark`, image path for `detect`.
  - `--hyp`: path to a hyp YAML (default `models/hyps/high.yaml`).
  - `--hypo`: inline overrides `k:v,k:v` or a YAML path (applied after `--hyp`).
  - `--epochs`, `--batch-size`, `--img-size`, `--device`, `--workers`.
  - `--cache`: dataset cache (`ram` or `disk`). `--cache-compression` for disk.
  - `--compile`: enable `torch.compile` with tuned inductor options.
  - `--resume`: checkpoint file or run dir (auto-picks `checkpoints/last.pt`/`best.pt`).
  - `--log`: CLI/TensorBoard logging controls (e.g., `INFO`, `DEBUG`, `WARNING,HEAVY`).
  - `--tb-only`: TensorBoard-only logging; minimal CLI/file logs.
  - `--log-dir`: base run dir (auto-incremented unless resuming).

## Examples
- Train (fresh): `python 9001.py train --data /path/to/dataset --hyp models/hyps/low.yaml --epochs 100 --batch-size 16`
- Train (resume): `python 9001.py train --resume runs/exp42`
- Validate: `python 9001.py val --data /path/to/dataset --weights runs/exp42/checkpoints/best.pt`
- Detect single image: `python 9001.py detect --data path/to/image.jpg --weights runs/exp42/checkpoints/best.pt --img-size 640`
- Benchmark: `python 9001.py benchmark --data /path/to/dataset --batch-size 16 --img-size 640`
- Inline overrides: `python 9001.py train --data ... --hyp models/hyps/low.yaml --hypo "dino.enabled:true,dino.quant:fp16,assign_mode:simota"`


## Runtime Attachment (core/runtime.py)
- `attach_runtime(model, imgsz)` ensures a single source of truth for runtime fields:
  - Finds the `Detect` module, infers level strides via a no-grad forward at `imgsz` and caches `detect.last_shapes` → `detect.strides`.
  - Sets `model.nc`, `model.reg_max`, `model.strides`, `model.detect_layer`.
  - Creates or updates a `DFLDecoder` with `reg_max`, `strides`, and `dfl_tau` from config.
  - Attaches a `Postprocessor` with thresholds from config (NMS, class-agnostic, pre/post top-k, etc.).

## Central Config (core/config.py)
- `YOLOConfig` holds normalized defaults across model, dataloaders, training, loss, and postprocess.
- Merge priority in `get_config(...)`:
  - Hyp dict (highest) → hyp file path → `cfg['hyp']` if present → in-repo defaults.
- Key groups:
  - Model core: `reg_max`, `nc`, `strides`, detect bias init, `decode_centered`.
  - Data: `letterbox_center`, `pad_value`, `shuffle`, `persistent_workers`, `prefetch_factor`, `cuda_prefetch`, `gpu_collate`.
  - Optimization: `use_ema`, `ema_decay`, `grad_clip`, `optimizer` (e.g., Lion, SGD), LR decoupling.
  - Augment: mosaic/mixup/copy_paste, geometric and color jitter, smart crop controls.
  - Loss/IoU: `box`, `cls`, `dfl`, `iou_type`, optional L1 with gating and ramp.
  - Postprocess: `conf_thresh`, `iou_thresh`, pre/post top-k, class-agnostic NMS, NMS-free flags.
  - DINO distillation: see below.

## DINOv3 Integration
- Teacher wrapper: `utils/dino_teacher.DINOTeacher`
  - Loads HF DINOv3 ViT models (e.g., `facebook/dinov3-vitb16-pretrain-lvd1689m`).
  - Optional weight-only quantization (torchao) with `quant: int4|int8|fp16|fp32|bf16`; defaults to bf16/fp16 where appropriate.
  - Returns patch tokens `[B,Ht,Wt,Ct]`, a global CLS embedding `[B,Ct]`, and a saliency map `[B,Ht,Wt]` from CLS-attention (fallback: token energy).
  - Simple preprocessing (resize, ImageNet mean/std), autocast on CUDA, and lazy init/unload for memory safety.
- Trainer wiring: `core/trainer.Trainer`
  - Reads `cfg['dino']` from `YOLOConfig` and, if `enabled`, instantiates the teacher at the chosen `resolution`, `quant`, and `dtype`.
  - Logs per-step/epoch DINO loss terms and share; supports prewarm and HF token via `dino.hf_token` or `HF_TOKEN` env.
  - Optional backbone LR gating (`dino.bb_lr_gate`) responsive to distill loss share.
- Loss integration: `utils/loss.DetectionLoss`
  - Uses `dino.objfor02` to bias assignment cost with teacher objectness prior (no extra loss).
  - `reg_weight_with_saliency`: weights regression/DFL terms by teacher saliency with `reg_weight_floor` and `reg_weight_power`.
  - Optional `centroid_align`: aligns predicted box centers with saliency centers in early epochs.
  - Prototype-based class soft targets (`dino.cls_proto`) and region contrastive (`dino.contrast`) are available in config defaults.

## Example Hyp: models/hyps/low.yaml
```
optimizer:
  SGD:
    lr0: 0.01        # base LR
    lrf: 0.02        # final LR fraction for cosine decay
    momentum: 0.937  # SGD momentum
    warmup_momentum: 0.8   # start momentum during warmup
    warmup_epochs: 3.0     # linear warmup duration (epochs)
    warmup_bias_lr: 0.1   # bias-group LR at start of warmup
    nesterov: true
    weight_decay: 0.0005
    decoupled_lr:
      backbone:
        lr_scale: 1.0
      head:
        lr_scale: 1.0

min_stride_box_alpha: 0.0

hsv_h: 0.015
hsv_s: 0.4
hsv_v: 0.1
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 0.9

assign_mode: simota
assign_radius: 1.4
assign_lambda_iou: 3.5
assign_topq: 10

dino:
  model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
  enabled: true
  objfor02: true
  resolution: 512
  reg_weight_with_saliency: true
  reg_weight_floor: 0.2
  reg_weight_power: 2.0
  centroid_align: true
  centroid_weight: 0.1
  centroid_max_epochs: 6
  max_epochs: 12
  every_n: 1
  quant: fp16
```

## Augmentation Validator
This repo doesn't use albumtations, or ultralytics. They're all "from scratch". So while the transforms are similar, they're structurally different. My mosaic is different. Copy paste is different etc. To test the validity of the transforms, you can use this tool.

- Script: `tools/validate_augmentations.py` — visualizes and sanity-checks the augmentation pipeline on your dataset.
- Outputs annotated images under `aug_outcomes/` and optional `metrics.csv` for the full pipeline.
- Default mode runs two suites and saves a few samples per case:
  - CPU ops: `mosaic`, `mixup`, `copy_paste` (uses dataloader’s training transforms on CPU).
  - GPU ops: affine-style transforms (`degrees`, `translate`, `scale`, `shear`, `perspective`, `fliplr`, `flipud`, HSV jitters). Uses CUDA prefetcher path when available and overlays timing.
- How it works:
  - Loads a base hyp (from `--hyp` if provided, otherwise defaults), zeroes most augmentation magnitudes, ensures `augment: true`, then applies one augmentation knob per test case.
  - Draws the YOLO-normalized boxes back on the letterboxed canvas and overlays key hyp/timing diagnostics.
  - Requires the dataset root to be detectable by `utils.dataloaders.create_dataloader` (YOLO or COCO format).
- Quick start:
  - CPU/GPU suites: `python tools/validate_augmentations.py --data /path/to/dataset --hyp models/hyps/low.yaml --cpu-batch 1 --gpu-batch 4`
  - Full pipeline: `python tools/validate_augmentations.py --data /path/to/dataset --full-pipeline --batch-size 32 --num-workers 8 --prefetch-factor 2 --gpu-collate --cuda-prefetch --img-size 640 --cache ram --ram-compress`
- Output structure:
  - `aug_outcomes/cpu_ops/`: images like `mosaic_only_1.jpg`, `mixup_only_*.jpg`, etc.
  - `aug_outcomes/gpu_ops/`: images like `rotate_symmetric_deg_*.jpg`, `combo_*`, with overlay of timings when available.
  - `aug_outcomes/full_pipeline/`: `pipeline_*.jpg` grids and `metrics.csv` with columns such as `cpu_fetch_ms`, `gpu_total_ms`, `gpu_aug_ms`, `h2d_bytes`, `buffer_depth`, `cuda_mem_*`, cache hit counts, and per-item letterbox/CPU-aug times.
- Useful flags:
  - `--hyp`: optional hyp YAML to seed settings (keeps only relevant keys for viz).
  - `--cpu-batch`, `--gpu-batch`: batch sizes for the two suites (default 1/4).
  - `--full-pipeline`: enable end-to-end dataloader path with `--gpu-collate`, `--cuda-prefetch`, `--prefetch-*`, `--cache`, `--ram-compress`.
  - `--samples`, `--per-grid`: number of grids and tiles per grid saved in full pipeline mode.

## Data Loading
- `utils.dataloaders.create_dataloader` auto-detects YOLO (labels/*.txt + images) or COCO JSON, reads `data.yaml` names if present, and supports RAM/disk caches.
- Prefetch: `cuda_prefetch`, `prefetch_factor`, `persistent_workers`, optional GPU-side collate/affine.

## Datasets
- Supported formats: YOLO and COCO.
- YOLO layout example:
  - `dataset/`
    - `images/` → `train/*.jpg`, `val/*.jpg`, `test/*.jpg`
    - `labels/` → `train/*.txt`, `val/*.txt`, `test/*.txt` (YOLO normalized `cls x y w h` per line)
    - Optional `data.yaml` for names/classes (recommended)
- COCO layout example:
  - `dataset/`
    - `images/train/*.jpg`, `images/val/*.jpg`, `images/test/*.jpg`
    - `annotations/instances_train.json`, `annotations/instances_val.json`, `annotations/instances_test.json`
- Minimal `data.yaml` (YOLO):
  - `names: ["person", "car", "dog"]`
  - If missing, names are inferred from COCO JSON or defaulted to `class_0..class_{nc-1}`.
- Point `--data` to the dataset root (`dataset/`). The loader auto-detects format and splits.
- Caching and speed:
  - `--cache ram|disk|hybrid` and `--cache-compression` control on-disk cache.
  - GPU collate/affine and CUDA prefetch are available (see Augmentation Validator section for profiling).


## Notes & Tips
- When using DINO, some Hugging Face model repos may require authentication; set `HF_TOKEN` or pass `dino.hf_token` in your hyp.
- `attach_runtime` should be called after loading weights and before training/validation/inference (handled by `9001.py`).
- For custom models, implement `DetectModelBase.parse_model`, expose a `Detect` head, and rely on `attach_runtime` to finalize runtime attributes.

