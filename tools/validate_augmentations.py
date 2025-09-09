import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
import shutil
from core.config import get_config
from utils.dataloaders import create_dataloader as create_dataloader_unified
from utils.prefetcher import CUDAPrefetcher

def denorm_and_draw(imgs, targets, class_names, draw_index: int = 0):
    """
    Draw a single augmented image (batch=1) and overlay YOLO-normalized boxes on the letterboxed canvas.
    """
    if imgs.ndim == 4:
        assert 0 <= draw_index < imgs.shape[0], "draw_index out of range for batch"
        img_tensor = imgs[draw_index]
    else:
        img_tensor = imgs
    img = img_tensor.permute(1, 2, 0).cpu().numpy().copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    Hc, Wc = img.shape[:2]
    if targets.numel():
        t = targets[targets[:, 0] == float(draw_index)]  # only this image
        for _, cls_idx, x, y, w, h in t.tolist():
            cx, cy = x * Wc, y * Hc
            ww, hh = w * Wc, h * Hc
            x1, y1 = int(cx - ww / 2), int(cy - hh / 2)
            x2, y2 = int(cx + ww / 2), int(cy + hh / 2)
            cls_name = class_names[int(cls_idx)
                                  ] if int(cls_idx) < len(class_names) else str(int(cls_idx))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, cls_name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

def overlay_text(img, lines, y0=28, dy=22, color=(255, 255, 255)):
    """Overlay multi-line text on top-left."""
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img

def _make_grid(images, grid: int = 4):
    """Make a simple 2x2 (or NxN) grid from a list of HxWxC images; pads with gray if needed."""
    if not images:
        return None
    n = min(grid, len(images))
    side = int(np.ceil(n**0.5))
    H, W, C = images[0].shape
    canvas = np.full((side * H, side * W, C), 114, dtype=images[0].dtype)
    for i in range(n):
        r, c = divmod(i, side)
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[i]
    return canvas

def run_test_case(
    data_path, base_hyp, test_name, aug_hyp, out_dir: Path, num_samples=5, seed=0, batch_size=1
):
    """Runs a single augmentation test case with visible diagnostics; saves N individual images."""
    print(f"--- Running test: {test_name} -> {out_dir} ---")
    hyp = {**base_hyp, **aug_hyp}

    create_dataloader_fn = create_dataloader_unified
    print(f"  Using unified dataloader (live training pipeline)")

    try:
        dl, info = create_dataloader_fn(
            data_path=data_path,
            split="train",
            img_size=640,
            batch_size=int(batch_size),
            hyp=hyp,
            augment=True,
            num_workers=0
        )
        try:
            import random as _random
            _random.seed(seed)
        except Exception:
            pass
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            it = CUDAPrefetcher(dl, device=torch.device('cuda'), amp=False)
        else:
            it = iter(dl)
        class_names = info.get('names', [])
        for i in range(num_samples):
            print(f"  Generating sample {i+1}/{num_samples}...")
            try:
                batch = next(it)
            except StopIteration:
                if torch.cuda.is_available():
                    it = CUDAPrefetcher(dl, device=torch.device('cuda'), amp=False)
                else:
                    it = iter(dl)
                batch = next(it)

            imgs, targets, _, _, _ = batch

            if not use_cuda:
                from utils.augment import apply_gpu_affine_on_batch
                pad_value = int(hyp.get('pad_value', get_config().hyp.get('pad_value', 114)))
                imgs = imgs.to(dtype=torch.float32).div_(255.0)
                imgs, targets = apply_gpu_affine_on_batch(imgs, targets, hyp, pad_value)

            img_vis = denorm_and_draw(imgs, targets, class_names, draw_index=0)
            diag = []
            for k in ("mosaic", "mixup", "copy_paste", "fliplr", "flipud"):
                if k in hyp and hyp.get(k, 0.0) > 0.0:
                    diag.append(f"{k}={hyp[k]}")
            sc_keys = (
                "smart_crop_prob", "smart_crop_topk", "smart_crop_jitter", "smart_crop_unbiased_p"
            )
            for k in sc_keys:
                if k in hyp:
                    diag.append(f"{k}={hyp[k]}")
            for k in (
                "degrees", "translate", "scale", "shear", "perspective", "hsv_h", "hsv_s", "hsv_v"
            ):
                if k in hyp and hyp.get(k, 0.0) != 0.0:
                    diag.append(f"{k}={hyp[k]}")
            timing = getattr(it, 'last_timing', None) if use_cuda else None
            if timing is not None:
                cpu_ms = timing.get('cpu_fetch_ms', None)
                gpu_total_ms = timing.get('gpu_total_ms', None)
                gpu_aug_ms = timing.get('gpu_aug_ms', None)
                tlines = []
                if cpu_ms is not None:
                    tlines.append(f"cpu_fetch_ms={cpu_ms:.1f}")
                if gpu_total_ms is not None:
                    tlines.append(f"gpu_total_ms={gpu_total_ms:.1f}")
                if gpu_aug_ms is not None:
                    tlines.append(f"gpu_affine_ms={gpu_aug_ms:.1f}")
                if tlines:
                    diag.append(' | '.join(tlines))
            diag.insert(0, f"test={test_name}")
            diag.insert(1, f"bs={batch_size}")
            img_vis = overlay_text(img_vis, diag)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{test_name}_{i+1}.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))  # noqa
        print(f"Saved {num_samples} visualizations for {test_name} -> {out_dir}\n")
    except Exception as e:
        print(f"!!! Test '{test_name}' failed with an error: {e}\n")
        import traceback
        traceback.print_exc()

def main(args):
    output_root = Path("aug_outcomes")
    if output_root.exists():
        shutil.rmtree(output_root)

    cpu_dir = output_root / "cpu_ops"
    gpu_dir = output_root / "gpu_ops"
    cpu_dir.mkdir(parents=True)
    gpu_dir.mkdir(parents=True)
    print(f"CPU ops results: {cpu_dir.resolve()}")
    print(f"GPU ops results: {gpu_dir.resolve()}")

    base_hyp = get_config(hyp_path=args.hyp).hyp if getattr(args, 'hyp', None) else get_config().hyp
    keep_keys = {
        'box',
        'cls',
        'dfl',
        'iou_type',
        'l1_weight',
        'pad_value',
        'letterbox_center',
        'augment',
        'smart_crop_topk',
        'smart_crop_jitter',
        'smart_crop_unbiased_p',
        'smart_crop_prob',
    }
    for k in list(base_hyp.keys()):
        if k not in keep_keys:
            if isinstance(base_hyp[k], (int, float)) or k == 'translate':
                base_hyp[k] = 0.0
    base_hyp['augment'] = True  # Ensure augmentation pipeline is enabled

    cpu_cases = {
        "baseline_no_aug": {},
        "mosaic_only": {"mosaic": 1.0},
        "mixup_only": {"mixup": 1.0},
        "copy_paste_only": {"copy_paste": 1.0},
    }

    gpu_cases = {
        "flip_left_right": {"fliplr": 1.0},
        "flip_up_down": {"flipud": 1.0},
        "hsv_hue_aggressive": {"hsv_h": 0.5},
        "hsv_saturation_aggressive": {"hsv_s": 0.9},
        "hsv_value_aggressive": {"hsv_v": 0.9},
        "hsv_all_aggressive": {"hsv_h": 0.5, "hsv_s": 0.9, "hsv_v": 0.9},
        "rotate_symmetric_deg": {"degrees": [-30, 30]},
        "rotate_left_only_deg": {"degrees": "-20.0"},
        "rotate_right_only_deg": {"degrees": "+20.0"},
        "scale_symmetric": {"scale": [-.9, .9]},
        "scale_down_only": {"scale": "-0.9"},
        "scale_up_only": {"scale": "+0.9"},
        "translate_symmetric": {"translate": [-.1, .1]},
        "shear_symmetric": {"shear": [-45, 45]},
        "perspective_aggressive": {"perspective": 0.1},
        "combo_mosaic_affine": {
            "mosaic": 1.0, "degrees": 20.0, "scale": 0.2, "translate": 0.1, "shear": 10.0
        },
        "combo_rotate_shear": {"degrees": 30.0, "shear": 15.0},
        "combo_flip_translate_hsv": {
            "fliplr": 1.0, "translate": 0.3, "hsv_h": 0.1, "hsv_s": 0.7, "hsv_v": 0.4
        },
    }

    print(f"\n{'='*60}\nTesting CPU ops (mosaic/mixup/copy_paste)\n{'='*60}\n")
    for name, aug_hyp in cpu_cases.items():
        run_test_case(
            args.data,
            base_hyp,
            name,
            aug_hyp,
            out_dir=cpu_dir,
            num_samples=5,
            seed=0,
            batch_size=args.cpu_batch
        )

    print(
        f"\n{'='*60}\nTesting GPU ops (affine-style: degrees/translate/scale/shear/perspective)\n{'='*60}\n"
    )
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, GPU affine/perspective tests will run on CPU fallback.")
    for name, aug_hyp in gpu_cases.items():
        run_test_case(
            args.data,
            base_hyp,
            name,
            aug_hyp,
            out_dir=gpu_dir,
            num_samples=5,
            seed=0,
            batch_size=args.gpu_batch
        )

def run_full_pipeline(args):
    """End-to-end pipeline validator using the real dataloader + optional prefetcher and GPU collate.
    Saves a few small sample grids per batch to visually verify transforms are identical to training."""
    out_dir = Path("aug_outcomes/full_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = (get_config(hyp_path=args.hyp).hyp
           if getattr(args, 'hyp', None) else get_config().hyp).copy()
    cfg['augment'] = True
    cfg['gpu_collate'] = bool(args.gpu_collate)
    cfg['cuda_prefetch'] = bool(args.cuda_prefetch)
    cfg['stats_enabled'] = True
    cfg['prefetch_max_batches'] = int(args.prefetch_max_batches)
    cfg['prefetch_mem_fraction'] = float(args.prefetch_mem_fraction)

    dl, info = create_dataloader_unified(
        data_path=args.data,
        split="train",
        img_size=args.img_size,
        batch_size=int(args.batch_size),
        hyp=cfg,
        augment=True,
        num_workers=int(args.num_workers),
        persistent_workers=True,
        prefetch_factor=int(args.prefetch_factor),
        cache=args.cache if args.cache.lower() != 'none' else None,
        ram_compression=None if args.ram_compress is False else 'npz'
    )

    use_cuda = torch.cuda.is_available()
    if use_cuda and cfg['cuda_prefetch']:
        it = CUDAPrefetcher(
            dl,
            device=torch.device('cuda'),
            amp=False,
            max_prefetch_batches=cfg['prefetch_max_batches'],
            mem_fraction=cfg['prefetch_mem_fraction']
        )
    else:
        it = iter(dl)

    class_names = info.get('names', [])
    samples = int(args.samples)
    per_grid = int(args.per_grid)
    saved = 0
    agg = {'cpu_ms': [], 'gpu_ms': [], 'gpu_aug_ms': [], 'h2d_bytes': [], 'buffer_depth': []}
    import csv
    csv_path = out_dir / "metrics.csv"
    write_header = not csv_path.exists()
    csv_f = open(csv_path, 'a', newline='')
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow([
            'step', 'bs', 'imgsz', 'workers', 'gpu_collate', 'cuda_prefetch',
            'prefetch_max_batches', 'prefetch_mem_fraction', 'cpu_fetch_ms', 'gpu_total_ms',
            'gpu_aug_ms', 'h2d_bytes', 'buffer_depth', 'cuda_mem_alloc', 'cuda_mem_reserved',
            'cuda_mem_free', 'cuda_mem_total', 'ram_cache_hits', 'disk_cache_hits', 'raw_loads',
            'letterbox_ms_total', 'cpu_aug_ms_total', 'dataset_items'
        ])
    while saved < samples:
        try:
            batch = next(it)
        except StopIteration:
            break
        imgs, targets, _, _, _ = batch
        k = min(per_grid, imgs.shape[0])
        tiles = [denorm_and_draw(imgs, targets, class_names, draw_index=i) for i in range(k)]
        grid = _make_grid(tiles, grid=k)
        diag = [
            f"bs={args.batch_size}",
            f"gpu_collate={cfg['gpu_collate']}",
            f"cuda_prefetch={cfg['cuda_prefetch']}",
            f"workers={args.num_workers} pf={args.prefetch_factor}",
        ]
        if hasattr(it, 'last_timing') and it.last_timing:
            t = it.last_timing
            if t.get('cpu_fetch_ms') is not None:
                diag.append(f"cpu_ms={t['cpu_fetch_ms']:.1f}")
            if t.get('gpu_total_ms') is not None:
                diag.append(f"gpu_ms={t['gpu_total_ms']:.1f}")
            if t.get('gpu_aug_ms') is not None:
                diag.append(f"gpu_aug_ms={t['gpu_aug_ms']:.1f}")
        cuda = torch.cuda.is_available()
        mem_alloc = mem_reserved = mem_free = mem_total = None
        if cuda:
            try:
                mem_alloc = torch.cuda.memory_allocated()
                mem_reserved = torch.cuda.memory_reserved()
                mem_free, mem_total = torch.cuda.mem_get_info()
            except Exception:
                pass
        timing = getattr(it, 'last_timing', None) if hasattr(it, 'last_timing') else None
        cpu_ms = gpu_ms = gpu_aug_ms = None
        h2d_bytes = buffer_depth = None
        if timing:
            cpu_ms = timing.get('cpu_fetch_ms')
            gpu_ms = timing.get('gpu_total_ms')
            gpu_aug_ms = timing.get('gpu_aug_ms')
            h2d_bytes = timing.get('h2d_bytes')
            buffer_depth = timing.get('buffer_depth')
            if cpu_ms is not None:
                agg['cpu_ms'].append(cpu_ms)
            if gpu_ms is not None:
                agg['gpu_ms'].append(gpu_ms)
            if gpu_aug_ms is not None:
                agg['gpu_aug_ms'].append(gpu_aug_ms)
            if h2d_bytes is not None:
                agg['h2d_bytes'].append(h2d_bytes)
            if buffer_depth is not None:
                agg['buffer_depth'].append(buffer_depth)
        ds = getattr(dl, 'dataset', None)
        stats = getattr(ds, 'stats', {}) if ds is not None else {}
        csv_w.writerow([
            saved + 1, args.batch_size, args.img_size, args.num_workers,
            int(cfg['gpu_collate']),
            int(cfg['cuda_prefetch']), cfg['prefetch_max_batches'], cfg['prefetch_mem_fraction'],
            cpu_ms, gpu_ms, gpu_aug_ms, h2d_bytes, buffer_depth, mem_alloc, mem_reserved, mem_free,
            mem_total,
            stats.get('ram_cache_hits'),
            stats.get('disk_cache_hits'),
            stats.get('raw_loads'),
            stats.get('letterbox_ms_total'),
            stats.get('cpu_aug_ms_total'),
            stats.get('items')
        ])
        csv_f.flush()
        diag_extra = []
        if cpu_ms is not None:
            diag_extra.append(f"cpu_ms={cpu_ms:.1f}")
        if gpu_ms is not None:
            diag_extra.append(f"gpu_ms={gpu_ms:.1f}")
        if gpu_aug_ms is not None:
            diag_extra.append(f"gpu_aug_ms={gpu_aug_ms:.1f}")
        if mem_alloc is not None and mem_total is not None:
            diag_extra.append(f"alloc={mem_alloc/1e9:.2f}G/{mem_total/1e9:.2f}G")
        grid = overlay_text(grid, diag + diag_extra)
        out_path = out_dir / f"pipeline_{saved+1}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        saved += 1
    print(f"Full pipeline outputs saved to: {out_dir.resolve()} ({saved} images)")
    try:
        ds = getattr(dl, 'dataset', None)
        st = getattr(ds, 'stats', {}) if ds is not None else {}

        def _mean(v):
            return (sum(v) / len(v)) if v else None

        m_cpu = _mean(agg['cpu_ms'])
        m_gpu = _mean(agg['gpu_ms'])
        m_aug = _mean(agg['gpu_aug_ms'])
        m_h2d_gb = _mean([b / 1e9 for b in agg['h2d_bytes']]) if agg['h2d_bytes'] else None
        m_buf = _mean(agg['buffer_depth'])
        print("--- Pipeline Summary ---")
        if m_cpu is not None:
            print(f"mean cpu_fetch_ms: {m_cpu:.1f}")
        if m_gpu is not None:
            print(f"mean gpu_total_ms: {m_gpu:.1f}")
        if m_aug is not None:
            print(f"mean gpu_aug_ms: {m_aug:.1f}")
        if m_h2d_gb is not None:
            print(f"mean h2d_bytes (GB): {m_h2d_gb:.3f}")
        if m_buf is not None:
            print(f"mean buffer_depth: {m_buf:.1f}")
        if st:
            items = max(int(st.get('items', 0)), 1)
            print(
                f"ram_cache_hits: {st.get('ram_cache_hits',0)}  disk_cache_hits: {st.get('disk_cache_hits',0)}  raw_loads: {st.get('raw_loads',0)}"
            )
            print(f"avg letterbox_ms/item: {float(st.get('letterbox_ms_total',0.0))/items:.2f}")
            print(f"avg cpu_aug_ms/item: {float(st.get('cpu_aug_ms_total',0.0))/items:.2f}")
    except Exception:
        pass
    try:
        csv_f.close()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv9001 Augmentation Visualizer")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to the dataset root (e.g., ../coco)"
    )
    parser.add_argument(
        "--hyp", type=str, default=None, help="Optional path to hyp YAML to override defaults"
    )
    parser.add_argument(
        "--gpu-batch", type=int, default=4, help="Batch size for GPU pipeline tests"
    )
    parser.add_argument(
        "--cpu-batch", type=int, default=1, help="Batch size for CPU pipeline tests"
    )
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline validator")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for full pipeline")
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Dataloader workers for full pipeline"
    )
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor per worker")
    parser.add_argument("--img-size", type=int, default=640, help="Square image size for pipeline")
    parser.add_argument("--gpu-collate", action="store_true", help="Assemble batches on GPU")
    parser.add_argument("--cuda-prefetch", action="store_true", help="Use CUDA prefetcher")
    parser.add_argument(
        "--prefetch-max-batches", type=int, default=3, help="Max CUDA batches buffered"
    )
    parser.add_argument(
        "--prefetch-mem-fraction", type=float, default=0.80, help="Max VRAM fraction to use"
    )
    parser.add_argument("--samples", type=int, default=4, help="How many grids to save")
    parser.add_argument("--per-grid", type=int, default=4, help="How many images per grid (tiles)")
    parser.add_argument(
        "--cache",
        type=str,
        default='none',
        choices=['none', 'ram', 'hybrid', 'disk'],
        help="Dataset cache policy"
    )
    parser.add_argument("--ram-compress", action="store_true", help="Compress RAM cache (npz)")
    args = parser.parse_args()
    if args.full_pipeline:
        run_full_pipeline(args)
    else:
        main(args)
