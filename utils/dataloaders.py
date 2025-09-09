"""
utils/dataloaders.py

YOLOv9001 data loading with minimal memory footprint.

Key features
------------
- Multi-format support (YOLO/COCO) with custom parsing - no supervision dependency
- Split inference from a single dataset root (images/{train*,val*,test*})
- Fixed-size images (imgsz x imgsz) via ExactLetterboxTransform at the end (no multi-res)
- Augment-first pipeline: random augments (scale/affine/mixup/copy-paste/mosaic) on RAW image/labels
- Caching: cache={"ram" | "disk" | "hybrid" | None}
    * RAM: keeps RAW images + RAW labels in memory (no letterbox stored)
    * DISK: saves RAW data as .npy under <root>/.yolo_cache/raw_v1/
    * HYBRID: uses up to 50% of current free RAM for in-memory caching; spills the rest to disk
- DDP-friendly (DistributedSampler if torch.distributed is initialized)
- Hard crash rules:
    * fail if cannot infer both train and val splits
    * fail if any split has zero images

Notes
-----
- Supports YOLO format: labels/<split>/*.txt files with normalized [cls, x, y, w, h]
- Supports COCO format: annotations/instances_<split>.json files
- Labels are stored as YOLO-normalized [cls, x, y, w, h] on the square canvas
- Letterboxing pastes with configurable centering (center/top-left) via hyp['letterbox_center']
"""

from __future__ import annotations

import os
import multiprocessing as mp
import io
import hashlib
import threading
from concurrent import futures
import re
import glob
import json
import platform
try:
    import psutil  # optional, for free memory detection
except Exception:
    psutil = None
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import cv2
import time
import functools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from tqdm import tqdm
from core.config import get_config
from utils.augment import (
    ExactLetterboxTransform,
    mosaic4_raw,
    copy_paste_raw,
    mixup_raw,
)

GPUMultiFormatDataset = None  # vulture: ignore[unused-variable] — reserved for optional GPU dataset
collate_fn_raw_gpu = None  # vulture: ignore[unused-variable] — reserved for optional GPU collate
from utils.logging import get_logger

persistent_object = type('Persistent', (), {})()
setattr(persistent_object, 'logger', None)
setattr(persistent_object, 'cache_lock', None)
setattr(persistent_object, 'log_mutex', None)

class _NoopLogger:
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop

class _ThreadSafeLogger:
    def __init__(self, base, lock):
        self._base = base
        self._lock = lock

    def __getattr__(self, name):
        attr = getattr(self._base, name, None)
        if callable(attr):

            def wrapped(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)

            return wrapped
        return attr

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _list_images(d: Union[str, Path]) -> List[str]:
    """List all image files in a directory."""
    d = str(d)
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(d, f"*{ext}")))
        files.extend(glob.glob(os.path.join(d, f"*{ext.upper()}")))
    return sorted(files)

def _free_mem_bytes() -> int:
    """
    Best-effort free memory detection across platforms. Returns 0 on failure.
    Priority:
      1) psutil.virtual_memory().free
      2) POSIX sysconf (Linux/macOS)
      3) Windows GlobalMemoryStatusEx via ctypes
    """
    try:
        if psutil is not None:
            return int(psutil.virtual_memory().free)
    except Exception:
        pass
    try:
        if os.name == "posix":
            pg = os.sysconf("SC_PAGE_SIZE")
            av = os.sysconf("SC_AVPHYS_PAGES")
            return int(pg) * int(av)
    except Exception:
        pass
    try:
        if platform.system().lower().startswith("win"):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)  # vulture: ignore[unused-attribute]
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullAvailPhys)
    except Exception:
        pass
    return 0

def infer_splits(root: Union[str, Path]) -> Dict[str, Path]:
    """
    Detect train/val/test split directories beneath <root>/images.
    
    Supports common names:
      train: ["train", "train2017", "training"]
      val  : ["val", "valid", "val2017", "validation"]
      test : ["test", "test2017"]
    """
    root = Path(root)
    img_root = root / "images"
    candidates = {
        "train": ["train", "train2017", "training"],
        "val": ["val", "valid", "val2017", "validation"],
        "test": ["test", "test2017"],
    }
    splits: Dict[str, Path] = {}
    for split, names in candidates.items():
        for name in names:
            d = img_root / name
            if d.is_dir():
                splits[split] = d
                break

    if "train" not in splits or "val" not in splits:
        raise ValueError(
            f"[dataloaders] Could not infer both train and val splits under {img_root}. "
            f"Expected subfolders like images/train, images/val, images/train2017, images/val2017."
        )
    return splits

def _detect_format(root: Path, images_dir: Path) -> str:
    """
    Detect dataset format (YOLO or COCO).
    Priority:
      1) YOLO if labels/<basename> exists with .txt files
      2) COCO if annotations/instances_<basename>.json exists
    """
    base = images_dir.name
    labels_dir = root / "labels" / base

    if labels_dir.is_dir() and any(labels_dir.glob("*.txt")):
        return "yolo"

    coco_json_candidates = [
        root / "annotations" / f"instances_{base}.json",
        root / "annotations" / f"instances-{base}.json",
        root / "annotations" / f"{base}.json",
    ]

    for json_path in coco_json_candidates:
        if json_path.is_file():
            return "coco"

    return "yolo"

def _load_names_from_yaml(root: Path) -> Optional[List[str]]:
    """Load class names from data.yaml if present."""
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return None
    try:
        import yaml
        with open(data_yaml, "r") as f:
            d = yaml.safe_load(f) or {}
        names = d.get("names", None)
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    except Exception:
        text = data_yaml.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
        if m:
            items = [x.strip().strip('"') for x in m.group(1).split(",")]
            return items
    return None

def _read_yolo_txt(txt_path: Path) -> np.ndarray:
    """Read YOLO format labels from txt file, handling both bbox and polygon formats."""
    if not txt_path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    lines = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue

            cls = float(parts[0])

            if len(parts) == 5:
                x, y, w, h = map(float, parts[1:5])
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Invalid bounding box with non-positive width or height in {txt_path}: [w={w}, h={h}]"
                    )
                lines.append([cls, x, y, w, h])
            else:
                coords = list(map(float, parts[1:]))
                if len(coords) >= 4 and len(coords) % 2 == 0:
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    x_min = max(0.0, min(1.0, x_min))
                    x_max = max(0.0, min(1.0, x_max))
                    y_min = max(0.0, min(1.0, y_min))
                    y_max = max(0.0, min(1.0, y_max))
                    w = max(1e-6, x_max - x_min)
                    h = max(1e-6, y_max - y_min)
                    x = x_min + 0.5 * w
                    y = y_min + 0.5 * h
                    lines.append([cls, x, y, w, h])

    if not lines:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(lines, dtype=np.float32)

def _parse_coco_json(
    json_path: Path, images_dir: Path
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[str]]:
    """
    Custom COCO JSON parser that returns annotations per image.
    Returns: (annotations_dict, class_names)
    Where annotations_dict maps: image_path -> (xyxy_abs, class_ids)
    """
    with open(json_path, 'r') as f:
        coco = json.load(f)

    id_to_path = {}
    for img_info in coco.get('images', []):
        img_path = str(images_dir / img_info['file_name'])
        id_to_path[img_info['id']] = img_path

    cat_id_to_idx = {}
    categories = sorted(coco.get('categories', []), key=lambda x: x['id'])
    for idx, cat in enumerate(categories):
        cat_id_to_idx[cat['id']] = idx
    names = [cat['name'] for cat in categories]

    annotations_dict = {}

    for img_path in id_to_path.values():
        annotations_dict[img_path] = (
            np.zeros((0, 4), dtype=np.float32), np.zeros((0, ), dtype=np.int64)
        )

    img_anns = {}
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in id_to_path:
            continue

        img_path = id_to_path[img_id]
        if img_path not in img_anns:
            img_anns[img_path] = []

        bbox = ann.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 4:
            x, y, w, h = bbox
            xyxy = [x, y, x + w, y + h]
            cls_id = cat_id_to_idx.get(ann.get('category_id', 0), 0)
            img_anns[img_path].append((xyxy, cls_id))

    for img_path, anns in img_anns.items():
        if anns:
            xyxy = np.array([a[0] for a in anns], dtype=np.float32)
            cls = np.array([a[1] for a in anns], dtype=np.int64)
            annotations_dict[img_path] = (xyxy, cls)

    return annotations_dict, names

class MultiFormatDataset(Dataset):
    """
    Dataset supporting YOLO and COCO formats with minimal memory usage.
    - Only stores image paths and annotation references
    - Loads and processes data on-demand in __getitem__
    - Optional RAM/disk caching for processed images
    """
    def __init__(
        self,
        images_dir: Path,
        root: Path,
        imgsz: int = 640,
        split: str = "train",
        cache: Optional[str] = None,
        hyp: Optional[Dict[str, Any]] = None,
        cache_root: Optional[Path] = None,
        logger=None,
        max_stride: int = 32,
        cache_compression: bool = True,
        cache_threads: int = 0,
        ram_compression: Optional[str] = None,
        ram_compress_level: int = 3,
        hybrid_ram_budget: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.logger = logger or get_logger()
        self.images_dir = images_dir
        self.root = root
        self.imgsz = int(imgsz)
        self.split = split
        self.cache = (cache or "").lower() if cache else None
        self.hyp = hyp or get_config().hyp
        self.train_mode = (self.split == "train")
        self.cache_compression = cache_compression
        self.cache_threads = int(cache_threads) if cache_threads is not None else 0
        self.ram_compression = (ram_compression or "").lower() if ram_compression else None
        self.ram_compress_level = int(ram_compress_level)
        if persistent_object.cache_lock is None:
            persistent_object.cache_lock = threading.Lock()
        self._cache_lock = persistent_object.cache_lock
        self.hybrid_ram_budget_shared = hybrid_ram_budget

        def _reserve_hybrid_budget(nbytes: int) -> bool:
            if self.hybrid_ram_budget_shared is None:
                return False
            try:
                nb = int(nbytes)
            except Exception:
                nb = int(nbytes)
            with self._cache_lock:
                left = int(self.hybrid_ram_budget_shared.get("bytes_left", 0))
                if left >= nb:
                    self.hybrid_ram_budget_shared["bytes_left"] = left - nb
                    return True
            return False

        self._reserve_hybrid_budget = _reserve_hybrid_budget

        self.letterbox_center = self.hyp.get('letterbox_center', True)
        self.max_stride = max_stride

        if self.imgsz % self.max_stride != 0:
            new_imgsz = int(np.ceil(self.imgsz / self.max_stride) * self.max_stride)
            self.logger.warning(
                "dataloader/imgsz",
                f"imgsz {self.imgsz} not divisible by max_stride {self.max_stride}; using {new_imgsz}"
            )
            self.imgsz = new_imgsz

        self.pad_value = int(self.hyp.get('pad_value', 114))

        self.format = _detect_format(root, images_dir)

        self.stats_enabled = bool(self.hyp.get('stats_enabled', False))
        self.stats = {
            'ram_cache_hits': 0,
            'disk_cache_hits': 0,
            'raw_loads': 0,
            'letterbox_ms_total': 0.0,
            'cpu_aug_ms_total': 0.0,
            'items': 0,
        } if self.stats_enabled else {}

        def _clamp01(x: float) -> float:
            try:
                import numpy as _np
                return float(_np.clip(float(x), 0.0, 1.0))
            except Exception:
                return float(x)

        for k in ("fliplr", "flipud", "copy_paste", "mixup", "mosaic"):
            if k in self.hyp:
                self.hyp[k] = _clamp01(self.hyp[k])

        for k in ("degrees", "shear", "translate", "scale", "perspective"):
            if k in self.hyp and not isinstance(self.hyp[k], (int, float, str, list, tuple)):
                self.logger.warning(
                    "dataloader/hyp_type", f"hyp['{k}'] has an unexpected type {type(self.hyp[k])}"
                )

        self.image_paths = sorted([Path(p) for p in _list_images(images_dir)])

        if not self.image_paths:
            raise ValueError(
                f"[dataloaders] No images found for split '{self.split}' at {images_dir}."
            )

        if self.format == "coco":
            base = images_dir.name
            coco_json_candidates = [
                root / "annotations" / f"instances_{base}.json",
                root / "annotations" / f"instances-{base}.json",
                root / "annotations" / f"{base}.json",
            ]
            self.coco_json = next((p for p in coco_json_candidates if p.is_file()), None)

            if self.coco_json:
                self.coco_annotations, coco_names = _parse_coco_json(self.coco_json, images_dir)
                self.names = coco_names or _load_names_from_yaml(root) or [
                    f"class_{i}" for i in range(80)
                ]
                self.logger.info(
                    f"dataloader/{self.split}",
                    f"format=coco images={len(self.image_paths)} ann={self.coco_json.name}"
                )
            else:
                self.coco_annotations = {}
                self.names = _load_names_from_yaml(root)
                if self.names is None:
                    raise ValueError(
                        "[dataloaders] Missing class names. Provide data.yaml with 'names' or COCO annotations."
                    )
                self.logger.warning(
                    f"dataloader/{self.split}", f"No COCO annotations found, using empty labels"
                )

        else:
            labels_dir = root / "labels" / images_dir.name
            self.label_paths = [labels_dir / (p.stem + ".txt") for p in self.image_paths]
            self._yolo_label_by_img = {p: lp for p, lp in zip(self.image_paths, self.label_paths)}

            self.names = _load_names_from_yaml(root)
            if self.names is None:
                max_cls = -1
                sample_size = min(100, len(self.label_paths))
                for lbl_path in self.label_paths[:sample_size]:
                    if lbl_path.is_file():
                        labels = _read_yolo_txt(lbl_path)
                        if labels.size:
                            max_cls = max(max_cls, int(labels[:, 0].max()))
                nc = max_cls + 1 if max_cls >= 0 else 80
                self.names = [f"class_{i}" for i in range(nc)]
                self.logger.warning(
                    f"dataloader/{self.split}",
                    f"Missing class names in data.yaml, inferring {nc} classes."
                )

            self.logger.info(
                f"dataloader/{self.split}",
                f"format=yolo images={len(self.image_paths)} labels={labels_dir}"
            )

        if self.format == "yolo":
            max_id = -1
            for p in self.label_paths[:min(500, len(self.label_paths))]:
                if p.is_file():
                    arr = _read_yolo_txt(p)
                    if arr.size:
                        max_id = max(max_id, int(arr[:, 0].max()))
            if max_id >= len(self.names):
                raise ValueError(
                    f"[dataloaders] data.yaml defines {len(self.names)} names but labels contain class id {max_id}."
                )

        self.nc = len(self.names)

        self._ram_cache: Dict[str, tuple] = {}
        self._disk_cache_dirs: List[Path] = []

        if self.cache in ("disk", "ram", "hybrid"):
            self._prepare_cache_dirs(cache_root if cache_root is not None else root)

        if self.cache == 'hybrid':
            if self.hybrid_ram_budget_shared is None:
                free_bytes = _free_mem_bytes()
                budget = int(0.50 * max(0, free_bytes))
                self.hybrid_ram_budget_shared = {"bytes_left": budget}
                self.logger.info(f"dataloader/{self.split}", f"Created local hybrid cache budget.")
            try:
                budget_bytes = self.hybrid_ram_budget_shared.get("bytes_left", 0)
                human = f"{budget_bytes/1e9:.2f} GB"
            except Exception:
                human = str(budget_bytes)
            self.logger.info(f"dataloader/{self.split}", f"Hybrid cache budget available = {human}")

        if self.cache in ('ram', 'disk', 'hybrid'):
            self._precache()

        self.logger.info(
            f"dataloader/{self.split}",
            f"prepared {len(self.image_paths)} samples (cache={self.cache}, format={self.format})"
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            state['logger'] = None
        if '_cache_lock' in state:
            state['_cache_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if persistent_object.log_mutex is None:
                persistent_object.log_mutex = threading.Lock()
            if persistent_object.logger is None:
                is_main = (getattr(mp.current_process(), 'name', '') == 'MainProcess')
                if is_main:
                    from utils.logging import get_logger as _get_logger
                    base_logger = _get_logger()
                    persistent_object.logger = _ThreadSafeLogger(
                        base_logger, persistent_object.log_mutex
                    )
                else:
                    persistent_object.logger = _NoopLogger()
            self.logger = persistent_object.logger
        except Exception:
            self.logger = _NoopLogger()
        if persistent_object.cache_lock is None:
            persistent_object.cache_lock = threading.Lock()
        self._cache_lock = persistent_object.cache_lock

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int], float, Tuple[float, float, float,
                                                                         float]]:
        """
        RAW -> augment(s) -> crop if needed -> ExactLetterboxTransform (scaleup=False during train)
        Returns: (tensor_image, yolo_norm_labels, (orig_h, orig_w), ratio, pad)
        """
        img_path = self.image_paths[idx]
        img_str_path = str(img_path)

        if self.cache in ('ram', 'hybrid') and img_str_path in self._ram_cache:
            item = self._ram_cache[img_str_path]
            if self.stats_enabled:
                self.stats['ram_cache_hits'] += 1
            if isinstance(item, (bytes, bytearray, memoryview)):
                try:
                    with io.BytesIO(item) as buf:
                        data = np.load(buf, allow_pickle=False)
                        raw_img = data['raw_img']
                        xyxy_abs = data['xyxy_abs'].astype(np.float32, copy=False)
                        cls = data['cls'].astype(np.int64, copy=False)
                        oh, ow = map(int, data['orig_shape'])
                except Exception as e:
                    self.logger.warning(
                        "dataloader/ram_cache_load_error",
                        f"In-memory cache decode failed for {img_path}: {e}"
                    )
                    raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(idx)
            else:
                raw_img, xyxy_abs, cls, (oh, ow) = item
                if raw_img.dtype != np.uint8:
                    raw_img = raw_img.astype(np.uint8, copy=False)
                raw_img = np.ascontiguousarray(raw_img)
                xyxy_abs = xyxy_abs.astype(np.float32, copy=False)
                cls = cls.astype(np.int64, copy=False)
        elif self.cache in ('disk', 'hybrid') and self._disk_cache_dirs:
            data_loaded = False
            for cand in self._iter_cache_candidates(img_path):
                if cand.exists():
                    try:
                        data = np.load(str(cand), allow_pickle=False)
                        raw_img = data['raw_img']
                        xyxy_abs = data['xyxy_abs'].astype(np.float32, copy=False)
                        cls = data['cls'].astype(np.int64, copy=False)
                        oh, ow = map(int, data['orig_shape'])
                        if self.stats_enabled:
                            self.stats['disk_cache_hits'] += 1
                        if raw_img.dtype != np.uint8:
                            raw_img = raw_img.astype(np.uint8, copy=False)
                        raw_img = np.ascontiguousarray(raw_img)
                        data_loaded = True
                        if self.cache == 'hybrid':
                            payload, payload_nbytes = self._make_ram_payload(
                                raw_img, xyxy_abs, cls, (oh, ow)
                            )
                            use_ram = self._reserve_hybrid_budget(payload_nbytes)
                            if use_ram:
                                self._ram_cache[img_str_path] = payload
                        break
                    except Exception as e:
                        self.logger.warning(
                            "dataloader/cache_load_error", f"Failed to load cache file {cand}: {e}"
                        )
            if not data_loaded:
                raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(idx)
                if self.stats_enabled:
                    self.stats['raw_loads'] += 1
                if self.cache == 'disk':
                    try:
                        primary = self._primary_cache_path(img_path)
                        primary.parent.mkdir(parents=True, exist_ok=True)
                        if self.cache_compression:
                            np.savez_compressed(
                                str(primary),
                                raw_img=raw_img.astype(np.uint8, copy=False),
                                xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                cls=cls.astype(np.int64, copy=False),
                                orig_shape=np.array([oh, ow], dtype=np.int32),
                            )
                        else:
                            np.savez(
                                str(primary),
                                raw_img=raw_img.astype(np.uint8, copy=False),
                                xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                cls=cls.astype(np.int64, copy=False),
                                orig_shape=np.array([oh, ow], dtype=np.int32),
                            )
                    except Exception as e:
                        self.logger.warning(
                            "dataloader/cache_save_error",
                            f"Failed to save cache file {primary}: {e}"
                        )
                elif self.cache == 'hybrid':
                    payload, payload_nbytes = self._make_ram_payload(
                        raw_img, xyxy_abs, cls, (oh, ow)
                    )
                    use_ram = self._reserve_hybrid_budget(payload_nbytes)
                    if use_ram:
                        self._ram_cache[img_str_path] = payload
                    else:
                        try:
                            primary = self._primary_cache_path(img_path)
                            primary.parent.mkdir(parents=True, exist_ok=True)
                            if self.cache_compression:
                                np.savez_compressed(
                                    str(primary),
                                    raw_img=raw_img.astype(np.uint8, copy=False),
                                    xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                    cls=cls.astype(np.int64, copy=False),
                                    orig_shape=np.array([oh, ow], dtype=np.int32),
                                )
                            else:
                                np.savez(
                                    str(primary),
                                    raw_img=raw_img.astype(np.uint8, copy=False),
                                    xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                    cls=cls.astype(np.int64, copy=False),
                                    orig_shape=np.array([oh, ow], dtype=np.int32),
                                )
                        except Exception as e:
                            self.logger.warning(
                                "dataloader/cache_save_error",
                                f"Failed to save cache file {primary}: {e}"
                            )
        else:
            raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(idx)
            if self.stats_enabled:
                self.stats['raw_loads'] += 1
            if self.cache == 'ram':
                if self.ram_compression == "npz":
                    try:
                        buf = io.BytesIO()
                        np.savez_compressed(
                            buf,
                            raw_img=raw_img.astype(np.uint8, copy=False),
                            xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                            cls=cls.astype(np.int64, copy=False),
                            orig_shape=np.array([oh, ow], dtype=np.int32),
                        )
                        self._ram_cache[img_str_path] = buf.getvalue()
                    except Exception as e:
                        self.logger.warning(
                            "dataloader/ram_cache_save_error",
                            f"Failed to compress RAM cache for {img_path}: {e}"
                        )
                        self._ram_cache[img_str_path] = (raw_img, xyxy_abs, cls, (oh, ow))
                else:
                    self._ram_cache[img_str_path] = (raw_img, xyxy_abs, cls, (oh, ow))
            elif self.cache == 'hybrid':
                payload, payload_nbytes = self._make_ram_payload(raw_img, xyxy_abs, cls, (oh, ow))
                use_ram = self._reserve_hybrid_budget(payload_nbytes)
                if use_ram:
                    self._ram_cache[img_str_path] = payload
                else:
                    try:
                        primary = self._primary_cache_path(img_path)
                        primary.parent.mkdir(parents=True, exist_ok=True)
                        if self.cache_compression:
                            np.savez_compressed(
                                str(primary),
                                raw_img=raw_img.astype(np.uint8, copy=False),
                                xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                cls=cls.astype(np.int64, copy=False),
                                orig_shape=np.array([oh, ow], dtype=np.int32),
                            )
                        else:
                            np.savez(
                                str(primary),
                                raw_img=raw_img.astype(np.uint8, copy=False),
                                xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                                cls=cls.astype(np.int64, copy=False),
                                orig_shape=np.array([oh, ow], dtype=np.int32),
                            )
                    except Exception as e:
                        self.logger.warning(
                            "dataloader/cache_save_error",
                            f"Failed to save cache file {primary}: {e}"
                        )

        img_aug, boxes_abs, cls_ids = raw_img, xyxy_abs, cls

        cpu_aug_t0 = time.perf_counter() if self.stats_enabled else None
        if self.train_mode and bool(self.hyp.get("augment", True)):
            if float(self.hyp.get("mosaic", 0.0)
                    ) > 0.0 and np.random.rand() < float(self.hyp["mosaic"]):
                img_aug, boxes_abs, cls_ids = mosaic4_raw(self, idx)

            if float(self.hyp.get("copy_paste", 0.0)
                    ) > 0.0 and np.random.rand() < float(self.hyp["copy_paste"]):
                img_aug, boxes_abs, cls_ids = copy_paste_raw(self, img_aug, boxes_abs, cls_ids)

            if float(self.hyp.get("mixup", 0.0)
                    ) > 0.0 and np.random.rand() < float(self.hyp["mixup"]):
                img_aug, boxes_abs, cls_ids = mixup_raw(self, img_aug, boxes_abs, cls_ids)
        if self.stats_enabled and cpu_aug_t0 is not None:
            self.stats['cpu_aug_ms_total'] += (time.perf_counter() - cpu_aug_t0) * 1000.0

        lb_t0 = time.perf_counter() if self.stats_enabled else None
        lb = ExactLetterboxTransform(
            img_size=self.imgsz,
            center=self.letterbox_center,
            pad_value=self.pad_value,
            scaleup=False,
            pad_to_stride=self.max_stride
        )
        canvas, yolo_norm, _, ratio, pad = lb(img_aug, boxes_abs, cls_ids)
        if self.stats_enabled and lb_t0 is not None:
            self.stats['letterbox_ms_total'] += (time.perf_counter() - lb_t0) * 1000.0
            self.stats['items'] += 1

        img_t = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().to(torch.uint8)
        labels_t = torch.from_numpy(yolo_norm).to(
            torch.float32
        ) if yolo_norm.size else torch.zeros((0, 5), dtype=torch.float32)
        return img_t, labels_t, (oh, ow), ratio, pad

    def _precache(self):
        """Pre-cache all items to RAM, disk, or hybrid (threaded, with progress bar)."""
        if len(self) == 0:
            return
        self.logger.info(
            f"dataloader/{self.split}", f"Caching {len(self)} images for faster training..."
        )
        if self.cache == 'hybrid' and self.hybrid_ram_budget_shared:
            try:
                human = f"{self.hybrid_ram_budget_shared['bytes_left']/1e9:.2f} GB"
            except Exception:
                human = str(self.hybrid_ram_budget_shared['bytes_left'])
            self.logger.info(f"dataloader/{self.split}", f"Initial hybrid RAM budget: {human}")

        def _cache_one(i: int):
            img_path = self.image_paths[i]
            img_str_path = str(img_path)
            if self.cache == 'ram':
                if img_str_path in self._ram_cache:
                    return
                raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(i)
                if self.ram_compression == "npz":
                    buf = io.BytesIO()
                    np.savez_compressed(
                        buf,
                        raw_img=raw_img.astype(np.uint8, copy=False),
                        xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                        cls=cls.astype(np.int64, copy=False),
                        orig_shape=np.array([oh, ow], dtype=np.int32),
                    )
                    with self._cache_lock:
                        self._ram_cache[img_str_path] = buf.getvalue()
                else:
                    with self._cache_lock:
                        self._ram_cache[img_str_path] = (raw_img, xyxy_abs, cls, (oh, ow))
            elif self.cache == 'hybrid':
                if img_str_path in self._ram_cache:
                    return

                raw_img, xyxy_abs, cls, (oh,
                                         ow), disk_loaded = (None, None, None, (None, None), False)

                for cand in self._iter_cache_candidates(img_path):
                    if cand.exists():
                        try:
                            data = np.load(str(cand), allow_pickle=False)
                            raw_img = data['raw_img']
                            xyxy_abs = data['xyxy_abs'].astype(np.float32, copy=False)
                            cls = data['cls'].astype(np.int64, copy=False)
                            oh, ow = map(int, data['orig_shape'])
                            disk_loaded = True
                            break
                        except Exception:
                            pass

                if not disk_loaded:
                    raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(i)

                payload, payload_nbytes = self._make_ram_payload(raw_img, xyxy_abs, cls, (oh, ow))
                use_ram = False
                with self._cache_lock:
                    if self.hybrid_ram_budget_shared is not None and self.hybrid_ram_budget_shared[
                        'bytes_left'] >= payload_nbytes:
                        self.hybrid_ram_budget_shared['bytes_left'] -= payload_nbytes
                        use_ram = True

                if use_ram:
                    with self._cache_lock:
                        self._ram_cache[img_str_path] = payload
                elif not disk_loaded:
                    primary = self._primary_cache_path(img_path)
                    primary.parent.mkdir(parents=True, exist_ok=True)
                    if self.cache_compression:
                        np.savez_compressed(
                            str(primary),
                            raw_img=raw_img.astype(np.uint8, copy=False),
                            xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                            cls=cls.astype(np.int64, copy=False),
                            orig_shape=np.array([oh, ow], dtype=np.int32),
                        )
                    else:
                        np.savez(
                            str(primary),
                            raw_img=raw_img.astype(np.uint8, copy=False),
                            xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                            cls=cls.astype(np.int64, copy=False),
                            orig_shape=np.array([oh, ow], dtype=np.int32),
                        )
            elif self.cache == 'disk' and self._disk_cache_dirs:
                for cand in self._iter_cache_candidates(img_path):
                    if cand.exists():
                        return
                raw_img, xyxy_abs, cls, (oh, ow) = self._load_raw(i)
                primary = self._primary_cache_path(img_path)
                primary.parent.mkdir(parents=True, exist_ok=True)
                if self.cache_compression:
                    np.savez_compressed(
                        str(primary),
                        raw_img=raw_img.astype(np.uint8, copy=False),
                        xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                        cls=cls.astype(np.int64, copy=False),
                        orig_shape=np.array([oh, ow], dtype=np.int32),
                    )
                else:
                    np.savez(
                        str(primary),
                        raw_img=raw_img.astype(np.uint8, copy=False),
                        xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                        cls=cls.astype(np.int64, copy=False),
                        orig_shape=np.array([oh, ow], dtype=np.int32),
                    )

        if self.cache_threads and self.cache_threads > 0:
            with futures.ThreadPoolExecutor(max_workers=self.cache_threads) as ex:
                list(
                    tqdm(
                        ex.map(_cache_one, range(len(self))),
                        total=len(self),
                        desc=f"Caching {self.split} data"
                    )
                )
        else:
            for i in tqdm(range(len(self)), desc=f"Caching {self.split} data"):
                _cache_one(i)

    def _make_ram_payload(
        self,
        raw_img: np.ndarray,
        xyxy_abs: np.ndarray,
        cls: np.ndarray,
        orig_shape: Tuple[int, int],
    ):
        """
        Build the RAM payload and compute its approximate size in bytes.
        If self.ram_compression == 'npz', we produce a compressed bytes payload (measured precisely).
        Otherwise, we produce the raw tuple and approximate nbytes from numpy arrays (+ small overhead).
        """
        if self.ram_compression == "npz":
            buf = io.BytesIO()
            np.savez_compressed(
                buf,
                raw_img=raw_img.astype(np.uint8, copy=False),
                xyxy_abs=xyxy_abs.astype(np.float32, copy=False),
                cls=cls.astype(np.int64, copy=False),
                orig_shape=np.asarray(orig_shape, dtype=np.int32),
            )
            payload = buf.getvalue()
            return payload, len(payload)
        else:
            approx_nbytes = int(raw_img.nbytes + xyxy_abs.nbytes + cls.nbytes + 16)
            return (raw_img, xyxy_abs, cls, orig_shape), approx_nbytes

    def _prepare_cache_dirs(self, base: Path):
        tag_cur = "compressed" if self.cache_compression else "uncompressed"
        tag_alt = "uncompressed" if self.cache_compression else "compressed"
        root_dir = base / ".yolo_cache"
        primary = root_dir / f"raw_v1_{tag_cur}" / self.split
        primary.mkdir(parents=True, exist_ok=True)
        self._disk_cache_dirs = [
            primary,
            root_dir / f"raw_v1_{tag_cur}",
            root_dir / f"raw_v1_{tag_alt}" / self.split,
            root_dir / f"raw_v1_{tag_alt}",
        ]

    def _cache_key(self, img_path: Path) -> str:
        try:
            rel = str(img_path.relative_to(self.root))
        except Exception:
            rel = str(img_path)
        h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        return f"{img_path.stem}.{h}.npz"

    def _primary_cache_path(self, img_path: Path) -> Path:
        return self._disk_cache_dirs[0] / self._cache_key(img_path)

    def _iter_cache_candidates(self, img_path: Path):
        key_new = self._cache_key(img_path)
        key_legacy = f"{img_path.stem}.npz"
        for d in self._disk_cache_dirs:
            yield d / key_new
        for d in self._disk_cache_dirs:
            yield d / key_legacy

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """Load RAW image (RGB uint8) and RAW labels (absolute xyxy in original pixels) for a single index."""
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)
        img = np.ascontiguousarray(img)
        h0, w0 = img.shape[:2]
        if self.format == "coco":
            if str(img_path) in getattr(self, "coco_annotations", {}):
                xyxy_abs, cls = self.coco_annotations[str(img_path)]
            else:
                xyxy_abs = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0, ), dtype=np.int64)
        else:
            labels = _read_yolo_txt(self._yolo_label_by_img[img_path])
            if labels.size:
                cls = labels[:, 0].astype(np.int64)
                x_center, y_center, w, h = labels[:,
                                                  1] * w0, labels[:, 2
                                                                 ] * h0, labels[:, 3
                                                                               ] * w0, labels[:, 4
                                                                                             ] * h0
                x1 = x_center - w / 2
                y1 = y_center - h / 2
                x2 = x_center + w / 2
                y2 = y_center + h / 2
                xyxy_abs = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
            else:
                xyxy_abs = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0, ), dtype=np.int64)
        return img, xyxy_abs, cls, (h0, w0)

def collate_fn(batch):
    imgs, labels, orig_shapes, ratios, pads = zip(*batch)
    imgs = torch.stack(imgs, 0)
    if any(l.numel() for l in labels):
        sizes = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
        all_labs = torch.cat([l for l in labels if l.numel()], 0).to(torch.float32)
        idx = torch.repeat_interleave(torch.arange(len(labels)), sizes).to(all_labs.device)
        targets = torch.cat([idx.unsqueeze(1).to(all_labs.dtype), all_labs], 1)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32, device=imgs.device)
    return imgs, targets, orig_shapes, ratios, pads

def collate_fn_gpu_affine(batch, hyp: Dict[str, Any], pad_value: int):
    """
    Module-level, picklable GPU collate that assembles a batch on CUDA and applies optional
    perspective/affine augmentations according to hyp. Falls back to CPU collate if CUDA is
    unavailable.
    """
    imgs, labels, orig_shapes, ratios, pads = zip(*batch)
    B = len(imgs)
    imgs = torch.stack(imgs, 0)  # uint8 CPU [B,3,H,W]
    H, W = imgs.shape[2], imgs.shape[3]

    if any(l.numel() for l in labels):
        sizes = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
        all_labs = torch.cat([l for l in labels if l.numel()], 0).to(torch.float32)
        bi = torch.repeat_interleave(torch.arange(B), sizes).to(all_labs.device).to(all_labs.dtype)
        targets = torch.cat([bi.unsqueeze(1), all_labs], 1)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32, device=imgs.device)

    if not torch.cuda.is_available():
        return collate_fn(batch)

    device = torch.device('cuda')
    imgs_f = imgs.to(device=device, dtype=torch.float32).div_(255.0)

    boxes_list: List[torch.Tensor] = []
    cls_list: List[torch.Tensor] = []
    if targets.numel():
        t = targets.to(device)
        bi_long = t[:, 0].to(torch.long)
        for i in range(B):
            ti = t[bi_long == i]
            if ti.numel():
                cls_i = ti[:, 1].to(torch.int64)
                cx, cy, ww, hh = ti[:, 2] * W, ti[:, 3] * H, ti[:, 4] * W, ti[:, 5] * H
                x1 = (cx - ww * 0.5).clamp_(0, W)
                y1 = (cy - hh * 0.5).clamp_(0, H)
                x2 = (cx + ww * 0.5).clamp_(0, W)
                y2 = (cy + hh * 0.5).clamp_(0, H)
                boxes_i = torch.stack([x1, y1, x2, y2], dim=1)
            else:
                cls_i = torch.zeros((0, ), device=device, dtype=torch.int64)
                boxes_i = torch.zeros((0, 4), device=device, dtype=torch.float32)
            cls_list.append(cls_i)
            boxes_list.append(boxes_i)
    else:
        for _ in range(B):
            cls_list.append(torch.zeros((0, ), device=device, dtype=torch.int64))
            boxes_list.append(torch.zeros((0, 4), device=device, dtype=torch.float32))

    from utils.augment import random_perspective_gpu_batch, random_affine_gpu_batch
    persp = hyp.get('perspective', 0.0)
    if float(persp or 0.0) != 0.0:
        imgs_f, boxes_list, keep_list = random_perspective_gpu_batch(
            imgs_f, boxes_list, persp, pad_value
        )
        cls_list = [
            c[k] if (c is not None and c.numel()) else c for c, k in zip(cls_list, keep_list)
        ]

    imgs_f, boxes_list, keep_list = random_affine_gpu_batch(
        imgs_f,
        boxes_list,
        int(H),
        hyp.get('degrees', 0.0),
        hyp.get('translate', 0.0),
        hyp.get('scale', 0.0),
        hyp.get('shear', 0.0),
        pad_value,
        hyp=hyp,
    )

    pieces = []
    for i in range(B):
        b = boxes_list[i]
        if b is None or b.numel() == 0:
            continue
        k = keep_list[i]
        if (
            cls_list[i] is not None and cls_list[i].numel() and k is not None and
            hasattr(k, 'numel') and k.numel() == cls_list[i].shape[0]
        ):
            cls_i = cls_list[i][k]
        else:
            cls_i = cls_list[i]
        if cls_i.shape[0] != b.shape[0]:
            n = min(cls_i.shape[0], b.shape[0])
            if n == 0:
                continue
            b = b[:n]
            cls_i = cls_i[:n]
        wv = (b[:, 2] - b[:, 0]).clamp_min(1.0)
        hv = (b[:, 3] - b[:, 1]).clamp_min(1.0)
        cx = b[:, 0] + 0.5 * wv
        cy = b[:, 1] + 0.5 * hv
        yolo = torch.stack([cls_i.to(imgs_f.dtype), cx / W, cy / H, wv / W, hv / H], dim=1)
        bi_full = torch.full((yolo.shape[0], 1), i, device=device, dtype=imgs_f.dtype)
        pieces.append(torch.cat([bi_full, yolo], 1))
    targets = torch.cat(pieces,
                        0) if pieces else torch.zeros((0, 6), device=device, dtype=imgs_f.dtype)

    return imgs_f.to(memory_format=torch.channels_last), targets, orig_shapes, ratios, pads

def build_collate_fn_with_gpu_affine(dataset: Dataset):
    """
    Build a collate function that applies GPU-side affine/flip augmentations.

    vulture: ignore[unused-function] — optional GPU dataloader path kept for future use.
    """
    hyp = getattr(dataset, 'hyp', {})
    pad_value = int(getattr(dataset, 'pad_value', 114))

    def _collate(batch):
        imgs, labels, orig_shapes, ratios, pads = zip(*batch)
        B = len(imgs)
        imgs = torch.stack(imgs, 0)  # uint8 CPU [B,3,H,W]
        H, W = imgs.shape[2], imgs.shape[3]

        if any(l.numel() for l in labels):
            sizes = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)
            all_labs = torch.cat([l for l in labels if l.numel()], 0).to(torch.float32)
            bi = torch.repeat_interleave(torch.arange(B),
                                         sizes).to(all_labs.device).to(all_labs.dtype)
            targets = torch.cat([bi.unsqueeze(1), all_labs], 1)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32, device=imgs.device)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            imgs_f = imgs.to(device=device, dtype=torch.float32).div_(255.0)

            boxes_list: List[torch.Tensor] = []
            cls_list: List[torch.Tensor] = []
            if targets.numel():
                t = targets.to(device)
                bi_long = t[:, 0].to(torch.long)
                for i in range(B):
                    ti = t[bi_long == i]
                    if ti.numel():
                        cls_i = ti[:, 1].to(torch.int64)
                        cx, cy, ww, hh = ti[:, 2] * W, ti[:, 3] * H, ti[:, 4] * W, ti[:, 5] * H
                        x1 = (cx - ww * 0.5).clamp_(0, W)
                        y1 = (cy - hh * 0.5).clamp_(0, H)
                        x2 = (cx + ww * 0.5).clamp_(0, W)
                        y2 = (cy + hh * 0.5).clamp_(0, H)
                        boxes_i = torch.stack([x1, y1, x2, y2], dim=1)
                    else:
                        cls_i = torch.zeros((0, ), device=device, dtype=torch.int64)
                        boxes_i = torch.zeros((0, 4), device=device, dtype=torch.float32)
                    cls_list.append(cls_i)
                    boxes_list.append(boxes_i)
            else:
                for _ in range(B):
                    cls_list.append(torch.zeros((0, ), device=device, dtype=torch.int64))
                    boxes_list.append(torch.zeros((0, 4), device=device, dtype=torch.float32))

            from utils.augment import random_perspective_gpu_batch, random_affine_gpu_batch
            persp = hyp.get('perspective', 0.0)
            if float(persp or 0.0) != 0.0:
                imgs_f, boxes_list, keep_list = random_perspective_gpu_batch(
                    imgs_f, boxes_list, persp, pad_value
                )
                cls_list = [
                    c[k] if (c is not None and c.numel()) else c
                    for c, k in zip(cls_list, keep_list)
                ]

            imgs_f, boxes_list, keep_list = random_affine_gpu_batch(
                imgs_f,
                boxes_list,
                int(H),
                hyp.get('degrees', 0.0),
                hyp.get('translate', 0.0),
                hyp.get('scale', 0.0),
                hyp.get('shear', 0.0),
                pad_value,
                hyp=hyp,
            )

            pieces = []
            for i in range(B):
                b = boxes_list[i]
                if b is None or b.numel() == 0:
                    continue
                k = keep_list[i]
                if (
                    cls_list[i] is not None and cls_list[i].numel() and k is not None and
                    hasattr(k, 'numel') and k.numel() == cls_list[i].shape[0]
                ):
                    cls_i = cls_list[i][k]
                else:
                    cls_i = cls_list[i]
                if cls_i.shape[0] != b.shape[0]:
                    n = min(cls_i.shape[0], b.shape[0])
                    if n == 0:
                        continue
                    b = b[:n]
                    cls_i = cls_i[:n]
                wv = (b[:, 2] - b[:, 0]).clamp_min(1.0)
                hv = (b[:, 3] - b[:, 1]).clamp_min(1.0)
                cx = b[:, 0] + 0.5 * wv
                cy = b[:, 1] + 0.5 * hv
                yolo = torch.stack([cls_i.to(imgs_f.dtype), cx / W, cy / H, wv / W, hv / H], dim=1)
                bi_full = torch.full((yolo.shape[0], 1), i, device=device, dtype=imgs_f.dtype)
                pieces.append(torch.cat([bi_full, yolo], 1))
            targets = torch.cat(pieces, 0) if pieces else torch.zeros((0, 6),
                                                                      device=device,
                                                                      dtype=imgs_f.dtype)

            return imgs_f.to(memory_format=torch.channels_last), targets, orig_shapes, ratios, pads
        else:
            return collate_fn(batch)

    return _collate

def create_dataloader(
    data_path: Union[str, Path],
    split: str,
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    cache: Optional[str] = None,
    pin_memory: bool = True,
    hyp: Optional[Dict[str, Any]] = None,
    max_stride: int = 32,
    seed: int | None = None,
    cache_compression: bool = True,
    cache_threads: Optional[int] = None,
    ram_compression: Optional[str] = None,
    ram_compress_level: int = 3,
    hybrid_ram_budget: Optional[Dict[str, int]] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Build a single dataloader for a specific split using MultiFormatDataset.
    Automatically detects YOLO or COCO format.
    """
    if cache_threads is None:
        cache_threads = num_workers
    logger = get_logger()
    root = Path(data_path)
    if not root.exists():
        raise ValueError(f"[dataloaders] Data path does not exist: {root}")

    config = get_config(hyp=hyp)
    dataset_hyp = config.hyp if augment and split == "train" else get_config().hyp

    splits = infer_splits(root)
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in {root/'images'}")
    images_dir = splits[split]

    is_train = split == "train"
    dataset = MultiFormatDataset(
        images_dir=images_dir,
        root=root,
        imgsz=img_size,
        split=split,
        cache=cache,
        hyp=dataset_hyp,
        cache_root=root,
        logger=logger,
        max_stride=max_stride,
        cache_compression=cache_compression,
        cache_threads=cache_threads,
        ram_compression=ram_compression,
        ram_compress_level=ram_compress_level,
        hybrid_ram_budget=hybrid_ram_budget,
    )

    if len(dataset) == 0:
        raise ValueError(f"[dataloaders] Empty split: {split} ({len(dataset)} images)")

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler = DistributedSampler(dataset, shuffle=is_train) if distributed else None

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))

    collate = collate_fn
    try:
        gpu_collate = bool(config.get('gpu_collate', False)) and torch.cuda.is_available()
    except Exception:
        gpu_collate = False
    mp_ctx = None
    if gpu_collate:
        collate = functools.partial(
            collate_fn_gpu_affine,
            hyp=dataset.hyp,
            pad_value=int(getattr(dataset, 'pad_value', 114))
        )
        try:
            setattr(dataset, 'gpu_collate', True)
        except Exception:
            pass
        if num_workers and num_workers > 0:
            try:
                mp_ctx = mp.get_context('spawn')
            except Exception:
                mp_ctx = None
            if mp_ctx is None:
                get_logger().warning(
                    'dataloader/gpu_collate_disabled',
                    'Falling back to CPU collate: could not acquire spawn context for workers'
                )
                collate = collate_fn
                try:
                    setattr(dataset, 'gpu_collate', False)
                except Exception:
                    pass
                gpu_collate = False

    if persistent_workers is None:
        persistent_workers = bool(config.get('persistent_workers', False))
    if prefetch_factor is None:
        prefetch_factor = int(config.get('prefetch_factor', 2))

    shuffle_flag = bool(config.get('shuffle', True))
    effective_pin_memory = bool(pin_memory and not gpu_collate)
    pin_mem_device = "cuda" if (effective_pin_memory and torch.__version__ >= "2.0.0") else ""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and is_train and shuffle_flag),
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=effective_pin_memory,
        drop_last=is_train,
        collate_fn=collate,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory_device=pin_mem_device,
        generator=g,
        multiprocessing_context=mp_ctx,
    )

    info = {
        "nc": dataset.nc,
        "names": dataset.names,
        "format": dataset.format,
        f"{split}_images": len(dataset),
    }
    logger.log_text(
        f"dataloader/{split}/summary",
        json.dumps({
            "batches": len(loader), "batch_size": batch_size, "nc": dataset.nc, "format":
                dataset.format
        },
                   indent=2)
    )
    return loader, info

def create_dataloaders(
    data_path: Union[str, Path],
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    cache: Optional[str] = None,
    pin_memory: bool = True,
    model_nc: Optional[int] = None,
    hyp: Optional[Dict[str, Any]] = None,
    max_stride: int = 32,
    cache_compression: bool = True,
    cache_threads: Optional[int] = None,
    ram_compression: Optional[str] = None,
    ram_compress_level: int = 3,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Build train/val dataloaders using MultiFormatDataset.
    - data_path: dataset root containing images/{train*,val*} and labels/ or annotations/
    - hyp: optional dict; if provided, hyp["scale"] enables resize augmentation
    """
    if cache_threads is None:
        cache_threads = num_workers
    logger = get_logger()
    logger.info("dataloader/init", f"Creating train/val dataloaders from {data_path}")

    hybrid_ram_budget = None
    if str(cache).lower() == 'hybrid':
        free_bytes = _free_mem_bytes()
        budget = int(0.50 * max(0, free_bytes))
        hybrid_ram_budget = {"bytes_left": budget}
        try:
            human = f"{budget/1e9:.2f} GB"
        except Exception:
            human = str(budget)
        logger.info("dataloader/init", f"Created shared hybrid cache budget: {human}")

    train_loader, train_info = create_dataloader(
        data_path,
        "train",
        img_size,
        batch_size,
        num_workers,
        augment,
        cache,
        pin_memory,
        hyp,
        max_stride,
        cache_compression=cache_compression,
        cache_threads=cache_threads,
        ram_compression=ram_compression,
        ram_compress_level=ram_compress_level,
        hybrid_ram_budget=hybrid_ram_budget,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader, val_info = create_dataloader(
        data_path,
        "val",
        img_size,
        batch_size,
        num_workers,
        False,
        cache,
        pin_memory,
        None,
        max_stride,
        cache_compression=cache_compression,
        cache_threads=cache_threads,
        ram_compression=ram_compression,
        ram_compress_level=ram_compress_level,
        hybrid_ram_budget=hybrid_ram_budget,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    info = {**train_info, **val_info}
    if train_info["nc"] != val_info["nc"]:
        logger.warning(
            "dataloader/nc_mismatch",
            f"Train nc={train_info['nc']} vs Val nc={val_info['nc']}. Using train."
        )
        info["nc"] = train_info["nc"]
        info["names"] = train_info["names"]

    if model_nc is not None and model_nc != info["nc"]:
        raise ValueError(f"[dataloaders] Model nc={model_nc} but dataset has nc={info['nc']}")

    return train_loader, val_loader, info

def create_dummy_dataloader(
    img_size: int = 640,
    batch_size: int = 16,
    num_classes: int = 80,
    length: int = 64
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Random images + random YOLO-normalized labels. For testing only.
    """
    class DummyDataset(Dataset):
        def __init__(self, img_size, num_classes, length, pad_value=114):
            self.img_size = img_size
            self.num_classes = num_classes
            self.length = length
            self.pad_value = pad_value

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            h, w = np.random.randint(300, 800), np.random.randint(400, 1000)
            image = np.full((h, w, 3), np.random.randint(100, 200), dtype=np.uint8)

            lb = ExactLetterboxTransform(
                img_size=self.img_size, center=True, pad_value=self.pad_value, scaleup=False
            )

            n = np.random.randint(0, 5)
            xyxy_abs = np.zeros((n, 4), dtype=np.float32)
            cls = np.zeros((n, ), dtype=np.int64)
            if n > 0:
                xyxy_abs[:, 0] = np.random.randint(0, w // 2, size=(n, ))
                xyxy_abs[:, 1] = np.random.randint(0, h // 2, size=(n, ))
                xyxy_abs[:, 2] = np.random.randint(w // 2, w, size=(n, ))
                xyxy_abs[:, 3] = np.random.randint(h // 2, h, size=(n, ))
                cls[:] = np.random.randint(0, self.num_classes, size=(n, ))

            canvas, yolo_norm, (oh, ow), ratio, pad = lb(image, xyxy_abs, cls)

            img_t = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().to(torch.uint8)
            lab_t = torch.from_numpy(yolo_norm).to(torch.float32) if yolo_norm.size else \
                    torch.zeros((0, 5), dtype=torch.float32)

            return img_t, lab_t, (oh, ow), ratio, pad

    ds = DummyDataset(img_size, num_classes, length)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn
    )
    info = {
        "nc": num_classes,
        "names": [f"class_{i}" for i in range(num_classes)],
        "format": "dummy",
        "train_images": length,
        "val_images": length,
    }
    return loader, info
