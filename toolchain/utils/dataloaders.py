# utils/dataloaders.py

import contextlib
import glob
import hashlib
import os
import random
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image
from torch.utils.data import Dataset, distributed
from tqdm import tqdm

from utils.augmentations import (
    Albumentations,
    augment_hsv,
    copy_paste,
    letterbox,
    mixup,
    random_perspective,
)
from utils.general import (
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    cv2,
    segments2boxes,
    xywhn2xyxy,
    xyxy2xywhn,
)
from utils.torch_utils import torch_distributed_zero_first

IMG_FORMATS = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm")
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))

# EXIF Orientation key
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break

# --- TurboJPEG (keep both FASTUPSAMPLE + FASTDCT flags) ---
try:
    from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

    _JPEG_TURBO = TurboJPEG()
    LOGGER.info("TurboJPEG enabled for faster JPG decoding.")
except ImportError:
    _JPEG_TURBO = None


def get_hash(paths: List[str]) -> str:
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.md5(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def exif_size(img: Image.Image) -> Tuple[int, int]:
    s = img.size
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:
            s = (s[1], s[0])
    return s


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _free_ram_fraction() -> float:
    mem = psutil.virtual_memory()
    return float(mem.available) / float(mem.total) if mem.total else 0.0


def _resolve_cache_mode(cache, prefix: str):
    """
    Normalize the cache mode: False | 'ram' | 'disk' | 'hybrid'
    Behavior:
      - For any mode that uses RAM ('ram' or 'hybrid'), if free RAM < 30% at the start of this pass,
        fall back to 'disk'.
      - Validation/Test are NOT forced to disk; they follow the same rule as train.
      - True is treated like 'ram'.
    """
    pref = (prefix or "").lower()
    is_train = "train" in pref

    # Normalize input
    mode = cache
    if mode is True:
        mode = "ram"
    if isinstance(mode, str):
        mode = mode.lower()
        if mode not in ("ram", "disk", "hybrid"):
            mode = False

    # RAM guard
    if mode in ("ram", "hybrid"):
        frac = _free_ram_fraction()
        if frac < 0.30:
            LOGGER.warning(
                f"{prefix}Free RAM < 30% ({frac:.1%}), falling back to disk cache for this pass."
            )
            mode = "disk"
        else:
            LOGGER.info(f"{prefix}Using {mode} cache (free RAM {frac:.1%}).")
    elif mode == "disk":
        LOGGER.info(f"{prefix}Using disk cache.")
    else:
        LOGGER.info(f"{prefix}No image caching.")

    return mode


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,  # False | True | 'ram' | 'disk' | 'hybrid'
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    prefix="",
    shuffle=False,
    **_ignored,  # allow compatibility args from older callsites (image_weights, quad, etc.)
):
    """
    Returns (InfiniteDataLoader, dataset)
    Accepts extra kwargs to stay compatible with legacy callers (ignored).
    """
    cache_mode = _resolve_cache_mode(cache, prefix)

    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,                # respect caller; your training uses fixed-size via letterbox(auto=False)
            cache_images=cache_mode,  # normalized: False|'ram'|'disk'|'hybrid'
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            prefix=prefix,
        )

    n = len(dataset)
    if batch_size < 1:
        batch_size = 1

    # make batch divide dataset length if possible (keeps epoch steps stable)
    if n % batch_size != 0:
        target = batch_size
        best = None
        delta = 0
        while True:
            down = target - delta
            up = target + delta
            picked = None
            if down >= 1 and n % down == 0:
                picked = down
            if up <= n and n % up == 0:
                if picked is None or abs(up - target) < abs(picked - target):
                    picked = up
            if picked is not None:
                best = picked
                break
            delta += 1
            if down < 1 and up > n:
                best = n
                break
        if best != batch_size:
            LOGGER.warning(
                f"Batch size {batch_size} does not evenly divide dataset length {n}. "
                f"Adjusting to nearest divisible size {best}."
            )
            batch_size = best

    batch_size = min(batch_size, n)
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
    )
    if nw > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    return InfiniteDataLoader(**loader_kwargs), dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImagesAndLabels(Dataset):
    cache_version = 0.8

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        cache_images=False,  # False | 'ram' | 'disk' | 'hybrid'
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
    ):
        self.img_size = int(img_size)
        self.augment = augment
        self.hyp = hyp or {}
        self.rect = rect
        self.mosaic = augment and not rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.pad = pad
        self.single_cls = single_cls
        self.albumentations = Albumentations(size=img_size) if augment else None

        self.im_files = self._get_image_files(prefix)
        self.label_files = [
            Path(x).with_suffix(".txt").as_posix().replace("images", "labels") for x in self.im_files
        ]

        cache_path = (
            Path(path).with_suffix(".cache")
            if isinstance(path, str)
            else Path(self.im_files[0]).parent.with_suffix(".cache")
        )
        self._cache_labels(cache_path, prefix)

        self.n = len(self.im_files)
        self.indices = range(self.n)

        self.ims: List[Union[np.ndarray, None]] = [None] * self.n
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache_images:
            self._init_image_cache(cache_images, prefix)

    # ---------- file discovery / labels cache ----------

    def _get_image_files(self, prefix):
        try:
            f = []
            for p in self.path if isinstance(self.path, list) else [self.path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        lines = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in lines]
                else:
                    raise FileNotFoundError(f"{prefix}{p} does not exist")
            im_files = sorted(
                [x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS]
            )
            assert im_files, f"{prefix}No images found"
            return im_files
        except Exception as e:
            raise Exception(f"{prefix}Error loading data from {self.path}: {e}")

    def _cache_labels(self, path, prefix):
        try:
            cache, exists = torch.load(path), True
            assert cache["version"] == self.cache_version
            assert cache["hash"] == get_hash(self.im_files + self.label_files)
        except Exception:
            cache, exists = self._create_label_cache(path, prefix), False

        nf, nm, ne, nc, n = cache.pop("results")
        if exists:
            LOGGER.info(
                f"Scanning '{path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            )

        assert nf > 0 or not self.augment, f"{prefix}No labels found in {path}, can not start training."

        [cache.pop(k) for k in ("hash", "version")]
        self.labels = [np.array(x["labels"], dtype=np.float32) for x in cache.values()]
        self.shapes = np.array([x["shape"] for x in cache.values()], dtype=np.float64)
        self.im_files = list(cache.keys())
        self.label_files = [
            Path(x).with_suffix(".txt").as_posix().replace("images", "labels") for x in self.im_files
        ]

        if self.single_cls:
            for i in range(len(self.labels)):
                if self.labels[i].size:
                    self.labels[i][:, 0] = 0

    def _create_label_cache(self, path, prefix):
        x = {}
        nm, nf, ne, nc = 0, 0, 0, 0
        pbar = tqdm(
            zip(self.im_files, self.label_files),
            desc=f"{prefix}Scanning images",
            total=len(self.im_files),
            bar_format=TQDM_BAR_FORMAT,
        )
        for im_file, lb_file in pbar:
            try:
                im = Image.open(im_file)
                im.verify()
                shape = exif_size(im)
                assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
                assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"

                l = []
                if os.path.exists(lb_file):
                    with open(lb_file) as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        if any([len(x) > 8 for x in l]):
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                        l = np.array(l, dtype=np.float32)
                    if l.size:
                        if l.ndim == 1:
                            if l.size == 5:
                                l = l.reshape(1, 5)
                            else:
                                l = np.zeros((0, 5), dtype=np.float32)
                        assert l.shape[1] == 5, "labels require 5 columns"
                        assert (l >= 0).all(), "negative labels"
                        assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate"
                        nf += 1
                    else:
                        l = np.zeros((0, 5), dtype=np.float32)
                        ne += 1
                else:
                    l = np.zeros((0, 5), dtype=np.float32)
                    nm += 1

                x[im_file] = {"labels": l, "shape": shape}
            except Exception as e:
                nc += 1
                LOGGER.warning(f"{prefix}WARNING ⚠️ Ignoring corrupt image/label: {im_file}: {e}")

        x["hash"] = get_hash(self.im_files + self.label_files)
        x["version"] = self.cache_version
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        torch.save(x, path)
        return x

    # ---------- caching ----------

    def _estimate_resized_bytes(self, i: int) -> int:
        """Estimate bytes after resizing to self.img_size, uint8 HxWx3."""
        h0, w0 = self.shapes[i]
        r = self.img_size / max(h0, w0)
        h = int(round(h0 * r))
        w = int(round(w0 * r))
        if h <= 0 or w <= 0:
            return 0
        return int(h * w * 3)  # uint8, 3 channels

    def _init_image_cache(self, cache_type: str, prefix: str):
        # If RAM-based mode but we already dipped under 30%, fall back to disk.
        if cache_type in ("ram", "hybrid") and _free_ram_fraction() < 0.30:
            LOGGER.warning(f"{prefix}Free RAM < 30% at dataset init; falling back to disk cache.")
            cache_type = "disk"

        if cache_type == "disk":
            results = ThreadPool(NUM_THREADS).imap(self.cache_image_to_disk, range(self.n))
            pbar = tqdm(
                enumerate(results),
                total=self.n,
                bar_format=TQDM_BAR_FORMAT,
                desc=f"{prefix}Caching images (disk)",
            )
            for _ in pbar:
                pass
            pbar.close()
            return

        if cache_type == "ram":
            results = ThreadPool(NUM_THREADS).imap(self._load_image_to_ram, range(self.n))
            pbar = tqdm(
                enumerate(results),
                total=self.n,
                bar_format=TQDM_BAR_FORMAT,
                desc=f"{prefix}Caching images (ram)",
            )
            for i, im in pbar:
                self.ims[i] = im
            pbar.close()
            return

        if cache_type == "hybrid":
            # Budget: use up to 50% of *current* free memory
            free_bytes = psutil.virtual_memory().available
            budget = int(0.50 * free_bytes)

            est_bytes = np.array([self._estimate_resized_bytes(i) for i in range(self.n)], dtype=np.int64)
            cum = est_bytes.cumsum()
            k = int((cum <= budget).sum())

            if k <= 0:
                LOGGER.warning(f"{prefix}Hybrid cache: budget too small for RAM, falling back to disk-only.")
                cache_type = "disk"
                results = ThreadPool(NUM_THREADS).imap(self.cache_image_to_disk, range(self.n))
                pbar = tqdm(
                    enumerate(results),
                    total=self.n,
                    bar_format=TQDM_BAR_FORMAT,
                    desc=f"{prefix}Caching images (disk)",
                )
                for _ in pbar:
                    pass
                pbar.close()
                return

            # First k -> RAM
            results_ram = ThreadPool(NUM_THREADS).imap(self._load_image_to_ram, range(k))
            pbar = tqdm(
                enumerate(results_ram),
                total=k,
                bar_format=TQDM_BAR_FORMAT,
                desc=f"{prefix}Caching images (hybrid-ram, ~{budget/1e9:.1f} GB)",
            )
            for i, im in pbar:
                self.ims[i] = im
            pbar.close()

            # Remaining -> disk
            if k < self.n:
                results_disk = ThreadPool(NUM_THREADS).imap(self.cache_image_to_disk, range(k, self.n))
                pbar2 = tqdm(
                    enumerate(results_disk),
                    total=(self.n - k),
                    bar_format=TQDM_BAR_FORMAT,
                    desc=f"{prefix}Caching images (hybrid-disk)",
                )
                for _ in pbar2:
                    pass
                pbar2.close()
            return

        # else: no caching
        return

    # ---------- Dataset protocol ----------

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp

        mosaic_flag = bool(self.mosaic and random.random() < hyp.get("mosaic", 0.0))

        if mosaic_flag:
            # Mosaic path
            img, labels = self._load_mosaic(index)  # already s x s
            h0, w0 = img.shape[:2]
            shapes = (h0, w0)  # training ignores shapes
        else:
            # Standard path
            img, (h0, w0), (h, w) = self.load_image(index)
            shape = self.img_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            labels = self.labels[index].copy()
            if labels.size:
                # to pixel xyxy with resize+pad applied
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1]
                )
            shapes = (h0, w0)

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp.get("degrees", 0.0),
                    translate=hyp.get("translate", 0.0),
                    scale=hyp.get("scale", 0.0),
                    shear=hyp.get("shear", 0.0),
                    perspective=hyp.get("perspective", 0.0),
                )

        # Normalize labels back to xywhn in [0,1], clip
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3
            )
            eps = 1e-6
            labels[:, 1:3] = np.clip(labels[:, 1:3], eps, 1.0 - eps)
            labels[:, 3:5] = np.clip(labels[:, 3:5], eps, 1.0)
            wh_ok = (labels[:, 3] > eps) & (labels[:, 4] > eps)
            labels = labels[wh_ok]

        # Albumentations (expects normalized xywh)
        if self.albumentations and len(labels):
            img, labels = self.albumentations(img, labels)
            if len(labels):
                labels[:, 1:3] = np.clip(labels[:, 1:3], 1e-6, 1.0 - 1e-6)
                labels[:, 3:5] = np.clip(labels[:, 3:5], 1e-6, 1.0)

        # Simple color/flip augs
        if self.augment:
            augment_hsv(img, hgain=hyp.get("hsv_h", 0.0), sgain=hyp.get("hsv_s", 0.0), vgain=hyp.get("hsv_v", 0.0))
            if random.random() < hyp.get("flipud", 0.0):
                img = np.flipud(img)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
            if random.random() < hyp.get("fliplr", 0.0):
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

        # Pack labels
        labels_out = torch.zeros((len(labels), 6), dtype=torch.float32)
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)

        # HWC -> CHW, BGR->RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        shapes_out = shapes  # (h0, w0)
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes_out

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    # ---------- image I/O ----------

    def _load_image_raw(self, path: str):
        # Use TurboJPEG if available (FASTUPSAMPLE | FASTDCT) to preserve previous speed behavior
        if _JPEG_TURBO and Path(path).suffix.lower() in (".jpg", ".jpeg"):
            with open(path, "rb") as f:
                return _JPEG_TURBO.decode(
                    f.read(), flags=(TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
                )
        return cv2.imread(path)

    def _load_image_to_ram(self, i: int):
        im = self._load_image_raw(self.im_files[i])
        assert im is not None, f"Image Not Found {self.im_files[i]}"
        r = self.img_size / max(im.shape[:2])
        if r != 1:
            im = cv2.resize(
                im,
                (int(im.shape[1] * r), int(im.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA,
            )
        return im

    def load_image(self, i: int):
        im = self.ims[i]
        if im is None:
            npy_f = self.npy_files[i]
            if npy_f.exists():
                im = np.load(npy_f)
            else:
                im = self._load_image_raw(self.im_files[i])
                assert im is not None, f"Image Not Found {self.im_files[i]}"

            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                im = cv2.resize(
                    im,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA,
                )
            return im, (h0, w0), im.shape[:2]
        return im, self.shapes[i], self.shapes[i]

    def cache_image_to_disk(self, i: int):
        f = self.npy_files[i]
        if not f.exists():
            im, _, _ = self.load_image(i)
            np.save(f.as_posix(), im)

    # ---------- mosaic (robust) ----------

    def _load_mosaic(self, index):
        """Robust 4-image mosaic with safe pasting to prevent broadcast mismatches."""
        labels4 = []
        s = int(self.img_size)
        yc = int(random.uniform(-s // 2, 2 * s + s // 2))
        xc = int(random.uniform(-s // 2, 2 * s + s // 2))

        indices = [index] + random.choices(self.indices, k=3)
        img4 = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)

        def safe_paste(dst, src, x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
            # ints
            x1a = int(x1a)
            y1a = int(y1a)
            x2a = int(x2a)
            y2a = int(y2a)
            x1b = int(x1b)
            y1b = int(y1b)
            x2b = int(x2b)
            y2b = int(y2b)

            # clip to dst/src bounds
            x1a = max(0, min(x1a, dst.shape[1]))
            x2a = max(0, min(x2a, dst.shape[1]))
            y1a = max(0, min(y1a, dst.shape[0]))
            y2a = max(0, min(y2a, dst.shape[0]))

            x1b = max(0, min(x1b, src.shape[1]))
            x2b = max(0, min(x2b, src.shape[1]))
            y1b = max(0, min(y1b, src.shape[0]))
            y2b = max(0, min(y2b, src.shape[0]))

            da_h = y2a - y1a
            da_w = x2a - x1a
            db_h = y2b - y1b
            db_w = x2b - x1b
            if da_h <= 0 or da_w <= 0 or db_h <= 0 or db_w <= 0:
                return None

            h = min(da_h, db_h)
            w = min(da_w, db_w)
            y2a = y1a + h
            x2a = x1a + w
            y2b = y1b + h
            x2b = x1b + w
            if h <= 0 or w <= 0:
                return None

            dst[y1a:y2a, x1a:x2a] = src[y1b:y2b, x1b:x2b]
            return (x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)

        for i, index in enumerate(indices):
            img, _, (h, w) = self.load_image(index)

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            else:  # i == 3, bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            pasted = safe_paste(img4, img, x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b)

            if pasted is None:
                padw = padh = 0
            else:
                x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b = pasted
                padw = x1a - x1b
                padh = y1a - y1b

            labels = self.labels[index].copy()
            if labels.ndim == 1:
                if labels.size == 0:
                    labels = labels.reshape(0, 5)
                elif labels.size == 5:
                    labels = labels.reshape(1, 5)
                else:
                    labels = np.zeros((0, 5), dtype=np.float32)
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        # concat/clip labels to [0, 2s]
        if len(labels4):
            labels4 = [lab if lab.ndim == 2 else np.zeros((0, 5), dtype=np.float32) for lab in labels4]
            labels4 = np.concatenate(labels4, 0)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        else:
            labels4 = np.zeros((0, 5), dtype=np.float32)

        # random crop back to s x s
        x_start = min(max(xc - s // 2, 0), 2 * s - s)
        y_start = min(max(yc - s // 2, 0), 2 * s - s)
        x_end = x_start + s
        y_end = y_start + s
        img_out = img4[y_start:y_end, x_start:x_end]

        if labels4.size:
            labels4[:, [1, 3]] -= x_start
            labels4[:, [2, 4]] -= y_start
            np.clip(labels4[:, 1:], 0, s, out=labels4[:, 1:])

        return img_out, labels4
