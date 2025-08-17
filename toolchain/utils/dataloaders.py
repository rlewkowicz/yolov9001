# dataloaders.py

import contextlib
import glob
import hashlib
import os
import random
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from PIL import Image, ExifTags
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import BatchAugmenter
from utils.general import (
    LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, segments2boxes, xywhn2xyxy
)
from utils.torch_utils import torch_distributed_zero_first

# --- Helper functions ---
IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'

def get_hash(paths):
    return hashlib.md5(''.join(paths).encode()).hexdigest()

def _imread_fast(path: str):
    try:
        im = Image.open(path).convert("RGB")
        return np.array(im)[:, :, ::-1] # to BGR
    except Exception:
        return cv2.imread(path)

def img2label_paths(img_paths):
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to a new shape for dataloader processing."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

# ##################################################################################################
# Collate Function for Batching and GPU Augmentation
# ##################################################################################################

class GPUCollate:
    def __init__(self, hyp, imgsz, augment=True):
        self.augment = augment
        self.imgsz = imgsz
        if self.augment:
            self.augmenter = BatchAugmenter(hyp, imgsz)
        
    def __call__(self, batch):
        # The batch is now a list of tuples, where the first element is a tensor
        im_tensors, labels, paths, segments = zip(*batch)
        
        # Stack the tensors received from the workers
        im_tensor = torch.stack(im_tensors, 0)
        
        # Format labels
        labels_out = []
        for i, label in enumerate(labels):
            if len(label) > 0:
                l = torch.from_numpy(label)
                new_labels = torch.zeros((l.shape[0], 6), dtype=l.dtype)
                new_labels[:, 0] = i
                new_labels[:, 1:] = l
                labels_out.append(new_labels)
        
        labels_tensor = torch.cat(labels_out, 0) if labels_out else torch.empty(0, 6)

        if self.augment:
            im_tensor, labels_tensor = self.augmenter(im_tensor, labels_tensor, segments)
            
        return im_tensor, labels_tensor, paths, None

# ##################################################################################################
# Infinite Dataloader (Restored from original)
# ##################################################################################################

class InfiniteDataLoader(dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
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

# ##################################################################################################
# RAM Manager (Restored from original)
# ##################################################################################################
class RamManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.mem = psutil.virtual_memory()
            self.total_available = self.mem.available
            self.allocated = 0
            self.initialized = True

    def request_ram_bytes(self, percentage: float) -> int:
        remaining = self.total_available - self.allocated
        request = int(remaining * percentage)
        self.allocated += request
        return request

# ##################################################################################################
# Main Dataset and DataLoader Creation
# ##################################################################################################

class LoadImagesAndLabels(Dataset):
    cache_version = 1.2 # Version bump for new __getitem__ return type

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, min_items=0, prefix='', ram_budget=0):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        # --- FIX: Restore original logic to resolve relative paths ---
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS])
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')
        
        self.label_files = img2label_paths(self.im_files)
        self.n = len(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
            assert cache['version'] == self.cache_version
        except Exception:
            cache = self.cache_labels(cache_path, prefix)

        self.labels = [cache[f][0] for f in self.im_files]
        self.segments = [cache[f][1] for f in self.im_files]
        self.indices = range(self.n)
        self.ims = [None] * self.n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache_images:
            if cache_images == 'hybrid':
                self.cache_images_hybrid(ram_budget, prefix)
            else:
                gb = 1 << 30
                fcn = self.cache_images_to_disk if cache_images == 'disk' else self.load_image_into_ram
                results = ThreadPool(NUM_THREADS).imap(fcn, range(self.n))
                pbar = tqdm(enumerate(results), total=self.n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
                total_bytes = 0
                for i, x in pbar:
                    if cache_images == 'disk':
                        if self.npy_files[i].exists():
                            total_bytes += self.npy_files[i].stat().st_size
                    else:
                        self.ims[i] = x
                        total_bytes += self.ims[i].nbytes
                    pbar.desc = f'{prefix}Caching images ({total_bytes / gb:.1f}GB {cache_images})'
                pbar.close()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        index = self.indices[index]
        if self.mosaic:
            img, labels, segments = self.load_mosaic(index)
        else:
            img = self.load_image(index)
            labels = self.labels[index].copy()
            segments = self.segments[index].copy()
        
        # Letterbox and convert to tensor inside the worker
        img = letterbox(img, self.img_size)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).flip(0).float() / 255.0

        return img_tensor, labels, self.im_files[index], segments

    def load_image(self, i):
        im = self.ims[i]
        if im is None:
            if self.npy_files[i].exists():
                im = np.load(self.npy_files[i])
            else:
                im = _imread_fast(self.im_files[i])
            assert im is not None, f'Image Not Found {self.im_files[i]}'
            self.ims[i] = im
        return im

    def load_mosaic(self, index):
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + random.choices(range(self.n), k=3)
        
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        for i, idx in enumerate(indices):
            img = self.load_image(idx)
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)

            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b

            labels = self.labels[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
                labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)
        labels4[:, 1:] = np.clip(labels4[:, 1:], 0, 2 * s)
        labels4[:, 1:] = xyxy2xywhn(labels4[:, 1:], w=s*2, h=s*2)
        return img4, labels4, []

    def cache_labels(self, path, prefix):
        x = {}
        pbar = tqdm(zip(self.im_files, self.label_files), desc=f'{prefix}Caching labels', total=self.n)
        for im_file, lb_file in pbar:
            try:
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                else:
                    lb = np.array(lb, dtype=np.float32); segments = []
                x[im_file] = [lb, segments]
            except Exception:
                x[im_file] = [np.empty((0, 5), dtype=np.float32), []]
        x['version'] = self.cache_version
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(path, x)
        return x

    def cache_images_hybrid(self, ram_budget, prefix):
        gb = 1 << 30
        self.shapes = np.array([s for s in [self._get_shape(f) for f in self.im_files] if s is not None])
        est_bytes_per_img = self.shapes.mean(0).prod() * 3
        total_est_bytes = est_bytes_per_img * self.n
        num_ram_cache = int(ram_budget / est_bytes_per_img) if est_bytes_per_img > 0 else 0
        ram_indices = random.sample(range(self.n), min(num_ram_cache, self.n))
        
        pbar = tqdm(range(self.n), total=self.n, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
        for i in pbar:
            if i in ram_indices:
                self.ims[i] = self.load_image_into_ram(i)
            else:
                self.cache_images_to_disk(i)
    
    def _get_shape(self, f):
        try: return Image.open(f).size
        except: return None
    
    def load_image_into_ram(self, i):
        return _imread_fast(self.im_files[i])

    def cache_images_to_disk(self, i):
        if not self.npy_files[i].exists():
            np.save(self.npy_files[i].as_posix(), _imread_fast(self.im_files[i]))

# ##################################################################################################
# create_dataloader function (Restored to original signature and structure)
# ##################################################################################################

def create_dataloader(
    path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False,
    pad=0.0, rect=False, rank=-1, workers=8, image_weights=False, close_mosaic=False,
    quad=False, min_items=0, prefix='', shuffle=False
):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False

    ram_budget = RamManager().request_ram_bytes(0.5) if cache == 'hybrid' else 0
    
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(
            path, imgsz, batch_size, augment=augment, hyp=hyp, rect=rect,
            image_weights=image_weights, cache_images=cache, single_cls=single_cls,
            stride=int(stride), pad=pad, min_items=min_items, prefix=prefix, ram_budget=ram_budget
        )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader_cls = DataLoader if image_weights or close_mosaic else InfiniteDataLoader
    
    collate_fn = GPUCollate(hyp=hyp, imgsz=imgsz, augment=augment)

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader_cls(
        dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, num_workers=nw,
        sampler=sampler, pin_memory=PIN_MEMORY, collate_fn=collate_fn,
        worker_init_fn=seed_worker, generator=generator
    ), dataset
