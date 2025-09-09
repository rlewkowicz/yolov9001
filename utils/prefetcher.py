"""
utils/prefetcher.py

CUDA prefetcher for overlapping data transfer with computation.
Also captures simple per-batch timing for CPU fetch and GPU augmentation.
"""
import time
import torch

class CUDAPrefetcher:
    """
    Asynchronously prefetch batches to a CUDA device with bounded GPU memory usage.

    - Uses a dedicated CUDA stream for copies/processing.
    - Maintains a small ring buffer of ready-on-GPU batches (up to max_prefetch_batches).
    - Never exceeds mem_fraction of total GPU memory (best-effort using mem_get_info).
    """
    def __init__(
        self, loader, device, amp=False, max_prefetch_batches: int = 3, mem_fraction: float = 0.8
    ):
        self.loader = loader
        self.iter = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.amp = amp
        self.max_prefetch = max(1, int(max_prefetch_batches))
        self.mem_fraction = float(mem_fraction)
        self.buffer = []  # list of dicts: {'batch': tuple, 'event': cudaEvent, 'timing': dict}
        self.done = False
        self.last_timing = None  # timing for the last batch returned by __next__  # vulture: ignore[unused-attribute]
        self._closed = False
        self._preload_fill()

    def _process_fetched(self, fetched, stats_on=False):
        """Move a fetched CPU batch to GPU and apply optional aug; returns (batch, timing dict, ready_event)."""
        imgs, targets, orig_shapes, ratios, pads = fetched
        with torch.cuda.stream(self.stream):
            if isinstance(imgs, torch.Tensor) and imgs.device.type == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)
            else:
                if imgs.dtype == torch.uint8:
                    imgs = imgs.to(self.device, non_blocking=True, dtype=torch.float32)
                    imgs.div_(255.0)
                else:
                    imgs = imgs.to(self.device, non_blocking=True)
                imgs = imgs.to(memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)

            evt_total_start = torch.cuda.Event(enable_timing=True) if stats_on else None
            evt_total_end = torch.cuda.Event(enable_timing=True) if stats_on else None
            evt_aug_start = torch.cuda.Event(enable_timing=True) if stats_on else None
            evt_aug_end = torch.cuda.Event(enable_timing=True) if stats_on else None
            if stats_on and evt_total_start is not None:
                evt_total_start.record(self.stream)
            try:
                ds = getattr(self.loader, 'dataset', None)
                hyp = getattr(ds, 'hyp', {}) if ds is not None else {}
                gpu_collate_active = bool(getattr(ds, 'gpu_collate', False))
                apply_aug = (not gpu_collate_active) and bool(hyp.get('augment', False)) \
                    and bool(getattr(ds, 'train_mode', False))
                if apply_aug:
                    from utils.augment import apply_gpu_affine_on_batch
                    if stats_on and evt_aug_start is not None:
                        evt_aug_start.record(self.stream)
                    pad_value = int(getattr(ds, 'pad_value', 114)) if ds is not None else 114
                    imgs, targets = apply_gpu_affine_on_batch(imgs, targets, hyp, pad_value)
                    if stats_on and evt_aug_end is not None:
                        evt_aug_end.record(self.stream)
            except Exception:
                pass
            if stats_on and evt_total_end is not None:
                evt_total_end.record(self.stream)

            ready_evt = torch.cuda.Event()
            ready_evt.record(self.stream)

        timing = None
        if stats_on:
            timing = {
                'gpu_total_ms_event': (evt_total_start, evt_total_end),
                'gpu_aug_ms_event': (evt_aug_start, evt_aug_end),
            }
        return (imgs, targets, orig_shapes, ratios, pads), timing, ready_evt

    def _gpu_usage_ok(self) -> bool:
        """Return True if current device memory usage is under the configured fraction."""
        try:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            used_frac = 1.0 - (float(free_b) / float(total_b))
            return used_frac < self.mem_fraction
        except Exception:
            try:
                total_b = torch.cuda.get_device_properties(self.device).total_memory
                reserved = torch.cuda.memory_reserved(self.device)
                return float(reserved) / float(total_b) < self.mem_fraction
            except Exception:
                return True  # if cannot query, don't block prefetching

    def _preload_fill(self):
        """Fill the buffer up to max_prefetch (or until memory threshold reached)."""
        while not self.done and len(self.buffer) < self.max_prefetch and self._gpu_usage_ok():
            ds = getattr(self.loader, 'dataset', None)
            hyp = getattr(ds, 'hyp', {}) if ds is not None else {}
            stats_on = bool(hyp.get('stats_enabled', False))

            cpu_fetch_t0 = time.perf_counter() if stats_on else None
            try:
                fetched = next(self.iter)
            except StopIteration:
                self.done = True
                break
            cpu_fetch_ms = (time.perf_counter() - cpu_fetch_t0) * 1000.0 if stats_on else None

            (imgs, targets, orig_shapes, ratios,
             pads), timing_events, ready_evt = self._process_fetched(fetched, stats_on=stats_on)

            timing = {
                'cpu_fetch_ms':
                    cpu_fetch_ms,
                'gpu_total_ms_event':
                    timing_events['gpu_total_ms_event'] if stats_on else None,
                'gpu_aug_ms_event':
                    timing_events['gpu_aug_ms_event'] if stats_on else None,
                'buffer_depth':
                    len(self.buffer) if stats_on else None,
                'h2d_bytes': (
                    fetched[0].numel() * fetched[0].element_size() if
                    (stats_on and isinstance(fetched[0], torch.Tensor) and
                     not fetched[0].is_cuda) else None
                ),
            } if stats_on else None
            self.buffer.append({
                'batch': (imgs, targets, orig_shapes, ratios, pads), 'event': ready_evt, 'timing':
                    timing
            })

    def _fetch_one_sync(self):
        """Fetch a single batch even if mem cap prevents prefetch; returns a ready-on-GPU batch."""
        try:
            fetched = next(self.iter)
        except StopIteration:
            self.done = True
            raise
        batch, _, ready_evt = self._process_fetched(fetched, stats_on=False)
        torch.cuda.current_stream(self.device).wait_event(ready_evt)
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        """Return one ready batch and keep preloading within memory limits."""
        if not self.buffer and self.done:
            raise StopIteration
        if not self.buffer:
            self._preload_fill()
            if not self.buffer:
                return self._fetch_one_sync()
        item = self.buffer.pop(0)
        torch.cuda.current_stream(self.device).wait_event(item['event'])
        timing = item.get('timing', None)
        cpu_ms = timing.get('cpu_fetch_ms') if timing else None
        gpu_total_ms = None
        gpu_aug_ms = None
        if timing:
            try:
                ts = timing.get('gpu_total_ms_event')
                if ts is not None:
                    gpu_total_ms = ts[0].elapsed_time(ts[1])
                ta = timing.get('gpu_aug_ms_event')
                if ta is not None:
                    gpu_aug_ms = ta[0].elapsed_time(ta[1])
            except Exception:
                pass
            self.last_timing = {  # vulture: ignore[unused-attribute]
                'cpu_fetch_ms': cpu_ms, 'gpu_total_ms': gpu_total_ms, 'gpu_aug_ms': gpu_aug_ms,
                'buffer_depth': timing.get('buffer_depth'), 'h2d_bytes': timing.get('h2d_bytes')
            }
        else:
            self.last_timing = None  # vulture: ignore[unused-attribute]

        self._preload_fill()
        return item['batch']

    def close(self):
        """Best-effort release of CUDA resources and references."""
        if getattr(self, "_closed", False):
            return
        try:
            self.last_timing = None  # vulture: ignore[unused-attribute]
            self.buffer.clear()
            self.iter = None
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            self.stream = None
        finally:
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
