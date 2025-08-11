from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile

def check_train_batch_size(model, imgsz=640, amp=True):
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)

def autobatch(model, imgsz=640, fraction=0.85, batch_size=16):
    """
    Heuristic GPU RAM-based autobatch search.
    Ensures dummy inputs and model live on the same device/dtype to avoid CUDA/CPU mismatches.
    Falls back cleanly if profiling fails.
    """
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for --imgsz {imgsz}")

    # Resolve device & dtype from the model
    try:
        p0 = next(model.parameters())
        device = p0.device
        dtype = p0.dtype
    except StopIteration:
        # No params (edge case): default to CUDA:0 if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

    # CPU or cuDNN benchmark guardrails
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(
            f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}"
        )
        return batch_size

    # GPU memory snapshot
    gb = 1 << 30
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / gb
    reserved_gb = torch.cuda.memory_reserved(device) / gb
    allocated_gb = torch.cuda.memory_allocated(device) / gb
    free_gb = total_gb - (reserved_gb + allocated_gb)
    LOGGER.info(
        f"{prefix}CUDA:{device.index if device.index is not None else 0} "
        f"({props.name}) {total_gb:.2f}G total, {reserved_gb:.2f}G reserved, "
        f"{allocated_gb:.2f}G allocated, {free_gb:.2f}G free"
    )

    # Candidate batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Build dummy inputs on the SAME device/dtype as the model
    imgsz = imgsz if isinstance(imgsz, int) else int(imgsz[0])
    imgs = [torch.empty(b, 3, imgsz, imgsz, device=device, dtype=dtype) for b in batch_sizes]

    # Profile memory vs batch size
    results = []
    try:
        results = profile(imgs, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f"{prefix}{e}")
        LOGGER.info(f"{prefix}Falling back to default batch-size {batch_size}")
        return batch_size

    # Extract memory usages (GB) from profile results
    mem_gb = [row[2] for row in results if row is not None]
    used_bs = batch_sizes[:len(mem_gb)]

    if len(mem_gb) < 2:
        # Not enough data to fit a line
        LOGGER.info(f"{prefix}Insufficient profile samples, using default batch-size {batch_size}")
        return batch_size

    # Fit linear model: mem = p0 * batch + p1
    try:
        p = np.polyfit(used_bs, mem_gb, deg=1)
    except Exception as e:
        LOGGER.warning(f"{prefix}polyfit failed: {e}")
        return batch_size

    # Estimate max batch that fits within 'fraction' of total memory
    # target_mem = reserved + allocated + mem(b) <= total * fraction
    target_free = total_gb * fraction - (reserved_gb + allocated_gb)
    est_b = int((target_free - p[1]) / max(p[0], 1e-9))  # guard divide-by-zero
    # Respect the last successful candidate if any profile failed mid-list
    if None in results:
        i = results.index(None)
        if est_b >= batch_sizes[i]:
            est_b = batch_sizes[max(i - 1, 0)]

    # Clamp & sanity
    if est_b < 1 or est_b > 1024:
        LOGGER.warning(
            f"{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}"
        )
        est_b = batch_size

    # Report final projected utilization
    proj_fraction = (np.polyval(p, est_b) + reserved_gb + allocated_gb) / max(total_gb, 1e-9)
    LOGGER.info(
        f"{prefix}Using batch-size {est_b} for {props.name} "
        f"{total_gb * proj_fraction:.2f}G/{total_gb:.2f}G ({proj_fraction * 100:.0f}%) ✅"
    )
    return est_b

