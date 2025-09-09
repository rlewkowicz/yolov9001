"""
core/runtime.py

Functions for attaching and managing model runtime attributes.
"""
import torch

def attach_runtime(model, imgsz=640):
    """
    Calibrates and attaches critical runtime attributes to the model.
    This ensures a single source of truth for parameters like strides, nc, etc.,
    which are then read by other components like the loss function and dataloader.

    Args:
        model (torch.nn.Module): The model to attach attributes to.
        imgsz (int): The image size to use for stride calibration.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # vulture: ignore[unused-attribute] (global backend flag)
    torch.backends.cudnn.allow_tf32 = True  # vulture: ignore[unused-attribute] (global backend flag)
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    detect = None
    if hasattr(model, 'layers'):
        for m in model.layers:
            if m.__class__.__name__ in ('Detect', 'MockDetect'):
                detect = m
                break
    if detect is None:
        raise RuntimeError("Detect layer not found on model")

    if not hasattr(detect, 'strides') or detect.strides is None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
            _ = model(dummy)

        if was_training:
            model.train()

        if not hasattr(detect, 'last_shapes') or not detect.last_shapes:
            raise RuntimeError(
                "Detect.last_shapes not populated after dry run. Cannot infer strides."
            )

        shapes = detect.last_shapes
        strides = [imgsz // s[0] for s in shapes]
        detect.strides = torch.tensor(
            strides, device=next(model.parameters()).device, dtype=torch.float32
        )

    model.nc = int(getattr(model, 'nc', getattr(detect, 'nc', 80)))
    model.reg_max = int(getattr(detect, 'reg_max', 16))
    model.strides = torch.as_tensor(detect.strides, device=next(model.parameters()).device)
    model.detect_layer = detect

    from utils.geometry import DFLDecoder
    from core.config import get_config
    config = get_config(hyp=model.hyp)
    dfl_tau = config.get('dfl_tau', 1.0)

    if not hasattr(model, 'dfl_decoder') or model.dfl_decoder is None:
        model.dfl_decoder = DFLDecoder(
            reg_max=model.reg_max,
            strides=model.strides.tolist(),
            device=next(model.parameters()).device,
            tau=dfl_tau,
        )
    else:
        model.dfl_decoder.reg_max = model.reg_max
        model.dfl_decoder.strides = model.strides
        model.dfl_decoder.tau = dfl_tau

    from utils.postprocess import Postprocessor
    model.postprocessor = Postprocessor(
        cfg=config,
        nc=model.nc,
        device=next(model.parameters()).device,
        model=model,
        decoder=model.dfl_decoder
    )
    try:
        setattr(model, 'config_obj', config)
    except Exception:
        pass
