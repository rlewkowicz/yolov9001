import argparse
import os
import platform
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
import torch.nn as nn
import torch.ao.quantization as tq

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.experimental import attempt_load
from models.yolo import RN_DualDDetect
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
)
from utils.torch_utils import select_device, smart_inference_mode

def export_formats():
    x = [["ONNX", "onnx", ".onnx", True, True], ["PT", "pt", ".pt", True, False]]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

def try_export(inner_func):
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(
                f"{prefix} export success {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)"
            )
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure {dt.t:.1f}s: {e}")
            return None, None

    return outer_func

def _replace_modules_recursively(m):
    for n, c in list(m.named_children()):
        name = c.__class__.__name__
        if "FakeQuantize" in name or "Observer" in name or name in ("QuantStub", "DeQuantStub"):
            setattr(m, n, nn.Identity())
        else:
            _replace_modules_recursively(c)

def _prepare_model_for_onnx_export(src_model):
    m = deepcopy(src_model).eval()
    for mod in m.modules():
        if hasattr(mod, "set_observer_enabled"):
            mod.set_observer_enabled(False)
        if hasattr(mod, "set_fake_quant_enabled"):
            mod.set_fake_quant_enabled(False)
        if hasattr(mod, "observer_enabled"):
            setattr(mod, "observer_enabled", False)
        if hasattr(mod, "fake_quant_enabled"):
            setattr(mod, "fake_quant_enabled", False)
    _replace_modules_recursively(m)
    return m

@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    check_requirements("onnx")
    import onnx
    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    export_model = _prepare_model_for_onnx_export(model)
    for _, m in export_model.named_modules():
        if isinstance(m, RN_DualDDetect):
            m.export = True
            if not hasattr(m, "export_logits") or not m.export_logits:
                setattr(m, "export_logits", True)
            if not hasattr(m, "export_split") or not m.export_split:
                setattr(m, "export_split", True)

    use_model = export_model.cpu().eval()
    use_im = im.cpu()
    with torch.no_grad():
        y = use_model(use_im)

    def _flatten(o):
        if isinstance(o, (list, tuple)):
            out = []
            for t in o:
                out.extend(_flatten(t))
            return out
        elif torch.is_tensor(o):
            return [o]
        else:
            raise TypeError(f"Unexpected output type during export: {type(o)}")

    outs = _flatten(y)
    if len(outs) != 2:
        raise RuntimeError("Expected dual outputs (boxes, logits).")

    output_names = ["boxes", "logits"]
    dynamic_axes = {"images": {0: "batch", 2: "height", 3: "width"}}
    if dynamic:
        for name in output_names:
            dynamic_axes[name] = {0: "batch", 2: "anchors"}

    torch.onnx.export(
        use_model,
        use_im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if dynamic else None,
    )

    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)

    d = {
        "stride": int(max(model.stride)),
        "names": model.names,
        "head_activation": "logits",
        "outputs": len(output_names),
    }
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(
                ("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1")
            )
            import onnxsim
            LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "onnx-simplifier check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")

    return f, model_onnx

def _load_state_dict_only(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and hasattr(obj["model"], "state_dict"):
            return obj["model"].state_dict()
        if "ema" in obj and hasattr(obj["ema"], "state_dict"):
            return obj["ema"].state_dict()
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise RuntimeError("Unsupported checkpoint format")

def _quant_ready(model):
    import importlib
    import torch.ao.quantization as tq
    fqmod = importlib.import_module("torch.ao.quantization.fake_quantize")
    FakeQuantize = getattr(fqmod, "FakeQuantize", tuple())
    FusedMovingAvgObsFakeQuantize = getattr(fqmod, "FusedMovingAvgObsFakeQuantize", FakeQuantize)
    has_any_fq = 0
    has_eager_attrs = 0
    has_qconfig_any = False
    has_stubs = 0
    for m in model.modules():
        if isinstance(m, (FakeQuantize, FusedMovingAvgObsFakeQuantize)):
            has_any_fq += 1
        if hasattr(m, "weight_fake_quant") or hasattr(m, "activation_post_process"):
            has_eager_attrs += 1
        if getattr(m, "qconfig", None) is not None:
            has_qconfig_any = True
        if isinstance(m, (tq.QuantStub, tq.DeQuantStub)):
            has_stubs += 1
    return (has_any_fq > 0) or (has_eager_attrs > 0) or (has_qconfig_any and has_stubs > 0)

@try_export
def export_pt_fp32_weights(weights_path, file, prefix=colorstr("PT:")):
    sd = _load_state_dict_only(weights_path)
    out_fp32 = file.with_name(file.stem + "_weights.pt")
    torch.save(sd, out_fp32)
    return out_fp32, None

@try_export
def export_pt_int8_torchscript(model, file, device, prefix=colorstr("PT-INT8:")):
    torch.backends.quantized.engine = "fbgemm"
    m = deepcopy(model).to("cpu").eval()
    if not _quant_ready(m):
        raise RuntimeError(
            "Model is not quantization-ready. Prepare QAT/PTQ with observers/stubs before --int8 export."
        )
    try:
        for _, mod in m.named_modules():
            if isinstance(mod, RN_DualDDetect):
                if hasattr(mod, "export_logits"):
                    mod.export_logits = False
                if hasattr(mod, "export_split"):
                    mod.export_split = False
    except Exception:
        pass
    qmodel = tq.convert(m, inplace=False)
    ts = torch.jit.script(qmodel)
    out_ts = file.with_name(file.stem + "_int8.ts.pt")
    ts.save(str(out_ts))
    return out_ts, None

@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",
    weights=ROOT / "yolo.pt",
    imgsz=(640, 640),
    batch_size=1,
    device="cpu",
    include=("onnx", ),
    half=False,
    inplace=False,
    dynamic=False,
    simplify=True,
    opset=19,
    int8=False,
):
    t = time.time()
    include = [x.lower() for x in include]
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)

    device = select_device(device)
    if half:
        assert device.type != "cpu", "--half only compatible with GPU export, i.e. use --device 0"
        assert not dynamic, "--half not compatible with --dynamic"

    model = attempt_load(weights, device=device, inplace=True, fuse=True)

    gs = int(max(model.stride))
    imgsz = [check_img_size(x, gs) for x in imgsz]
    im = torch.zeros(batch_size, 3, *imgsz).to(device)

    model.eval()
    for _, m in model.named_modules():
        if isinstance(m, RN_DualDDetect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
            if not hasattr(m, "export_logits") or not m.export_logits:
                setattr(m, "export_logits", True)
            if not hasattr(m, "export_split") or not m.export_split:
                setattr(m, "export_split", True)

    for _ in range(2):
        y = model(im)
    if half:
        im, model = im.half(), model.half()

    shape = (y[0] if isinstance(y, (tuple, list)) else y).shape
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"
    )

    if "onnx" in include:
        export_onnx(model, im, file, opset, dynamic, simplify)
    if "pt" in include:
        if int8:
            export_pt_int8_torchscript(model, file, device)
        else:
            export_pt_fp32_weights(str(file), file)

    LOGGER.info(f"\nExport complete ({time.time() - t:.1f}s)")
    LOGGER.info(f"Results saved to {colorstr('bold', file.parent.resolve())}")
    LOGGER.info(f"Visualize:       https://netron.app")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path"
    )
    parser.add_argument("--weights", type=str, default=ROOT / "yolo.pt", help="model.pt path")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set Detect() inplace=True")
    parser.add_argument("--dynamic", default=False, action="store_true", help="ONNX: dynamic axes")
    parser.add_argument(
        "--simplify", default=True, action="store_true", help="ONNX: simplify model"
    )
    parser.add_argument("--opset", type=int, default=19, help="ONNX: opset version")
    parser.add_argument(
        "--include", nargs="+", default=["onnx"], help="export formats, e.g. --include pt onnx"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="when --include pt: export TorchScript INT8 instead of FP32 weights"
    )
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
