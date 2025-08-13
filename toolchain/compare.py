#!/usr/bin/env python3
# compare.py  — FP32 vs QDQ comparison for YOLO-style heads with split outputs
# Supports:
#   - ONNX with 1 output [N, 4+nc, S]  (concat)
#   - ONNX with 2 outputs: boxes [N,4,S], logits [N,nc,S]  (split)
#   - Cosine similarity on the raw heads
#   - Side-by-side Top-5 with Ultralytics NMS
#   - QDQ param extraction (final DQ on the *class/logits* output)
#   - Optional PT int8_calib.head comparison

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
import yaml
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.quant_utils import model_has_pre_process_metadata
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.spatial.distance import cosine

# ---- Ultralytics utils (assumed available) ----
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) in sys.path else FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    check_img_size,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.plots import Annotator, colors


# ======================= Small helpers =======================

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 1.0
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return 1.0 - cosine(a, b)


def print_side_by_side_table(rows, headers):
    col_widths = [
        max(len(str(row[i])) for row in rows + [headers]) + 2 for i in range(len(headers))
    ]
    line = "".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))


def get_providers(opt_device: str):
    avail = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in avail and opt_device != "cpu":
        return ["CUDAExecutionProvider"] + (["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else [])
    return ["CPUExecutionProvider"]


def head_mode_from_meta(session) -> str:
    """
    Returns 'logits' or 'probs' if metadata 'head_activation' is present.
    Defaults to 'logits'.
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map
        v = meta.get("head_activation", "")
        if isinstance(v, str) and v.lower().startswith("prob"):
            return "probs"
        if isinstance(v, str) and v.lower().startswith("logit"):
            return "logits"
    except Exception:
        pass
    return "logits"


def boxes_format_from_meta(session) -> str:
    """
    Returns 'xywh' or 'xyxy' if metadata 'boxes_format' is present. Defaults to 'xywh'.
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map
        v = meta.get("boxes_format", "")
        if isinstance(v, str) and v.lower().startswith("xyx"):
            return "xyxy"
        if isinstance(v, str) and v.lower().startswith("xyw"):
            return "xywh"
    except Exception:
        pass
    return "xywh"


def names_from_meta(session, fallback_nc=80):
    try:
        meta = session.get_modelmeta().custom_metadata_map
        n = meta.get("names", None)
        if n is None:
            return [str(i) for i in range(fallback_nc)]
        # try to eval dict/list string safely
        try:
            out = eval(n, {"__builtins__": {}}, {})
            if isinstance(out, dict):
                # ensure dense order 0..nc-1 for NMS pretty print
                nc = max(int(k) for k in out.keys()) + 1 if out else fallback_nc
                return [out.get(i, str(i)) for i in range(nc)]
            if isinstance(out, (list, tuple)):
                return list(out)
        except Exception:
            pass
    except Exception:
        pass
    return [str(i) for i in range(fallback_nc)]


# =================== Calibration & Quantization ===================

class ImageCalibrator:
    def __init__(self, calib_files, input_name, imgsz=(640, 640), stride=32):
        self.input_name = input_name
        self.imgsz = imgsz
        self.stride = stride
        self.files = calib_files
        self.iterator = iter(self.files)

    def get_next(self):
        try:
            img_path = next(self.iterator)
            loader = LoadImages(img_path, img_size=self.imgsz, stride=self.stride, auto=False)
            _, im, _, _, _ = next(iter(loader))
            if im.dtype != np.float32:
                im = im.astype(np.float32) / 255.0
            if len(im.shape) == 3:
                im = np.expand_dims(im, 0)
            return {self.input_name: im}
        except StopIteration:
            return None


def _screen_calib_images(model_path, candidate_image_paths, imgsz, providers):
    # simple health-screen: run once and drop images that cause errors
    sess = onnxruntime.InferenceSession(str(model_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    good = []
    loader = LoadImages(candidate_image_paths, img_size=imgsz, auto=False)
    for path, im, _, _, _ in loader:
        x = im.astype(np.float32) / 255.0
        if x.ndim == 3:
            x = x[None]
        try:
            _ = sess.run([o.name for o in sess.get_outputs()], {input_name: x})
            good.append(path)
        except Exception:
            pass
    LOGGER.info(f"[Calib screening] {len(good)}/{len(candidate_image_paths)} images passed.")
    return good


def quantize_qdq(fp32_onnx, data_yaml, imgsz, device, per_channel=True, calib="minmax", act_signed=False, force=False):
    fp32_path = Path(fp32_onnx)
    out_qdq = fp32_path.with_stem(f"{fp32_path.stem}_int8_qdq")
    if out_qdq.exists():
        if force:
            try:
                os.remove(out_qdq)
                LOGGER.info(f"--quantize specified: removed existing {out_qdq}")
            except Exception:
                pass
        else:
            return str(out_qdq)

    providers = get_providers(device)
    tmp_sess = onnxruntime.InferenceSession(str(fp32_path), providers=providers)
    input_name = tmp_sess.get_inputs()[0].name
    del tmp_sess

    # read calib list
    data_yaml_path = Path(data_yaml)
    with open(data_yaml_path, errors="ignore") as f:
        data_dict = yaml.safe_load(f)
    dataset_root = (data_yaml_path.parent / ".." / data_dict["path"]).resolve()
    calib_txt = dataset_root / data_dict["train"]
    with open(calib_txt) as f:
        candidates = [str((dataset_root / p).resolve()) for p in [line.strip() for line in f][:1000]]

    # screen images once
    good_imgs = _screen_calib_images(fp32_path, candidates, imgsz, providers)
    if not good_imgs:
        raise RuntimeError("No stable images found for calibration.")

    # pre-process if needed
    model_path_for_quant = fp32_path
    model = onnx.load(str(fp32_path))
    if not model_has_pre_process_metadata(model):
        preprocessed = fp32_path.with_stem(f"{fp32_path.stem}-preprocessed")
        quant_pre_process(
            input_model_path=str(fp32_path),
            output_model_path=str(preprocessed),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )
        model_path_for_quant = preprocessed

    tmp_q = fp32_path.with_stem(f"{fp32_path.stem}_int8_tmp")
    calib_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile
    }

    LOGGER.info("Running Quantization (QDQ)...")
    quantize_static(
        model_input=str(model_path_for_quant),
        model_output=str(tmp_q),
        calibration_data_reader=ImageCalibrator(good_imgs, input_name, imgsz=imgsz, stride=32),
        quant_format=QuantFormat.QDQ,
        activation_type=(QuantType.QInt8 if act_signed else QuantType.QUInt8),
        weight_type=QuantType.QInt8,
        per_channel=bool(per_channel),
        nodes_to_exclude=[],
        calibrate_method=calib_map.get(calib.lower(), CalibrationMethod.MinMax),
        extra_options={
            "ActivationSymmetric": bool(act_signed),
            "WeightSymmetric": True,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": True,
        },
    )

    if model_path_for_quant != fp32_path:
        try:
            os.remove(model_path_for_quant)
        except Exception:
            pass
    try:
        if out_qdq.exists():
            os.remove(out_qdq)
    except Exception:
        pass
    os.replace(tmp_q, out_qdq)
    return str(out_qdq)


# ==================== QDQ param extractor (class head) ====================

def _producer_map(g):
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    return prod


def get_output_qdq_params(model_path: str, prefer_output: str | None = None):
    """
    If the model has two outputs, we walk back from the *class/logits* one.
    If one output, we walk back from that.
    Returns scale/zero_point for the DequantizeLinear that feeds that output.
    """
    m = onnx.load(model_path)
    g = m.graph
    if not g.output:
        return None

    # choose output
    out_vi = None
    if prefer_output:
        for o in g.output:
            if o.name == prefer_output:
                out_vi = o
                break
    if out_vi is None:
        out_vi = g.output[0]

    prod = _producer_map(g)
    node = prod.get(out_vi.name, None)

    passthrough = {"Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Cast"}
    hops = 0
    while node is not None and node.op_type in passthrough and node.input and hops < 20:
        node = prod.get(node.input[0], None)
        hops += 1

    if node is None or node.op_type != "DequantizeLinear" or len(node.input) < 3:
        return None

    sc_name, zp_name = node.input[1], node.input[2]

    def find_init(name):
        for ini in g.initializer:
            if ini.name == name:
                return numpy_helper.to_array(ini)
        return None

    scale_arr = find_init(sc_name)
    zp_arr = find_init(zp_name)
    if scale_arr is None or zp_arr is None:
        return None

    axis = 1
    for a in node.attribute:
        if a.name == "axis":
            axis = int(a.i)

    return {
        "scale": float(scale_arr) if scale_arr.size == 1 else scale_arr.astype(np.float32).reshape(-1),
        "zero_point": int(zp_arr) if zp_arr.size == 1 else zp_arr.reshape(-1),
        "axis": axis,
        "zp_dtype": zp_arr.dtype,
    }


# ==================== Optional: read PT head calib ====================

def dq_from_pt(pt_path, expect_channels=None):
    """
    Reads ckpt['int8_calib']['head'] with keys: 'min', 'max', 'per_channel' (optional).
    Returns scale/zero_point computed from min/max (matching ONNX Q/DQ convention):
      scale = (max-min)/255 ; zp = round(-min/scale)
    """
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    calib = None
    if isinstance(ckpt, dict):
        calib = ckpt.get("int8_calib", None)
        if isinstance(calib, dict):
            calib = calib.get("head", None)
    if calib is None or ("min" not in calib or "max" not in calib):
        return None

    per_channel = bool(calib.get("per_channel", True))
    mins = calib.get("min")
    maxs = calib.get("max")

    if per_channel:
        mins = np.array(mins, dtype=np.float32).reshape(-1)
        maxs = np.array(maxs, dtype=np.float32).reshape(-1)
        if expect_channels is not None and mins.shape[0] != expect_channels:
            pass
        scale = (maxs - mins) / 255.0
        scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
        zp = np.round(-mins / scale).astype(np.int32)
        zp = np.clip(zp, -128, 255)
        return {"scale": scale, "zero_point": zp, "axis": 1}
    else:
        mins = float(mins if not isinstance(mins, (list, tuple)) else mins[0])
        maxs = float(maxs if not isinstance(maxs, (list, tuple)) else maxs[0])
        sc = (maxs - mins) / 255.0 if (maxs - mins) > 1e-12 else 1.0
        zp = int(np.clip(round(-mins / sc), -128, 255))
        return {"scale": float(sc), "zero_point": int(zp), "axis": 1}


def print_pt_vs_qdq_qparams(pt_dq, qdq_params, names, max_classes_to_show=10):
    if pt_dq is None:
        print("\n[PT calib] No PT int8_calib.head found (pass --pt to enable).")
        return
    if qdq_params is None:
        print("\n[QDQ] No final DequantizeLinear found at output.")
        return

    nc = len(names)

    sc_q = qdq_params["scale"]
    zp_q = qdq_params["zero_point"]
    if isinstance(sc_q, np.ndarray):
        sc_q = sc_q.reshape(-1)
    if isinstance(zp_q, np.ndarray):
        zp_q = zp_q.reshape(-1)

    def slice_cls(arr, default):
        if isinstance(arr, np.ndarray):
            if arr.size >= (nc + 4):  # expect [C] = 4 box + nc cls
                return arr[-nc:]
            elif arr.size == nc:
                return arr
            else:
                return np.full((nc,), default, dtype=np.float32)
        else:
            return np.full((nc,), float(arr), dtype=np.float32)

    cls_sc_qdq = slice_cls(sc_q, float(sc_q) if not isinstance(sc_q, np.ndarray) else 1.0)
    cls_zp_qdq = slice_cls(zp_q, int(zp_q) if not isinstance(zp_q, np.ndarray) else 0)

    sc_p = pt_dq["scale"]
    zp_p = pt_dq["zero_point"]
    cls_sc_pt = slice_cls(sc_p, float(sc_p) if not isinstance(sc_p, np.ndarray) else 1.0)
    cls_zp_pt = slice_cls(zp_p, int(zp_p) if not isinstance(zp_p, np.ndarray) else 0)

    print(f"\nPT vs QDQ qparams for class logits (first {max_classes_to_show}):")
    print(f"{'cls':<4} {'sc_pt':>10} {'zp_pt':>7} {'sc_qdq':>10} {'zp_qdq':>8}")
    print("-" * 50)
    for k in range(min(nc, max_classes_to_show)):
        print(f"{k:<4d} {float(cls_sc_pt[k]):10.6g} {int(cls_zp_pt[k]):7d} "
              f"{float(cls_sc_qdq[k]):10.6g} {int(cls_zp_qdq[k]):8d}")


# ========================= IO & postprocess =========================

def prepare_input_for_session(session, source, imgsz):
    loader = LoadImages(source, img_size=imgsz, auto=False)
    _, im, _, _, _ = next(iter(loader))
    x = im.astype(np.float32) / 255.0
    if x.ndim == 3:
        x = x[None]
    return x, im  # model_input (NCHW float), original BGR


def _to_cfirst(a: np.ndarray) -> np.ndarray:
    """Return [N, C, S] given [N, C, S] or [N, S, C]."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 3:
        return a.reshape(a.shape[0], a.shape[1], -1)
    return a if a.shape[1] < a.shape[2] else np.transpose(a, (0, 2, 1))


def run_head_split(session, x):
    """
    Strict YOLO split: 2 outputs (boxes [N,4,S], logits [N,nc,S]).
    """
    outs = session.get_outputs()
    names = [o.name for o in outs]
    ys = session.run(names, {session.get_inputs()[0].name: x})
    ys = [np.asarray(y, dtype=np.float32) for y in ys]
    ys = [_to_cfirst(y) for y in ys]  # -> [N, C, S]

    # Identify boxes/logits by C
    idx_boxes = next((i for i, a in enumerate(ys) if a.shape[1] == 4), None)
    if idx_boxes is None:
        raise RuntimeError("Two outputs present but no boxes tensor with C==4 found.")
    idx_logits = 1 - idx_boxes
    boxes, logits = ys[idx_boxes], ys[idx_logits]
    return boxes, logits


def run_head_concat(session, x, nc_hint: int | None):
    """
    Single output concat: [N, 4+nc, S] or [N, S, 4+nc].
    We use nc_hint (from metadata names) to split reliably.
    """
    y = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: x})[0]
    y = _to_cfirst(y)  # [N, C, S]
    if nc_hint is None:
        C = y.shape[1]
        if C <= 4:
            raise RuntimeError(f"Single-output head has C={C} (<=4). Cannot split.")
        nc_hint = C - 4
    boxes = y[:, :4, :]
    logits = y[:, -nc_hint:, :]
    return boxes.astype(np.float32), logits.astype(np.float32)


def run_head(session, x, names: list[str]):
    """
    Always returns (boxes, logits) in C-first form:
      boxes  -> [N, 4,  S]  (xywh center, pixels)
      logits -> [N, nc, S]  (raw logits unless model exported probs)
    """
    outs = session.get_outputs()
    if len(outs) == 2:
        return run_head_split(session, x)
    else:
        nc_hint = len(names) if isinstance(names, (list, tuple)) else None
        return run_head_concat(session, x, nc_hint)


def assemble_for_nms(boxes_cfirst: np.ndarray,
                     logits_cfirst: np.ndarray,
                     mode: str = "logits",
                     boxes_format: str = "xywh") -> torch.Tensor:
    """
    Convert (boxes, logits) -> tensor for Ultralytics NMS:
      input:  boxes  [N, 4,  S]  (xywh center or xyxy as per boxes_format)
              logits [N, nc, S]
      output: torch.FloatTensor [N, S, 4+nc],
              boxes in XYWH(center), classes as PROBABILITIES.
    """
    b = np.asarray(boxes_cfirst, dtype=np.float32)
    c = np.asarray(logits_cfirst, dtype=np.float32)

    # Ensure probs
    if mode == "logits":
        c = 1.0 / (1.0 + np.exp(-c))

    # If boxes are xyxy, convert to xywh(center)
    if str(boxes_format).lower().startswith("xyx"):
        x1, y1, x2, y2 = b[:, 0, :], b[:, 1, :], b[:, 2, :], b[:, 3, :]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w  = (x2 - x1)
        h  = (y2 - y1)
        b = np.stack([cx, cy, w, h], axis=1)

    pred_cfirst = np.concatenate([b, c], axis=1)         # [N, 4+nc, S]
    pred = np.transpose(pred_cfirst, (0, 2, 1))          # [N, S, 4+nc]
    return torch.from_numpy(pred).float()


def run_one(session, im_in, im0, names, mode, boxes_format, conf_thres, iou_thres, max_det, save_path):
    """
    End-to-end on one image. Returns:
      - flattened head (for cosine), assembled as [N, 4+nc, S]
      - top-5 summary (list)
    """
    boxes, logits = run_head(session, im_in, names)              # C-first
    head_flat = np.concatenate([boxes, logits], axis=1)          # [N, 4+nc, S] for cosine

    # NMS tensor
    pred_t = assemble_for_nms(boxes, logits, mode=mode, boxes_format=boxes_format)

    # NMS
    dets = non_max_suppression(pred_t, conf_thres, iou_thres, max_det=max_det)

    # Draw Top-5
    im_draw = im0.copy()
    annotator = Annotator(im_draw, line_width=3, example=str(names))
    top5 = []

    def name_for(cls_index: int):
        if isinstance(names, dict):
            return names.get(cls_index, str(cls_index))
        elif isinstance(names, (list, tuple)):
            return names[cls_index] if 0 <= cls_index < len(names) else str(cls_index)
        else:
            return str(cls_index)

    for det in dets:
        if len(det):
            model_shape = (im_in.shape[2], im_in.shape[3])  # (H, W)
            det[:, :4] = scale_boxes(model_shape, det[:, :4], im_draw.shape).round()
            det_sorted = det[det[:, 4].argsort(descending=True)]
            for j, (*xyxy, conf, cls) in enumerate(det_sorted[:5]):
                c = int(cls)
                label = f"{name_for(c)} {float(conf):.3f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
                top5.append((name_for(c), float(conf), [float(k) for k in xyxy]))

    try:
        cv2.imwrite(str(save_path), annotator.result())
    except Exception:
        # Some OpenCV builds throw for large canvas; ignore but keep going
        pass

    return head_flat, top5


def dump_logits_vs_qparams(label, head_cfirst, names, qdq_params, max_classes_to_show=10):
    """
    head_cfirst: [N, 4+nc, S], where last nc are class logits (pre-sigmoid).
    """
    a = head_cfirst.reshape(1, head_cfirst.shape[1], -1).astype(np.float32)
    nc = len(names)
    cls = a[:, -nc:, :].reshape(nc, -1)  # [nc, S]

    if qdq_params is None:
        print(f"\n{label}: No QDQ params found (final DQ not detected).")
        return

    sc = qdq_params["scale"]
    zp = qdq_params["zero_point"]
    zp_dtype = qdq_params.get("zp_dtype", None)

    # Prepare per-class scales/zp
    if isinstance(sc, np.ndarray) and sc.size >= (a.shape[1]):  # per-channel [C]
        cls_sc = sc[-nc:]
        cls_zp = zp[-nc:] if isinstance(zp, np.ndarray) else np.full_like(cls_sc, int(zp))
    else:
        s = float(sc) if not isinstance(sc, np.ndarray) else float(sc.reshape(()))
        z = int(zp) if not isinstance(zp, np.ndarray) else int(zp.reshape(()))
        cls_sc = np.full((nc,), s, dtype=np.float32)
        cls_zp = np.full((nc,), z, dtype=np.int32)

    # qmin/qmax bounds by dtype
    if zp_dtype == np.uint8:
        qmin_q, qmax_q = 0, 255
    elif zp_dtype == np.int8:
        qmin_q, qmax_q = -128, 127
    else:
        qmin_q, qmax_q = 0, 255

    print(f"\nClass logits vs QDQ qparams ({label}, per-channel first {max_classes_to_show}):")
    print(f"{'cls':<4} {'mean':>9} {'min':>10} {'max':>10} {'scale':>9} {'zp':>5} {'qmin':>9} {'qmax':>9}")
    print("-" * 70)
    for k in range(min(nc, max_classes_to_show)):
        v = cls[k, :]
        mean_v = float(v.mean())
        min_v = float(v.min())
        max_v = float(v.max())
        qmin = float((qmin_q - int(cls_zp[k])) * float(cls_sc[k]))
        qmax = float((qmax_q - int(cls_zp[k])) * float(cls_sc[k]))
        print(f"{k:<4d} {mean_v:9.4f} {min_v:10.4f} {max_v:10.4f} {float(cls_sc[k]):9.6g} {int(cls_zp[k]):5d} {qmin:9.3f} {qmax:9.3f}")


# ============================== Main flow ==============================

def compare_models(opt):
    fp32_path = Path(opt.weights)
    int8_path_obj = fp32_path.with_stem(f"{fp32_path.stem}_int8_qdq")
    if opt.quantize or not int8_path_obj.exists():
        int8_path = quantize_qdq(
            fp32_onnx=str(fp32_path),
            data_yaml=opt.data,
            imgsz=opt.imgsz,
            device=opt.device,
            per_channel=opt.per_channel,
            calib=opt.calib,
            act_signed=opt.act_signed,
            force=opt.quantize,
        )
    else:
        int8_path = str(int8_path_obj)
        LOGGER.info(f"Found existing INT8 model at {int8_path}, skipping quantization.")

    providers = get_providers(opt.device)
    sess_fp32 = onnxruntime.InferenceSession(str(fp32_path), providers=providers)
    sess_qdq = onnxruntime.InferenceSession(str(int8_path), providers=providers)

    # names & nc
    names = names_from_meta(sess_fp32)

    # postprocess modes + boxes formats
    mode_fp32 = head_mode_from_meta(sess_fp32)        # default logits
    mode_qdq = head_mode_from_meta(sess_qdq)          # default logits
    boxes_fp32 = boxes_format_from_meta(sess_fp32) or opt.boxes
    boxes_qdq  = boxes_format_from_meta(sess_qdq) or opt.boxes

    # inputs
    im_in, im0 = prepare_input_for_session(sess_fp32, opt.source, opt.imgsz)
    im_in_qdq, _ = prepare_input_for_session(sess_qdq, opt.source, opt.imgsz)

    # run & compare
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = Path(opt.source).stem if not Path(opt.source).is_dir() else "image"

    head_fp32_cfirst, top5_fp32 = run_one(
        sess_fp32, im_in, im0, names, mode_fp32, boxes_fp32,
        opt.conf_thres, opt.iou_thres, opt.max_det,
        save_dir / f"{base}_fp32.jpg"
    )
    head_qdq_cfirst, top5_qdq = run_one(
        sess_qdq, im_in_qdq, im0.copy(), names, mode_qdq, boxes_qdq,
        opt.conf_thres, opt.iou_thres, opt.max_det,
        save_dir / f"{base}_int8.jpg"
    )

    # cosine similarity (heads flattened consistently)
    cs = safe_cosine(head_fp32_cfirst, head_qdq_cfirst)
    print(f"\nCosine similarity (FP32 head vs QDQ head): {cs:.6f}")
    print(f"Postprocess mode: FP32={mode_fp32} | QDQ={mode_qdq}")
    print(f"Boxes format: FP32={boxes_fp32} | QDQ={boxes_qdq}\n")

    # top-5 table
    rows = []
    for i in range(max(len(top5_fp32), len(top5_qdq))):
        lf = top5_fp32[i] if i < len(top5_fp32) else ("-", "-", "-")
        ri = top5_qdq[i] if i < len(top5_qdq) else ("-", "-", "-")
        rows.append((
            i + 1,
            lf[0], f"{lf[1]:.3f}" if isinstance(lf[1], float) else "-", str(lf[2]),
            ri[0], f"{ri[1]:.3f}" if isinstance(ri[1], float) else "-", str(ri[2]),
        ))
    print("Top-5 detections (side-by-side):")
    print_side_by_side_table(
        rows,
        ["Rank", "FP32 class", "FP32 conf", "FP32 box [xyxy]", "QDQ class", "QDQ conf", "QDQ box [xyxy]"]
    )
    print(f"\nSaved: {save_dir / (base + '_fp32.jpg')} and {save_dir / (base + '_int8.jpg')}")

    # Dump class logits vs qparams (walk back from the logits output if present)
    prefer_logits_out = None
    try:
        # Try to find an output named exactly "logits"
        for o in onnx.load(str(int8_path)).graph.output:
            if o.name.lower() == "logits":
                prefer_logits_out = o.name
                break
    except Exception:
        pass

    qdq_params = get_output_qdq_params(int8_path, prefer_output=prefer_logits_out)
    dump_logits_vs_qparams("FP32 head", head_fp32_cfirst, names, qdq_params, max_classes_to_show=10)
    dump_logits_vs_qparams("QDQ head", head_qdq_cfirst, names, qdq_params, max_classes_to_show=10)

    # Optional: PT calib and compare
    pt_path = Path(opt.pt) if opt.pt else fp32_path.with_suffix(".pt")
    if pt_path and pt_path.exists():
        pt_dq = dq_from_pt(str(pt_path), expect_channels=None)
        if pt_dq is not None:
            print_pt_vs_qdq_qparams(pt_dq, qdq_params, names, max_classes_to_show=10)
        else:
            print("\n[PT calib] Found PT file but no int8_calib.head {min,max} inside.")
    else:
        if opt.pt:
            print(f"\n[PT calib] PT file not found: {opt.pt}")


def parse_opt():
    p = argparse.ArgumentParser(description="FP32 vs QDQ INT8 comparison with YOLO split heads")
    p.add_argument("--weights", type=str, required=True, help="Path to FP32 ONNX")
    p.add_argument("--source", type=str, required=True, help="Path to a single image")
    p.add_argument("--data", type=str, default=str(ROOT / "toolchain/data/coco.yaml"), help="dataset.yaml for calibration")
    p.add_argument("--imgsz", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    p.add_argument("--device", default="cpu", help="CUDA device, e.g. 0 or cpu")
    p.add_argument("--quantize", action="store_true", help="force re-quantization (delete & rebuild *_int8_qdq.onnx)")
    p.add_argument("--per-channel", action="store_true", default=True, help="per-channel weight quantization")
    p.add_argument("--calib", type=str, default="minmax", choices=["minmax", "entropy", "percentile"])
    p.add_argument("--conf-thres", type=float, default=0.1, help="confidence threshold")
    p.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300, help="max detections per image")
    p.add_argument("--project", default=ROOT / "runs/compare", help="save dir")
    p.add_argument("--name", default="exp", help="subdir name")
    p.add_argument("--act-signed", action="store_true", help="use signed int8 activations (symmetric)")
    p.add_argument("--boxes", type=str, default="xywh", choices=["xywh", "xyxy"], help="format of exported boxes (fallback if no meta)")
    p.add_argument("--pt", type=str, default=None, help="(optional) path to training .pt with int8_calib.head to compare qparams")
    opt = p.parse_args()
    opt.imgsz = check_img_size(opt.imgsz)
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    compare_models(opt)
