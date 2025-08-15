#!/usr/bin/env python3

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
    print_args,
    scale_boxes,
)
from utils.plots import Annotator, colors

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
        return ["CUDAExecutionProvider"
               ] + (["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else [])
    return ["CPUExecutionProvider"]

def names_from_meta(session, fallback_nc=80):
    try:
        meta = session.get_modelmeta().custom_metadata_map
        n = meta.get("names", None)
        if n is None:
            return [str(i) for i in range(fallback_nc)]
        try:
            out = eval(n, {"__builtins__": {}}, {})
            if isinstance(out, dict):
                nc = max(int(k) for k in out.keys()) + 1 if out else fallback_nc
                return [out.get(i, str(i)) for i in range(nc)]
            if isinstance(out, (list, tuple)):
                return list(out)
        except Exception:
            pass
    except Exception:
        pass
    return [str(i) for i in range(fallback_nc)]

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

def quantize_qdq(
    fp32_onnx,
    data_yaml,
    imgsz,
    device,
    calib="minmax",
    act_signed=False,
    force=False,
):
    fp32_path = Path(fp32_onnx)
    out_qdq = fp32_path.with_stem(f"{fp32_path.stem}_int8_qdq")
    if out_qdq.exists() and not force:
        LOGGER.info(f"Found existing INT8 model at {out_qdq}, skipping quantization.")
        return str(out_qdq)

    if out_qdq.exists() and force:
        try:
            os.remove(out_qdq)
            LOGGER.info(f"--quantize specified: removed existing {out_qdq}")
        except Exception as e:
            LOGGER.warning(f"Could not remove existing quantized model: {e}")

    providers = get_providers(device)
    tmp_sess = onnxruntime.InferenceSession(str(fp32_path), providers=providers)
    input_name = tmp_sess.get_inputs()[0].name
    del tmp_sess

    data_yaml_path = Path(data_yaml)
    with open(data_yaml_path, errors="ignore") as f:
        data_dict = yaml.safe_load(f)
    dataset_root = (data_yaml_path.parent / ".." / data_dict["path"]).resolve()
    calib_txt = dataset_root / data_dict["train"]
    with open(calib_txt) as f:
        candidates = [
            str((dataset_root / p).resolve()) for p in [line.strip() for line in f][:1000]
        ]

    good_imgs = _screen_calib_images(fp32_path, candidates, imgsz, providers)
    if not good_imgs:
        raise RuntimeError("No stable images found for calibration. Check model and dataset.")

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
        "percentile": CalibrationMethod.Percentile,
    }

    LOGGER.info("Running Quantization (QDQ)...")
    quantize_static(
        model_input=str(model_path_for_quant),
        model_output=str(tmp_q),
        calibration_data_reader=ImageCalibrator(good_imgs, input_name, imgsz=imgsz, stride=32),
        quant_format=QuantFormat.QDQ,
        activation_type=(QuantType.QInt8 if act_signed else QuantType.QUInt8),
        weight_type=QuantType.QInt8,
        per_channel=True,
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

    os.replace(tmp_q, out_qdq)
    LOGGER.info(f"Quantization complete. Model saved to {out_qdq}")
    return str(out_qdq)

def _producer_map(g):
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    return prod

def get_output_qdq_params(model_path: str, prefer_output: str | None = None):
    m = onnx.load(model_path)
    g = m.graph
    if not g.output:
        return None

    out_vi = None
    if prefer_output:
        for o in g.output:
            if o.name == prefer_output:
                out_vi = o
                break
    if out_vi is None:
        if len(g.output) == 2:
            for o in g.output:
                if o.type.tensor_type.shape.dim[1].dim_value != 4:
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

    scale_arr, zp_arr = find_init(sc_name), find_init(zp_name)
    if scale_arr is None or zp_arr is None:
        return None

    axis = 1
    for a in node.attribute:
        if a.name == "axis":
            axis = int(a.i)

    return {
        "scale":
            float(scale_arr) if scale_arr.size == 1 else scale_arr.astype(np.float32).reshape(-1),
        "zero_point":
            int(zp_arr) if zp_arr.size == 1 else zp_arr.reshape(-1),
        "axis":
            axis,
        "zp_dtype":
            zp_arr.dtype,
    }

def dq_from_pt(pt_path, expect_channels=None):
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    calib = ckpt.get("int8_calib", {}).get("head") if isinstance(ckpt, dict) else None
    if calib is None or ("min" not in calib or "max" not in calib):
        return None

    per_channel = True
    mins, maxs = calib.get("min"), calib.get("max")

    if per_channel:
        mins, maxs = np.array(mins,
                              dtype=np.float32).reshape(-1), np.array(maxs,
                                                                      dtype=np.float32).reshape(-1)
        scale = (maxs - mins) / 255.0
        scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
        zp = np.clip(np.round(-mins / scale), -128, 255).astype(np.int32)
        return {"scale": scale, "zero_point": zp, "axis": 1}
    else:
        mins, maxs = float(mins[0] if isinstance(mins, (list, tuple)) else mins
                          ), float(maxs[0] if isinstance(maxs, (list, tuple)) else maxs)
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
    sc_q, zp_q = qdq_params["scale"], qdq_params["zero_point"]
    if isinstance(sc_q, np.ndarray):
        sc_q = sc_q.ravel()
    if isinstance(zp_q, np.ndarray):
        zp_q = zp_q.ravel()

    def slice_cls(arr, default_val, dtype=np.float32):
        if not isinstance(arr, np.ndarray):
            return np.full((nc, ), float(arr), dtype=dtype)
        if arr.size >= (nc + 4):
            return arr[-nc:]
        elif arr.size == nc:
            return arr
        return np.full((nc, ), default_val, dtype=dtype)

    cls_sc_qdq = slice_cls(sc_q, float(sc_q) if not isinstance(sc_q, np.ndarray) else 1.0)
    cls_zp_qdq = slice_cls(
        zp_q, int(zp_q) if not isinstance(zp_q, np.ndarray) else 0, dtype=np.int32
    )
    sc_p, zp_p = pt_dq["scale"], pt_dq["zero_point"]
    cls_sc_pt = slice_cls(sc_p, float(sc_p) if not isinstance(sc_p, np.ndarray) else 1.0)
    cls_zp_pt = slice_cls(
        zp_p, int(zp_p) if not isinstance(zp_p, np.ndarray) else 0, dtype=np.int32
    )

    print(f"\nPT vs QDQ qparams for class logits (first {max_classes_to_show}):")
    print(f"{'cls':<4} {'sc_pt':>10} {'zp_pt':>7} {'sc_qdq':>10} {'zp_qdq':>8}")
    print("-" * 50)
    for k in range(min(nc, max_classes_to_show)):
        print(
            f"{k:<4d} {float(cls_sc_pt[k]):10.6g} {int(cls_zp_pt[k]):7d} {float(cls_sc_qdq[k]):10.6g} {int(cls_zp_qdq[k]):8d}"
        )

def prepare_input_for_session(session, source, imgsz):
    loader = LoadImages(source, img_size=imgsz, auto=False)
    _, im, im0, _, _ = next(iter(loader))
    x = im.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 3:
        x = x[None]
    return x, im0

def _to_cfirst(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 3:
        return a.reshape(a.shape[0], a.shape[1], -1)
    return a if a.shape[1] < a.shape[2] else np.transpose(a, (0, 2, 1))

def run_head(session, x, _names_unused):
    outs = session.get_outputs()
    out_names = [o.name for o in outs]
    inp_name = session.get_inputs()[0].name

    if "boxes" in out_names and "logits" in out_names:
        y_boxes, y_logits = session.run(["boxes", "logits"], {inp_name: x})
        b = _to_cfirst(y_boxes).astype(np.float32)
        l = _to_cfirst(y_logits).astype(np.float32)
        return b, l

    ys = session.run(out_names, {inp_name: x})
    ys = [_to_cfirst(y) for y in ys]
    idx_boxes = next((i for i, a in enumerate(ys) if a.shape[1] == 4), 0)
    idx_logits = 1 - idx_boxes
    return ys[idx_boxes].astype(np.float32), ys[idx_logits].astype(np.float32)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def iou(box1, boxes2):
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)

def custom_non_max_suppression(boxes_cfirst, logits_cfirst, conf_thres, iou_thres, max_det=300):
    boxes_xywh = np.transpose(boxes_cfirst[0], (1, 0))
    logits = np.transpose(logits_cfirst[0], (1, 0))

    probs = 1.0 / (1.0 + np.exp(-logits))
    class_conf = np.max(probs, axis=1)
    class_ids = np.argmax(probs, axis=1)

    valid_indices = class_conf > conf_thres
    boxes_xywh = boxes_xywh[valid_indices]
    class_conf = class_conf[valid_indices]
    class_ids = class_ids[valid_indices]

    if boxes_xywh.shape[0] == 0:
        return [torch.empty((0, 6))]

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    final_detections = []
    unique_classes = np.unique(class_ids)

    for c in unique_classes:
        class_mask = class_ids == c
        class_boxes = boxes_xyxy[class_mask]
        class_scores = class_conf[class_mask]

        order = class_scores.argsort()[::-1]

        keep_indices = []
        while order.size > 0:
            i = order[0]
            keep_indices.append(i)

            ious = iou(class_boxes[i], class_boxes[order[1:]])

            inds_to_remove = np.where(ious > iou_thres)[0]
            order = np.delete(order, np.concatenate(([0], inds_to_remove + 1)))

        for idx in keep_indices:
            final_detections.append(np.concatenate((class_boxes[idx], [class_scores[idx]], [c])))

    if not final_detections:
        return [torch.empty((0, 6))]

    final_detections = np.array(final_detections)
    final_detections = final_detections[final_detections[:, 4].argsort()[::-1]]
    final_detections = final_detections[:max_det]

    return [torch.from_numpy(final_detections)]

def safe_save_image(img_like, save_path: Path, ref_shape=None):
    try:
        arr = img_like
        try:
            from PIL import Image
            if isinstance(arr, Image.Image):
                arr = np.array(arr)
        except Exception:
            pass

        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[0] in (
            1, 3
        ) and arr.shape[0] < arr.shape[-1] and arr.shape[1] != 1:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2BGR)

        if arr.ndim != 3 or arr.shape[2] != 3:
            if ref_shape is not None and len(ref_shape) >= 3:
                H, W, C = ref_shape[:3]
                if arr.size == H * W * C:
                    arr = arr.reshape(H, W, C)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Unexpected image shape {arr.shape}")

        if arr.dtype != np.uint8:
            if arr.dtype in (np.float32, np.float64):
                if arr.max() <= 1.0 + 1e-6:
                    arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)

        arr = np.ascontiguousarray(arr)

        ok = cv2.imwrite(str(save_path), arr)
        if not ok:
            raise RuntimeError("cv2.imwrite returned False")

    except Exception:
        try:
            from PIL import Image
            img = Image.fromarray(arr[..., ::-1]
                                 ) if arr.ndim == 3 and arr.shape[2] == 3 else Image.fromarray(arr)
            img.save(str(save_path))
        except Exception as e2:
            npy_path = str(save_path.with_suffix(".npy"))
            np.save(npy_path, arr)
            LOGGER.warning(
                f"Could not save annotated image to {save_path}; saved raw array to {npy_path} instead. Error: {e2}"
            )

def run_one(session, im_in, im0, names, conf_thres, iou_thres, max_det, save_path):
    boxes, logits = run_head(session, im_in, names)

    head_flat_for_cosine = np.concatenate([boxes, logits], axis=1)

    dets = custom_non_max_suppression(boxes, logits, conf_thres, iou_thres, max_det)

    im_draw = im0.copy()
    annotator = Annotator(im_draw, line_width=3, example=str(names))
    top5 = []

    def name_for(cls_index: int):
        if isinstance(names, dict):
            return names.get(cls_index, f"cls_{cls_index}")
        elif isinstance(names, (list, tuple)) and 0 <= cls_index < len(names):
            return names[cls_index]
        return f"cls_{cls_index}"

    if dets and len(dets[0]):
        det = dets[0]
        if not isinstance(det, torch.Tensor):
            det = torch.from_numpy(det)
        det = det.float().cpu()

        model_shape = (im_in.shape[2], im_in.shape[3])
        det[:, :4] = scale_boxes(model_shape, det[:, :4], im_draw.shape).round()

        order = det[:, 4].argsort(descending=True)
        det_top5 = det[order][:5]

        for *xyxy, conf, cls in det_top5:
            c = int(cls.item() if isinstance(cls, torch.Tensor) else cls)
            label = f"{name_for(c)} {float(conf):.3f}"
            annotator.box_label(xyxy, label, color=colors(c, True))
            top5.append((name_for(c), float(conf), [int(v) for v in xyxy]))

    im_out = annotator.result()
    cv2.imwrite(str(save_path), im_out)

    return head_flat_for_cosine, top5

def dump_logits_vs_qparams(label, head_cfirst_concat, names, qdq_params, max_classes_to_show=10):
    a = head_cfirst_concat.reshape(1, head_cfirst_concat.shape[1], -1).astype(np.float32)
    nc = len(names)
    cls_logits = a[:, -nc:, :].reshape(nc, -1)

    if qdq_params is None:
        print(f"\n{label}: No QDQ params found (final DQ not detected).")
        return

    sc, zp, zp_dtype = qdq_params["scale"], qdq_params["zero_point"], qdq_params.get(
        "zp_dtype", np.uint8
    )

    if isinstance(sc, np.ndarray) and sc.size >= nc:
        cls_sc = sc[-nc:] if sc.size > nc else sc
        cls_zp = zp[-nc:] if isinstance(zp, np.ndarray) and zp.size > nc else zp
    else:
        s = float(sc.item() if not isinstance(sc, float) else sc)
        z = int(zp.item() if not isinstance(zp, int) else zp)
        cls_sc, cls_zp = np.full((nc, ), s, dtype=np.float32), np.full((nc, ), z, dtype=np.int32)

    qmin_q, qmax_q = (-128, 127) if zp_dtype == np.int8 else (0, 255)

    print(f"\nClass logits vs QDQ qparams ({label}, first {max_classes_to_show}):")
    print(
        f"{'cls':<4} {'mean':>9} {'min':>10} {'max':>10} {'scale':>9} {'zp':>5} {'q_min':>9} {'q_max':>9}"
    )
    print("-" * 75)
    for k in range(min(nc, max_classes_to_show)):
        v = cls_logits[k, :]
        mean_v, min_v, max_v = float(v.mean()), float(v.min()), float(v.max())
        qmin_dequant = float((qmin_q - int(cls_zp[k])) * float(cls_sc[k]))
        qmax_dequant = float((qmax_q - int(cls_zp[k])) * float(cls_sc[k]))
        print(
            f"{k:<4d} {mean_v:9.4f} {min_v:10.4f} {max_v:10.4f} {float(cls_sc[k]):9.6g} {int(cls_zp[k]):5d} {qmin_dequant:9.3f} {qmax_dequant:9.3f}"
        )

def compare_models(opt):
    fp32_path = Path(opt.weights)
    int8_path_obj = fp32_path.with_stem(f"{fp32_path.stem}_int8_qdq")

    if opt.quantize or not int8_path_obj.exists():
        int8_path = quantize_qdq(
            fp32_onnx=str(fp32_path),
            data_yaml=opt.data,
            imgsz=opt.imgsz,
            device=opt.device,
            calib=opt.calib,
            act_signed=opt.act_signed,
            force=opt.quantize,
        )
    else:
        int8_path = str(int8_path_obj)

    providers = get_providers(opt.device)
    sess_fp32 = onnxruntime.InferenceSession(str(fp32_path), providers=providers)
    sess_qdq = onnxruntime.InferenceSession(str(int8_path), providers=providers)

    names = names_from_meta(sess_fp32)

    im_in, im0 = prepare_input_for_session(sess_fp32, opt.source, opt.imgsz)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = Path(opt.source).stem if not Path(opt.source).is_dir() else "image"

    head_fp32_cfirst, top5_fp32 = run_one(
        sess_fp32, im_in, im0, names, opt.conf_thres, opt.iou_thres, opt.max_det,
        save_dir / f"{base}_fp32.jpg"
    )
    head_qdq_cfirst, top5_qdq = run_one(
        sess_qdq, im_in, im0.copy(), names, opt.conf_thres, opt.iou_thres, opt.max_det,
        save_dir / f"{base}_int8.jpg"
    )

    cs = safe_cosine(head_fp32_cfirst, head_qdq_cfirst)
    print(f"\nCosine similarity (FP32 head vs QDQ head): {cs:.6f}\n")

    rows = []
    for i in range(max(len(top5_fp32), len(top5_qdq))):
        lf = top5_fp32[i] if i < len(top5_fp32) else ("-", "-", "-")
        ri = top5_qdq[i] if i < len(top5_qdq) else ("-", "-", "-")
        rows.append((
            i + 1,
            lf[0],
            f"{lf[1]:.3f}" if isinstance(lf[1], float) else "-",
            str(lf[2]),
            ri[0],
            f"{ri[1]:.3f}" if isinstance(ri[1], float) else "-",
            str(ri[2]),
        ))
    print("Top-5 detections (side-by-side):")
    print_side_by_side_table(
        rows, [
            "Rank", "FP32 class", "FP32 conf", "FP32 box [xyxy]", "QDQ class", "QDQ conf",
            "QDQ box [xyxy]"
        ]
    )
    print(f"\nSaved annotated images to: {save_dir}")

    prefer_logits_out = next(
        (o.name for o in onnx.load(str(int8_path)).graph.output if o.name.lower() == "logits"), None
    )
    qdq_params = get_output_qdq_params(int8_path, prefer_output=prefer_logits_out)
    dump_logits_vs_qparams("FP32 head", head_fp32_cfirst, names, qdq_params, max_classes_to_show=10)
    dump_logits_vs_qparams("QDQ head", head_qdq_cfirst, names, qdq_params, max_classes_to_show=10)

    pt_path = Path(opt.pt) if opt.pt else fp32_path.with_suffix(".pt")
    if pt_path and pt_path.exists():
        pt_dq = dq_from_pt(str(pt_path))
        if pt_dq is not None:
            print_pt_vs_qdq_qparams(pt_dq, qdq_params, names, max_classes_to_show=10)
        else:
            print("\n[PT calib] Found PT file but no int8_calib.head {min,max} inside.")
    elif opt.pt:
        print(f"\n[PT calib] PT file not found: {opt.pt}")

def parse_opt():
    p = argparse.ArgumentParser(description="FP32 vs QDQ INT8 comparison with YOLO split heads")
    p.add_argument("--weights", type=str, required=True, help="Path to FP32 ONNX model")
    p.add_argument(
        "--source", type=str, required=True, help="Path to a single image for comparison"
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data/coco.yaml"),
        help="dataset.yaml for calibration"
    )
    p.add_argument(
        "--imgsz", "--img-size", nargs="+", type=int, default=[640], help="Inference size h,w"
    )
    p.add_argument(
        "--device", default="cpu", help="Hardware device for inference, e.g., 'cuda' or 'cpu'"
    )
    p.add_argument(
        "--quantize", action="store_true", help="Force re-quantization even if INT8 model exists"
    )
    p.add_argument(
        "--calib",
        type=str,
        default="minmax",
        choices=["minmax", "entropy", "percentile"],
        help="Calibration method",
    )
    p.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold for NMS")
    p.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument(
        "--max-det", type=int, default=300, help="Maximum number of detections per image"
    )
    p.add_argument("--project", default=ROOT / "runs/compare", help="Directory to save results")
    p.add_argument("--name", default="exp", help="Subdirectory name for results")
    p.add_argument(
        "--act-signed", action="store_true", help="Use signed int8 (symmetric) for activations"
    )
    p.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Optional: path to .pt file with int8_calib.head to compare qparams"
    )
    opt = p.parse_args()
    opt.imgsz = check_img_size(opt.imgsz)
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    compare_models(opt)
