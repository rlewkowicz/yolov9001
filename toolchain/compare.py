#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import onnx
import onnxruntime
import yaml
from onnx import shape_inference
from onnxruntime.quantization import QuantType, quantize_static, CalibrationMethod, QuantFormat
from onnxruntime.quantization.quant_utils import model_has_pre_process_metadata
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch
import ast

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) in sys.path else FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import LoadImages
from utils.general import LOGGER, print_args, check_img_size, increment_path, scale_boxes, non_max_suppression
from utils.plots import Annotator, colors

def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity that is well-defined for zero vectors."""
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

def get_providers(opt_device):
    avail = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider"
               ] + (["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else [])
    return ["CPUExecutionProvider"]

def create_debug_session(model_path, providers, disable_optim=False):
    if model_path is None:
        raise ValueError("create_debug_session: model_path is None")
    model_path = str(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"create_debug_session: model not found: {model_path}")

    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)

    tensor_type_map = {}
    for tensor in model.graph.initializer:
        tensor_type_map[tensor.name] = tensor.data_type
    for tensor_info in list(model.graph.input) + list(model.graph.value_info
                                                     ) + list(model.graph.output):
        if tensor_info.name in tensor_type_map:
            continue
        tensor_type_map[tensor_info.name] = tensor_info.type.tensor_type.elem_type

    all_tensor_names = {tensor.name for tensor in model.graph.initializer}
    for node in model.graph.node:
        all_tensor_names.update(node.input)
        all_tensor_names.update(node.output)
    original_outputs = {o.name for o in model.graph.output}
    for name in sorted(list(all_tensor_names)):
        if name and name in tensor_type_map and name not in original_outputs:
            data_type = tensor_type_map.get(name)
            output_tensor_info = onnx.helper.make_tensor_value_info(name, data_type, None)
            model.graph.output.append(output_tensor_info)

    import onnxruntime as ort
    so = ort.SessionOptions()
    if disable_optim:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=so, providers=providers)
    output_names = [output.name for output in session.get_outputs()]
    return session, output_names

def create_session(model_path, providers, disable_optim=False):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if disable_optim:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)

def list_all_tensors_no_nan_inf(model_path, candidate_image_paths, imgsz, providers):
    session, output_names = create_debug_session(str(model_path), providers)
    input_name = session.get_inputs()[0].name
    good = []
    total = 0
    loader = LoadImages(candidate_image_paths, img_size=imgsz, auto=False)
    for path, im, _, _, _ in tqdm(loader, desc="Screening images"):
        total += 1
        x = im
        if x.dtype != np.float32:
            x = x.astype(np.float32) / 255.0
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        try:
            outs = session.run(output_names, {input_name: x})
            bad = any(np.isnan(arr).any() or np.isinf(arr).any() for arr in outs)
            if not bad:
                good.append(path)
        except Exception:
            pass
    LOGGER.info(
        f"[Calib screening] {len(good)}/{total} images passed (no NaN/Inf in any intermediate tensor)."
    )
    return good

def read_calib_from_metadata(fp32_model_path):
    try:
        m = onnx.load(fp32_model_path)
        kv = {p.key: p.value for p in m.metadata_props}
        if not kv.get("int8_calib_present", "False") in ("True", "true", "1"):
            return None
        per_channel = kv.get("int8_per_channel", "True") in ("True", "true", "1")
        ch = int(kv.get("int8_channels", "0"))
        head_min = kv.get("int8_head_min", None)
        head_max = kv.get("int8_head_max", None)
        if head_min is not None:
            head_min = ast.literal_eval(head_min)
        if head_max is not None:
            head_max = ast.literal_eval(head_max)
        return {"per_channel": per_channel, "channels": ch, "min": head_min, "max": head_max}
    except Exception:
        return None

def force_uint8_outputs(model_path_in, model_path_out, calib_meta=None):
    model = onnx.load(model_path_in)
    graph = model.graph

    per_channel = False
    mins = None
    maxs = None
    if calib_meta is not None and calib_meta.get("min") is not None and calib_meta.get(
        "max"
    ) is not None:
        per_channel = bool(calib_meta.get("per_channel", True))
        mins = np.array(calib_meta["min"], dtype=np.float32).reshape(-1) if isinstance(
            calib_meta["min"], (list, tuple, np.ndarray)
        ) else np.array([float(calib_meta["min"])], dtype=np.float32)
        maxs = np.array(calib_meta["max"], dtype=np.float32).reshape(-1) if isinstance(
            calib_meta["max"], (list, tuple, np.ndarray)
        ) else np.array([float(calib_meta["max"])], dtype=np.float32)

    for i, o in enumerate(graph.output):
        if o.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
            continue  # already uint8
        scale_name = f"out{i}_scale"
        zp_name = f"out{i}_zero_point"
        q_out_name = f"{o.name}_uint8"

        if per_channel and mins is not None and maxs is not None:
            scale = (maxs - mins) / 255.0
            scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
            zp = np.round(-mins / scale).astype(np.int32)
            zp = np.clip(zp, 0, 255).astype(np.uint8)
            scale_init = onnx.helper.make_tensor(
                scale_name, onnx.TensorProto.FLOAT, [scale.shape[0]], scale.tolist()
            )
            zp_init = onnx.helper.make_tensor(
                zp_name, onnx.TensorProto.UINT8, [zp.shape[0]], zp.tolist()
            )
            q_node = onnx.helper.make_node(
                "QuantizeLinear",
                inputs=[o.name, scale_name, zp_name],
                outputs=[q_out_name],
                name=f"QuantizeLinear_Out{i}",
                axis=1,
            )
        elif mins is not None and maxs is not None:
            sc = float((maxs[0] - mins[0]) / 255.0) if maxs[0] - mins[0] > 1e-12 else 1.0
            zp = int(np.clip(round(-mins[0] / sc), 0, 255))
            scale_init = onnx.helper.make_tensor(scale_name, onnx.TensorProto.FLOAT, [1], [sc])
            zp_init = onnx.helper.make_tensor(zp_name, onnx.TensorProto.UINT8, [1], [zp])
            q_node = onnx.helper.make_node(
                "QuantizeLinear",
                inputs=[o.name, scale_name, zp_name],
                outputs=[q_out_name],
                name=f"QuantizeLinear_Out{i}"
            )
        else:
            scale_init = onnx.helper.make_tensor(
                scale_name, onnx.TensorProto.FLOAT, [1], [1.0 / 255.0]
            )
            zp_init = onnx.helper.make_tensor(zp_name, onnx.TensorProto.UINT8, [1], [0])
            q_node = onnx.helper.make_node(
                "QuantizeLinear",
                inputs=[o.name, scale_name, zp_name],
                outputs=[q_out_name],
                name=f"QuantizeLinear_Out{i}"
            )

        graph.initializer.extend([scale_init, zp_init])
        graph.node.append(q_node)

        o.name = q_out_name
        o.type.tensor_type.elem_type = onnx.TensorProto.UINT8

    onnx.save(model, model_path_out)

def dq_from_pt(pt_path, expect_channels=None):
    """
    Load per-channel (or per-tensor) min/max from training checkpoint and derive (scale, zero_point).
    Returns dict {'scale': float or np.ndarray, 'zero_point': int or np.ndarray, 'axis': 1}
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
            return None
        scale = (maxs - mins) / 255.0
        scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
        zp = np.round(-mins / scale).astype(np.int32)
        zp = np.clip(zp, 0, 255).astype(np.uint8)
        return {"scale": scale, "zero_point": zp, "axis": 1}
    else:
        mins = float(mins if not isinstance(mins, (list, tuple)) else mins[0])
        maxs = float(maxs if not isinstance(maxs, (list, tuple)) else maxs[0])
        sc = (maxs - mins) / 255.0 if (maxs - mins) > 1e-12 else 1.0
        zp = int(np.clip(round(-mins / sc), 0, 255))
        return {"scale": float(sc), "zero_point": int(zp), "axis": 1}

def print_dq(label, dq):
    if dq is None:
        print(f"\n[Dequant] {label}: None")
        return
    sc = dq["scale"]
    zp = dq["zero_point"]
    ax = dq.get("axis", 1)
    if isinstance(sc, np.ndarray):
        print(
            f"\n[Dequant] {label}: axis={ax}, per-channel={sc.shape[0]} | "
            f"scale[min={sc.min():.6g}, max={sc.max():.6g}] "
            f"zp[min={int(zp.min())}, max={int(zp.max())}]"
        )
    else:
        print(f"\n[Dequant] {label}: axis={ax}, per-tensor | scale={float(sc):.6g}, zp={int(zp)}")

def quantize_model(opt):
    model_input_path = Path(opt.weights)
    model_output_path_uint8 = model_input_path.with_stem(f"{model_input_path.stem}_int8_io_uint8")

    data_yaml_path = Path(opt.data)
    with open(data_yaml_path, errors="ignore") as f:
        data_dict = yaml.safe_load(f)
    dataset_root = (data_yaml_path.parent / ".." / data_dict['path']).resolve()
    calib_data_list_path = dataset_root / data_dict['train']
    with open(calib_data_list_path) as f:
        candidate_files = [line.strip() for line in f][:1000]
    candidate_image_paths = [str(dataset_root / p) for p in candidate_files]

    providers = get_providers(opt.device)

    tmp_sess = onnxruntime.InferenceSession(str(model_input_path), providers=providers)
    input_name = tmp_sess.get_inputs()[0].name
    model_meta = tmp_sess.get_modelmeta().custom_metadata_map
    stride = int(model_meta.get("stride", 32))
    out_shape = tmp_sess.get_outputs()[0].shape  # e.g., [1, C, N]
    expect_c = out_shape[1] if isinstance(out_shape,
                                          (list, tuple)) and len(out_shape) >= 2 and isinstance(
                                              out_shape[1], int
                                          ) else None
    del tmp_sess

    good_calib_images = list_all_tensors_no_nan_inf(
        model_input_path, candidate_image_paths, opt.imgsz, providers
    )
    LOGGER.info(
        f"Calibration images passing NaN/Inf check: {len(good_calib_images)}/{len(candidate_image_paths)}"
    )
    if not good_calib_images:
        raise ValueError("No stable images found for calibration.")

    model_path_for_quant = model_input_path
    model = onnx.load(str(model_input_path))
    if not model_has_pre_process_metadata(model):
        preprocessed_model_path = model_input_path.with_stem(
            f"{model_input_path.stem}-preprocessed"
        )
        quant_pre_process(
            input_model_path=str(model_input_path),
            output_model_path=str(preprocessed_model_path),
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )
        model_path_for_quant = preprocessed_model_path

    tmp_q_path = model_input_path.with_stem(f"{model_input_path.stem}_int8_tmp")
    LOGGER.info("Running Quantization (MinMax, QOperator, Act=QUInt8, W=QInt8)...")
    quantize_static(
        model_input=str(model_path_for_quant),
        model_output=str(tmp_q_path),
        calibration_data_reader=ImageCalibrator(
            calib_files=good_calib_images, input_name=input_name, imgsz=opt.imgsz, stride=stride
        ),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=opt.per_channel,
        nodes_to_exclude=[],
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False, "WeightSymmetric": True, "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": True
        },
    )

    pt_path = opt.pt or str(Path(model_input_path).with_suffix(".pt"))
    calib_meta = None
    dq = dq_from_pt(pt_path,
                    expect_channels=expect_c) if pt_path and Path(pt_path).exists() else None
    if dq is not None:
        sc, zp = dq["scale"], dq["zero_point"]
        if isinstance(sc, np.ndarray):
            sc = sc.astype(np.float32).reshape(-1)
            zp = zp.astype(np.float32).reshape(-1)
            mins = (-zp * sc).tolist()
            maxs = ((255.0 - zp) * sc).tolist()
            calib_meta = {"per_channel": True, "min": mins, "max": maxs}
        else:
            sc = float(sc)
            zp = float(zp)
            minv = -zp * sc
            maxv = (255.0 - zp) * sc
            calib_meta = {"per_channel": False, "min": float(minv), "max": float(maxv)}
        LOGGER.info("Using PT checkpoint calibration for output QuantizeLinear.")
    else:
        calib_meta = read_calib_from_metadata(str(model_input_path))
        if calib_meta is None:
            LOGGER.info("No PT/ONNX output calibration found; using default 1/255 for outputs.")

    LOGGER.info(f"Writing final INT8 model with uint8 outputs -> {model_output_path_uint8}")
    force_uint8_outputs(str(tmp_q_path), str(model_output_path_uint8), calib_meta=calib_meta)

    for p in [
        tmp_q_path, model_path_for_quant if model_path_for_quant != model_input_path else None
    ]:
        try:
            if p and Path(p).exists():
                os.remove(p)
        except Exception:
            pass

    if not Path(model_output_path_uint8).exists():
        raise RuntimeError(
            f"Quantization finished but output file not found: {model_output_path_uint8}"
        )

    return str(model_output_path_uint8)

def run_and_collect_outputs(session, input_name, input_image, output_names):
    outputs = session.run(output_names, {input_name: input_image})
    return {name: out for name, out in zip(output_names, outputs)}

def session_io_info(session):
    ins = session.get_inputs()
    outs = session.get_outputs()
    in_info = [(i.name, i.type, [d if isinstance(d, int) else d for d in i.shape]) for i in ins]
    out_info = [(o.name, o.type, [d if isinstance(d, int) else d for d in o.shape]) for o in outs]
    return in_info, out_info

def prepare_input_for_session(session, source, imgsz):
    loader = LoadImages(source, img_size=imgsz, auto=False)
    _, im, _, _, _ = next(iter(loader))
    exp = session.get_inputs()[0]
    dtype = exp.type
    if "tensor(uint8)" in dtype:
        if im.dtype != np.uint8:
            im = im.astype(np.uint8)
        if len(im.shape) == 3:
            im = np.expand_dims(im, 0)
        return im
    else:
        x = im.astype(np.float32) / 255.0
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        return x

def preprocess_int8_for_session(session, source, imgsz):
    loader = LoadImages(source, img_size=imgsz, auto=False)
    _, im, _, _, _ = next(iter(loader))
    if im.dtype != np.uint8:
        im = im.astype(np.uint8)
    if len(im.shape) == 3:
        im = np.expand_dims(im, 0)
    return im

def diag_partition_stats(fp32_head, int8_head_deq, nc):
    a = fp32_head.astype(np.float32)
    b = int8_head_deq.astype(np.float32)
    a_box, a_cls = a[:, :4, :], a[:, -nc:, :]
    b_box, b_cls = b[:, :4, :], b[:, -nc:, :]

    def stats(x):
        xf = x.reshape(-1).astype(np.float32)
        return float(xf.min()), float(xf.max()), float(xf.mean()), float(xf.std())

    ab_min, ab_max, ab_mean, ab_std = stats(a_box)
    bb_min, bb_max, bb_mean, bb_std = stats(b_box)
    ac_min, ac_max, ac_mean, ac_std = stats(a_cls)
    bc_min, bc_max, bc_mean, bc_std = stats(b_cls)
    fa = a_box.flatten()
    fb = b_box.flatten()
    ca = 1.0 - cosine(fa, fb) if fa.size and fb.size else 1.0
    fa = a_cls.flatten()
    fb = b_cls.flatten()
    cc = 1.0 - cosine(fa, fb) if fa.size and fb.size else 1.0
    print(
        "\nBox partition stats FP32:",
        {"min": ab_min, "max": ab_max, "mean": ab_mean, "std": ab_std}
    )
    print(
        "Box partition stats INT8 deq:",
        {"min": bb_min, "max": bb_max, "mean": bb_mean, "std": bb_std}
    )
    print("Box partition cosine:", ca)
    print(
        "\nClass partition stats FP32:",
        {"min": ac_min, "max": ac_max, "mean": ac_mean, "std": ac_std}
    )
    print(
        "Class partition stats INT8 deq:",
        {"min": bc_min, "max": bc_max, "mean": bc_mean, "std": bc_std}
    )
    print("Class partition cosine:", cc)

def non_max_suppression_int8(
    output_uint8, scale, zp, axis=1, names_nc=None, conf_thres=0.1, iou_thres=0.45, max_det=300
):
    x = output_uint8.astype(np.float32)
    s = np.array(scale, dtype=np.float32).reshape(-1)
    z = np.array(zp, dtype=np.float32).reshape(-1)

    if s.size == 1 and z.size == 1:
        x = (x - z[0]) * s[0]
    else:
        if axis == 1:
            x = (x - z.reshape(1, -1, 1)) * s.reshape(1, -1, 1)
        elif axis == 0:
            x = (x - z.reshape(-1, 1, 1)) * s.reshape(-1, 1, 1)
        else:
            x = (x - z.reshape(1, -1, 1)) * s.reshape(1, -1, 1)

    pred = torch.from_numpy(x).float()
    dets = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
    return dets

def run_detect_one(
    session, im, im0, conf_thres, iou_thres, max_det, save_dir, save_name, names, dq=None
):
    input_name = session.get_inputs()[0].name
    out_meta = session.get_outputs()[0]
    out_dtype = out_meta.type
    out_name = out_meta.name

    output = session.run([out_name], {input_name: im})[0]

    if out_dtype == "tensor(uint8)":
        if dq is None:
            dq = {"scale": 1.0, "zero_point": 0, "axis": 1}
        sc = dq["scale"]
        zp = dq["zero_point"]
        axis = dq.get("axis", 1)
        dets = non_max_suppression_int8(
            output,
            sc,
            zp,
            axis=axis,
            names_nc=len(names),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )
    else:
        pred = torch.from_numpy(output).float()
        dets = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

    im_draw = im0.copy()
    annotator = Annotator(im_draw, line_width=3, example=str(names))
    top5 = []
    for det in dets:
        if len(det):
            im_shape = (im.shape[2], im.shape[3]) if im.ndim == 4 else (im.shape[1], im.shape[2])
            det[:, :4] = scale_boxes(im_shape, det[:, :4], im_draw.shape).round()
            det_sorted = det[det[:, 4].argsort(descending=True)]
            for j, (*xyxy, conf, cls) in enumerate(det_sorted[:5]):
                c = int(cls)
                label = f"{names[c]} {float(conf):.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
                top5.append((names[c], float(conf), [float(k) for k in xyxy]))
    out_img = annotator.result()
    cv2.imwrite(str(save_dir / save_name), out_img)
    return top5

def extract_output_scale_zp(model_path):
    from onnx import numpy_helper
    m = onnx.load(model_path)
    name_to_init = {i.name: i for i in m.graph.initializer}
    out_names = [o.name for o in m.graph.output]
    scales = {}
    zps = {}
    axes = {}
    for n in m.graph.node:
        if n.op_type == "QuantizeLinear" and n.output and n.output[0] in out_names:
            s_name = n.input[1] if len(n.input) > 1 else None
            z_name = n.input[2] if len(n.input) > 2 else None
            ax = 1
            for a in n.attribute:
                if a.name == "axis":
                    ax = a.i
            axes[n.output[0]] = ax
            if s_name in name_to_init:
                s = numpy_helper.to_array(name_to_init[s_name]).astype(np.float32).reshape(-1)
                scales[n.output[0]] = s if s.size > 1 else float(s[0])
            if z_name in name_to_init:
                z = numpy_helper.to_array(name_to_init[z_name]).astype(np.uint8).reshape(-1)
                zps[n.output[0]] = z if z.size > 1 else int(z[0])
    out = {}
    for o in out_names:
        sc = scales.get(o, None)
        zp = zps.get(o, None)
        ax = axes.get(o, 1)
        if sc is None or zp is None:
            continue
        out[o] = {"scale": sc, "zero_point": zp, "axis": ax}
    return out

def dequantize_array(arr_uint8, scale, zp, axis=1):
    x = arr_uint8.astype(np.float32)
    s = np.array(scale, dtype=np.float32).reshape(-1)
    z = np.array(zp, dtype=np.float32).reshape(-1)
    if s.size == 1 and z.size == 1:
        return (x - z[0]) * s[0]
    if axis == 1:
        return (x - z.reshape(1, -1, 1)) * s.reshape(1, -1, 1)
    if axis == 0:
        return (x - z.reshape(-1, 1, 1)) * s.reshape(-1, 1, 1)
    return (x - z.reshape(1, -1, 1)) * s.reshape(1, -1, 1)

def head_metrics(fp32_head, int8_head_deq, names_nc=None, topk=100):
    a = fp32_head.reshape(1, fp32_head.shape[1], -1).astype(np.float32)
    b = int8_head_deq.reshape(1, int8_head_deq.shape[1], -1).astype(np.float32)
    fa = a.flatten()
    fb = b.flatten()
    cs = safe_cosine(fa, fb)
    mae = float(np.mean(np.abs(fa - fb))) if fa.size else 0.0

    if names_nc is None:
        return cs, mae, None, None

    nc = int(names_nc)
    if a.shape[1] <= nc:
        return cs, mae, None, None

    ac = a[:, -nc:, :]
    bc = b[:, -nc:, :]
    acs = 1 / (1 + np.exp(-ac))
    bcs = 1 / (1 + np.exp(-bc))
    ai = np.argmax(acs, axis=1).reshape(-1)
    bi = np.argmax(bcs, axis=1).reshape(-1)
    agree = float(np.mean((ai == bi).astype(np.float32)))

    k = min(topk, ai.size)
    at = np.argpartition(-acs.max(axis=1).reshape(-1), k - 1)[:k]
    bt = np.argpartition(-bcs.max(axis=1).reshape(-1), k - 1)[:k]
    topk_jacc = float(
        len(set(at.tolist()) & set(bt.tolist())) / max(1, len(set(at.tolist()) | set(bt.tolist())))
    )
    return cs, mae, agree, topk_jacc

def tensor_stats(x, is_uint8=False):
    x_flat = x.reshape(-1)
    mn = float(np.min(x_flat))
    mx = float(np.max(x_flat))
    mean = float(np.mean(x_flat))
    std = float(np.std(x_flat))
    if is_uint8:
        z0 = float(np.mean((x_flat == 0).astype(np.float32)))
        z255 = float(np.mean((x_flat == 255).astype(np.float32)))
    else:
        z0 = 0.0
        z255 = 0.0
    lo = float(np.min(x_flat))
    hi = float(np.max(x_flat))
    if is_uint8:
        hist, _ = np.histogram(x_flat, bins=64, range=(0, 255))
    else:
        rng_hi = hi if hi > lo else lo + 1e-6
        hist, _ = np.histogram(x_flat, bins=64, range=(lo, rng_hi))
    p = hist.astype(np.float32)
    p = p / max(1.0, np.sum(p))
    entropy = float(-np.sum(np.where(p > 0, p * np.log2(p), 0.0)))
    return {
        "min": mn, "max": mx, "mean": mean, "std": std, "p0": z0, "p255": z255, "entropy": entropy
    }

def threshold_sweep_counts(head_float, names_nc, thresholds):
    nc = int(names_nc)
    hf = head_float.reshape(1, head_float.shape[1], -1).astype(np.float32)
    if hf.shape[1] <= nc:
        scores = hf.max(axis=1).reshape(-1)
    else:
        cls = hf[:, -nc:, :]
        scores = (1 / (1 + np.exp(-cls))).max(axis=1).reshape(-1)
    out = []
    for t in thresholds:
        out.append((
            t, int(np.sum(scores >= t)),
            float(scores[scores >= t].mean() if np.any(scores >= t) else 0.0)
        ))
    return out

def quant_dequant_roundtrip(fp32_head, scale, zp, axis=1):
    x = fp32_head.astype(np.float32)
    s = np.array(scale, dtype=np.float32).reshape(-1)
    z = np.array(zp, dtype=np.float32).reshape(-1)
    if s.size == 1 and z.size == 1:
        q = np.clip(np.round(x / s[0]) + z[0], 0, 255).astype(np.uint8)
        x2 = (q.astype(np.float32) - z[0]) * s[0]
    else:
        if axis != 1:
            axis = 1
        q = np.clip(np.round(x / s.reshape(1, -1, 1)) + z.reshape(1, -1, 1), 0,
                    255).astype(np.uint8)
        x2 = (q.astype(np.float32) - z.reshape(1, -1, 1)) * s.reshape(1, -1, 1)
    mae = float(np.mean(np.abs(x - x2)))
    cos_val = safe_cosine(x.flatten(), x2.flatten())
    return mae, cos_val

def get_nc_and_regmax(session, fallback_nc=80, fallback_regmax=None):
    nc = None
    regmax = fallback_regmax
    try:
        meta = session.get_modelmeta().custom_metadata_map
        nms = eval(meta.get("names", "None"))
        if nms is not None:
            nc = len(nms)
    except Exception:
        pass
    if nc is None:
        nc = fallback_nc
    return nc, regmax

def compare_models(opt):
    fp32_path = opt.weights
    int8_path_obj = Path(fp32_path).with_stem(f"{Path(fp32_path).stem}_int8_io_uint8")

    if opt.quantize or not int8_path_obj.exists():
        if not opt.data:
            raise ValueError(
                "INT8 quantization requires a --data argument for the calibration dataset."
            )
        int8_path = quantize_model(opt)
    else:
        int8_path = str(int8_path_obj)
        LOGGER.info(f"Found existing INT8 model at {int8_path}, skipping quantization.")

    providers = get_providers(opt.device)

    session_fp32_dbg, fp32_output_names = create_debug_session(
        fp32_path, providers, disable_optim=opt.disable_optim
    )
    session_int8_dbg, int8_output_names = create_debug_session(
        int8_path, providers, disable_optim=opt.disable_optim
    )

    runtime_fp32 = create_session(fp32_path, providers, disable_optim=opt.disable_optim)
    runtime_int8 = create_session(int8_path, providers, disable_optim=opt.disable_optim)

    input_name_fp32 = session_fp32_dbg.get_inputs()[0].name
    im_fp32 = prepare_input_for_session(runtime_fp32, opt.source, opt.imgsz)
    in_type_int8 = runtime_int8.get_inputs()[0].type
    if "tensor(uint8)" in in_type_int8:
        im_int8 = preprocess_int8_for_session(runtime_int8, opt.source, opt.imgsz)
    else:
        im_int8 = prepare_input_for_session(runtime_int8, opt.source, opt.imgsz)

    out_shape = runtime_fp32.get_outputs()[0].shape  # e.g., [1, 84, 8400]
    expect_c = None
    if isinstance(out_shape,
                  (list, tuple)) and len(out_shape) >= 2 and isinstance(out_shape[1], int):
        expect_c = out_shape[1]

    pt_path = opt.pt
    if pt_path is None:
        candidate = Path(fp32_path).with_suffix(".pt")
        pt_path = str(candidate) if candidate.exists() else None
    dq = dq_from_pt(pt_path, expect_channels=expect_c) if pt_path else None
    if dq is not None:
        print_dq("Selected from PT", dq)
    else:
        if pt_path:
            print("\n[Dequant] PT file found but no usable calibration stats; falling back...")
        else:
            print("\n[Dequant] PT file not provided/found; falling back...")

        dq_map = extract_output_scale_zp(int8_path)
        out0_name = runtime_int8.get_outputs()[0].name
        dq = dq_map.get(out0_name, None)
        if dq is not None:
            print_dq("Selected from ONNX output QuantizeLinear", dq)
        else:
            dq = {"scale": 1.0 / 255.0, "zero_point": 0, "axis": 1}
            print_dq("Fallback identity", dq)

    outputs_fp32 = run_and_collect_outputs(
        session_fp32_dbg, input_name_fp32, im_fp32, fp32_output_names
    )
    outputs_int8 = run_and_collect_outputs(
        session_int8_dbg,
        session_int8_dbg.get_inputs()[0].name, im_int8, int8_output_names
    )

    print("\n" + "=" * 45 + " ONNX LAYER-BY-LAYER COMPARISON " + "=" * 45)
    print(f"{'Layer (Tensor Name)':<75} {'Cosine Sim':>15s} {'MAE':>15s}")
    print('-' * 110)
    common_names = sorted(list(set(outputs_fp32.keys()) & set(outputs_int8.keys())))
    for name in common_names:
        fp32_val, int8_val = outputs_fp32.get(name), outputs_int8.get(name)
        if fp32_val is None or int8_val is None:
            continue
        if fp32_val.shape != int8_val.shape:
            continue
        flat_fp32 = fp32_val.ravel().astype(np.float32)
        flat_int8 = int8_val.ravel().astype(np.float32)
        if flat_fp32.size == 0 or flat_int8.size == 0:
            continue
        cos_sim = safe_cosine(flat_fp32, flat_int8)
        mae = float(np.mean(np.abs(flat_fp32 - flat_int8)))
        color_start = "\033[91m" if (not np.isnan(cos_sim) and cos_sim < 0.95) else ""
        color_end = "\033[0m" if color_start else ""
        cos_print = f"{cos_sim:15.6f}" if not np.isnan(cos_sim) else f"{'nan':>15s}"
        print(f"{name:<75} {color_start}{cos_print}{color_end} {mae:15.6f}")

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(opt.source).stem if not Path(opt.source).is_dir() else "image"

    in_info_fp32, out_info_fp32 = session_io_info(runtime_fp32)
    in_info_int8, out_info_int8 = session_io_info(runtime_int8)

    io_rows = []
    max_len = max(len(in_info_fp32), len(in_info_int8))
    for i in range(max_len):
        l = in_info_fp32[i] if i < len(in_info_fp32) else ("-", "-", "-")
        r = in_info_int8[i] if i < len(in_info_int8) else ("-", "-", "-")
        io_rows.append((f"IN{i}", l[0], l[1], str(l[2]), r[0], r[1], str(r[2])))
    print("\nInput tensors:")
    print_side_by_side_table(
        io_rows,
        ["Slot", "FP32 name", "FP32 type", "FP32 shape", "INT8 name", "INT8 type", "INT8 shape"]
    )

    oo_rows = []
    max_len_o = max(len(out_info_fp32), len(out_info_int8))
    for i in range(max_len_o):
        l = out_info_fp32[i] if i < len(out_info_fp32) else ("-", "-", "-")
        r = out_info_int8[i] if i < len(out_info_int8) else ("-", "-", "-")
        oo_rows.append((f"OUT{i}", l[0], l[1], str(l[2]), r[0], r[1], str(r[2])))
    print("\nOutput tensors:")
    print_side_by_side_table(
        oo_rows,
        ["Slot", "FP32 name", "FP32 type", "FP32 shape", "INT8 name", "INT8 type", "INT8 shape"]
    )

    names = None
    try:
        meta = runtime_fp32.get_modelmeta().custom_metadata_map
        names = eval(meta.get("names", "None"))
    except Exception:
        names = None
    if names is None:
        names = [str(i) for i in range(1000)]

    im0 = cv2.imread(opt.source)
    fp32_top5 = run_detect_one(
        runtime_fp32,
        im_fp32,
        im0,
        opt.conf_thres,
        opt.iou_thres,
        opt.max_det,
        save_dir,
        f"{base_name}_fp32.jpg",
        names,
        dq=None,
    )
    im0_int8 = im0.copy()
    int8_top5 = run_detect_one(
        runtime_int8,
        im_int8,
        im0_int8,
        opt.conf_thres,
        opt.iou_thres,
        opt.max_det,
        save_dir,
        f"{base_name}_int8.jpg",
        names,
        dq=dq,
    )

    det_rows = []
    for i in range(max(len(fp32_top5), len(int8_top5))):
        lf = fp32_top5[i] if i < len(fp32_top5) else ("-", "-", "-")
        ri = int8_top5[i] if i < len(int8_top5) else ("-", "-", "-")
        det_rows.append((
            i + 1, lf[0], f"{lf[1]:.3f}" if isinstance(lf[1], float) else "-", str(lf[2]), ri[0],
            f"{ri[1]:.3f}" if isinstance(ri[1], float) else "-", str(ri[2])
        ))
    print("\nTop-5 detections (side-by-side):")
    print_side_by_side_table(
        det_rows, [
            "Rank", "FP32 class", "FP32 conf", "FP32 box [xyxy]", "INT8 class", "INT8 conf",
            "INT8 box [xyxy]"
        ]
    )
    print(
        f"\nSaved: {save_dir / (base_name + '_fp32.jpg')} and {save_dir / (base_name + '_int8.jpg')}"
    )

    if opt.diagnostics:
        in_fp32_stats = tensor_stats(im_fp32, is_uint8=False)
        in_int8_stats = tensor_stats(im_int8, is_uint8=False)
        print("\nInput stats FP32:", in_fp32_stats)
        print("Input stats INT8:", in_int8_stats)

        out_fp32_0 = runtime_fp32.run([runtime_fp32.get_outputs()[0].name],
                                      {runtime_fp32.get_inputs()[0].name: im_fp32})[0]
        out_int8_0_raw = runtime_int8.run([runtime_int8.get_outputs()[0].name],
                                          {runtime_int8.get_inputs()[0].name: im_int8})[0]

        sc = dq["scale"]
        zp = dq["zero_point"]
        ax = dq.get("axis", 1)
        out_int8_0 = dequantize_array(out_int8_0_raw, sc, zp, axis=ax)

        if opt.dump_head:
            np.save(str(save_dir / "fp32_head.npy"), out_fp32_0)
            np.save(str(save_dir / "int8_head_uint8.npy"), out_int8_0_raw)
            np.save(str(save_dir / "int8_head_deq.npy"), out_int8_0)

        nc, _ = get_nc_and_regmax(runtime_fp32, fallback_nc=len(names))
        diag_partition_stats(out_fp32_0, out_int8_0, nc)
        cs, mae, agree, topk_j = head_metrics(out_fp32_0, out_int8_0, names_nc=nc, topk=100)
        print("\nFinal head metrics:")
        print(
            "cosine:", f"{cs:.6f}" if not np.isnan(cs) else "nan", "mae:", f"{mae:.6f}",
            "argmax_agree:", "-" if agree is None else f"{agree:.6f}", "topk_jaccard:",
            "-" if topk_j is None else f"{topk_j:.6f}"
        )

        print("\nHead stats FP32:", tensor_stats(out_fp32_0, is_uint8=False))
        print("Head stats INT8 uint8:", tensor_stats(out_int8_0_raw, is_uint8=True))
        print("Head stats INT8 deq:", tensor_stats(out_int8_0, is_uint8=False))

        mae_r, cos_r = quant_dequant_roundtrip(out_fp32_0, sc, zp, axis=ax)
        print("\nFP32 quant-dequant roundtrip:", {"mae": mae_r, "cosine": cos_r})

        thresholds = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
        sweep_fp32 = threshold_sweep_counts(out_fp32_0, nc, thresholds)
        sweep_int8 = threshold_sweep_counts(out_int8_0, nc, thresholds)
        sweep_rows = []
        for i in range(len(thresholds)):
            tf, cf, mf = sweep_fp32[i]
            ti, ci, mi = sweep_int8[i]
            sweep_rows.append((tf, ci, f"{mi:.4f}", cf, f"{mf:.4f}"))
        print("\nDetection count vs threshold (INT8 vs FP32 on dequantized heads):")
        print_side_by_side_table(
            sweep_rows, ["thr", "INT8 count", "INT8 mean", "FP32 count", "FP32 mean"]
        )

        if opt.ablate_floatize_int8:
            pred = torch.from_numpy(out_int8_0.astype(np.float32)).float()
            dets = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, max_det=opt.max_det)
            have = any([len(d) for d in dets])
            print("\nAblation floatized-INT8 through float postprocess detections:", int(have))

def main(opt):
    compare_models(opt)

def parse_opt():
    parser = argparse.ArgumentParser(
        description="ONNX FP32 vs. INT8 Layer-By-Layer and Detection Comparison Tool"
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument(
        "--source", type=str, required=True, help="Path to a single image for comparison."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "toolchain/data/coco.yaml"),
        help="Path to dataset.yaml for INT8 calibration."
    )
    parser.add_argument(
        "--imgsz", "--img-size", nargs="+", type=int, default=[640], help="Inference size h,w."
    )
    parser.add_argument("--device", default="cpu", help="CUDA device, i.e., 0 or cpu.")
    parser.add_argument(
        "--quantize",
        default=True,
        action="store_true",
        help="Force re-quantization even if an INT8 model exists."
    )
    parser.add_argument(
        "--per-channel",
        default=True,
        action="store_true",
        help="Enable per-channel quantization for INT8 conversion."
    )
    parser.add_argument("--conf-thres", type=float, default=0.1, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument(
        "--project", default=ROOT / "runs/compare", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostics")
    parser.add_argument(
        "--ablate-floatize-int8",
        action="store_true",
        help="Run float postprocess on dequantized INT8 head"
    )
    parser.add_argument("--dump-head", action="store_true", help="Dump head tensors to npy")
    parser.add_argument(
        "--disable-optim", action="store_true", help="Disable ORT graph optimizations"
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Path to training .pt with int8_calib; if not set, will try sibling of --weights"
    )
    opt = parser.parse_args()
    opt.imgsz = check_img_size(opt.imgsz)
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
