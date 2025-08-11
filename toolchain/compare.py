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
from onnx import helper, numpy_helper, shape_inference, TensorProto
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.quant_utils import model_has_pre_process_metadata
from onnxruntime.quantization.shape_inference import quant_pre_process
from scipy.spatial.distance import cosine
from tqdm import tqdm

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


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
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
    if "CUDAExecutionProvider" in avail and opt_device != "cpu":
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

    so = onnxruntime.SessionOptions()
    if disable_optim:
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        model.SerializeToString(), sess_options=so, providers=providers
    )
    output_names = [output.name for output in session.get_outputs()]
    return session, output_names


def create_session(model_path, providers, disable_optim=False):
    so = onnxruntime.SessionOptions()
    if disable_optim:
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(str(model_path), sess_options=so, providers=providers)


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


def _producers_map(graph):
    prod = {}
    for n in graph.node:
        for o in n.output:
            prod[o] = n
    return prod


def _consumers_map(graph):
    cons = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return cons


def _rank_of(graph, name):
    vis = list(graph.value_info) + list(graph.input) + list(graph.output)
    for vi in vis:
        if vi.name == name:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                return len(tt.shape.dim)
    return None


def _walk_back_passthrough(prod_map, start_tensor):
    passthrough = {
        "Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Cast",
        "QuantizeLinear", "DequantizeLinear"
    }
    t = start_tensor
    node = prod_map.get(t, None)
    hops = 0
    while node is not None and node.op_type in passthrough and hops < 256:
        if not node.input:
            break
        t = node.input[0]
        node = prod_map.get(t, None)
        hops += 1
    return node, t


def force_uint8_outputs(model_path_in, model_path_out, calib_meta):
    """
    Converter-friendly tail surgery with calibrated uint8 output (logits):
      - Recover logits (input to Sigmoid/HardSigmoid).
      - Transpose each branch to [N,S,C], Concat on the channel axis, Transpose back to [N,C,S].
      - QuantizeLinear with per-channel qparams (class channels forced symmetric).
      - Prune the entire old tail downstream of original Concat/DQ to avoid orphan Q/DQ nodes.
    """
    if calib_meta is None or "min" not in calib_meta or "max" not in calib_meta:
        raise ValueError("force_uint8_outputs: calib_meta with 'min' and 'max' is required.")

    m = onnx.load(model_path_in)
    m = shape_inference.infer_shapes(m)
    g = m.graph
    prod = _producers_map(g)
    cons = _consumers_map(g)

    if not g.output:
        raise RuntimeError("Model has no outputs.")
    out_vi = g.output[0]
    float_out_name = out_vi.name

    dq = prod.get(float_out_name, None)
    if not dq or dq.op_type != "DequantizeLinear":
        raise RuntimeError("Expected final DequantizeLinear feeding the model output.")

    parent_node, _ = _walk_back_passthrough(prod, dq.input[0])
    if parent_node is None or parent_node.op_type not in ("Concat", "QLinearConcat"):
        raise RuntimeError("Tail pattern unsupported: expected (Q)Concat near model output.")

    orig_concat = parent_node

    # Helpers for identity-safe node collection/removal (NodeProto is unhashable)
    def _append_unique_node(lst, node):
        if node is None:
            return
        for n in lst:
            if n is node:
                return
        lst.append(node)

    def _safe_remove_node(graph, node):
        try:
            graph.node.remove(node)
        except ValueError:
            pass

    # Collect all downstream nodes from the original tail (orig Concat and its terminal DQ)
    to_remove = []
    queue_vals = list(orig_concat.output) + list(dq.output)
    _append_unique_node(to_remove, orig_concat)
    _append_unique_node(to_remove, dq)
    while queue_vals:
        t = queue_vals.pop()
        for cnode in cons.get(t, []):
            _append_unique_node(to_remove, cnode)
            queue_vals.extend(list(cnode.output))

    # Gather Concat data inputs
    if orig_concat.op_type == "Concat":
        data_inputs = list(orig_concat.input)
    else:
        pins = list(orig_concat.input)
        if len(pins) < 5 or (len(pins) - 2) % 3 != 0:
            raise RuntimeError("QLinearConcat inputs not in expected triplets after first two.")
        # Take only data tensors: [y_scale, y_zp, x0, x0_scale, x0_zp, x1, x1_scale, x1_zp, ...]
        data_inputs = [pins[i] for i in range(2, len(pins), 3)]

    # Identify class prob branch (Sigmoid/HardSigmoid) and take its logits input
    box_float_inputs = []
    cls_logits_float = None
    for t in data_inputs:
        n, t_float = _walk_back_passthrough(prod, t)
        if n is not None and n.op_type in ("HardSigmoid", "Sigmoid"):
            _, logits_t = _walk_back_passthrough(prod, n.input[0])
            cls_logits_float = logits_t
        else:
            box_float_inputs.append(t_float)
    if cls_logits_float is None:
        raise RuntimeError("Could not find class probability branch (HardSigmoid/Sigmoid).")

    candidate_inputs = box_float_inputs + [cls_logits_float]

    # Transpose every input to [N,S,C] (from typical [N,C,S]); keep shapes already [N,S,C]
    transposed_inputs = []
    for idx, tin in enumerate(candidate_inputs):
        r = _rank_of(g, tin)
        if r == 3:
            t_out = f"{tin}_toNSC"
            tp = helper.make_node(
                "Transpose", inputs=[tin], outputs=[t_out],
                name=f"Transpose_toNSC_{idx}", perm=[0, 2, 1]
            )
            g.node.append(tp)
            transposed_inputs.append(t_out)
        else:
            # Unknown rank or already [N,S,C]; pass through
            transposed_inputs.append(tin)

    # New Concat in [N,S,C] space on channel axis (axis=2 to avoid negative-axis headaches in converters)
    nsc_concat_out = float_out_name + "_NSC_concat"
    new_concat = helper.make_node(
        "Concat",
        inputs=transposed_inputs,
        outputs=[nsc_concat_out],
        name="New_Concat_Logits_NSC",
        axis=2,
    )

    # Transpose back to [N,C,S]
    ncs_out = float_out_name + "_NCS"
    tp_back = helper.make_node(
        "Transpose",
        inputs=[nsc_concat_out],
        outputs=[ncs_out],
        name="Transpose_back_to_NCS",
        perm=[0, 2, 1],
    )

    # Build output scale/zp from calib_meta; force symmetric class logits (channels >= 4)
    mins = np.array(calib_meta["min"], dtype=np.float32)
    maxs = np.array(calib_meta["max"], dtype=np.float32)
    if mins.ndim == 1 and mins.size > 4:
        cls_start = 4
        a = np.maximum(np.abs(mins[cls_start:]), np.abs(maxs[cls_start:]))
        a = np.maximum(a, 1e-6)
        mins[cls_start:] = -a
        maxs[cls_start:] = a

    scale = (maxs - mins) / 255.0
    scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
    zp = np.round(-mins / scale).astype(np.int32)
    zp = np.clip(zp, 0, 255).astype(np.uint8)

    scale_name = "final_out_scale"
    zp_name = "final_out_zp"
    out_uint8 = float_out_name + "_uint8"

    # Remove any old inits with the same names
    for i in range(len(g.initializer) - 1, -1, -1):
        if g.initializer[i].name in (scale_name, zp_name):
            del g.initializer[i]

    g.initializer.extend([
        onnx.helper.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale.shape, scale),
        onnx.helper.make_tensor(zp_name, onnx.TensorProto.UINT8, zp.shape, zp),
    ])

    q_node = helper.make_node(
        "QuantizeLinear",
        inputs=[ncs_out, scale_name, zp_name],
        outputs=[out_uint8],
        name="QuantizeLinear_FinalOutput",
    )
    # Per-channel along channel axis (N,C,S => axis=1)
    if scale.ndim > 0 and scale.size > 1:
        q_node.attribute.append(helper.make_attribute("axis", 1))

    # Prune the old tail AFTER we have added any helper Transposes above (they read upstream tensors)
    for n in list(to_remove):
        _safe_remove_node(g, n)

    # Append new tail nodes at the end to keep topo order (inputs already produced upstream)
    g.node.extend([new_concat, tp_back, q_node])

    # Switch model output to our uint8 tensor
    out_vi.name = out_uint8
    out_vi.type.tensor_type.elem_type = TensorProto.UINT8

    # Re-infer shapes and validate
    m = shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, model_path_out)
    LOGGER.info(
        "Successfully restructured ONNX graph (NSC concat, logits->uint8 with calibrated qparams) and saved to %s",
        model_path_out,
    )


def dq_from_pt(pt_path, expect_channels=None):
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
        return {"scale": float(sc), "zero_point": int(zp)}


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
            f"scale[min={sc.min():.6g}, max={sc.max():.6g}] zp[min={int(zp.min())}, max={int(zp.max())}]"
        )
    else:
        print(f"\n[Dequant] {label}: axis={ax}, per-tensor | scale={float(sc):.6g}, zp={int(zp)}")


def quantize_model(opt):
    model_input_path = Path(opt.weights)
    model_output_path_uint8 = model_input_path.with_stem(f"{model_input_path.stem}_int8_qdq_io_uint8")

    data_yaml_path = Path(opt.data)
    with open(data_yaml_path, errors="ignore") as f:
        data_dict = yaml.safe_load(f)
    dataset_root = (data_yaml_path.parent / ".." / data_dict["path"]).resolve()
    calib_data_list_path = dataset_root / data_dict["train"]
    with open(calib_data_list_path) as f:
        candidate_files = [line.strip() for line in f][:1000]
    candidate_image_paths = [str(dataset_root / p) for p in candidate_files]

    providers = get_providers(opt.device)
    tmp_sess = onnxruntime.InferenceSession(str(model_input_path), providers=providers)
    input_name = tmp_sess.get_inputs()[0].name
    model_meta = tmp_sess.get_modelmeta().custom_metadata_map
    stride = int(model_meta.get("stride", 32))
    out_shape = tmp_sess.get_outputs()[0].shape
    expect_c = out_shape[1] if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2 and isinstance(out_shape[1], int) else None
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
    LOGGER.info("Running Quantization (MinMax, QDQ, Act=QUInt8, W=QInt8)...")
    quantize_static(
        model_input=str(model_path_for_quant),
        model_output=str(tmp_q_path),
        calibration_data_reader=ImageCalibrator(
            calib_files=good_calib_images, input_name=input_name, imgsz=opt.imgsz, stride=stride
        ),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=opt.per_channel,
        nodes_to_exclude=[],
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
            "ForceQuantizeNoInputCheck": True,
        },
    )

    pt_path = opt.pt or str(Path(model_input_path).with_suffix(".pt"))
    dq = dq_from_pt(pt_path, expect_channels=expect_c) if pt_path and Path(pt_path).exists() else None
    if dq is None:
        raise ValueError(
            "PT checkpoint with int8_calib.head {min,max} is required for output quantization."
        )
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
    total_channels = a.shape[1]
    box_channels = total_channels - nc
    LOGGER.info(
        f"Interpreting output tensor with {box_channels} box channels and {nc} class channels."
    )
    a_box, a_cls = a[:, :box_channels, :], a[:, box_channels:, :]
    b_box, b_cls = b[:, :box_channels, :], b[:, box_channels:, :]

    def stats(x):
        xf = x.reshape(-1).astype(np.float32)
        if xf.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        return float(xf.min()), float(xf.max()), float(xf.mean()), float(xf.std())

    ab_min, ab_max, ab_mean, ab_std = stats(a_box)
    bb_min, bb_max, bb_mean, bb_std = stats(b_box)
    ac_min, ac_max, ac_mean, ac_std = stats(a_cls)
    bc_min, bc_max, bc_mean, bc_std = stats(b_cls)
    ca = safe_cosine(a_box.flatten(), b_box.flatten())
    cc = safe_cosine(a_cls.flatten(), 1.0 / (1.0 + np.exp(-b_cls)).flatten())
    print(
        "\nBox partition stats FP32:",
        {"min": ab_min, "max": ab_max, "mean": ab_mean, "std": ab_std}
    )
    print(
        "Box partition stats INT8 deq:",
        {"min": bb_min, "max": bb_max, "mean": bb_mean, "std": bb_std}
    )
    print(f"Box partition cosine: {ca:.6f}")
    print(
        "\nClass partition stats FP32:",
        {"min": ac_min, "max": ac_mean, "mean": ac_mean, "std": ac_std}
    )
    print(
        "Class partition stats INT8 deq:",
        {"min": bc_min, "max": bc_max, "mean": bc_mean, "std": bc_std}
    )
    print(f"Class partition cosine: {cc:.6f}")


def non_max_suppression_int8(
    output_uint8, scale, zp, axis=1, names_nc=None, conf_thres=0.1, iou_thres=0.45, max_det=300
):
    x = output_uint8.astype(np.float32)
    s = np.array(scale, dtype=np.float32)
    z = np.array(zp, dtype=np.float32)
    if s.ndim == 0 or s.size == 1:
        s_val = s.item() if s.ndim > 0 else s
        z_val = z.item() if z.ndim > 0 else z
        dequantized_x = (x - z_val) * s_val
    else:
        shape = [1] * x.ndim
        if axis < 0:
            axis += x.ndim
        if axis >= x.ndim or axis < 0:
            raise ValueError(f"Invalid axis {axis} for a tensor with {x.ndim} dimensions.")
        shape[axis] = -1
        s = s.reshape(shape)
        z = z.reshape(shape)
        dequantized_x = (x - z) * s
    nc = int(names_nc)
    box_channels = dequantized_x.shape[1] - nc
    box_part = dequantized_x[:, :box_channels, :]
    cls_logits = dequantized_x[:, box_channels:, :]
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
    final_pred = np.concatenate([box_part, cls_probs], axis=1)
    dets = non_max_suppression(
        torch.from_numpy(final_pred).float(), conf_thres, iou_thres, max_det=max_det
    )
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
            raise RuntimeError("uint8 output requires dq parameters from PT/graph")
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
            max_det=max_det,
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


def dequantize_array(arr_uint8, scale, zp, axis=1):
    x = arr_uint8.astype(np.float32)
    s = np.array(scale, dtype=np.float32)
    z = np.array(zp, dtype=np.float32)
    if s.ndim == 0 or s.size == 1:
        s_val = s.item() if s.ndim > 0 else s
        z_val = z.item() if z.ndim > 0 else z
        return (x - z_val) * s_val
    else:
        shape = [1] * x.ndim
        if axis < 0:
            axis += x.ndim
        if axis >= x.ndim or axis < 0:
            raise ValueError(f"Invalid axis {axis} for a tensor with {x.ndim} dimensions.")
        shape[axis] = -1
        s = s.reshape(shape)
        z = z.reshape(shape)
        return (x - z) * s


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


def get_uint8_output_qparams(model_path: str):
    m = onnx.load(model_path)
    g = m.graph
    if not g.output:
        return None
    out_name = g.output[0].name

    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n

    passthrough = {"Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Cast"}
    node = prod.get(out_name, None)
    hops = 0
    while node is not None and node.op_type in passthrough and hops < 10:
        if not node.input:
            break
        node = prod.get(node.input[0], None)
        hops += 1

    if node is None or node.op_type != "QuantizeLinear":
        return None
    if len(node.input) < 3:
        return None
    scale_name = node.input[1]
    zp_name = node.input[2]

    def find_init(name):
        for ini in g.initializer:
            if ini.name == name:
                return numpy_helper.to_array(ini)
        return None

    scale = find_init(scale_name)
    zp = find_init(zp_name)
    if scale is None or zp is None:
        return None

    axis = 1
    for a in node.attribute:
        if a.name == "axis":
            axis = int(a.i)

    if scale.size == 1 and zp.size == 1:
        return {
            "scale": float(scale.reshape(())), "zero_point": int(np.uint8(zp.reshape(()))), "axis":
                axis
        }
    else:
        return {
            "scale": scale.astype(np.float32).reshape(-1), "zero_point":
                zp.astype(np.uint8).reshape(-1), "axis": axis
        }


def quant_dequant_roundtrip(fp32_head, scale, zp, axis=1):
    x = fp32_head.astype(np.float32)
    s = np.array(scale, dtype=np.float32)
    z = np.array(zp, dtype=np.float32)
    if s.ndim == 0 or s.size == 1:
        s_val = s.item() if s.ndim > 0 else s
        z_val = z.item() if z.ndim > 0 else z
        q = np.clip(np.round(x / s_val) + z_val, 0, 255).astype(np.uint8)
        x2 = (q.astype(np.float32) - z_val) * s_val
    else:
        shape = [1] * x.ndim
        if axis < 0:
            axis += x.ndim
        if axis >= x.ndim or axis < 0:
            raise ValueError(f"Invalid axis {axis} for a tensor with {x.ndim} dimensions.")
        shape[axis] = -1
        s_r = s.reshape(shape)
        z_r = z.reshape(shape)
        q = np.clip(np.round(x / s_r) + z_r, 0, 255).astype(np.uint8)
        x2 = (q.astype(np.float32) - z_r) * s_r
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
    int8_path_obj = Path(fp32_path).with_stem(f"{Path(fp32_path).stem}_int8_qdq_io_uint8")
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

    dq = get_uint8_output_qparams(int8_path)
    if dq is None:
        raise RuntimeError("Could not read final output scale/zp from INT8 graph.")
    print_dq("Selected from INT8 graph", dq)

    outputs_fp32 = run_and_collect_outputs(
        session_fp32_dbg, input_name_fp32, im_fp32, fp32_output_names
    )
    outputs_int8 = run_and_collect_outputs(
        session_int8_dbg,
        session_int8_dbg.get_inputs()[0].name, im_int8, int8_output_names
    )

    print("\n" + "=" * 45 + " ONNX LAYER-BY-LAYER COMPARISON " + "=" * 45)
    print(f"{'Layer (Tensor Name)':<75} {'Cosine Sim':>15s} {'MAE':>15s}")
    print("-" * 110)
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

    print("\n" + "=" * 45 + " LAST 20 LAYER STATS DUMP " + "=" * 45)
    last_20_names = common_names[-20:]
    for name in last_20_names:
        fp32_val = outputs_fp32.get(name)
        int8_val = outputs_int8.get(name)
        if fp32_val is None or int8_val is None or fp32_val.size == 0:
            continue
        print(f"\n--- Layer: {name} ---")
        print(f"  Shape: {fp32_val.shape}")
        print(f"  FP32 Stats: {tensor_stats(fp32_val, is_uint8=False)}")
        print(f"  INT8 Stats: {tensor_stats(int8_val, is_uint8=False)}")

    print("\n" + "=" * 100)
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
        dq=get_uint8_output_qparams(int8_path),
    )

    det_rows = []
    for i in range(max(len(fp32_top5), len(int8_top5))):
        lf = fp32_top5[i] if i < len(fp32_top5) else ("-", "-", "-")
        ri = int8_top5[i] if i < len(int8_top5) else ("-", "-", "-")
        det_rows.append((
            i + 1,
            lf[0],
            f"{lf[1]:.3f}" if isinstance(lf[1], float) else "-",
            str(lf[2]),
            ri[0],
            f"{ri[1]:.3f}" if isinstance(ri[1], float) else "-",
            str(ri[2]),
        ))
    print("\nTop-5 detections (side-by-side):")
    print_side_by_side_table(
        det_rows,
        [
            "Rank", "FP32 class", "FP32 conf", "FP32 box [xyxy]", "INT8 class", "INT8 conf",
            "INT8 box [xyxy]"
        ],
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

        graph_dq = get_uint8_output_qparams(int8_path)
        sc = graph_dq["scale"]
        zp = graph_dq["zero_point"]
        ax = graph_dq.get("axis", 1)
        out_int8_0 = dequantize_array(out_int8_0_raw, sc, zp, axis=ax)

        nc, _ = get_nc_and_regmax(runtime_fp32, fallback_nc=len(names))
        diag_partition_stats(out_fp32_0, out_int8_0, nc)
        cs, mae, agree, topk_j = head_metrics(out_fp32_0, out_int8_0, names_nc=nc, topk=100)
        print("\nFinal head metrics:")
        print(
            "cosine:",
            f"{cs:.6f}" if not np.isnan(cs) else "nan",
            "mae:",
            f"{mae:.6f}",
            "argmax_agree:",
            "-" if agree is None else f"{agree:.6f}",
            "topk_jaccard:",
            "-" if topk_j is None else f"{topk_j:.6f}",
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
        help="Path to dataset.yaml for INT8 calibration.",
    )
    parser.add_argument(
        "--imgsz", "--img-size", nargs="+", type=int, default=[640], help="Inference size h,w."
    )
    parser.add_argument("--device", default="cpu", help="CUDA device, i.e., 0 or cpu.")
    parser.add_argument(
        "--quantize",
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
        help="Run float postprocess on dequantized INT8 head",
    )
    parser.add_argument("--dump-head", action="store_true", help="Dump head tensors to npy")
    parser.add_argument(
        "--disable-optim", action="store_true", help="Disable ORT graph optimizations"
    )
    parser.add_argument(
        "--pt",
        type=str,
        default=None,
        help="Path to training .pt with int8_calib; if not set, will try sibling of --weights",
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
