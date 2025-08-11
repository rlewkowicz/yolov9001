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
import torch.nn.functional as F
import ast

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) in sys.path else FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import LoadImages
from utils.general import LOGGER, print_args, check_img_size, increment_path, scale_boxes, non_max_suppression
from utils.plots import Annotator, colors
from utils.tal.anchor_generator import make_anchors, dist2bbox

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
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) + 2 for i in range(len(headers))]
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
    if "CUDAExecutionProvider" in avail and opt_device != 'cpu':
        return ["CUDAExecutionProvider"] + (["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else [])
    return ["CPUExecutionProvider"]

def create_debug_session(model_path, providers, disable_optim=False):
    model_path = str(model_path)
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    tensor_type_map = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
    for tensor_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if tensor_info.name not in tensor_type_map:
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
    session = onnxruntime.InferenceSession(model.SerializeToString(), sess_options=so, providers=providers)
    return session, [output.name for output in session.get_outputs()]

def create_session(model_path, providers, disable_optim=False):
    so = onnxruntime.SessionOptions()
    if disable_optim:
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    return onnxruntime.InferenceSession(str(model_path), sess_options=so, providers=providers)

def list_all_tensors_no_nan_inf(model_path, candidate_image_paths, imgsz, providers):
    session, output_names = create_debug_session(str(model_path), providers)
    input_name = session.get_inputs()[0].name
    good = []
    loader = LoadImages(candidate_image_paths, img_size=imgsz, auto=False)
    for path, im, _, _, _ in tqdm(loader, desc="Screening images"):
        x = im.astype(np.float32) / 255.0
        if len(x.shape) == 3:
            x = np.expand_dims(x, 0)
        try:
            outs = session.run(output_names, {input_name: x})
            if not any(np.isnan(arr).any() or np.isinf(arr).any() for arr in outs):
                good.append(path)
        except Exception:
            pass
    LOGGER.info(f"[Calib screening] {len(good)}/{len(loader)} images passed (no NaN/Inf in any intermediate tensor).")
    return good

def force_uint8_outputs(model_path_in, model_path_out, calib_meta=None):
    model = onnx.load(model_path_in)
    graph = model.graph
    output_node = graph.output[0]
    if output_node.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
        onnx.save(model, model_path_out)
        return
    final_dequant = None
    for n in reversed(graph.node):
        if n.output and n.output[0] == output_node.name and n.op_type == "DequantizeLinear":
            final_dequant = n
            break
    if final_dequant is None:
        raise RuntimeError("Final DequantizeLinear not found")
    pre_dq_name = final_dequant.input[0]
    final_concat = None
    for n in reversed(graph.node):
        if n.output and n.output[0] == pre_dq_name and n.op_type == "Concat":
            final_concat = n
            break
    if final_concat is None:
        raise RuntimeError("Concat feeding final DequantizeLinear not found")
    cls_in_name = None
    box_in_name = None
    for inp in final_concat.input:
        src = None
        for n in reversed(graph.node):
            if n.output and n.output[0] == inp:
                src = n
                break
        if src is not None and src.op_type == "Hardsigmoid":
            cls_in_name = src.input[0]
        else:
            box_in_name = inp if box_in_name is None else box_in_name
    if cls_in_name is None:
        raise RuntimeError("Hardsigmoid feeding class branch not found")
    new_concat_out = "concat_boxes_logits"
    new_concat = onnx.helper.make_node(
        "Concat",
        inputs=[box_in_name, cls_in_name],
        outputs=[new_concat_out],
        name="Concat_Boxes_Logits",
        axis=1,
    )
    mins = np.array(calib_meta["min"], dtype=np.float32)
    maxs = np.array(calib_meta["max"], dtype=np.float32)
    scale = (maxs - mins) / 255.0
    scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
    zp = np.round(-mins / scale).astype(np.uint8)
    scale_name = "final_out_scale"
    zp_name = "final_out_zp"
    out_q_name = f"{output_node.name}_uint8"
    q_node = onnx.helper.make_node(
        "QuantizeLinear",
        inputs=[new_concat_out, scale_name, zp_name],
        outputs=[out_q_name],
        name="QuantizeLinear_FinalOutput",
        axis=1,
    )
    graph.node.remove(final_dequant)
    graph.node.remove(final_concat)
    graph.node.extend([new_concat, q_node])
    graph.initializer.extend([
        onnx.helper.make_tensor(scale_name, onnx.TensorProto.FLOAT, scale.shape, scale),
        onnx.helper.make_tensor(zp_name, onnx.TensorProto.UINT8, zp.shape, zp),
    ])
    output_node.name = out_q_name
    output_node.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    onnx.checker.check_model(model)
    onnx.save(model, model_path_out)
    
def get_nc_and_regmax(session, fallback_nc=80, fallback_regmax=None):
    nc = None
    regmax = fallback_regmax
    try:
        meta = session.get_modelmeta().custom_metadata_map
        names = eval(meta.get("names", "None"))
        if names is not None:
            nc = len(names)
    except Exception:
        pass
    if nc is None:
        nc = fallback_nc
    return nc, regmax
    
def dq_from_pt(pt_path, expect_channels=None):
    try:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    calib_root = ckpt.get("int8_calib", {}) if isinstance(ckpt, dict) else {}
    calib = calib_root.get("head_logits", None)
    if calib is None:
        calib = calib_root.get("head", None)
    if not calib or "min" not in calib or "max" not in calib:
        return None
    mins = np.array(calib["min"], dtype=np.float32).reshape(-1)
    maxs = np.array(calib["max"], dtype=np.float32).reshape(-1)
    if expect_channels is not None and mins.shape[0] != expect_channels:
        return None
    scale = (maxs - mins) / 255.0
    scale = np.where(scale <= 1e-12, 1.0, scale).astype(np.float32)
    zp = np.round(-mins / scale).clip(0, 255).astype(np.uint8)
    return {"scale": scale, "zero_point": zp, "axis": 1}


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
    stride = int(tmp_sess.get_modelmeta().custom_metadata_map.get("stride", 32))
    del tmp_sess
    good_calib_images = list_all_tensors_no_nan_inf(model_input_path, candidate_image_paths, opt.imgsz, providers)
    if not good_calib_images:
        raise ValueError("No stable images found for calibration.")
    
    tmp_q_path = model_input_path.with_stem(f"{model_input_path.stem}_int8_tmp")
    LOGGER.info("Running Quantization (MinMax, QOperator, Act=QUInt8, W=QInt8)...")
    quantize_static(
        model_input=str(model_input_path),
        model_output=str(tmp_q_path),
        calibration_data_reader=ImageCalibrator(calib_files=good_calib_images, input_name=input_name, imgsz=opt.imgsz, stride=stride),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=opt.per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True, "EnableSubgraph": True}
    )
    
    pt_path = opt.pt or str(Path(model_input_path).with_suffix(".pt"))
    calib_meta = None
    if pt_path and Path(pt_path).exists():
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        if ckpt and 'int8_calib' in ckpt:
            calib_meta = ckpt['int8_calib'].get('head')
    if calib_meta:
        LOGGER.info("Using PT checkpoint calibration for final output quantization.")
    else:
        raise ValueError(f"Could not load valid calibration data from {pt_path}. Please regenerate it.")
    
    LOGGER.info(f"Writing final INT8 model with uint8 outputs -> {model_output_path_uint8}")
    force_uint8_outputs(str(tmp_q_path), str(model_output_path_uint8), calib_meta=calib_meta)
    
    if tmp_q_path.exists():
        os.remove(tmp_q_path)
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
    x = im.astype(np.float32) / 255.0
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)
    return x

def dequantize_array(arr_uint8, scale, zp, axis=1):
    x = arr_uint8.astype(np.float32)
    s = np.array(scale, dtype=np.float32)
    z = np.array(zp, dtype=np.float32)
    if s.ndim == 0 or s.size == 1:
        return (x - z.item()) * s.item()
    else:
        shape = [1] * x.ndim
        shape[axis] = -1
        return (x - z.reshape(shape)) * s.reshape(shape)

def diag_partition_stats(fp32_head, int8_head_deq, nc):
    a = fp32_head
    b = int8_head_deq
    
    box_channels = a.shape[1] - nc
    LOGGER.info(f"Interpreting output tensor with {box_channels} box channels and {nc} class channels.")
    
    a_box, a_cls_prob = a[:, :box_channels, :], a[:, box_channels:, :]
    b_box, b_cls_logits = b[:, :box_channels, :], b[:, box_channels:, :]
    
    b_cls_prob = 1 / (1 + np.exp(-b_cls_logits))

    def stats(x):
        xf = x.flatten()
        return {"min": float(xf.min()), "max": float(xf.max()), "mean": float(xf.mean()), "std": float(xf.std())}

    print("\nBox partition stats FP32:", stats(a_box))
    print("Box partition stats INT8 deq:", stats(b_box))
    print(f"Box partition cosine: {safe_cosine(a_box.flatten(), b_box.flatten()):.6f}")
    
    print("\nClass partition stats FP32 (probs):", stats(a_cls_prob))
    print("Class partition stats INT8 deq (logits):", stats(b_cls_logits))
    print("Class partition stats INT8 deq (probs):", stats(b_cls_prob))
    print(f"Class partition cosine (probs): {safe_cosine(a_cls_prob.flatten(), b_cls_prob.flatten()):.6f}")

def non_max_suppression_int8(output_uint8, dq, names_nc, conf_thres=0.1, iou_thres=0.45, max_det=300):
    dequantized_logits = dequantize_array(output_uint8, dq["scale"], dq["zero_point"], axis=dq.get("axis", 1))
    
    box_channels = dequantized_logits.shape[1] - names_nc
    box_part = dequantized_logits[:, :box_channels, :]
    cls_part = dequantized_logits[:, box_channels:, :]
    
    cls_probs = 1 / (1 + np.exp(-cls_part))
    
    final_pred = np.concatenate([box_part, cls_probs], axis=1)
    
    return non_max_suppression(torch.from_numpy(final_pred), conf_thres, iou_thres, max_det=max_det)

def run_detect_one(session, im, im0, conf_thres, iou_thres, max_det, save_dir, save_name, names, dq=None):
    input_name = session.get_inputs()[0].name
    out_meta = session.get_outputs()[0]
    output = session.run([out_meta.name], {input_name: im})[0]

    if out_meta.type == "tensor(uint8)":
        dets = non_max_suppression_int8(output, dq, len(names), conf_thres, iou_thres, max_det)
    else:
        dets = non_max_suppression(torch.from_numpy(output), conf_thres, iou_thres, max_det=max_det)

    annotator = Annotator(im0.copy(), line_width=3, example=str(names))
    top5 = []
    for det in dets:
        if len(det):
            im_shape = im.shape[2:]
            det[:, :4] = scale_boxes(im_shape, det[:, :4], annotator.im.shape).round()
            det_sorted = det[det[:, 4].argsort(descending=True)]
            for *xyxy, conf, cls in det_sorted[:5]:
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(c, True))
                top5.append((names[c], float(conf), [float(k) for k in xyxy]))
    cv2.imwrite(str(save_dir / save_name), annotator.result())
    return top5

def main(opt):
    compare_models(opt)

def compare_models(opt):
    fp32_path = opt.weights
    int8_path_obj = Path(fp32_path).with_stem(f"{Path(fp32_path).stem}_int8_io_uint8")
    
    pt_path = opt.pt or str(Path(fp32_path).with_suffix(".pt"))
    if not (pt_path and Path(pt_path).exists()):
        raise FileNotFoundError(f"PT file with calibration data not found at {pt_path}")

    if opt.quantize or not int8_path_obj.exists():
        int8_path = quantize_model(opt)
    else:
        int8_path = str(int8_path_obj)
        LOGGER.info(f"Found existing INT8 model at {int8_path}, skipping quantization.")

    providers = get_providers(opt.device)
    runtime_fp32 = create_session(fp32_path, providers, disable_optim=opt.disable_optim)
    runtime_int8 = create_session(int8_path, providers, disable_optim=opt.disable_optim)
    
    im_fp32 = prepare_input_for_session(runtime_fp32, opt.source, opt.imgsz)
    
    out_shape = runtime_fp32.get_outputs()[0].shape
    expect_c = out_shape[1] if len(out_shape) >= 2 else None
    dq = dq_from_pt(pt_path, expect_channels=expect_c)
    if not dq:
        raise ValueError("Failed to extract valid dequantization parameters from PT file.")
    print_dq("Selected from PT", dq)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(opt.source).stem
    
    names = [str(i) for i in range(80)]
    try:
        meta = runtime_fp32.get_modelmeta().custom_metadata_map
        names = eval(meta.get("names", str(names)))
    except Exception:
        pass

    im0 = cv2.imread(opt.source)
    fp32_top5 = run_detect_one(runtime_fp32, im_fp32, im0, opt.conf_thres, opt.iou_thres, opt.max_det, save_dir, f"{base_name}_fp32.jpg", names)
    int8_top5 = run_detect_one(runtime_int8, im_fp32, im0, opt.conf_thres, opt.iou_thres, opt.max_det, save_dir, f"{base_name}_int8.jpg", names, dq=dq)
    
    det_rows = []
    for i in range(max(len(fp32_top5), len(int8_top5))):
        lf = fp32_top5[i] if i < len(fp32_top5) else ("-", "-", "-")
        ri = int8_top5[i] if i < len(int8_top5) else ("-", "-", "-")
        det_rows.append((i + 1, lf[0], f"{lf[1]:.3f}" if isinstance(lf[1], float) else "-", str(lf[2]), ri[0], f"{ri[1]:.3f}" if isinstance(ri[1], float) else "-", str(ri[2])))
    print("\nTop-5 detections (side-by-side):")
    print_side_by_side_table(det_rows, ["Rank", "FP32 class", "FP32 conf", "FP32 box [xyxy]", "INT8 class", "INT8 conf", "INT8 box [xyxy]"])
    print(f"\nSaved: {save_dir / (base_name + '_fp32.jpg')} and {save_dir / (base_name + '_int8.jpg')}")

    if opt.diagnostics:
        out_fp32_0 = runtime_fp32.run(None, {runtime_fp32.get_inputs()[0].name: im_fp32})[0]
        out_int8_0_raw = runtime_int8.run(None, {runtime_int8.get_inputs()[0].name: im_fp32})[0]
        out_int8_0_deq = dequantize_array(out_int8_0_raw, dq["scale"], dq["zero_point"], axis=dq.get("axis", 1))
        nc, _ = get_nc_and_regmax(runtime_fp32, fallback_nc=len(names))
        diag_partition_stats(out_fp32_0, out_int8_0_deq, nc)

def print_dq(label, dq):
    if dq is None:
        print(f"\n[Dequant] {label}: None")
        return
    sc = dq["scale"]
    zp = dq["zero_point"]
    ax = dq.get("axis", "N/A")
    if isinstance(sc, np.ndarray):
        print(f"\n[Dequant] {label}: axis={ax}, per-channel={sc.shape[0]} | scale[min={sc.min():.6g}, max={sc.max():.6g}] zp[min={int(zp.min())}, max={int(zp.max())}]")
    else:
        print(f"\n[Dequant] {label}: axis={ax}, per-tensor | scale={float(sc):.6g}, zp={int(zp)}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument("--source", type=str, required=True, help="Path to a single image for comparison.")
    parser.add_argument("--data", type=str, default=str(ROOT / "data/coco.yaml"), help="Path to dataset.yaml for INT8 calibration.")
    parser.add_argument("--imgsz", "--img-size", nargs="+", type=int, default=[640], help="Inference size h,w.")
    parser.add_argument("--device", default="cpu", help="CUDA device, i.e., 0 or cpu.")
    parser.add_argument("--quantize", action="store_true", help="Force re-quantization even if an INT8 model exists.")
    parser.add_argument("--per-channel", default=True, action="store_true", help="Enable per-channel quantization for INT8 conversion.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--project", default=ROOT / "runs/compare", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostics")
    parser.add_argument("--disable-optim", action="store_true", help="Disable ORT graph optimizations")
    parser.add_argument("--pt", type=str, default=None, help="Path to training .pt with int8_calib; if not set, will try sibling of --weights")
    opt = parser.parse_args()
    opt.imgsz = check_img_size(opt.imgsz)
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)