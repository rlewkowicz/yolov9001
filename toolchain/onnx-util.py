import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import torch
from tqdm import tqdm
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if str(FILE.parents[1]) in sys.path else FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadStreams,
    create_dataloader,
)
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    colorstr,
    increment_path,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ap_per_class, box_iou
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

_NUMPY_FROM_ORT = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}

def _to_cfirst(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 3:
        return a.reshape(a.shape[0], a.shape[1], -1)
    return a if a.shape[1] < a.shape[2] else np.transpose(a, (0, 2, 1))

def _get_uint8_output_qparams_for_output(model_path: str, out_name: str):
    try:
        m = onnx.load(model_path)
    except Exception:
        return None
    g = m.graph
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    node = prod.get(out_name, None)
    passthrough = {"Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Cast"}
    hops = 0
    while node is not None and node.op_type in passthrough and hops < 10:
        if not node.input:
            break
        node = prod.get(node.input[0], None)
        hops += 1
    if node is None or node.op_type != "QuantizeLinear" or len(node.input) < 3:
        return None
    sc_name, zp_name = node.input[1], node.input[2]
    def find_init(name):
        for ini in g.initializer:
            if ini.name == name:
                return numpy_helper.to_array(ini)
        return None
    scale = find_init(sc_name)
    zp = find_init(zp_name)
    if scale is None or zp is None:
        return None
    axis = 1
    for a in node.attribute:
        if a.name == "axis":
            axis = int(a.i)
    if scale.size == 1 and zp.size == 1:
        return {"scale": float(scale.reshape(())), "zero_point": int(np.uint8(zp.reshape(()))), "axis": axis}
    else:
        return {"scale": scale.astype(np.float32).reshape(-1), "zero_point": zp.astype(np.uint8).reshape(-1), "axis": axis}

def _floatize_uint8_logits(y_uint8: np.ndarray, qparams: dict) -> np.ndarray:
    x = y_uint8.astype(np.float32)
    s = np.array(qparams["scale"], dtype=np.float32)
    z = np.array(qparams["zero_point"], dtype=np.float32)
    axis = int(qparams.get("axis", 1))
    if s.ndim <= 0 or s.size == 1:
        s_val = s.item() if s.ndim > 0 else float(s)
        z_val = z.item() if s.ndim > 0 else float(z)
        x = (x - z_val) * s_val
    else:
        if axis < 0:
            axis += x.ndim
        shape = [1] * x.ndim
        shape[axis] = -1
        s = s.reshape(shape)
        z = z.reshape(shape)
        x = (x - z) * s
    return x

def _infer_input(session: onnxruntime.InferenceSession, model_path: str):
    inp = session.get_inputs()[0]
    input_name = inp.name
    ort_type = inp.type
    np_dtype = _NUMPY_FROM_ORT.get(ort_type, np.float32)
    if np_dtype in (np.float32, np.float16):
        mode = "float"
    elif np_dtype == np.uint8:
        mode = "uint8"
    else:
        mode = "float"
    return input_name, np_dtype, mode

def _prep_input_numpy(im_chw: np.ndarray, np_dtype, mode: str) -> np.ndarray:
    if im_chw.ndim == 3:
        im_chw = np.expand_dims(im_chw, 0)
    if mode == "float":
        out = im_chw.astype(np.float32) / 255.0
        if np_dtype == np.float16:
            out = out.astype(np.float16)
        elif np_dtype != np.float32:
            out = out.astype(np_dtype)
        return out
    elif mode == "uint8":
        return im_chw.astype(np.uint8)
    else:
        return im_chw.astype(np.float32) / 255.0

def _run_dual_outputs(session, ort_input, names_hint=None):
    outs = session.get_outputs()
    out_names = [o.name for o in outs]
    out_types = {o.name: o.type for o in outs}
    if "boxes" in out_names and "logits" in out_names:
        y_boxes, y_logits = session.run(["boxes", "logits"], {session.get_inputs()[0].name: ort_input})
        return ("boxes", _to_cfirst(y_boxes).astype(np.float32), out_types["boxes"]), ("logits", _to_cfirst(y_logits).astype(np.float32), out_types["logits"])
    ys = session.run(out_names, {session.get_inputs()[0].name: ort_input})
    ys_c = [_to_cfirst(y) for y in ys]
    idx_boxes = None
    for i, a in enumerate(ys_c):
        if a.shape[1] == 4:
            idx_boxes = i
            break
    if idx_boxes is None:
        idx_boxes = 0
    idx_logits = 1 - idx_boxes if len(ys_c) == 2 else next(i for i in range(len(ys_c)) if i != idx_boxes)
    name_b, name_l = out_names[idx_boxes], out_names[idx_logits]
    return (name_b, ys_c[idx_boxes].astype(np.float32), out_types[name_b]), (name_l, ys_c[idx_logits].astype(np.float32), out_types[name_l])

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _custom_nms_from_dual(boxes_cfirst, logits_cfirst, conf_thres, iou_thres, max_det=300):
    B = boxes_cfirst.shape[0]
    S = boxes_cfirst.shape[2]
    NC = logits_cfirst.shape[1]
    out = []
    for b in range(B):
        bx = boxes_cfirst[b].T
        lg = logits_cfirst[b].T
        probs = _sigmoid(lg)
        class_conf = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)
        mask = class_conf > conf_thres
        bx = bx[mask]
        class_conf_f = class_conf[mask]
        class_ids_f = class_ids[mask]
        if bx.shape[0] == 0:
            out.append(torch.empty((0, 6)))
            continue
        boxes_xyxy = xywh2xyxy(bx)
        final = []
        for c in np.unique(class_ids_f):
            m = class_ids_f == c
            cb = boxes_xyxy[m]
            cs = class_conf_f[m]
            order = cs.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(cb[i, 0], cb[order[1:], 0])
                yy1 = np.maximum(cb[i, 1], cb[order[1:], 1])
                xx2 = np.minimum(cb[i, 2], cb[order[1:], 2])
                yy2 = np.minimum(cb[i, 3], cb[order[1:], 3])
                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                inter = w * h
                area_i = (cb[i, 2] - cb[i, 0]) * (cb[i, 3] - cb[i, 1])
                area_o = (cb[order[1:], 2] - cb[order[1:], 0]) * (cb[order[1:], 3] - cb[order[1:], 1])
                iou = inter / (area_i + area_o - inter + 1e-6)
                inds = np.where(iou <= iou_thres)[0]
                order = order[inds + 1]
            for idx in keep:
                final.append(np.concatenate([cb[idx], [cs[idx]], [float(c)]]))
        if not final:
            out.append(torch.empty((0, 6)))
            continue
        arr = np.array(final)
        arr = arr[arr[:, 4].argsort()[::-1]]
        arr = arr[:max_det]
        out.append(torch.from_numpy(arr))
    return out

def detect(opt):
    source = str(opt.source)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / "labels" if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Loading {opt.weights} for ONNX Runtime inference...")
    device = select_device(opt.device)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type != "cpu" else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(opt.weights, providers=providers)
    model_meta = session.get_modelmeta().custom_metadata_map or {}
    names = eval(model_meta["names"]) if "names" in model_meta else {i: f"class{i}" for i in range(1000)}
    stride = int(model_meta["stride"]) if "stride" in model_meta else 32
    imgsz = check_img_size(opt.imgsz, s=stride)
    input_name, np_dtype, mode = _infer_input(session, opt.weights)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False) if webcam else LoadImages(source, img_size=imgsz, stride=stride, auto=False)
    nc = len(names) if isinstance(names, (list, tuple, dict)) else 80
    if isinstance(names, dict):
        nc = len(names)
    outs_info = session.get_outputs()
    out_types = {o.name: o.type for o in outs_info}
    dt = (Profile(), Profile(), Profile())
    first_qparams_checked = False
    logits_qparams = None
    logits_out_name = None
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            ort_input = _prep_input_numpy(im, np_dtype, mode)
        with dt[1]:
            (boxes_name, boxes_cfirst, boxes_type), (logits_name, logits_cfirst, logits_type) = _run_dual_outputs(session, ort_input)
            if not first_qparams_checked:
                logits_out_name = logits_name
                if logits_type == "tensor(uint8)":
                    logits_qparams = _get_uint8_output_qparams_for_output(opt.weights, logits_out_name)
                first_qparams_checked = True
            if logits_type == "tensor(uint8)":
                logits_cfirst = _floatize_uint8_logits(logits_cfirst, logits_qparams)
            probs_cfirst = _sigmoid(logits_cfirst)
            pred_list = _custom_nms_from_dual(boxes_cfirst.astype(np.float32), probs_cfirst.astype(np.float32), opt.conf_thres if opt.conf_thres is not None else 0.25, opt.iou_thres if opt.iou_thres is not None else 0.45, max_det=opt.max_det)
        with dt[2]:
            pass
        for i, det in enumerate(pred_list):
            p, im0 = ((Path(path[i]), im0s[i].copy()) if webcam else (Path(path), im0s.copy()))
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem)
            s += f"{ort_input.shape[2]}x{ort_input.shape[3]} "
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))
            if len(det):
                det = det.float().cpu()
                det[:, :4] = scale_boxes(ort_input.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy_box, conf, cls in det:
                    if opt.save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy_box).view(1, 4))).view(-1).tolist()
                        line = (int(cls), *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if opt.save_img:
                        c = int(cls)
                        label = f"{names[c]} {float(conf):.2f}"
                        annotator.box_label(xyxy_box, label, color=colors(c, True))
            im0 = annotator.result()
            if opt.save_img and dataset.mode == "image":
                cv2.imwrite(save_path, im0)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")
    t = tuple(x.t / len(dataset) * 1e3 for x in dt)
    LOGGER.info("Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape (1, 3, %d, %d)" % (t + (imgsz[0], imgsz[1])))
    if opt.save_txt or opt.save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if opt.save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def process_batch(detections, labels, iouv):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = (torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy())
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct

def _cuda_sync(device):
    if torch.cuda.is_available() and getattr(device, "type", "cpu") != "cpu":
        torch.cuda.synchronize()

def val(opt):
    data, weights, imgsz, conf_thres, iou_thres, max_det, workers, batch_size = (
        opt.data,
        opt.weights,
        opt.imgsz,
        opt.conf_thres,
        opt.iou_thres,
        opt.max_det,
        opt.workers,
        opt.batch_size,
    )
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Loading {weights} for ONNX Runtime validation...")
    device = select_device(opt.device)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type != "cpu" else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(weights, providers=providers)
    model_meta = session.get_modelmeta().custom_metadata_map or {}
    names = eval(model_meta["names"]) if "names" in model_meta else {i: f"class{i}" for i in range(1000)}
    stride = int(model_meta["stride"]) if "stride" in model_meta else 32
    imgsz = check_img_size(imgsz, s=stride)
    input_name, np_dtype, mode = _infer_input(session, weights)
    data = check_dataset(data)
    nc = int(data["nc"])
    dataloader = create_dataloader(
        data["val"],
        imgsz[0],
        batch_size if batch_size is not None else 1,
        stride,
        pad=0.5,
        rect=False,
        workers=workers,
        prefix=colorstr("val: "),
    )[0]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    seen = 0
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    stats, ap, ap_class = [], [], []
    warmup_batches = 3
    preprocess_times, inference_times, postprocess_times = [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    outs_info = session.get_outputs()
    out_types = {o.name: o.type for o in outs_info}
    first_qparams_checked = False
    logits_qparams = None
    logits_out_name = None
    names_nc = len(names) if not isinstance(names, dict) else len(names.keys())
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        _cuda_sync(device)
        t_start_pre = time.time()
        nb, _, height, width = im.shape
        if mode == "float":
            im_for_ort = (im.float() / 255.0).cpu().numpy()
            if np_dtype == np.float16:
                im_for_ort = im_for_ort.astype(np.float16)
        else:
            im_for_ort = im.to(torch.uint8).cpu().numpy()
        targets = targets.to(device)
        _cuda_sync(device)
        t_end_pre = time.time()
        if batch_i >= warmup_batches:
            preprocess_times.append(t_end_pre - t_start_pre)
        _cuda_sync(device)
        t_start_inf = time.time()
        (boxes_name, boxes_cfirst, boxes_type), (logits_name, logits_cfirst, logits_type) = _run_dual_outputs(session, im_for_ort)
        _cuda_sync(device)
        t_end_inf = time.time()
        if batch_i >= warmup_batches:
            inference_times.append(t_end_inf - t_start_inf)
        _cuda_sync(device)
        t_start_post = time.time()
        if not first_qparams_checked:
            logits_out_name = logits_name
            if logits_type == "tensor(uint8)":
                logits_qparams = _get_uint8_output_qparams_for_output(weights, logits_out_name)
            first_qparams_checked = True
        if logits_type == "tensor(uint8)":
            logits_cfirst = _floatize_uint8_logits(logits_cfirst, logits_qparams)
        probs_cfirst = _sigmoid(logits_cfirst)
        pred_list = _custom_nms_from_dual(boxes_cfirst.astype(np.float32), probs_cfirst.astype(np.float32), conf_thres if conf_thres is not None else 0.001, iou_thres if iou_thres is not None else 0.7, max_det=max_det)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        for si, det in enumerate(pred_list):
            det = det.to(device)
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            npr = len(det)
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue
            predn = det.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, det[:, 4].to(device), det[:, 5].to(device), labels[:, 0].to(device)))
        _cuda_sync(device)
        t_end_post = time.time()
        if batch_i >= warmup_batches:
            postprocess_times.append(t_end_post - t_start_post)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] if len(stats) else []
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    nt = np.bincount(stats[3].astype(int), minlength=nc) if len(stats) else np.zeros(nc)
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING: No labels found in {data['val']}, cannot compute metrics.")
    if len(stats) and stats[0].any() and nc > 1:
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    if preprocess_times:
        num = len(preprocess_times)
        avg_pre = (sum(preprocess_times) / num) * 1000
        avg_inf = (sum(inference_times) / num) * 1000
        avg_post = (sum(postprocess_times) / num) * 1000
        LOGGER.info(f"Average Speed (over last {num} batches):")
        LOGGER.info(f"  - Pre-processing:  {avg_pre:.2f}ms per batch")
        LOGGER.info(f"  - Inference:       {avg_inf:.2f}ms per batch")
        LOGGER.info(f"  - Post-processing: {avg_post:.2f}ms per batch")
    else:
        LOGGER.warning(f"Not enough batches ({len(dataloader)}) to calculate average speed after warm-up ({warmup_batches} batches).")

def main(opt):
    if opt.val and opt.data is None:
        raise ValueError("Validation requires a --data argument.")
    if opt.detect and opt.source is None:
        raise ValueError("Detection requires a --source argument.")
    if not opt.detect and not opt.val:
        if opt.source:
            opt.detect = True
        elif opt.data:
            opt.val = True
        else:
            raise ValueError("Please specify a task with --detect or --val.")
    if opt.val:
        if opt.conf_thres is None:
            opt.conf_thres = 0.001
        if opt.iou_thres is None:
            opt.iou_thres = 0.7
        if not hasattr(opt, "batch_size") or opt.batch_size is None:
            opt.batch_size = 1
    else:
        if opt.conf_thres is None:
            opt.conf_thres = 0.25
        if opt.iou_thres is None:
            opt.iou_thres = 0.45
        opt.batch_size = 1
    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2
    if opt.detect:
        detect(opt)
    elif opt.val:
        val(opt)

def parse_opt():
    parser = argparse.ArgumentParser(description="ONNX Model Inference and Validation Script (dual-output boxes+logits)")
    parser.add_argument("--weights", type=str, default=ROOT / "best.onnx", help="ONNX model path")
    parser.add_argument("--imgsz", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=None, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("--project", default=ROOT / "runs/onnx", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--source", type=str, default=None, help="file/dir/URL/glob/screen/0(webcam) for detection")
    parser.add_argument("--save-img", action="store_true", help="save annotated images")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--data", type=str, default=None, help="dataset.yaml path for validation")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size for validation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--val", action="store_true", help="Run validation on a dataset")
    group.add_argument("--detect", action="store_true", help="Run detection on a source")
    opt = parser.parse_args()
    opt.imgsz = [int(x) for x in opt.imgsz]
    if len(opt.imgsz) == 1:
        opt.imgsz = opt.imgsz * 2
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
