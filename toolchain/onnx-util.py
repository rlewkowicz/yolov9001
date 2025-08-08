import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_static
from tqdm import tqdm
import time
import shutil
import onnx

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
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ap_per_class, box_iou
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

class ImageCalibrator:
    def __init__(self, calib_dir_path, input_name, imgsz=(640, 640), stride=32):
        self.input_name = input_name
        self.dataloader = LoadImages(str(calib_dir_path), img_size=imgsz, stride=stride, auto=False)
        self.iterator = iter(self.dataloader)

    def get_next(self):
        try:
            _, im, _, _, _ = next(self.iterator)
            im = im.astype(np.float32) / 255.0
            if len(im.shape) == 3:
                im = np.expand_dims(im, 0)
            return {self.input_name: im}
        except StopIteration:
            return None

def detect(opt):
    (
        source,
        weights,
        imgsz,
        save_img,
        save_txt,
        line_thickness,
        conf_thres,
        iou_thres,
        max_det,
    ) = (
        opt.source,
        opt.weights,
        opt.imgsz,
        opt.save_img,
        opt.save_txt,
        opt.line_thickness,
        opt.conf_thres,
        opt.iou_thres,
        opt.max_det,
    )
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Loading {weights} for ONNX Runtime inference...")
    device = select_device(opt.device)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.type != "cpu" else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(weights, providers=providers)
    model_meta = session.get_modelmeta().custom_metadata_map
    names = eval(model_meta["names"])
    stride = int(model_meta["stride"])
    imgsz = check_img_size(imgsz, s=stride)
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=False)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = im.astype(np.float32) / 255.0
            if len(im.shape) == 3:
                im = np.expand_dims(im, 0)
        with dt[1]:
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            pred = session.run([output_name], {input_name: im})[0]
            pred = torch.from_numpy(pred)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        for i, det in enumerate(pred):
            p, im0 = ((Path(path[i]), im0s[i].copy()) if webcam else (Path(path), im0s.copy()))
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem)
            s += f"{im.shape[2]}x{im.shape[3]} "
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = ((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())
                        line = (int(cls), *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")
                    if save_img:
                        c = int(cls)
                        label = f"{names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")
    t = tuple(x.t / len(dataset) * 1e3 for x in dt)
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape (1, 3, {imgsz[0]}, {imgsz[1]})"
        % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

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
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device.type != "cpu" else ["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession(weights, providers=providers)
    model_meta = session.get_modelmeta().custom_metadata_map
    names = eval(model_meta["names"])
    stride = int(model_meta["stride"])
    imgsz = check_img_size(imgsz, s=stride)
    data = check_dataset(data)
    nc = int(data["nc"])
    dataloader = create_dataloader(
        data["val"],
        imgsz[0],
        batch_size,
        stride,
        pad=0.5,
        rect=False,
        workers=workers,
        prefix=colorstr("val: "),
    )[0]
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    seen = 0
    s = ("%22s" + "%11s" * 6) % (
        "Class",
        "Images",
        "Instances",
        "P",
        "R",
        "mAP50",
        "mAP50-95",
    )
    stats, ap, ap_class = [], [], []
    warmup_batches = 3
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        torch.cuda.synchronize()
        t_start_pre = time.time()
        im = im.to(device).float() / 255.0
        targets = targets.to(device)
        nb, _, height, width = im.shape
        im_numpy = im.cpu().numpy()
        torch.cuda.synchronize()
        t_end_pre = time.time()
        if batch_i >= warmup_batches:
            preprocess_times.append(t_end_pre - t_start_pre)
        torch.cuda.synchronize()
        t_start_inf = time.time()
        pred = session.run([output_name], {input_name: im_numpy})[0]
        torch.cuda.synchronize()
        t_end_inf = time.time()
        if batch_i >= warmup_batches:
            inference_times.append(t_end_inf - t_start_inf)
        torch.cuda.synchronize()
        t_start_post = time.time()
        pred = torch.from_numpy(pred).to(device)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        for si, det in enumerate(pred):
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
            stats.append((correct, det[:, 4], det[:, 5], labels[:, 0]))
        torch.cuda.synchronize()
        t_end_post = time.time()
        if batch_i >= warmup_batches:
            postprocess_times.append(t_end_post - t_start_post)
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
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
        num_timed_batches = len(preprocess_times)
        avg_pre = (sum(preprocess_times) / num_timed_batches) * 1000
        avg_inf = (sum(inference_times) / num_timed_batches) * 1000
        avg_post = (sum(postprocess_times) / num_timed_batches) * 1000
        LOGGER.info(f"Average Speed (over last {num_timed_batches} batches):")
        LOGGER.info(f"  - Pre-processing:  {avg_pre:.2f}ms per batch")
        LOGGER.info(f"  - Inference:       {avg_inf:.2f}ms per batch")
        LOGGER.info(f"  - Post-processing: {avg_post:.2f}ms per batch")
    else:
        LOGGER.warning(
            f"Not enough batches ({len(dataloader)}) to calculate average speed after warm-up ({warmup_batches} batches)."
        )

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

def main(opt):
    if opt.int8:
        if not opt.data:
            raise ValueError("INT8 quantization requires a --data argument for calibration.")

        model_input_path = Path(opt.weights)
        model_output_path = model_input_path.with_stem(f"{model_input_path.stem}_int8")

        if model_output_path.exists():
            LOGGER.info(f"INT8 model already exists at {model_output_path}. Skipping quantization.")
        else:
            LOGGER.info(f"Starting INT8 quantization for {model_input_path}...")

            data_yaml_path = Path(opt.data)
            with open(data_yaml_path) as f:
                data_dict = yaml.safe_load(f)

            dataset_root = (data_yaml_path.parent / ".." / data_dict['path']).resolve()

            calib_data_list_path = dataset_root / data_dict['train']
            with open(calib_data_list_path) as f:
                calib_files = [line.strip() for line in f][:200]

            calib_image_paths = [str(dataset_root / p) for p in calib_files]
            LOGGER.info(
                f"Using {len(calib_image_paths)} images from the training set for calibration."
            )

            calib_dir = Path('./calib_images_temp')
            calib_dir.mkdir(exist_ok=True)

            for img_path_str in calib_image_paths:
                img_path = Path(img_path_str)
                try:
                    os.symlink(img_path.resolve(), calib_dir / img_path.name)
                except FileExistsError:
                    pass

            try:
                temp_session = onnxruntime.InferenceSession(
                    str(model_input_path), providers=["CPUExecutionProvider"]
                )
                input_name = temp_session.get_inputs()[0].name

                calibrator = ImageCalibrator(
                    calib_dir_path=calib_dir, input_name=input_name, imgsz=opt.imgsz
                )

                nodes_to_exclude = []
                model = onnx.load(model_input_path)
                for node in model.graph.node:
                    if node.op_type in ['HardSwish', 'Softmax']:
                        nodes_to_exclude.append(node.name)
                LOGGER.info(f"Excluding {len(nodes_to_exclude)} nodes from quantization.")

                quantize_static(
                    model_input=model_input_path,
                    model_output=model_output_path,
                    calibration_data_reader=calibrator,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    nodes_to_exclude=nodes_to_exclude,
                )
                LOGGER.info(f"Successfully created INT8 model: {model_output_path}")
            finally:
                shutil.rmtree(calib_dir)

        opt.weights = str(model_output_path)

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
    parser = argparse.ArgumentParser(description="ONNX Model Inference and Validation Script")
    parser.add_argument(
        "--int8", action="store_true", help="Enable INT8 quantization before running"
    )
    parser.add_argument("--weights", type=str, default=ROOT / "best.onnx", help="ONNX model path")
    parser.add_argument(
        "--imgsz", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w"
    )
    parser.add_argument("--conf-thres", type=float, default=None, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or cpu")
    parser.add_argument(
        "--project", default=ROOT / "runs/onnx", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="file/dir/URL/glob/screen/0(webcam) for detection",
    )
    parser.add_argument("--save-img", action="store_true", help="save annotated images")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="dataset.yaml path for validation/calibration"
    )
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
