import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

def save_one_txt(predn, save_conf, shape, file):
    gn = torch.tensor(shape)[[1, 0, 1, 0]]
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])
    box[:, :2] -= box[:, 2:] / 2
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            "image_id": image_id,
            "category_id": class_map[int(p[5])],
            "bbox": [round(x, 3) for x in b],
            "score": round(p[4], 5),
        })


def process_batch(detections, labels, iouv):
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
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
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.7,
    max_det=300,
    task="val",
    device="",
    workers=8,
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    project=ROOT / "runs/val",
    name="exp",
    exist_ok=False,
    half=True,
    dnn=False,
    min_items=0,
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    fuse=False,
    use_lut=True,
    calib="",
):
    """Works with the fixed-shape training/val loaders we’ve been using.

    If dataloader provides shapes as (h0, w0) only, we *do not* rescale preds/labels back;
    evaluation is performed in network space. If shapes are ((h0,w0), (ratio,pad)), we rescale.
    """
    training = model is not None
    if training:
        device, pt, jit, engine = next(model.parameters()).device, True, False, False
        half &= device.type != "cpu"
        model.half() if half else model.float()
        data = check_dataset(data)
    else:
        device = select_device(device, batch_size=batch_size)
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        model = DetectMultiBackend(
            weights, device=device, dnn=dnn, data=data, fp16=half, fuse=False
        )
        stride = model.stride
        pt = True
        imgsz = check_img_size(imgsz, s=stride)
        half = model.fp16
        device = model.device
        data = check_dataset(data)

    if fuse and pt and not training:
        LOGGER.info("Fusing model for validation...")
        model.model.fuse()

    # Optionally enable DFL LUT in eval
    if use_lut:
        target_model = model if training else model.model
        for m in target_model.modules():
            if hasattr(m, "enable_int8_lut"):
                if hasattr(m, "update_lut_from_ema"):
                    with torch.no_grad():
                        m.update_lut_from_ema()
                m.enable_int8_lut(True)

    if calib:
        LOGGER.warning("`--calib` provided but calibration loader is not available; ignoring.")

    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith("val2017.txt")
    nc = 1 if single_cls else int(data["nc"])
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    if not training:
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)
        task = task if task in ("train", "val", "test") else "val"
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    dt = Profile(), Profile(), Profile()
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()
            im /= 255
            nb, _, height, width = im.shape

        with dt[1]:
            output = model(im) if training else model(im, augment=augment)

            preds = None
            aux_feats = None
            main_feats = None

            if isinstance(output, tuple):
                head_out, feat_out = output
                preds = head_out[-1] if isinstance(head_out, list) else head_out
                if isinstance(feat_out, (list, tuple)):
                    if len(feat_out) == 2 and isinstance(feat_out[0], (list, tuple)) and isinstance(
                        feat_out[1], (list, tuple)
                    ):
                        aux_feats, main_feats = feat_out[0], feat_out[1]
                    else:
                        main_feats = list(feat_out)
            elif isinstance(output, list):
                preds = output[-1] if output and torch.is_tensor(output[-1]) else output
            else:
                preds = output

        if compute_loss is not None and aux_feats is not None and main_feats is not None:
            loss += compute_loss([aux_feats, main_feats], targets)[1]

        # Targets to pixels for current tensor size (labels are normalized)
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        with dt[2]:
            preds = non_max_suppression(
                preds,
                conf_thres,
                iou_thres,
                labels=lb,
                multi_label=True,
                agnostic=single_cls,
                max_det=max_det,
            )

        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]
            path = Path(paths[si])

            # Accept either ((h0,w0),(ratio,pad)) or just (h0,w0)
            shape_info = shapes[si]
            has_ratio_pad = (
                isinstance(shape_info, (tuple, list)) and
                len(shape_info) == 2 and
                isinstance(shape_info[1], (tuple, list))
            )
            if has_ratio_pad:
                shape, ratio_pad = shape_info
            else:
                shape = im[si].shape[1:]  # network space; used for save_txt only
                ratio_pad = None

            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()

            # Only rescale if we actually have (ratio, pad)
            if has_ratio_pad:
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, ratio_pad)

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                if has_ratio_pad:
                    scale_boxes(im[si].shape[1:], tbox, shape, ratio_pad)
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            if save_txt:
                save_one_txt(
                    predn,
                    save_conf,
                    shape,
                    file=save_dir / "labels" / f"{path.stem}.txt",
                )
            if save_json:
                class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
                save_one_json(predn, jdict, path, class_map)
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)
            plot_images(
                im,
                output_to_target(preds),
                paths,
                save_dir / f"val_batch{batch_i}_pred.jpg",
                names,
            )

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] if len(stats) else []
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names
        )
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp = mr = map50 = map = 0.0
        ap_class = []
        tp = fp = p = r = f1 = np.array([])
    nt = np.bincount(stats[3].astype(int), minlength=nc) if len(stats) else np.zeros(nc, dtype=int)

    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning("WARNING ⚠️ no labels found in set; metrics are undefined")

    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    t = tuple(x.t / max(seen, 1) * 1e3 for x in dt)
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info("Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape %s" %
                    (*t, str(shape)))

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end", nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    model.float()
    if not training:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / max(len(dataloader), 1)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolo.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--min-items", type=int, default=0, help="Experimental")
    parser.add_argument("--fuse", action="store_true", help="fuse model before validation")
    parser.add_argument("--use-lut", action="store_true", help="enable LUT-softmax for DFL")
    parser.add_argument("--calib", type=str, default="", help="path to quant calibration file")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    if opt.task in ("train", "val", "test"):
        if opt.conf_thres > 0.001:
            LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.info("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not predictions alone")
        run(**vars(opt))
    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"
        if opt.task == "speed":
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)
        elif opt.task == "study":
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)
                np.savetxt(f, y, fmt="%10.4g")
            os.system("zip -r study.zip study_*.txt")
            plot_val_study(x=x)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
