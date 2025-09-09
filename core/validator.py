import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseRunner
from utils.metrics import (
    DetMetrics,
    box_iou,
    process_batch,
    process_batch_batched,
    ConfusionMatrixTorch,
    process_batch_area_bins,
    process_batch_area_bins_batched,
    ap_per_class as ap_per_class_torch,
)
from utils.prefetcher import CUDAPrefetcher
from utils.boxes import (
    yolo_to_xyxy,
    scale_boxes_from_canvas_to_original,
    clip_boxes_,
)
import cv2
import numpy as np

class Validator(BaseRunner):
    """Validation loop for YOLO models."""
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.metrics = {}
        self.nc = model.nc
        self.names = model.names if hasattr(model, 'names') else {i: str(i) for i in range(self.nc)}
        self.det_metrics = DetMetrics(names=self.names)
        self._area_bins = [(0.0, 32.0**2), (32.0**2, 96.0**2), (96.0**2, float('inf'))]
        self._area_names = ["S", "M", "L"]
        cm_conf = float(self.cfg.get('cm_conf', 0.25))
        cm_iou = float(self.cfg.get('cm_iou', 0.20))
        self.cm = ConfusionMatrixTorch(nc=self.nc, device=self.device, conf=cm_conf, iou_thr=cm_iou)

        self.postprocessor = self.model.get_postprocessor(device=self.device)

        if 'conf_thresh' in kwargs:
            self.postprocessor.conf_thresh = kwargs['conf_thresh']
        if 'iou_thresh' in kwargs:
            self.postprocessor.iou_thresh = kwargs['iou_thresh']

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, model: nn.Module = None):
        """Run validation and compute metrics."""
        eval_model = model if model is not None else self.model

        try:
            _ = eval_model.get_postprocessor(device=self.device)
        except Exception:
            try:
                from .runtime import attach_runtime
                attach_runtime(eval_model, imgsz=int(self.cfg.get('img_size', 640)))
            except Exception:
                pass

        use_injected = False
        try:
            from utils.postprocess import Postprocessor as _PP  # type: ignore
            use_injected = not isinstance(self.postprocessor, _PP)
        except Exception:
            use_injected = True

        if use_injected:
            try:
                self.postprocessor.model = eval_model
                self.postprocessor.nc = int(getattr(eval_model, 'nc', self.nc))
            except Exception:
                pass
        else:
            try:
                self.postprocessor = eval_model.get_postprocessor(device=self.device)
            except Exception:
                self.postprocessor.model = eval_model
            self.postprocessor.model = eval_model
            self.postprocessor.nc = int(getattr(eval_model, 'nc', self.postprocessor.nc))

        try:
            self.nc = int(getattr(eval_model, 'nc', self.nc))
            self.names = getattr(eval_model, 'names', self.names)
            self.cm = ConfusionMatrixTorch(
                nc=self.nc,
                device=self.device,
                conf=float(self.cfg.get('cm_conf', 0.25)),
                iou_thr=float(self.cfg.get('cm_iou', 0.20))
            )
            self.det_metrics = DetMetrics(names=self.names)
        except Exception:
            pass

        was_training = eval_model.training
        eval_model.eval()

        self.det_metrics.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

        sample_captured = False
        self.sample_images = None
        self.sample_boxes = None
        self.sample_images_orig = None
        self.sample_boxes_orig = None

        self.cm.reset()
        self._score_hist = []
        self._iou_hist = []
        self._area_stats = {
            name: {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
            for name in self._area_names
        }

        self.cfg.get('img_size', 640)
        if hasattr(dataloader.dataset, 'imgsz'):
            dataloader.dataset.imgsz

        ds_nc = getattr(getattr(dataloader, 'dataset', None), 'nc', None)
        if ds_nc is not None:
            model_nc = int(getattr(eval_model, 'nc', self.postprocessor.nc))
            if model_nc != int(ds_nc):
                raise ValueError(
                    f"[Validator] Class count mismatch: model nc={model_nc} vs dataset nc={ds_nc}. "
                    f"Ensure dataset labels/names and model.nc align."
                )

        loader = dataloader
        use_prefetch = bool(self.cfg.get('cuda_prefetch', True)) and self.device.type == 'cuda'
        if use_prefetch:
            max_prefetch = int(self.cfg.get("prefetch_max_batches", 3))
            mem_frac = float(self.cfg.get("prefetch_mem_fraction", 0.80))
            loader = CUDAPrefetcher(
                loader, self.device, max_prefetch_batches=max_prefetch, mem_fraction=mem_frac
            )

        try:
            nb = len(dataloader)
            pbar = tqdm(enumerate(loader), total=nb, desc='valid', leave=True)
            for i, (images, targets, orig_shapes, ratios, pads) in pbar:
                images = self.preprocess(images)
                if images.dtype == torch.uint8:
                    images = images.to(torch.float32).div_(255.0)
                images = images.to(memory_format=torch.channels_last)
                canvas_shape = images.shape[2:]

                model_output = eval_model(images)
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    outputs, feat_shapes = model_output
                else:
                    outputs = model_output
                    feat_shapes = getattr(
                        getattr(eval_model, 'detect_layer', None), 'last_shapes', None
                    )

                if isinstance(outputs, list):
                    feat_shapes = tuple((o.shape[-2], o.shape[-1]) for o in outputs)
                    outputs = torch.cat([o.view(o.shape[0], o.shape[1], -1) for o in outputs],
                                        dim=2)
                elif torch.is_tensor(outputs):
                    if feat_shapes is None:
                        try:
                            dl = getattr(eval_model, 'detect_layer', None)
                            if dl is not None and getattr(dl, 'last_shapes', None) is not None:
                                feat_shapes = tuple(dl.last_shapes)
                        except Exception:
                            pass

                predictions = self.postprocessor(outputs, canvas_shape[0], feat_shapes=feat_shapes)

                targets = targets.to(self.device)

                eval_in_orig = bool(self.cfg.get('eval_in_original_space', False))
                if not eval_in_orig:
                    det_boxes_list = []
                    det_scores_list = []
                    det_cls_list = []
                    det_batch_idx = []
                    for bi, pred in enumerate(predictions):
                        if pred['boxes'].numel():
                            det_boxes_list.append(pred['boxes'].float())
                            det_scores_list.append(pred['scores'].view(-1))
                            det_cls_list.append(pred['class_ids'].to(torch.float32).view(-1))
                            det_batch_idx.append(
                                torch.full((pred['boxes'].shape[0], ),
                                           bi,
                                           device=self.device,
                                           dtype=torch.float32)
                            )
                    if det_boxes_list:
                        det_boxes = torch.cat(det_boxes_list, 0)
                        det_scores = torch.cat(det_scores_list, 0)
                        det_cls = torch.cat(det_cls_list, 0)
                        det_b = torch.cat(det_batch_idx, 0)
                        if bool(self.cfg.get("clip_pred_to_canvas", True)) and det_boxes.numel():
                            Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
                            clip_boxes_(det_boxes, (Hc, Wc), pixel_edges=False)
                        dets = torch.cat([
                            det_boxes,
                            det_scores.view(-1, 1),
                            det_cls.view(-1, 1),
                            det_b.view(-1, 1)
                        ],
                                         dim=1)  # [K,7]
                    else:
                        dets = torch.empty((0, 7), device=self.device)

                    if targets.numel():
                        lab_b = targets[:, 0:1].to(torch.float32)
                        lab_cls = targets[:, 1:2].to(torch.float32)
                        gt_boxes_canvas = yolo_to_xyxy(targets[:, 2:], canvas_shape)
                        gt_boxes_canvas[:,
                                        [0, 2]] = gt_boxes_canvas[:,
                                                                  [0, 2]].clamp(0, canvas_shape[1])
                        gt_boxes_canvas[:,
                                        [1, 3]] = gt_boxes_canvas[:,
                                                                  [1, 3]].clamp(0, canvas_shape[0])
                        labels = torch.cat([lab_b, lab_cls, gt_boxes_canvas], dim=1)  # [L,6]
                    else:
                        labels = torch.empty((0, 6), device=self.device)

                    iouv_vec = self.det_metrics.iouv.to(self.device)
                    tp_all = process_batch_batched(dets, labels, iouv_vec)

                    if dets.shape[0] > 0:
                        self._score_hist.append(dets[:, 4].detach().float().cpu())

                    self.det_metrics.stats['tp'].append(tp_all)
                    if dets.shape[0] > 0:
                        self.det_metrics.stats['conf'].append(dets[:, 4])
                        self.det_metrics.stats['pred_cls'].append(dets[:, 5].to(torch.long))
                    else:
                        self.det_metrics.stats['conf'].append(torch.empty(0, device=self.device))
                        self.det_metrics.stats['pred_cls'].append(
                            torch.empty(0, device=self.device, dtype=torch.long)
                        )
                    if labels.shape[0] > 0:
                        self.det_metrics.stats['target_cls'].append(labels[:, 1].to(torch.long))
                    else:
                        self.det_metrics.stats['target_cls'].append(
                            torch.empty(0, device=self.device, dtype=torch.long)
                        )

                    try:
                        if labels.numel():
                            areas = (labels[:, 4] - labels[:, 2]
                                    ).clamp_min(0) * (labels[:, 3] - labels[:, 1]).clamp_min(0)
                            edges = torch.tensor([0.0, 32.0**2, 96.0**2,
                                                  float('inf')],
                                                 device=self.device)
                            bin_ids = torch.bucketize(areas, edges, right=False) - 1
                            bin_ids = bin_ids.clamp(0, len(self._area_names) - 1)
                            tp_bins = process_batch_area_bins_batched(
                                dets, labels, iouv_vec, bin_ids, num_bins=len(self._area_names)
                            )  # [B,K,T]
                            for b, name in enumerate(self._area_names):
                                self._area_stats[name]['tp'].append(tp_bins[b])
                                if dets.shape[0] > 0:
                                    self._area_stats[name]['conf'].append(dets[:, 4])
                                    self._area_stats[name]['pred_cls'].append(
                                        dets[:, 5].to(torch.long)
                                    )
                                else:
                                    self._area_stats[name]['conf'].append(
                                        torch.empty(0, device=self.device)
                                    )
                                    self._area_stats[name]['pred_cls'].append(
                                        torch.empty(0, device=self.device, dtype=torch.long)
                                    )
                                mask = (bin_ids == b)
                                if mask.any():
                                    self._area_stats[name]['target_cls'].append(
                                        labels[mask, 1].to(torch.long)
                                    )
                                else:
                                    self._area_stats[name]['target_cls'].append(
                                        torch.empty(0, device=self.device, dtype=torch.long)
                                    )
                        else:
                            for b, name in enumerate(self._area_names):
                                self._area_stats[name]['tp'].append(
                                    torch.zeros((dets.shape[0], iouv_vec.numel()),
                                                dtype=torch.bool,
                                                device=self.device)
                                )
                                if dets.shape[0] > 0:
                                    self._area_stats[name]['conf'].append(dets[:, 4])
                                    self._area_stats[name]['pred_cls'].append(
                                        dets[:, 5].to(torch.long)
                                    )
                                else:
                                    self._area_stats[name]['conf'].append(
                                        torch.empty(0, device=self.device)
                                    )
                                    self._area_stats[name]['pred_cls'].append(
                                        torch.empty(0, device=self.device, dtype=torch.long)
                                    )
                                self._area_stats[name]['target_cls'].append(
                                    torch.empty(0, device=self.device, dtype=torch.long)
                                )
                    except Exception:
                        pass

                    for bi, pred in enumerate(predictions):
                        gt_mask = (targets[:, 0].to(torch.long) == bi)
                        gt_labels = targets[gt_mask, 1:]
                        if gt_labels.numel() == 0 and (not pred['boxes'].numel()):
                            continue
                        gt_boxes_canvas = yolo_to_xyxy(
                            gt_labels[:, 1:], canvas_shape
                        ) if gt_labels.numel() else torch.empty((0, 4), device=self.device)
                        if gt_boxes_canvas.numel():
                            gt_boxes_canvas[:, [0, 2]] = gt_boxes_canvas[:, [0, 2]].clamp(
                                0, canvas_shape[1]
                            )
                            gt_boxes_canvas[:, [1, 3]] = gt_boxes_canvas[:, [1, 3]].clamp(
                                0, canvas_shape[0]
                            )
                        gt_cls = gt_labels[:, 0] if gt_labels.numel() else torch.empty(
                            (0, ), device=self.device, dtype=torch.long
                        )
                        self.cm.update(
                            pred.get('boxes', torch.empty((0, 4), device=self.device)),
                            pred.get('scores', torch.empty((0, ), device=self.device)),
                            pred.get(
                                'class_ids', torch.empty((0, ),
                                                         device=self.device,
                                                         dtype=torch.long)
                            ), gt_boxes_canvas, gt_cls
                        )

                    if not sample_captured:
                        imgs_vis = images.detach().float().cpu().clamp(0, 1)
                        n_show = min(4, imgs_vis.shape[0])
                        boxes_list_canvas = []
                        for j in range(n_show):
                            boxes_list_canvas.append(predictions[j]['boxes'].detach().float().cpu())
                        self.sample_images = imgs_vis[:n_show]
                        self.sample_boxes = torch.nn.utils.rnn.pad_sequence([
                            b for b in boxes_list_canvas
                        ],
                                                                            batch_first=True,
                                                                            padding_value=0.0)
                        if bool(self.cfg.get("log_val_images_original", True)):
                            orig_imgs = []
                            orig_boxes = []
                            for j in range(n_show):
                                Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
                                pj = pads[j]
                                oj = orig_shapes[j]
                                left, top, right, bottom = int(pj[0]), int(pj[1]), int(pj[2]), int(
                                    pj[3]
                                )
                                img_canvas = (imgs_vis[j] * 255.0).byte()  # CHW
                                img_np = img_canvas.permute(1, 2, 0).numpy()  # HWC uint8
                                roi = img_np[top:Hc - bottom, left:Wc - right]
                                H0, W0 = int(oj[0]), int(oj[1])
                                if roi.size and (roi.shape[0] > 0 and roi.shape[1] > 0):
                                    resized = cv2.resize(
                                        roi, (W0, H0), interpolation=cv2.INTER_LINEAR
                                    )
                                else:
                                    resized = np.full((H0, W0, 3), 114, dtype=np.uint8)
                                boxes_canvas = predictions[j]['boxes'].detach().float()
                                boxes_orig = scale_boxes_from_canvas_to_original(
                                    boxes_canvas, (Hc, Wc), (H0, W0), (left, top), (right, bottom)
                                ).cpu()
                                clip_boxes_(boxes_orig, (H0, W0), pixel_edges=True)
                                orig_imgs.append(
                                    torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                                )
                                orig_boxes.append(boxes_orig)
                            if len(orig_imgs):
                                self.sample_images_orig = orig_imgs
                                self.sample_boxes_orig = orig_boxes
                        sample_captured = True
                    continue  # skip per-image slow path

                for i_pred, pred in enumerate(predictions):

                    orig_shape = orig_shapes[i_pred]
                    pad = pads[i_pred]

                    pred['boxes'] = pred['boxes'].float()
                    batch_idx = targets[:, 0] == i_pred
                    gt_labels = targets[batch_idx, 1:]

                    if bool(self.cfg.get("clip_pred_to_canvas", True)) and pred['boxes'].numel():
                        Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
                        clip_boxes_(pred['boxes'], (Hc, Wc), pixel_edges=False)

                    if i == 0 and pred['boxes'].numel() and gt_labels.numel():
                        assert pred['class_ids'].dtype == torch.long, "class_ids must be torch.long"
                        gt_boxes_canvas_check = yolo_to_xyxy(gt_labels[:, 1:], canvas_shape)
                        assert pred['boxes'].shape[-1] == 4 and gt_boxes_canvas_check.shape[-1] == 4
                        pred_boxes_orig = scale_boxes_from_canvas_to_original(
                            pred['boxes'], canvas_shape, orig_shape, (pad[0], pad[1]),
                            (pad[2], pad[3])
                        )
                        assert torch.all(
                            pred['boxes'][:, 2:] >= pred['boxes'][:, :2]
                        ), "Pred canvas xyxy invalid (x2< x1 or y2< y1)"
                        assert torch.all(
                            pred_boxes_orig[:, 2:] >= pred_boxes_orig[:, :2]
                        ), "Pred original xyxy invalid"
                        self.logger.debug(
                            "val/box_check", {
                                "pred_canvas_minmax":
                                    [float(pred['boxes'].min()),
                                     float(pred['boxes'].max())],
                                "pred_orig_minmax":
                                    [float(pred_boxes_orig.min()),
                                     float(pred_boxes_orig.max())],
                            }
                        )

                    if gt_labels.numel() == 0:
                        if pred['boxes'].numel() > 0:
                            T = getattr(self.det_metrics, "num_iou_thrs", 10)
                            self.det_metrics.stats['tp'].append(
                                torch.zeros_like(pred['class_ids']
                                                ).bool().unsqueeze(1).expand(-1, T)
                            )
                            self.det_metrics.stats['conf'].append(pred['scores'])
                            self.det_metrics.stats['pred_cls'].append(pred['class_ids'])
                            self.det_metrics.stats['target_cls'].append(
                                torch.empty(0, dtype=torch.long, device=self.device)
                            )
                        continue

                    gt_boxes_canvas = yolo_to_xyxy(gt_labels[:, 1:], canvas_shape)  # canvas space
                    gt_boxes_canvas[:, [0, 2]] = gt_boxes_canvas[:,
                                                                 [0, 2]].clamp(0, canvas_shape[1])
                    gt_boxes_canvas[:, [1, 3]] = gt_boxes_canvas[:,
                                                                 [1, 3]].clamp(0, canvas_shape[0])
                    gt_cls = gt_labels[:, 0]

                    if pred['boxes'].numel() == 0:
                        T = getattr(self.det_metrics, "num_iou_thrs", 10)
                        self.det_metrics.stats['tp'].append(
                            torch.empty(0, T, dtype=torch.bool, device=self.device)
                        )
                        self.det_metrics.stats['conf'].append(torch.empty(0, device=self.device))
                        self.det_metrics.stats['pred_cls'].append(
                            torch.empty(0, device=self.device)
                        )
                        self.det_metrics.stats['target_cls'].append(gt_cls)
                        try:
                            labels_all = torch.cat([
                                gt_cls.view(-1, 1).to(torch.float32), gt_boxes_canvas
                            ],
                                                   dim=1)
                            areas = (labels_all[:, 3] -
                                     labels_all[:, 1]) * (labels_all[:, 4] - labels_all[:, 2])
                            for (lo, hi), name in zip(self._area_bins, self._area_names):
                                mask = (areas >= lo) & (areas < hi)
                                tgt_bin = labels_all[mask][:, 0].to(torch.long)
                                self._area_stats[name]['tp'].append(
                                    torch.empty(0, T, dtype=torch.bool, device=self.device)
                                )
                                self._area_stats[name]['conf'].append(
                                    torch.empty(0, device=self.device)
                                )
                                self._area_stats[name]['pred_cls'].append(
                                    torch.empty(0, device=self.device)
                                )
                                self._area_stats[name]['target_cls'].append(tgt_bin)
                        except Exception:
                            pass
                        continue

                    use_orig_metrics = bool(self.cfg.get("metrics_on_original", False))
                    if use_orig_metrics:
                        pred_boxes_eval = scale_boxes_from_canvas_to_original(
                            pred['boxes'], canvas_shape, orig_shape, (pad[0], pad[1]),
                            (pad[2], pad[3])
                        )
                        H0, W0 = int(orig_shape[0]), int(orig_shape[1])
                        clip_boxes_(pred_boxes_eval, (H0, W0), pixel_edges=True)
                        gt_boxes_eval = scale_boxes_from_canvas_to_original(
                            gt_boxes_canvas, canvas_shape, orig_shape, (pad[0], pad[1]),
                            (pad[2], pad[3])
                        )
                        clip_boxes_(gt_boxes_eval, (H0, W0), pixel_edges=True)
                        iou = box_iou(pred_boxes_eval, gt_boxes_eval)
                        det_boxes_for_tp = pred_boxes_eval
                        labels_for_tp = gt_boxes_eval
                        self.cm.update(
                            pred_boxes_eval, pred['scores'], pred['class_ids'], gt_boxes_eval,
                            gt_cls
                        )
                    else:
                        iou = box_iou(pred['boxes'], gt_boxes_canvas)
                        det_boxes_for_tp = pred['boxes']
                        labels_for_tp = gt_boxes_canvas
                        self.cm.update(
                            pred['boxes'], pred['scores'], pred['class_ids'], gt_boxes_canvas,
                            gt_cls
                        )

                    iouv_vec = self.det_metrics.iouv.to(self.device)
                    dets = torch.cat([
                        det_boxes_for_tp, pred['scores'].view(-1, 1),
                        pred['class_ids'].to(torch.float32).view(-1, 1)
                    ],
                                     dim=1)
                    labels = torch.cat([gt_cls.view(-1, 1).to(torch.float32), labels_for_tp], dim=1)
                    tp = process_batch(dets, labels, iouv_vec)

                    if pred['scores'].numel() > 0:
                        self._score_hist.append(pred['scores'].detach().float().cpu())
                    if iou.numel() > 0:
                        self._iou_hist.append(iou.detach().float().cpu())

                    self.det_metrics.stats['tp'].append(tp)
                    self.det_metrics.stats['conf'].append(pred['scores'])
                    self.det_metrics.stats['pred_cls'].append(pred['class_ids'])
                    self.det_metrics.stats['target_cls'].append(gt_cls)

                    try:
                        areas = (labels[:, 3] - labels[:, 1]
                                ).clamp_min(0) * (labels[:, 4] - labels[:, 2]).clamp_min(0)
                        edges = torch.tensor([0.0, 32.0**2, 96.0**2,
                                              float('inf')],
                                             device=self.device)
                        bin_ids = torch.bucketize(areas, edges, right=False) - 1
                        bin_ids = bin_ids.clamp(0, len(self._area_names) - 1)
                        tp_bins = process_batch_area_bins(
                            dets, labels, iouv_vec, bin_ids, num_bins=len(self._area_names)
                        )  # [B,N,T]
                        for b, name in enumerate(self._area_names):
                            self._area_stats[name]['tp'].append(tp_bins[b])
                            self._area_stats[name]['conf'].append(pred['scores'])
                            self._area_stats[name]['pred_cls'].append(pred['class_ids'])
                            mask = (bin_ids == b)
                            if mask.any():
                                self._area_stats[name]['target_cls'].append(
                                    labels[mask, 0].to(torch.long)
                                )
                            else:
                                self._area_stats[name]['target_cls'].append(
                                    torch.empty(0, dtype=torch.long, device=self.device)
                                )
                    except Exception:
                        pass

                if not sample_captured:
                    imgs_vis = images.detach().float().cpu().clamp(0, 1)
                    n_show = min(4, imgs_vis.shape[0])
                    boxes_list_canvas = []
                    for j in range(n_show):
                        boxes_list_canvas.append(predictions[j]['boxes'].detach().float().cpu())
                    self.sample_images = imgs_vis[:n_show]
                    self.sample_boxes = torch.nn.utils.rnn.pad_sequence([
                        b for b in boxes_list_canvas
                    ],
                                                                        batch_first=True,
                                                                        padding_value=0.0)
                    if bool(self.cfg.get("log_val_images_original", True)):
                        orig_imgs = []
                        orig_boxes = []
                        for j in range(n_show):
                            Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
                            pj = pads[j]
                            oj = orig_shapes[j]
                            left, top, right, bottom = int(pj[0]), int(pj[1]), int(pj[2]), int(
                                pj[3]
                            )
                            img_canvas = (imgs_vis[j] * 255.0).byte()  # CHW
                            img_np = img_canvas.permute(1, 2, 0).numpy()  # HWC uint8
                            roi = img_np[top:Hc - bottom, left:Wc - right]
                            H0, W0 = int(oj[0]), int(oj[1])
                            if roi.size and (roi.shape[0] > 0 and roi.shape[1] > 0):
                                resized = cv2.resize(roi, (W0, H0), interpolation=cv2.INTER_LINEAR)
                            else:
                                resized = np.full((H0, W0, 3), 114, dtype=np.uint8)
                            boxes_canvas = predictions[j]['boxes'].detach().float()
                            boxes_orig = scale_boxes_from_canvas_to_original(
                                boxes_canvas, (Hc, Wc), (H0, W0), (left, top), (right, bottom)
                            ).cpu()
                            clip_boxes_(boxes_orig, (H0, W0), pixel_edges=True)
                            orig_imgs.append(
                                torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                            )
                            orig_boxes.append(boxes_orig)
                        if len(orig_imgs):
                            self.sample_images_orig = orig_imgs
                            self.sample_boxes_orig = orig_boxes
                    sample_captured = True
        finally:
            if use_prefetch and isinstance(loader, CUDAPrefetcher):
                try:
                    loader.close()
                except Exception:
                    pass

        if was_training:
            eval_model.train()

        self.det_metrics.process()
        try:
            mp, mr, map50_val, map_val = self.det_metrics.box.mean_results()
        except Exception:
            mp, mr = 0.0, 0.0
        scalars = {
            'mAP50-95': float(self.det_metrics.box.map),
            'mAP50': float(self.det_metrics.box.map50),
            'mAP75': float(self.det_metrics.box.map75),
            'precision_macro': float(mp),
            'recall_macro': float(mr),
            'fitness': float(self.det_metrics.fitness),
        }

        dists = {}
        if self._score_hist:
            dists['confidence'] = torch.cat(self._score_hist, 0)
        if self._iou_hist:
            dists['iou'] = torch.cat([t.view(-1) for t in self._iou_hist], 0)
        self._score_hist = []
        self._iou_hist = []

        try:
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.det_metrics.stats.items()}
        except Exception:
            stats = {}

        try:
            ap_overall_t = self.det_metrics.box.ap
            ap50_overall_t = self.det_metrics.box.ap50
            idx_overall_t = self.det_metrics.box.ap_class_index
            ap_overall = ap_overall_t.detach().cpu().numpy(
            ) if isinstance(ap_overall_t, torch.Tensor) else np.array([])
            ap50_overall = ap50_overall_t.detach().cpu().numpy(
            ) if isinstance(ap50_overall_t, torch.Tensor) else np.array([])
            idx_overall = idx_overall_t.detach().cpu().numpy(
            ) if isinstance(idx_overall_t, torch.Tensor) else np.array([], dtype=int)
        except Exception:
            ap_overall, ap50_overall, idx_overall = np.array([]), np.array([]), np.array([],
                                                                                         dtype=int)
        per_class = {
            'ap': ap_overall,
            'ap50': ap50_overall,
            'idx': idx_overall,
        }

        per_bins = {}
        try:
            for name in self._area_names:
                stats_bin = self._area_stats.get(name, None)
                if not stats_bin or not stats_bin['tp']:
                    continue
                try:
                    tp_cat = torch.cat(stats_bin['tp'], 0).to(torch.bool)
                    conf_cat = torch.cat(stats_bin['conf'], 0)
                    pred_cls_cat = torch.cat(stats_bin['pred_cls'], 0).to(torch.long)
                    tgt_cls_cat = torch.cat(stats_bin['target_cls'], 0).to(torch.long)
                except Exception:
                    continue
                if tp_cat.numel() == 0:
                    ap_t = torch.empty((0, self.det_metrics.iouv.numel()), dtype=torch.float64)
                    uc_t = torch.empty((0, ), dtype=torch.long)
                else:
                    _, _, _, _, _, ap_t, uc_t = ap_per_class_torch(
                        tp_cat, conf_cat, pred_cls_cat, tgt_cls_cat
                    )
                ap_cls = ap_t.mean(dim=1) if ap_t.numel() else torch.empty(0, dtype=torch.float64)
                j50 = self.det_metrics.box._idx_for_iou(0.50)
                ap50_t = ap_t[:, j50] if ap_t.numel() else torch.empty(0, dtype=torch.float64)
                per_bins[name] = {
                    'ap': ap_cls.detach().cpu().numpy(),
                    'ap50': ap50_t.detach().cpu().numpy(),
                    'idx': uc_t.detach().cpu().numpy(),
                }
        except Exception:
            per_bins = {}

        if isinstance(self.names, dict):
            names_list = [self.names[i] for i in range(len(self.names))]
        else:
            names_list = list(self.names)

        _payload = {
            'scalars': scalars,
            'distributions': dists,
            'stats': stats,
            'cm': self.cm.array.clone().detach().cpu(),
            'per_class': per_class,
            'per_class_bins': per_bins,
            'names': names_list,
        }

        for k in ('mAP50-95', 'mAP50', 'mAP75', 'fitness', 'precision_macro', 'recall_macro'):
            if k in scalars:
                _payload[k] = scalars[k]

        self.metrics = {
            k: scalars[k]
            for k in ('mAP50-95', 'mAP50', 'mAP75', 'fitness') if k in scalars
        }
        return _payload

    def print_results(self):
        """Print validation results."""
        if not self.metrics:
            self.logger.info("val/status", "No validation metrics available. Run validate() first.")
            return

        self.logger.info("val/results", "Validation Results:")
        self.logger.info("val/separator", "-" * 50)
        for key, value in self.metrics.items():
            self.logger.info(f"val/{key}", f"{value:.4f}")
        self.logger.info("val/separator", "-" * 50)
