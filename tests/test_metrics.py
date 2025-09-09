import numpy as np
import torch

from utils.metrics import (box_iou, ap_per_class, process_batch, compute_confusion_matrix, Metric)
from utils.boxes import yolo_to_xyxy

def test_box_iou_identity():
    a = torch.tensor([[0., 0., 10., 10.]])
    iou = box_iou(a, a)
    assert iou.shape == (1, 1)
    assert torch.allclose(iou, torch.ones_like(iou), atol=0, rtol=0)

def test_box_iou_disjoint():
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[20., 20., 30., 30.]])
    iou = box_iou(a, b)
    assert iou.shape == (1, 1)
    assert torch.allclose(iou, torch.zeros_like(iou), atol=0, rtol=0)

def test_box_iou_broadcast_and_bounds():
    a = torch.tensor([[0., 0., 10., 10.], [5., 5., 15., 15.]])
    b = torch.tensor([[0., 0., 10., 10.], [10., 10., 20., 20.], [3., 3., 7., 7.]])
    iou = box_iou(a, b)
    assert iou.shape == (2, 3)
    assert torch.all(iou >= 0) and torch.all(iou <= 1)

def test_ap_per_class_perfect_score_guard():
    """Tests the exact-perfect guard in ap_per_class returns AP=1.0."""
    T = 10
    tp = np.ones((1, T), dtype=bool)  # 1 pred, correct at all thresholds
    conf = np.array([0.99], dtype=np.float64)
    pred_cls = np.array([0], dtype=np.int64)
    target_cls = np.array([0], dtype=np.int64)
    p, r, p_best, r_best, f1_best, ap, cls = ap_per_class(tp, conf, pred_cls, target_cls)
    assert ap.shape == (1, T) and np.allclose(ap, 1.0)

def test_ap_per_class_pr_curve():
    """Tests ap_per_class against a known PR curve."""
    T = 10
    tp = np.array([[True] * T, [False] * T, [True] * T], dtype=bool)
    conf = np.array([0.9, 0.8, 0.7])
    pred_cls = np.array([0, 0, 0])
    target_cls = np.array([0, 0])

    _, _, _, _, _, ap, _ = ap_per_class(tp, conf, pred_cls, target_cls)

    observed_ap = 0.8283333333333331
    assert ap.shape == (1, T)
    assert np.allclose(ap, observed_ap)

def test_process_batch_correctness_multi_iou():
    labels = torch.tensor([
        [0., 10., 10., 20., 20.],  # [cls, x1,y1,x2,y2]
    ])
    dets = torch.tensor([
        [10., 10., 20., 20., 0.9, 0.],  # perfect match
        [0., 0., 5., 5., 0.8, 0.],  # no overlap
    ])
    iouv = torch.linspace(0.5, 0.95, 10)
    correct = process_batch(dets, labels, iouv)
    assert correct.shape == (2, 10)
    assert correct[0].all()  # perfect match is correct at all IoUs
    assert (~correct[1]).all()  # second is incorrect at all IoUs

def test_ap_per_class_empty_and_shapes():
    tp = torch.empty(0, 10, dtype=torch.bool).numpy()
    conf = torch.empty(0).numpy()
    pred_cls = torch.empty(0).numpy()
    target_cls = torch.tensor([0]).numpy()

    res = ap_per_class(tp, conf, pred_cls, target_cls)
    p, r, f1, ap, ucls = res[2], res[3], res[4], res[5], res[6]
    assert p.shape == (1, )  # nc==1 but no preds means zeros path
    assert ap.shape == (1, 10)
    assert ucls.shape == (1, )

def test_ap_per_class_single_perfect_class():
    labels = torch.tensor([[0., 10., 10., 20., 20.]])
    dets = torch.tensor([[10., 10., 20., 20., 0.99, 0.]])
    iouv = torch.linspace(0.5, 0.95, 10)
    correct = process_batch(dets, labels, iouv).cpu().numpy()
    conf = dets[:, 4].cpu().numpy()
    pred_cls = dets[:, 5].cpu().numpy()
    target_cls = labels[:, 0].cpu().numpy()

    res = ap_per_class(correct, conf, pred_cls, target_cls)
    p_best, r_best, f1_best, ap = res[2], res[3], res[4], res[5]

    assert np.allclose(ap, 1.0)
    assert np.allclose(p_best, 1.0, atol=1e-3)
    assert np.allclose(r_best, 1.0, atol=1e-3)
    assert np.allclose(f1_best, 1.0, atol=1e-3)

def test_ap_per_class_shared_operating_point():
    labels = torch.tensor([
        [0., 10., 10., 20., 20.],
        [1., 40., 40., 60., 60.],
    ])
    dets = torch.tensor([
        [10., 10., 20., 20., 0.99, 0.],  # TP for class 0
        [0., 0., 5., 5., 0.95, 0.],  # FP for class 0
        [40., 40., 60., 60., 0.90, 1.],  # TP for class 1
        [35., 35., 45., 45., 0.85, 1.],  # Partial overlap -> FP for stricter IoU
    ])
    iouv = torch.linspace(0.5, 0.95, 10)
    correct = process_batch(dets, labels, iouv).cpu().numpy()
    conf = dets[:, 4].cpu().numpy()
    pred_cls = dets[:, 5].cpu().numpy()
    target_cls = labels[:, 0].cpu().numpy()

    res = ap_per_class(correct, conf, pred_cls, target_cls)
    p_best, r_best, f1_best, ap = res[2], res[3], res[4], res[5]

    assert ap.shape == (2, 10)
    assert p_best.shape == (2, ) and r_best.shape == (2, ) and f1_best.shape == (2, )
    assert np.all((f1_best >= 0.0) & (f1_best <= 1.0))

def test_compute_confusion_matrix_basic():
    true_cls = torch.tensor([0, 1])
    pred_cls = torch.tensor([0, 0])
    iou = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    cm = compute_confusion_matrix(pred_cls, true_cls, iou, iou_thr=0.5, nc=2)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[2, 0] == 1
    assert cm[1, 2] == 1

def test_metric_properties_and_indexing():
    m = Metric()
    m.nc = 3
    T = 10
    all_ap = np.stack([
        np.linspace(0.1, 1.0, T),
        np.linspace(0.2, 0.9, T),
        np.linspace(0.3, 0.8, T),
    ],
                      axis=0)
    iouv = torch.linspace(0.5, 0.95, T).cpu().numpy()
    m.iouv = iouv

    p_best = np.array([0.5, 0.6, 0.7])
    r_best = np.array([0.6, 0.7, 0.8])
    f1_best = 2 * p_best * r_best / np.maximum(p_best + r_best, 1e-12)
    ap_class_index = np.array([2, 0, 1])  # arbitrary reordering

    m.update((None, None, p_best, r_best, f1_best, all_ap, ap_class_index))

    assert np.isclose(m.map, all_ap.mean())
    j50 = int(np.argmin(np.abs(m.iouv - 0.50)))
    assert np.isclose(m.map50, all_ap[:, j50].mean())
    assert np.allclose(m.ap50, all_ap[:, j50])
    maps = m.maps
    assert maps.shape == (m.nc, )
    assert np.isclose(maps[2], all_ap[0].mean())  # because ap_class_index[0] == 2

    j50 = int(np.argmin(np.abs(m.iouv - 0.50)))
    assert np.isclose(m.map50, all_ap[:, j50].mean())
    assert np.allclose(m.ap50, all_ap[:, j50])
    maps = m.maps
    assert maps.shape == (m.nc, )
    assert np.isclose(maps[2], all_ap[0].mean())  # because ap_class_index[0] == 2

def test_process_batch_matches_and_empty_paths():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = W = 256
    gt_yolo = torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device)  # centered
    gt_cls = torch.tensor([3.0], device=device)
    gt_xyxy = yolo_to_xyxy(gt_yolo, (H, W)).float()
    labels = torch.cat([gt_cls.view(-1, 1), gt_xyxy], dim=1)  # [1,5]
    dets = torch.cat([gt_xyxy, torch.tensor([[0.9]], device=device),
                      gt_cls.view(-1, 1)], dim=1)  # [1,6]
    iouv = torch.tensor([0.5, 0.75, 0.9], device=device)
    tp = process_batch(dets, labels, iouv)
    assert tp.shape == (1, 3) and tp.all(), "Expected perfect matches across IoU thresholds"
    tp2 = process_batch(torch.empty(0, 6, device=device), labels, iouv)
    assert tp2.shape == (0, 3) and tp2.numel() == 0
    tp3 = process_batch(dets, torch.empty(0, 5, device=device), iouv)
    assert tp3.shape == (1, 3) and (~tp3
                                   ).all(), "No GTs => all predictions are false across thresholds"
