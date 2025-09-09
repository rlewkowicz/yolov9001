import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from unittest.mock import patch

from utils.loss import DetectionLoss
from tests.config import DEVICE, NUM_CLASSES, IMG_SIZE

class MockModelForLoss(nn.Module):
    def __init__(self, hyp, nc=3, reg_max=16, strides=[8, 16, 32]):
        super().__init__()
        self.hyp = hyp
        self.nc = nc
        self.reg_max = reg_max
        self.strides = torch.tensor(strides)
        from utils.geometry import DFLDecoder
        self.dfl_decoder = DFLDecoder(reg_max=reg_max)
        self.dummy_param = nn.Parameter(torch.empty(0))

    def parameters(self):
        return iter([self.dummy_param])

def _fresh_loss(hyp, nc=3, device="cpu"):
    model = MockModelForLoss(hyp, nc=nc).to(device)
    return DetectionLoss(model)

def test_vfl_gamma_zero_reduces_to_weighted_bce():
    hyp = {"cls_pw": 1.0, "vfl_alpha": 0.75, "vfl_gamma": 2.0, "cls_type": "vfl"}
    loss_fn = _fresh_loss(hyp=hyp)
    loss_fn.vfl_gamma = 0.0
    B, A, C = 1, 64, loss_fn.nc
    pred = torch.randn(B, A, C)
    tgt = torch.zeros(B, A, C)
    tgt[0, :5, 0] = 0.7  # 5 positives
    fg_mask = torch.zeros(B, A, dtype=torch.bool)
    fg_mask[0, :5] = True
    pred_prob = pred.sigmoid()
    pos_mask = tgt > 0
    weight = torch.where(
        pos_mask, tgt * loss_fn.hyp.get("cls_pw", 1.0),
        loss_fn.vfl_alpha * torch.ones_like(pred_prob)
    )
    bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction='none')
    expected = (bce * weight).sum() / weight.sum().clamp_min(1e-6)
    got = loss_fn.cls_loss(pred, tgt, fg_mask)
    assert torch.allclose(got, expected, rtol=1e-6)

def test_build_dfl_targets_edges_and_smoothing():
    loss_fn = _fresh_loss(hyp={"dfl_eps": 1e-3, "dfl_label_smooth": 0.2})
    eps = loss_fn.dfl_eps
    reg_max = loss_fn.reg_max
    t = torch.tensor([0.0, reg_max - 1 - eps - 1e-6, 3.5], dtype=torch.float32)
    T = loss_fn.build_dfl_targets(t)
    assert torch.allclose(T.sum(1), torch.ones_like(t), atol=1e-6)
    assert T.shape[1] == reg_max

def test_loss_forward_pass(loss_fn, model, dummy_input, dummy_targets):
    """Tests the full forward pass of the DetectionLoss module."""
    model.train()
    with torch.no_grad():
        preds = model(dummy_input)

    labels = list(dummy_targets)
    bs = len(labels)
    for i in range(bs):
        if labels[i].numel():
            bi = torch.full((labels[i].shape[0], 1), i, dtype=labels[i].dtype, device=DEVICE)
            labels[i] = torch.cat([bi, labels[i]], 1)
    targets = torch.cat([l for l in labels if l.numel()], 0) if any(
        l.numel() for l in labels
    ) else torch.zeros((0, 6), device=DEVICE)

    total_loss, loss_items = loss_fn(preds, targets)
    assert isinstance(total_loss, torch.Tensor) and total_loss.dim() == 0
    assert torch.isfinite(total_loss) and total_loss.item() > 0

def test_loss_empty_targets(loss_fn, model, dummy_input):
    """Tests that the loss function handles batches with no ground truth objects."""
    model.train()
    with torch.no_grad():
        preds = model(dummy_input)
    empty_targets = torch.zeros((0, 6), device=DEVICE)
    total_loss, loss_items = loss_fn(preds, empty_targets)
    assert torch.isfinite(total_loss) and total_loss.item() >= 0

def test_total_loss_respects_component_weights(loss_fn, model, dummy_input, dummy_targets):
    """Tests that loss component weights are correctly applied."""
    model.train()
    with torch.no_grad():
        preds = model(dummy_input)

    labels = list(dummy_targets)
    bs = len(labels)
    for i in range(bs):
        if labels[i].numel():
            bi = torch.full((labels[i].shape[0], 1), i, dtype=labels[i].dtype, device=DEVICE)
            labels[i] = torch.cat([bi, labels[i]], 1)
    targets = torch.cat([l for l in labels if l.numel()], 0) if any(
        l.numel() for l in labels
    ) else torch.zeros((0, 6), device=DEVICE)

    bw, cw, dw = loss_fn.box_weight, loss_fn.cls_weight, loss_fn.dfl_weight
    base, _ = loss_fn(preds, targets)
    base = base.item()
    loss_fn.box_weight = 0.0
    less_box, _ = loss_fn(preds, targets)
    assert less_box.item() <= base + 1e-5
    loss_fn.box_weight = bw

def test_box_loss_invariant_to_uniform_score_scale(loss_fn):
    """Tests that box_loss is invariant to uniform scaling of target_scores."""
    anchors, _ = loss_fn.dfl_decoder.get_anchors([(IMG_SIZE // 8, IMG_SIZE // 8),
                                                  (IMG_SIZE // 16, IMG_SIZE // 16),
                                                  (IMG_SIZE // 32, IMG_SIZE // 32)])
    pred_dist = torch.randn(1, anchors.shape[0], 4 * loss_fn.reg_max, device=DEVICE)
    pred_bboxes = torch.rand(1, anchors.shape[0], 4, device=DEVICE) * IMG_SIZE
    target_bboxes = torch.rand(1, anchors.shape[0], 4, device=DEVICE) * IMG_SIZE
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    fg_mask[0, :64] = True
    target_scores = torch.zeros(1, anchors.shape[0], NUM_CLASSES, device=DEVICE)
    target_scores[fg_mask] = 0.4
    loss1 = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores, fg_mask)
    loss2 = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores * 2.5, fg_mask)
    assert torch.allclose(loss1, loss2, rtol=1e-6)

def test_box_loss_no_fg_returns_zero(loss_fn):
    """Tests that box_loss is zero when there are no foreground anchors."""
    anchors, _ = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    pred_dist = torch.randn(1, anchors.shape[0], 4 * loss_fn.reg_max, device=DEVICE)
    pred_bboxes = torch.randn(1, anchors.shape[0], 4, device=DEVICE)
    target_bboxes = torch.randn(1, anchors.shape[0], 4, device=DEVICE)
    target_scores = torch.zeros(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    val = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores, fg_mask)
    assert val.item() == 0.0

def test_cls_loss_negative_only_normalization(loss_fn):
    """Tests cls_loss normalization when only negative samples are present."""
    loss_fn.hyp['cls_type'] = 'vfl'
    anchors, _ = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    pred_scores = torch.randn(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    target_scores = torch.zeros(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    alpha, gamma = loss_fn.vfl_alpha, loss_fn.vfl_gamma
    neg_weight = alpha * pred_scores.sigmoid().pow(gamma)
    bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
    expected = (bce * neg_weight).sum() / neg_weight.sum().clamp_min(1e-6)
    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected, rtol=1e-5)
    loss_fn.hyp['cls_type'] = 'bce'  # Reset

def test_cls_loss_qfl_branch_matches_expected(loss_fn):
    """Tests the Quality Focal Loss (QFL) implementation against a manual calculation."""
    loss_fn.hyp['cls_type'] = 'qfl'
    anchors, _ = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    pred_scores = torch.randn(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    target_scores = torch.zeros(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    fg_mask[0, 0] = True
    target_scores[0, 0, 0] = 0.7
    pred_prob = pred_scores.sigmoid()
    pos_mask = target_scores > 0
    weight = torch.zeros_like(pred_scores)
    weight[pos_mask] = (target_scores[pos_mask] - pred_prob[pos_mask]).abs().pow(loss_fn.qfl_beta)
    weight[~pos_mask] = loss_fn.vfl_alpha * pred_prob[~pos_mask].pow(loss_fn.vfl_gamma)
    bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
    expected = (bce * weight).sum() / weight.sum().clamp_min(1e-6)
    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected, rtol=1e-6)
    loss_fn.hyp['cls_type'] = 'vfl'  # Reset for other tests

def test_dfl_loss_invariant_to_logit_shift(loss_fn, model, dummy_input, dummy_targets):
    """Tests that DFL loss is invariant to a uniform shift in input logits via the forward pass."""
    model.train()
    with torch.no_grad():
        preds = model(dummy_input)

    preds_shifted = []
    for p in preds:
        dist_part, cls_part = p.split([loss_fn.reg_max * 4, loss_fn.nc], 1)
        dist_part_shifted = dist_part + 123.0
        preds_shifted.append(torch.cat([dist_part_shifted, cls_part], 1))

    labels = list(dummy_targets)
    bs = len(labels)
    for i in range(bs):
        if labels[i].numel():
            bi = torch.full((labels[i].shape[0], 1), i, dtype=labels[i].dtype, device=DEVICE)
            labels[i] = torch.cat([bi, labels[i]], 1)
    targets = torch.cat([l for l in labels if l.numel()], 0) if any(
        l.numel() for l in labels
    ) else torch.zeros((0, 6), device=DEVICE)

    _, loss_items_base = loss_fn(preds, targets)
    _, loss_items_shifted = loss_fn(preds_shifted, targets)

    assert torch.allclose(
        torch.tensor(loss_items_base['dfl']),
        torch.tensor(loss_items_shifted['dfl']),
        rtol=1e-5,
        atol=1e-5
    )

def test_dfl_loss_no_fg_returns_zero(loss_fn):
    """Tests that DFL loss is zero when there are no foreground anchors."""
    anchors, _ = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    pred_dist = torch.randn(1, anchors.shape[0], 4 * loss_fn.reg_max, device=DEVICE)
    target_bboxes = torch.zeros(1, anchors.shape[0], 4, device=DEVICE)
    target_scores = torch.zeros(1, anchors.shape[0], loss_fn.nc, device=DEVICE)
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    val = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores, fg_mask, anchors)
    assert val.item() == 0.0

def test_build_dfl_targets_smoothing(loss_fn):
    """Tests that label smoothing moves the DFL target distribution toward uniform."""
    t = torch.tensor([3.5], device=DEVICE)
    base = loss_fn.build_dfl_targets(t, smooth=0.0)
    smth = loss_fn.build_dfl_targets(t, smooth=0.2)
    uniform = torch.full_like(base, 1.0 / base.shape[1])
    assert (smth - uniform).abs().sum() < (base - uniform).abs().sum()

def test_cls_loss_vfl_positive_and_negatives(
    loss_fn, device="cuda" if torch.cuda.is_available() else "cpu"
):
    loss_fn.hyp['cls_type'] = 'vfl'
    A = 200  # anchors
    C = max(loss_fn.nc, 5)
    pred_scores = torch.randn(1, A, C, device=device)
    target_scores = torch.zeros(1, A, C, device=device)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=device)

    pos_idx = torch.arange(0, 10, device=device)
    fg_mask[0, pos_idx] = True
    target_scores[0, pos_idx, 0] = 0.8  # IoU-like target

    alpha, gamma = loss_fn.vfl_alpha, loss_fn.vfl_gamma
    cls_pw = float(loss_fn.hyp.get("cls_pw", 1.0))
    pred_prob = pred_scores.sigmoid()
    pos_mask = target_scores > 0
    pos_weight = target_scores * cls_pw
    neg_weight = alpha * pred_prob.pow(gamma)
    weight = torch.where(pos_mask, pos_weight, neg_weight)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_scores, target_scores, reduction='none'
    )
    expected = (bce * weight).sum() / weight.sum().clamp_min(1e-6)

    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected, rtol=1e-6)
    loss_fn.hyp['cls_type'] = 'bce'  # Reset

def test_build_dfl_targets_properties(loss_fn):
    t = torch.tensor([0.0, 3.49, loss_fn.reg_max - 1 - loss_fn.dfl_eps - 1e-6],
                     device=loss_fn.device)
    T = loss_fn.build_dfl_targets(t, smooth=loss_fn.dfl_label_smooth)
    assert torch.allclose(T.sum(dim=1), torch.ones_like(t, dtype=T.dtype), atol=1e-6)
    assert T.shape[1] == loss_fn.reg_max

def test_cls_loss_cls_pw_scaling(loss_fn, device="cuda" if torch.cuda.is_available() else "cpu"):
    """This test now verifies that cls_pw has NO effect when cls_type is 'bce'."""
    loss_fn.hyp['cls_type'] = 'bce'
    A = 20
    C = max(loss_fn.nc, 2)
    pred_scores = torch.randn(1, A, C, device=device)
    target_scores = torch.zeros(1, A, C, device=device)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=device)

    pos_idx = torch.arange(0, 5, device=device)
    fg_mask[0, pos_idx] = True
    target_scores[0, pos_idx, 0] = torch.rand(5, device=device) * 0.5 + 0.5  # IoU [0.5, 1.0]

    loss_fn.hyp['cls_pw'] = 1.0
    loss1 = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)

    loss_fn.hyp['cls_pw'] = 2.0
    loss2 = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)

    assert torch.allclose(loss1, loss2)  # Should be equal as cls_pw is ignored in BCE mode
    loss_fn.hyp['cls_type'] = 'bce'  # Reset to default

def test_dfl_decode_parity_train_vs_infer(loss_fn):
    import torch
    from utils.geometry import DFLDecoder, decode_distances, dfl_expectation
    from utils.boxes import BoundingBox
    anchors, strides = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    bs, na, reg_max = 2, anchors.shape[0], loss_fn.reg_max
    logits = torch.randn(bs, na, 4 * reg_max, device=DEVICE)
    bins = torch.arange(reg_max, dtype=torch.float32, device=DEVICE)

    dist_train = dfl_expectation(
        logits.view(bs, na, 4, reg_max),
        reg_max,
        tau=float(loss_fn.hyp.get("dfl_tau", 1.0)),
        bins=bins
    )
    dist_train_px = dist_train * strides.view(1, -1, 1)
    anchors_px = anchors * strides
    boxes_train = BoundingBox.dist2bbox(dist_train_px, anchors_px.unsqueeze(0), xywh=False)

    decoder = DFLDecoder(
        reg_max=reg_max,
        strides=strides.tolist(),
        device=logits.device.type,
        tau=float(loss_fn.hyp.get("dfl_tau", 1.0))
    )
    boxes_infer = decode_distances(
        anchors, strides, logits, reg_max, tau=decoder.tau, decoder=decoder
    )

    assert torch.allclose(boxes_train, boxes_infer, atol=1e-5)

def test_cls_loss_extremes(loss_fn):
    import torch
    A, C = 1024, max(loss_fn.nc, 5)
    pred_scores = torch.empty(1, A, C).normal_(mean=0, std=1)
    pred_scores[0, :100] = 50.0  # saturating + logits
    pred_scores[0, 100:200] = -50.0  # saturating - logits
    target_scores = torch.zeros(1, A, C)
    fg_mask = torch.zeros(1, A, dtype=torch.bool)
    vfl = loss_fn.cls_loss(pred_scores.clone(), target_scores, fg_mask)
    loss_fn.hyp['cls_type'] = 'qfl'
    qfl = loss_fn.cls_loss(pred_scores.clone(), target_scores, fg_mask)
    assert torch.isfinite(vfl) and torch.isfinite(qfl)
    loss_fn.hyp['cls_type'] = 'vfl'

def test_dfl_target_boundary_coverage(loss_fn):
    import torch
    rm = loss_fn.reg_max
    t = torch.tensor([-1.0, rm - 1 + 10.0, 0.0, rm - 1 - 1e-6], dtype=torch.float32)
    T = loss_fn.build_dfl_targets(t)
    assert T.shape == (t.numel(), rm)
    assert torch.allclose(T.sum(1), torch.ones(t.numel()))
    assert T[1].argmax().item() in {rm - 1 - 1, rm - 1}

def test_l1_loss_path(loss_fn):
    """Tests the L1 loss path for numerical safety and correctness."""
    loss_fn.l1_weight = 1.0
    anchors, strides = loss_fn.dfl_decoder.get_anchors([(80, 80), (40, 40), (20, 20)])
    pred_dist = torch.randn(1, anchors.shape[0], 4 * loss_fn.reg_max, device=DEVICE)
    pred_bboxes = torch.rand(1, anchors.shape[0], 4, device=DEVICE) * IMG_SIZE
    target_bboxes = torch.rand(1, anchors.shape[0], 4, device=DEVICE) * IMG_SIZE
    fg_mask = torch.zeros(1, anchors.shape[0], dtype=torch.bool, device=DEVICE)
    fg_mask[0, :64] = True
    target_scores = torch.zeros(1, anchors.shape[0], NUM_CLASSES, device=DEVICE)
    target_scores[fg_mask] = 0.4
    loss = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores, fg_mask)
    assert torch.isfinite(loss) and loss.item() >= 0
    loss_fn.l1_weight = 0.0

def test_quality_weight_normalizer_fallback(loss_fn):
    """Tests the fallback normalization in _quality_weight_and_norm."""
    A = 10
    target_scores = torch.zeros(1, A, loss_fn.nc, device=loss_fn.device)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=loss_fn.device)
    fg_mask[0, :5] = True  # 5 foreground samples
    w, denom = loss_fn._quality_weight_and_norm(target_scores, fg_mask)
    assert torch.isfinite(denom) and denom.item() == 5

def test_cls_loss_degenerate_masks(loss_fn):
    """Tests classification loss for stability with empty foreground masks."""
    B, N, C = 1, 8, loss_fn.nc
    pred_scores = torch.randn(B, N, C, device=loss_fn.device)
    target_scores = torch.zeros(B, N, C, device=loss_fn.device)
    fg_mask = torch.zeros(B, N, dtype=torch.bool, device=loss_fn.device)  # All false

    loss_fn.hyp['cls_type'] = 'vfl'
    v1 = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)

    loss_fn.hyp['cls_type'] = 'qfl'
    v2 = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)

    loss_fn.hyp['cls_type'] = 'vfl'  # Reset
    assert torch.isfinite(v1) and v1.item() >= 0
    assert torch.isfinite(v2) and v2.item() >= 0

def test_cls_loss_bce_soft_targets_and_normalization(loss_fn):
    """Tests the BCE loss branch with realistic soft targets and YOLOv8-style normalization."""
    loss_fn.hyp['cls_type'] = 'bce'
    B, A, C = 1, 100, loss_fn.nc
    pred_scores = torch.randn(B, A, C, device=DEVICE)
    target_scores = torch.zeros(B, A, C, device=DEVICE)
    fg_mask = torch.zeros(B, A, dtype=torch.bool, device=DEVICE)

    fg_mask[0, 0] = True
    target_scores[0, 0, 1] = 0.85  # 1st positive, class 1, iou=0.85
    fg_mask[0, 1] = True
    target_scores[0, 1, 2] = 0.60  # 2nd positive, class 2, iou=0.60

    num_pos = fg_mask.sum()
    assert num_pos == 2

    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='sum')
    expected_loss = bce_loss / num_pos

    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected_loss)
    loss_fn.hyp['cls_type'] = 'vfl'  # Reset

def test_cls_loss_bce_no_positives(loss_fn):
    """Tests the BCE loss branch when there are no positive anchors."""
    loss_fn.hyp['cls_type'] = 'bce'
    B, A, C = 1, 100, loss_fn.nc
    pred_scores = torch.randn(B, A, C, device=DEVICE)
    target_scores = torch.zeros(B, A, C, device=DEVICE)
    fg_mask = torch.zeros(B, A, dtype=torch.bool, device=DEVICE)  # No positives

    num_pos = fg_mask.sum().clamp_min(1)
    assert num_pos == 1

    bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='sum')
    expected_loss = bce_loss / num_pos

    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected_loss)
    loss_fn.hyp['cls_type'] = 'vfl'  # Reset

def test_cls_loss_invalid_type_defaults_to_vfl(loss_fn):
    """Tests that an invalid cls_type safely defaults to VFL."""
    loss_fn.hyp['cls_type'] = 'invalid_type'
    A = 10
    C = loss_fn.nc
    pred_scores = torch.randn(1, A, C, device=DEVICE)
    target_scores = torch.zeros(1, A, C, device=DEVICE)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=DEVICE)
    fg_mask[0, 0] = True
    target_scores[0, 0, 0] = 0.9

    alpha, gamma = loss_fn.vfl_alpha, loss_fn.vfl_gamma
    cls_pw = float(loss_fn.hyp.get("cls_pw", 1.0))
    pred_prob = pred_scores.sigmoid()
    pos_mask = target_scores > 0
    pos_weight = target_scores * cls_pw
    neg_weight = alpha * pred_prob.pow(gamma)
    weight = torch.where(pos_mask, pos_weight, neg_weight)
    bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction='none')
    expected_vfl = (bce * weight).sum() / weight.sum().clamp_min(1e-6)

    got = loss_fn.cls_loss(pred_scores, target_scores, fg_mask)
    assert torch.allclose(got, expected_vfl)
    loss_fn.hyp['cls_type'] = 'vfl'  # Reset

def test_box_and_dfl_loss_uses_quality_weighting(loss_fn):
    """Verify that box and DFL losses are weighted by the target quality score."""
    A = 10
    pred_bboxes = torch.rand(1, A, 4, device=DEVICE)
    target_bboxes = torch.rand(1, A, 4, device=DEVICE)
    pred_dist = torch.randn(1, A, 4 * loss_fn.reg_max, device=DEVICE)
    anchors = torch.rand(A, 2, device=DEVICE)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=DEVICE)
    fg_mask[0, :2] = True  # 2 positives

    target_scores_low = torch.zeros(1, A, loss_fn.nc, device=DEVICE)
    target_scores_low[0, 0, 0] = 0.2
    target_scores_low[0, 1, 0] = 0.3

    box_loss_low = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores_low, fg_mask)
    dfl_loss_low = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores_low, fg_mask, anchors)

    target_scores_high = torch.zeros(1, A, loss_fn.nc, device=DEVICE)
    target_scores_high[0, 0, 0] = 0.8
    target_scores_high[0, 1, 0] = 0.9

    box_loss_high = loss_fn.box_loss(pred_bboxes, target_bboxes, target_scores_high, fg_mask)
    dfl_loss_high = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores_high, fg_mask, anchors)

    assert box_loss_low != box_loss_high
    assert dfl_loss_low != dfl_loss_high

def test_loss_forward_handles_non_finite_preds(loss_fn, model, dummy_targets):
    """Ensures the loss forward pass is robust to non-finite prediction inputs."""
    bs = 2
    C = model.nc
    R = loss_fn.reg_max
    preds = [
        torch.randn(bs, C + R * 4, 80 // int(s.item()), 80 // int(s.item()), device=DEVICE)
        for s in loss_fn.strides
    ]
    preds[0][0, 0, 0, 0] = torch.nan
    preds[1][0, 1, 1, 1] = torch.inf

    labels = list(dummy_targets)
    for i in range(bs):
        if labels[i].numel():
            bi = torch.full((labels[i].shape[0], 1), i, dtype=labels[i].dtype, device=DEVICE)
            labels[i] = torch.cat([bi, labels[i]], 1)
    targets = torch.cat([l for l in labels if l.numel()], 0)

    total_loss, loss_items = loss_fn(preds, targets)
    assert torch.isfinite(total_loss), "Total loss should be finite even with non-finite preds"
    assert all(np.isfinite(v) for v in loss_items.values()), "All loss items should be finite"

@patch('utils.loss.TaskAlignedAssigner')
def test_loss_clamp_target_scores(MockAssigner, model):
    """
    Tests that the DetectionLoss clamps target_scores from the assigner to prevent
    out-of-bounds values [0,1] and NaNs from causing issues in the cls_loss.
    """
    loss_fn = DetectionLoss(model, imgsz=640)
    loss_fn.dfl_strict_targets = False  # Disable strict check for this test
    B, C = 2, model.nc

    dummy_preds = [
        torch.randn(B, C + loss_fn.reg_max * 4, 80 // int(s), 80 // int(s), device=DEVICE)
        for s in loss_fn.strides
    ]
    feat_shapes = [(p.shape[-2], p.shape[-1]) for p in dummy_preds]
    anchors, _ = loss_fn.dfl_decoder.get_anchors(feat_shapes)
    N = anchors.shape[0]

    mock_scores = torch.rand(B, N, C, device=DEVICE) * 1.5 - 0.25  # Range approx [-0.25, 1.25]
    mock_scores[0, 0, 0] = torch.nan

    mock_assigner_output = {
        "cls": {
            "target_scores": mock_scores, "fg_mask":
                torch.ones(B, N, dtype=torch.bool, device=DEVICE)
        }, "reg": {
            "target_bboxes": torch.rand(B, N, 4, device=DEVICE), "fg_mask":
                torch.ones(B, N, dtype=torch.bool, device=DEVICE)
        }
    }
    mock_assigner_instance = MockAssigner.return_value
    mock_assigner_instance.return_value = mock_assigner_output

    dummy_targets = torch.zeros(0, 6, device=DEVICE)  # empty targets are fine

    total_loss, loss_items = loss_fn(dummy_preds, dummy_targets)

    assert torch.isfinite(total_loss)
    assert loss_items['cls'] >= 0

def test_dfl_targets_strict_mode_exposes_oob(loss_fn):
    loss_fn.dfl_strict_targets = True
    loss_fn.dfl_clip_tolerance = 0.0
    reg_max = loss_fn.reg_max
    A = 64
    pred_dist = torch.randn(1, A, 4 * reg_max, device=DEVICE)
    anchors = torch.zeros(A, 2, device=DEVICE)
    anchors[:, :] = 1.0  # small anchor grid in 'grid units'
    target_bboxes = torch.full((1, A, 4), 10 * reg_max, device=DEVICE)
    target_scores = torch.zeros(1, A, loss_fn.nc, device=DEVICE)
    fg_mask = torch.zeros(1, A, dtype=torch.bool, device=DEVICE)
    fg_mask[0, :A // 2] = True
    with pytest.raises(AssertionError):
        _ = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores, fg_mask, anchors)
    loss_fn.dfl_strict_targets = False
    val = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores, fg_mask, anchors)
    assert torch.isfinite(val)

def test_build_dfl_targets_boundaries(loss_fn):
    """Thorough boundary checks for DFL target-building."""
    reg_max = loss_fn.reg_max
    eps = loss_fn.dfl_eps
    lo, hi = (0.0, float(reg_max - 1) - max(float(eps), 1e-6))
    t = torch.tensor([0.0, hi, 1.4999, 2.5001, hi - 1e-7],
                     dtype=torch.float32,
                     device=loss_fn.device)
    T = loss_fn.build_dfl_targets(t)
    assert T.shape == (t.numel(), reg_max)
    assert torch.allclose(T.sum(1), torch.ones_like(t), atol=1e-6)
    idxs = T.nonzero()
    for i in range(t.numel()):
        bins = idxs[idxs[:, 0] == i][:, 1]
        assert 1 <= bins.numel() <= 2
        if bins.numel() == 2:
            assert (bins[1] - bins[0]).item() == 1

def test_dfl_targets_strict_mode_edges(loss_fn):
    """Test strict mode with various out-of-bounds targets."""
    loss_fn.dfl_strict_targets = True
    loss_fn.dfl_clip_tolerance = 0.0
    reg_max = loss_fn.reg_max
    A = 8
    anchors = torch.zeros(A, 2, device=loss_fn.device)
    target_bboxes = torch.tensor(
        [[
            [-10., -10., -5., -5.],  # negative; large l/t distances
            [0., 0., 1e-6, 1e-6],  # tiny
            [0., 0., 1e6, 1e6],  # huge r/b
            [0., 0., (reg_max + 10.), (reg_max + 10.)],  # above high bound in grid units
            [1., 1., 2., 2.],
            [2., 2., 3., 3.],
            [3., 3., 4., 4.],
            [4., 4., 5., 5.],
        ]],
        device=loss_fn.device
    )
    pred_dist = torch.randn(1, A, 4 * reg_max, device=loss_fn.device)
    target_scores = torch.ones(1, A, loss_fn.nc, device=loss_fn.device) * 0.5
    fg_mask = torch.ones(1, A, dtype=torch.bool, device=loss_fn.device)
    with pytest.raises(AssertionError):
        _ = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores, fg_mask, anchors)
    loss_fn.dfl_strict_targets = False
    val = loss_fn.dfl_loss(pred_dist, target_bboxes, target_scores, fg_mask, anchors)
    assert torch.isfinite(val)
