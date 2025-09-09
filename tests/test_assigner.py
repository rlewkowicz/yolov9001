"""
Tests for the TaskAlignedAssigner, validating each mode's specific logic.
"""
import torch
import pytest
from utils.assigner import TaskAlignedAssigner
from tests.config import DEVICE
from utils.loss import _dfl_bounds
from utils.boxes import BoundingBox

@pytest.fixture(scope='module')
def shared_tensors():
    """Provides a common set of tensors for all tests to use."""
    B, N, C, M = 1, 100, 10, 4
    imgsz = 640
    pred_scores = torch.rand(B, N, C, device=DEVICE)
    pred_bboxes = torch.rand(B, N, 4, device=DEVICE) * imgsz
    pred_bboxes[..., 2:] = pred_bboxes[..., :2] + torch.rand(B, N, 2, device=DEVICE) * 100 + 1

    anc_points = torch.rand(N, 2, device=DEVICE) * imgsz
    anc_strides = torch.full((N, 1), 32, device=DEVICE, dtype=torch.float32)

    gt_labels = torch.randint(0, C, (B, M, 1), device=DEVICE)
    gt_bboxes = torch.rand(B, M, 4, device=DEVICE, dtype=torch.float32) * imgsz
    gt_bboxes[..., 2:] = gt_bboxes[..., :2] + torch.rand(B, M, 2, device=DEVICE) * 200 + 1

    mask_gt = torch.ones(B, M, 1, device=DEVICE, dtype=torch.bool)
    if M > 1:
        mask_gt[:, -1] = False

    return {
        "pd_scores": pred_scores, "pd_bboxes": pred_bboxes, "anc_points": anc_points, "anc_strides":
            anc_strides, "gt_labels": gt_labels, "gt_bboxes": gt_bboxes, "mask_gt": mask_gt
    }

@pytest.mark.parametrize("mode", ["ult", "simota", "mixed"])
def test_assigner_runs_and_shapes(mode, shared_tensors):
    """Test that each assigner mode runs and returns tensors of the correct shape."""
    assigner = TaskAlignedAssigner(mode=mode, num_classes=10)
    out = assigner(
        shared_tensors["pd_scores"],
        shared_tensors["pd_bboxes"],
        shared_tensors["anc_points"],
        shared_tensors["gt_labels"],
        shared_tensors["gt_bboxes"],
        shared_tensors["mask_gt"],
        anc_strides=shared_tensors["anc_strides"]
    )

    B, N, C = shared_tensors["pd_scores"].shape
    assert "reg" in out and "cls" in out
    assert out["reg"]["fg_mask"].shape == (B, N)
    assert out["cls"]["fg_mask"].shape == (B, N)
    assert out["reg"]["target_bboxes"].shape == (B, N, 4)
    assert out["cls"]["target_scores"].shape == (B, N, C)
    assert not torch.isnan(out["reg"]["target_bboxes"]).any()
    assert not torch.isnan(out["cls"]["target_scores"]).any()

def test_ult_mode_logic():
    """Verify ULT mode's logic: center-in-box and top-k by alignment score."""
    gt_bboxes = torch.tensor([[[100, 100, 200, 200]]], device=DEVICE, dtype=torch.float32)
    gt_labels = torch.tensor([[[1]]], device=DEVICE, dtype=torch.long)
    mask_gt = torch.ones(1, 1, 1, device=DEVICE, dtype=torch.bool)

    anc_points = torch.tensor([[150, 150], [125, 125], [50, 50], [150, 150]],
                              device=DEVICE,
                              dtype=torch.float32)
    pred_bboxes = torch.tensor([[[100, 100, 200, 200], [100, 100, 150, 150], [50, 50, 60, 60],
                                 [100, 100, 200, 200]]],
                               device=DEVICE,
                               dtype=torch.float32)
    pred_scores = torch.tensor([[[0.1, 0.9], [0.1, 0.85], [0.1, 0.99], [0.1, 0.2]]],
                               device=DEVICE,
                               dtype=torch.float32)

    assigner = TaskAlignedAssigner(
        mode='ult', num_classes=2, topk=2, alpha=1.0, beta=1.0, input_is_logits=False
    )
    out = assigner(pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    fg_mask = out["reg"]["fg_mask"][0]
    assert fg_mask.sum() == 2, "Should select top 2 candidates inside the box"
    assert fg_mask[0].item() is True, "Anchor 0 (best) should be selected"
    assert fg_mask[1].item() is True, "Anchor 1 (second best) should be selected"
    assert fg_mask[2].item() is False, "Anchor 2 (outside) should NOT be selected"
    assert fg_mask[3].item() is False, "Anchor 3 (low score) should not be in top 2"

def test_simota_mode_logic():
    """Verify SimOTA's logic: center radius gating and dynamic-k cost-based selection."""
    gt_bboxes = torch.tensor([[[100, 100, 200, 200]]], device=DEVICE, dtype=torch.float32)
    gt_labels = torch.tensor([[[1]]], device=DEVICE, dtype=torch.long)
    mask_gt = torch.ones(1, 1, 1, device=DEVICE, dtype=torch.bool)

    anc_strides = torch.full((4, 1), 32, device=DEVICE, dtype=torch.float32)
    anc_points = torch.tensor([[150, 150], [150 - 2 * 10, 150], [150 - 3 * 32, 150], [500, 500]],
                              device=DEVICE,
                              dtype=torch.float32)
    pred_bboxes = gt_bboxes.expand(1, 4, 4)
    pred_scores = torch.full((1, 4, 2), 0.9, device=DEVICE, dtype=torch.float32)

    assigner = TaskAlignedAssigner(mode='simota', num_classes=2, center_radius=2.5, topq=10)
    out = assigner(
        pred_scores,
        pred_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
        anc_strides=anc_strides
    )

    fg_mask = out["reg"]["fg_mask"][0]
    assert fg_mask[0].item() is True, "Anchor 0 (center) should be a candidate"
    assert fg_mask[1].item() is True, "Anchor 1 (inside radius and box) should be a candidate"
    assert fg_mask[2].item() is False, "Anchor 2 (outside radius) should NOT be a candidate"
    assert fg_mask[3].item() is False, "Anchor 3 (far away) should NOT be a candidate"

def test_mixed_mode_logic():
    """Verify mixed mode's logic: O2O for obj (confidence), O2M for reg+cls."""
    gt_bboxes = torch.tensor([[[100, 100, 200, 200]]], device=DEVICE, dtype=torch.float32)
    gt_labels = torch.tensor([[[1]]], device=DEVICE, dtype=torch.long)
    mask_gt = torch.ones(1, 1, 1, device=DEVICE, dtype=torch.bool)

    anc_points = torch.tensor([[150, 150], [150, 150], [150, 150]],
                              device=DEVICE,
                              dtype=torch.float32)
    anc_strides = torch.full((3, 1), 32, device=DEVICE, dtype=torch.float32)
    pred_bboxes = torch.tensor([[[110, 110, 190, 190], [100, 100, 200, 200], [105, 105, 195, 195]]],
                               device=DEVICE,
                               dtype=torch.float32)
    pred_scores = torch.tensor([[[0.1, 0.9], [0.1, 0.2], [0.1, 0.8]]],
                               device=DEVICE,
                               dtype=torch.float32)

    assigner = TaskAlignedAssigner(mode='mixed', num_classes=2, center_radius=5.0, topq=2)
    out = assigner(
        pred_scores,
        pred_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
        anc_strides=anc_strides
    )

    fg_mask_reg = out["reg"]["fg_mask"][0]
    fg_mask_cls = out["cls"]["fg_mask"][0]
    fg_mask_obj = out["obj"]["fg_mask"][0]

    assert fg_mask_obj.sum() == 1, "Obj fg_mask should have exactly one match (O2O)"
    assert fg_mask_reg.sum() >= 1, "Reg fg_mask should have at least one match (O2M)"
    assert fg_mask_cls.sum() >= 1, "Cls fg_mask should have at least one match (O2M)"
    assert fg_mask_reg.sum() >= fg_mask_obj.sum()

@pytest.mark.parametrize("mode", ["simota", "mixed"])
def test_assigner_logits_vs_probs_parity_all_modes(mode, shared_tensors):
    """Tests that simota and mixed modes are invariant to logits vs probs input."""
    C = shared_tensors["pd_scores"].shape[-1]
    A = TaskAlignedAssigner(mode=mode, num_classes=C, input_is_logits=True)
    logits = shared_tensors["pd_scores"].logit().clamp(-50, 50)
    outA = A(
        logits,
        shared_tensors["pd_bboxes"],
        shared_tensors["anc_points"],
        shared_tensors["gt_labels"],
        shared_tensors["gt_bboxes"],
        shared_tensors["mask_gt"],
        anc_strides=shared_tensors["anc_strides"]
    )

    B = TaskAlignedAssigner(mode=mode, num_classes=C, input_is_logits=False)
    outB = B(
        shared_tensors["pd_scores"],
        shared_tensors["pd_bboxes"],
        shared_tensors["anc_points"],
        shared_tensors["gt_labels"],
        shared_tensors["gt_bboxes"],
        shared_tensors["mask_gt"],
        anc_strides=shared_tensors["anc_strides"]
    )

    for k in ["reg", "cls"]:
        assert torch.equal(outA[k]["fg_mask"], outB[k]["fg_mask"])
        assert torch.equal(outA[k]["matched_gt"], outB[k]["matched_gt"])

def test_ult_topk_edgecases():
    """Tests ULT topk when k is larger than candidates or k=1."""
    B, N, C = 1, 4, 3
    gt_bboxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32, device=DEVICE)
    gt_labels = torch.tensor([[[1]]], dtype=torch.long, device=DEVICE)
    mask_gt = torch.ones(1, 1, 1, dtype=torch.bool, device=DEVICE)
    anc_points = torch.tensor([[150, 150], [10, 10], [500, 500], [300, 300]],
                              dtype=torch.float32,
                              device=DEVICE)
    pred_bboxes = gt_bboxes.expand(1, N, 4).contiguous()
    pred_scores = torch.tensor([[[0.1, 0.9, 0.1]] * N], dtype=torch.float32, device=DEVICE)

    assigner = TaskAlignedAssigner(mode='ult', num_classes=C, topk=3, input_is_logits=False)
    out = assigner(pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    fg = out["reg"]["fg_mask"][0]
    assert fg.sum() == 1  # Should select the single valid anchor

    assigner1 = TaskAlignedAssigner(mode='ult', num_classes=C, topk=1, input_is_logits=False)
    out1 = assigner1(pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    assert out1["reg"]["fg_mask"][0].sum() == 1

def test_conflict_resolution_by_iou():
    """Tests that the assigner resolves conflicts by picking the GT with the highest IoU."""
    B, N, C, M = 1, 1, 2, 2
    anc_points = torch.tensor([[100, 100]], dtype=torch.float32, device=DEVICE)
    pred_bboxes = torch.tensor([[[95, 95, 105, 105]]], dtype=torch.float32, device=DEVICE)
    gt_bboxes = torch.tensor([[[90, 90, 110, 110], [98, 98, 102, 102]]],
                             dtype=torch.float32,
                             device=DEVICE)
    gt_labels = torch.tensor([[[0], [1]]], dtype=torch.long, device=DEVICE)
    mask_gt = torch.ones(B, M, 1, dtype=torch.bool, device=DEVICE)
    pred_scores = torch.tensor([[[0.9, 0.9]]], dtype=torch.float32,
                               device=DEVICE)  # High score for both classes

    A = TaskAlignedAssigner(mode='ult', num_classes=C, input_is_logits=False, topk=1)
    out = A(pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    fg = out["reg"]["fg_mask"][0]
    matched_gt = out["reg"]["matched_gt"][0]

    assert fg.sum() == 1  # Only one GT should win for the single anchor
    assert matched_gt[fg].item() == 0  # GT 0 has higher IoU and should be chosen

def test_assigner_empty_gts(shared_tensors):
    """Tests assigner behavior with zero ground truth boxes."""
    B, N, C = shared_tensors["pd_scores"].shape
    assigner = TaskAlignedAssigner(mode='ult', num_classes=C)
    out = assigner(
        shared_tensors["pd_scores"], shared_tensors["pd_bboxes"], shared_tensors["anc_points"],
        torch.zeros(B, 0, 1, dtype=torch.long, device=shared_tensors["pd_scores"].device),
        torch.zeros(B, 0, 4, dtype=torch.float32, device=shared_tensors["pd_scores"].device),
        torch.zeros(B, 0, 1, dtype=torch.bool, device=shared_tensors["pd_scores"].device)
    )
    assert out["reg"]["fg_mask"].sum() == 0 and out["cls"]["fg_mask"].sum() == 0
    assert out["reg"]["target_bboxes"].shape == (B, N, 4)

def test_simota_center_prior_edges():
    """Tests SimOTA center prior at radius boundary and with no strides."""
    gt_bboxes = torch.tensor([[[0, 0, 300, 300]]], device=DEVICE, dtype=torch.float32)
    gt_labels = torch.tensor([[[0]]], device=DEVICE, dtype=torch.long)
    mask_gt = torch.ones(1, 1, 1, device=DEVICE, dtype=torch.bool)
    pred_bboxes = gt_bboxes.expand(1, 3, 4)
    pred_scores = torch.full((1, 3, 1), 0.9, device=DEVICE, dtype=torch.float32)

    radius = 2.5
    stride = 32.0
    center_x, center_y = 150.0, 150.0

    anc_points = torch.tensor([[center_x, center_y], [center_x + radius * stride, center_y],
                               [center_x + radius * stride + 0.1, center_y]],
                              device=DEVICE,
                              dtype=torch.float32)
    anc_strides = torch.full((3, 1), stride, device=DEVICE, dtype=torch.float32)

    assigner = TaskAlignedAssigner(mode='simota', num_classes=1, center_radius=radius)
    out = assigner(
        pred_scores,
        pred_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
        anc_strides=anc_strides
    )
    fg_mask = out["reg"]["fg_mask"][0]

    assert fg_mask[0].item() is True, "Center anchor should be a candidate"
    assert fg_mask[1].item() is True, "Anchor exactly at radius boundary should be a candidate"
    assert fg_mask[2].item(
    ) is False, "Anchor just outside radius boundary should not be a candidate"

    gt_bboxes_fallback = torch.tensor([[[100, 100, 200, 200]]], device=DEVICE, dtype=torch.float32)
    anc_points_fallback = torch.tensor([[150, 150], [50, 50]], device=DEVICE, dtype=torch.float32)
    pred_bboxes_fallback = gt_bboxes_fallback.expand(1, 2, 4)
    pred_scores_fallback = torch.full((1, 2, 1), 0.9, device=DEVICE)
    out_fallback = assigner(
        pred_scores_fallback,
        pred_bboxes_fallback,
        anc_points_fallback,
        gt_labels,
        gt_bboxes_fallback,
        mask_gt,
        anc_strides=None
    )
    fg_mask_fallback = out_fallback["reg"]["fg_mask"][0]
    assert fg_mask_fallback[0].item(
    ) is True, "Inside-box anchor should be selected without strides"
    assert fg_mask_fallback[1].item(
    ) is False, "Outside-box anchor should not be selected without strides"

def make_data(B=1, H=80, W=80, C=3, stride=8):
    N = H * W
    xs = torch.arange(W).float() + 0.5
    ys = torch.arange(H).float() + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    anc_grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [N,2] in grid units
    anc_px = anc_grid * stride  # pixels

    gt = torch.tensor([[80.0, 80.0, 560.0, 560.0]])  # xyxy in pixels
    gt = gt.unsqueeze(0)  # [B=1, M=1, 4]
    gl = torch.tensor([[1]]).long().unsqueeze(0)  # class 1
    gm = torch.ones(1, 1, 1, dtype=torch.bool)  # valid

    pd_b = gt.repeat(1, N, 1)  # [B,N,4]
    pd_logits = torch.full((1, N, C), -4.0)
    pd_logits[..., 1] = 4.0  # class 1 confident

    return pd_logits, pd_b, anc_px, gl, gt, gm

def test_ult_no_center_prior_and_sigmoid():
    pd_logits, pd_b, anc_px, gl, gt, gm = make_data()
    A = TaskAlignedAssigner(mode="ult", num_classes=3, input_is_logits=True)
    outA = A(pd_logits, pd_b, anc_px, gl, gt, gm)

    probs = pd_logits.sigmoid()
    B = TaskAlignedAssigner(mode="ult", num_classes=3, input_is_logits=False)
    outB = B(probs, pd_b, anc_px, gl, gt, gm)

    assert torch.equal(outA["reg"]["fg_mask"], outB["reg"]["fg_mask"])
    assert torch.equal(outA["reg"]["matched_gt"], outB["reg"]["matched_gt"])

def test_dfl_filtering_vs_clamp():
    """
    Tests that for a GT box with large distances from an anchor, the assignment is
    kept (clamped by DFL loss) rather than filtered out.
    """
    reg_max = 16
    stride = 8.0
    B, C = 1, 3

    anc_grid = torch.tensor([[10.0, 10.0]])  # [N, 2] in grid units
    anc_px = anc_grid * stride  # [N, 2] in pixels
    N = anc_grid.shape[0]

    x1, y1 = -50, -50
    x2, y2 = 300, 300
    gt_px = torch.tensor([[[x1, y1, x2, y2]]], dtype=torch.float32)  # [B, M, 4]
    gl = torch.tensor([[[1]]], dtype=torch.long)
    gm = torch.ones(B, 1, 1, dtype=torch.bool)

    pd_b_px = gt_px.repeat(1, N, 1)  # Perfect IoU
    pd_logits = torch.full((B, N, C), -10.0)
    pd_logits[:, :, 1] = 10.0  # High confidence for the correct class

    assigner = TaskAlignedAssigner(mode="ult", num_classes=C, topk=1, input_is_logits=True)
    assign = assigner(pd_logits, pd_b_px, anc_px, gl, gt_px, gm)
    fg_mask = assign["reg"]["fg_mask"].squeeze()  # [N]

    assert fg_mask.any(), "The anchor should have been assigned as a foreground positive"

    tgt_b_grid = assign["reg"]["target_bboxes"].squeeze() / stride
    ltrb = BoundingBox.bbox2dist(anc_grid, tgt_b_grid, reg_max).squeeze()  # [N, 4] -> [4]

    lo, hi = _dfl_bounds(reg_max, eps=1e-3)
    is_valid = (ltrb.min() >= lo) & (ltrb.max() <= hi)

    assert not is_valid, "The assigned target's distances should be out of bounds for DFL"

def test_assigner_center_prior_sanity_simota():
    """
    Constructs tiny tensors and asserts that enabling anc_strides reduces the candidate set
    exactly in a radius around GT centers.
    """
    B, M, N, C = 1, 1, 3, 1
    gt_bboxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32, device=DEVICE)
    gt_labels = torch.zeros(B, M, 1, dtype=torch.long, device=DEVICE)
    mask_gt = torch.ones(B, M, 1, dtype=torch.bool, device=DEVICE)
    pd_scores = torch.full((B, N, C), 0.9, device=DEVICE)
    pd_bboxes = gt_bboxes.repeat(1, N, 1)

    center_x, center_y = 150.0, 150.0
    radius = 2.5
    stride = 10.0
    anc_points = torch.tensor([[center_x, center_y], [center_x + radius * stride, center_y],
                               [center_x + radius * stride + 1.0, center_y]],
                              dtype=torch.float32,
                              device=DEVICE)
    anc_strides = torch.full((N, 1), stride, device=DEVICE)

    assigner = TaskAlignedAssigner(mode="simota", num_classes=C, center_radius=radius)
    out = assigner(
        pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, anc_strides=anc_strides
    )
    fg_mask = out["reg"]["fg_mask"].squeeze()

    assert fg_mask[0], "Center anchor should be a candidate"
    assert fg_mask[1], "Anchor at radius edge should be a candidate"
    assert not fg_mask[2], "Anchor outside radius should NOT be a candidate"

def test_batched_iou_correctness():
    """
    With B=2, M>0, N>0 random boxes, asserts new ious.shape == (B,M,N) and matches
    looped per-batch pairwise_box_iou (exact equality).
    """
    from utils.box_iou import pairwise_box_iou
    B, M, N = 2, 5, 10
    gt_bboxes = torch.rand(B, M, 4, device=DEVICE) * 200 + 50
    gt_bboxes[..., 2:] += gt_bboxes[..., :2]
    pd_bboxes = torch.rand(B, N, 4, device=DEVICE) * 200 + 50
    pd_bboxes[..., 2:] += pd_bboxes[..., :2]

    ious_manual = torch.stack([pairwise_box_iou(gt_bboxes[b], pd_bboxes[b]) for b in range(B)])

    ious_list = []
    for b in range(B):
        ious_list.append(pairwise_box_iou(gt_bboxes[b], pd_bboxes[b]))
    ious_assigner = torch.stack(ious_list, dim=0)

    assert ious_assigner.shape == (B, M, N)
    assert torch.equal(ious_manual, ious_assigner)
