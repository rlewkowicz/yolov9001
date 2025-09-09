"""
Tests for model architecture, initialization, and forward pass behavior.
"""
import math
import torch
import pytest
from tests.config import NUM_CLASSES, REG_MAX

def test_model_initialization(model):
    """
    Tests that the model initializes correctly and has the expected attributes.
    """
    assert model is not None, "Model fixture failed to initialize"
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 1e6, "Model should have a reasonable number of parameters"

    assert hasattr(model, 'nc'), "Model must have 'nc' (number of classes) attribute"
    assert model.nc == NUM_CLASSES, f"Model nc ({model.nc}) does not match test config ({NUM_CLASSES})"

    assert hasattr(model, 'reg_max'), "Model must have 'reg_max' attribute"
    assert model.reg_max == REG_MAX, f"Model reg_max ({model.reg_max}) does not match test config ({REG_MAX})"

    assert hasattr(model, 'strides'), "Model must have 'strides' attribute"
    assert len(model.strides) > 0, "Model strides should not be empty"

def test_model_forward_pass_train_mode(model, dummy_input):
    """
    Tests the model's forward pass in training mode.
    The output should be a list of tensors, one for each detection head.
    """
    model.train()
    with torch.no_grad():
        outputs = model(dummy_input)

    assert isinstance(outputs, list), "Model output in train mode should be a list"
    assert len(outputs) == len(model.strides), "Should have one output tensor per stride level"

    expected_channels = REG_MAX * 4 + NUM_CLASSES
    for i, out in enumerate(outputs):
        bs, c, h, w = out.shape
        assert bs == dummy_input.shape[0], "Output batch size should match input"
        assert c == expected_channels, f"Output channels ({c}) mismatch expected ({expected_channels})"
        assert h == dummy_input.shape[2] // model.strides[i]
        assert w == dummy_input.shape[3] // model.strides[i]

def test_model_forward_pass_eval_mode(model, dummy_input):
    """
    Tests the model's forward pass in evaluation mode.
    The output should be a tuple: (concatenated_tensor, list_of_shapes).
    """
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, tuple), "Model output in eval mode should be a tuple"
    assert len(output) == 2, "Output tuple should have two elements"

    preds, feat_shapes = output
    assert isinstance(preds, torch.Tensor), "First element of output should be a tensor"
    assert isinstance(feat_shapes, tuple), "Second element of output should be a tuple"
    assert len(feat_shapes) == len(model.strides), "Shapes list should match number of strides"

    bs, channels, num_anchors = preds.shape
    expected_channels = REG_MAX * 4 + NUM_CLASSES

    assert bs == dummy_input.shape[0]
    assert channels == expected_channels

    total_anchors = 0
    for i, stride in enumerate(model.strides):
        h, w = dummy_input.shape[2] // stride, dummy_input.shape[3] // stride
        total_anchors += h * w
        assert feat_shapes[i] == (h, w), f"Shape mismatch at stride {stride}"

    assert num_anchors == total_anchors, "Concatenated output has wrong number of anchors"

def test_detect_bias_initialization_strict(model):
    if not hasattr(model, 'detect_layer') or model.detect_layer is None:
        pytest.skip("Model doesn't have a detect layer")
    detect = model.detect_layer
    prior_prob = 0.01
    first_bin, other_bins = 2.0, -2.0

    cv2_bias_ids_before = [id(conv[-1].bias) for conv in detect.cv2]
    cv3_bias_ids_before = [id(conv[-1].bias) for conv in detect.cv3]

    detect.initialize_biases(prior_prob=prior_prob, first_bin=first_bin, other_bins=other_bins)

    assert cv2_bias_ids_before == [id(conv[-1].bias) for conv in detect.cv2]
    assert cv3_bias_ids_before == [id(conv[-1].bias) for conv in detect.cv3]

    cls_bias_expected = -math.log((1 - prior_prob) / prior_prob)
    for conv in detect.cv3:
        b = conv[-1].bias.detach()
        assert torch.allclose(b, torch.full_like(b, cls_bias_expected), rtol=0, atol=0)

    for conv in detect.cv2:
        b = conv[-1].bias.detach().cpu()
        assert b.numel() == 4 * detect.reg_max
        per_side = b.view(4, detect.reg_max)[0]
        assert per_side[0].item() == pytest.approx(first_bin)
        if detect.reg_max > 1:
            assert torch.allclose(per_side[1:], torch.full_like(per_side[1:], other_bins))

def test_stride_calibration(model):
    """
    Test that stride calibration correctly updates the model and Detect layer.
    """
    assert hasattr(model, 'strides'), "Model should have strides attribute"
    assert len(model.strides) > 0, "Model strides should not be empty"

    for stride in model.strides:
        assert stride in [4, 8, 16, 32, 64], f"Unexpected stride value: {stride}"

    if hasattr(model, 'detect_layer') and model.detect_layer is not None:
        assert hasattr(model.detect_layer, 'strides'), "Detect layer should have strides"
        model_strides = model.strides.tolist() if isinstance(model.strides,
                                                             torch.Tensor) else list(model.strides)
        detect_strides = model.detect_layer.strides.tolist() if isinstance(
            model.detect_layer.strides, torch.Tensor
        ) else list(model.detect_layer.strides)
        assert model_strides == detect_strides, \
            "Detect layer strides should match model strides"

    if hasattr(model, 'calibrate_strides'):
        original_strides = model.strides.clone() if isinstance(model.strides,
                                                               torch.Tensor) else list(
                                                                   model.strides
                                                               )

        model.calibrate_strides(img_size=512)

        assert hasattr(model, 'strides'), "Model should still have strides after recalibration"
        current_strides = model.strides.tolist() if isinstance(model.strides,
                                                               torch.Tensor) else list(
                                                                   model.strides
                                                               )
        assert all(isinstance(s, (int, float)) for s in current_strides), \
            "All strides should be numeric"

        original_count = len(original_strides) if isinstance(original_strides,
                                                             list) else original_strides.shape[0]
        assert len(current_strides) == original_count, \
            "Number of strides should remain the same"

def test_detect_training_vs_eval_shapes(model):
    yolo_model = model
    yolo_model.train()
    x = torch.randn(2, 3, 320, 320, device=next(yolo_model.parameters()).device)
    outs = yolo_model(x)  # list per level during training
    assert isinstance(outs, list) and all(o.ndim == 4 for o in outs)
    yolo_model.eval()
    with torch.no_grad():
        y, shapes = yolo_model(x)
    assert y.ndim == 3 and isinstance(shapes, (list, tuple)) and len(shapes) == len(outs)
