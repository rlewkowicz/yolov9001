import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader, Dataset

from core.validator import Validator
from utils.dataloaders import create_dataloader, collate_fn
from utils.boxes import yolo_to_xyxy
from tests.test_data import _make_tiny_yolo_dataset
from utils.augment import ExactLetterboxTransform
from utils.geometry import DFLDecoder

class MockPostprocessor:
    """A mock postprocessor to inject specific predictions for testing."""
    def __init__(self, predictions, nc):
        self.predictions = predictions
        self.nc = nc
        self.model = None

    def __call__(self, outputs, *args, **kwargs):
        return self.predictions

class MockValidatorDataset(Dataset):
    """A mock dataset to wrap a single batch of data for validator testing."""
    def __init__(self, batch):
        self.images, self.labels, self.orig_shapes, self.ratios, self.pads = batch
        self.labels_per_image = []
        for i in range(self.images.shape[0]):
            self.labels_per_image.append(self.labels[self.labels[:, 0] == i, 1:])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels_per_image[idx], self.orig_shapes[idx], self.ratios[
            idx], self.pads[idx]

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Creates a tiny YOLO dataset valid for the entire test module."""
    tmp_path = tmp_path_factory.mktemp("dataset")
    _make_tiny_yolo_dataset(tmp_path)
    imgsz = 256
    loader, info = create_dataloader(
        tmp_path, "train", imgsz, batch_size=2, num_workers=0, augment=False
    )
    return loader, info, imgsz

class TestValidatorMetricsE2E:
    @pytest.fixture
    def setup(self, mock_model_factory):
        """Provides a validator and a deterministic data batch for E2E tests."""
        imgsz, nc = 256, 3
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = mock_model_factory(nc=nc, device=device)
        validator = Validator(model, device=device)
        info = {'nc': nc}

        images = torch.randn(2, 3, imgsz, imgsz)
        labels = torch.tensor([[0, 1, 0.5, 0.5, 0.5, 0.5], [1, 0, 0.25, 0.25, 0.1, 0.1]])
        orig_shapes = [(imgsz, imgsz), (imgsz, imgsz)]
        ratios = [1.0, 1.0]
        pads = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
        batch = (images, labels, orig_shapes, ratios, pads)

        return validator, batch, info, imgsz, device

    def _get_gt_on_canvas(self, batch, imgsz, image_index=0):
        """Extracts and transforms a GT box to canvas coordinates."""
        _, targets, orig_shapes, ratios, pads = batch
        target_for_image = targets[targets[:, 0] == image_index]
        if not target_for_image.numel():
            pytest.fail(f"Image {image_index} has no GTs.")
        orig_shape, ratio, pad = orig_shapes[image_index], ratios[image_index], pads[image_index]
        gt_cls, gt_yolo = target_for_image[:, 1], target_for_image[:, 2:]
        gt_on_canvas = yolo_to_xyxy(gt_yolo, orig_shape)
        gt_on_canvas *= ratio
        gt_on_canvas[:, [0, 2]] += pad[0]
        gt_on_canvas[:, [1, 3]] += pad[1]
        gt_on_canvas.clamp_(0, imgsz)
        return gt_on_canvas, gt_cls

    def _run_validate_on_batch(self, validator, batch, predictions, nc):
        """Wraps a single batch in a proper DataLoader and runs validation."""
        validator.postprocessor = MockPostprocessor(predictions, nc)
        dataset = MockValidatorDataset(batch)
        dataset.nc = nc
        loader = DataLoader(dataset, batch_size=batch[0].shape[0], collate_fn=collate_fn)
        return validator.validate(loader)

    def test_perfect_match(self, setup):
        validator, batch, info, imgsz, device = setup
        gt_b0, gt_c0 = self._get_gt_on_canvas(batch, imgsz, image_index=0)
        gt_b1, gt_c1 = self._get_gt_on_canvas(batch, imgsz, image_index=1)

        perfect_preds = [{
            'boxes': gt_b0.to(device), 'scores': torch.tensor([0.99], device=device), 'class_ids':
                gt_c0.to(device, dtype=torch.long)
        }, {
            'boxes': gt_b1.to(device), 'scores': torch.tensor([0.99], device=device), 'class_ids':
                gt_c1.to(device, dtype=torch.long)
        }]
        metrics = self._run_validate_on_batch(validator, batch, perfect_preds, info['nc'])
        assert 'mAP50-95' in metrics and np.isclose(metrics['mAP50-95'], 1.0)

    def test_imperfect_match_and_miss(self, setup):
        validator, batch, info, imgsz, device = setup
        gt_b0, gt_c0 = self._get_gt_on_canvas(batch, imgsz, image_index=0)
        imperfect_boxes = gt_b0.clone()
        imperfect_boxes[:, :2] += 2

        imperfect_preds = [
            {
                'boxes': imperfect_boxes.to(device), 'scores': torch.tensor([0.99], device=device),
                'class_ids': gt_c0.to(device, dtype=torch.long)
            },
            {
                'boxes': torch.empty(0, 4, device=device), 'scores': torch.empty(0, device=device),
                'class_ids': torch.empty(0, dtype=torch.long, device=device)
            }  # Missed detection
        ]
        metrics = self._run_validate_on_batch(validator, batch, imperfect_preds, info['nc'])
        assert np.isclose(metrics['mAP50'], 0.5)  # Perfect IoU for one, zero for the other
        assert metrics['mAP50-95'] <= 0.5  # Imperfect IoU lowers the average

    def test_class_mismatch_and_fp(self, setup):
        validator, batch, info, imgsz, device = setup
        gt_b0, gt_c0 = self._get_gt_on_canvas(batch, imgsz, image_index=0)
        wrong_class = (gt_c0.int() + 1) % info['nc']

        mixed_preds = [
            {
                'boxes': gt_b0.to(device), 'scores': torch.tensor([0.99], device=device),
                'class_ids': wrong_class.to(device, dtype=torch.long)
            },  # Class mismatch
            {
                'boxes': torch.tensor([[0, 0, 10, 10]], device=device), 'scores':
                    torch.tensor([0.99], device=device), 'class_ids':
                        torch.tensor([0], device=device, dtype=torch.long)
            }  # FP
        ]
        metrics = self._run_validate_on_batch(validator, batch, mixed_preds, info['nc'])
        assert np.isclose(metrics['mAP50-95'], 0.0)

def test_validator_unpacked_and_types(tiny_dataset, mock_model_factory):
    loader, info, _ = tiny_dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mock_model_factory(nc=info["nc"], reg_max=16, strides=[8, 16, 32], device=device)
    validator = Validator(model, device=device)
    validator.validate(loader)
    stats = validator.det_metrics.stats
    if not stats['tp']:
        return
    assert all(isinstance(t, torch.Tensor) for t in stats['tp'])
    assert all(t.dim() == 2 for t in stats['tp'])
    assert all(c.dim() == 1 for c in stats['conf'])

def test_validator_class_space_mismatch_raises(mock_model_factory):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mock_model_factory(nc=80, device=device)
    dataset = type(
        'ds', (Dataset, ), {
            '__len__':
                lambda s: 1, 'nc':
                    5, '__getitem__':
                        lambda s, i:
                        (torch.rand(3, 32, 32), torch.rand(1, 5), (32, 32), 1.0, (0.0, 0.0))
        }
    )()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    validator = Validator(model, device=device)
    with pytest.raises(ValueError, match="Class count mismatch"):
        validator.validate(dataloader)

def test_validator_syncs_with_ema_model(mock_model_factory):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model = mock_model_factory(nc=80, reg_max=16, device=device)
    ema_model = mock_model_factory(nc=10, reg_max=8, device=device)

    class MockSyncDataset(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), torch.zeros(1, 5), (32, 32), 1.0, (0.0, 0.0, 0.0, 0.0)

    dataset = MockSyncDataset()
    dataset.nc = 10
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    validator = Validator(base_model, device=device)
    validator.validate(dataloader, model=ema_model)

    assert validator.postprocessor.nc == ema_model.nc
    assert validator.postprocessor.decoder.reg_max == ema_model.reg_max

def test_validator_with_padded_batches(mock_model_factory):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mock_model_factory(nc=10, device=device)

    class DummyPaddedDataset(Dataset):
        def __init__(self, n_img=4, h=32, w=32, nc=10, max_bbox=5):
            self.n_img, self.h, self.w, self.nc, self.max_bbox = n_img, h, w, nc, max_bbox
            self.n_bboxes = torch.randint(1, max_bbox + 1, (n_img, ))

        def __len__(self):
            return self.n_img

        def __getitem__(self, i):
            img = torch.rand(3, self.h, self.w)
            n_box = self.n_bboxes[i].item()
            boxes = torch.rand(n_box, 4)
            cls = torch.randint(0, self.nc, (n_box, 1), dtype=torch.float32)
            return img, torch.cat([cls, boxes], 1), (self.h, self.w), 1.0, (0.0, 0.0, 0.0, 0.0)

    dataloader = DataLoader(DummyPaddedDataset(), batch_size=2, collate_fn=collate_fn)
    validator = Validator(model, device=device, cfg={'img_size': 32})

    try:
        validator.validate(dataloader)
    except Exception as e:
        pytest.fail(f"Validator failed with padded batches: {e}")

class GoldenDataset(Dataset):
    """A simple dataset that returns one synthetic image and its GT label."""
    def __init__(self, img, labels, imgsz, r, pad, H0, W0):
        self.img = img
        self.labels = labels
        self.imgsz = imgsz
        self.orig_shape = (H0, W0)
        self.r = r
        self.pad = pad

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_t = torch.from_numpy(self.img).permute(2, 0, 1).float() / 255.0
        labels_t = torch.from_numpy(self.labels).float()
        return img_t, labels_t, self.orig_shape, self.r, self.pad

def collate_fn_golden(batch):
    imgs, labels, orig_shapes, ratios, pads = zip(*batch)
    targets = torch.cat(
        [
            torch.full((labels[0].shape[0], 1), 0),  # batch index 0
            labels[0]
        ],
        dim=1
    )
    return torch.stack(imgs, 0), targets, orig_shapes, ratios, pads

@pytest.mark.cuda
def test_e2e_validator_golden_map(mock_model_factory):
    """
    Tests the entire validation pipeline end-to-end with a perfect prediction.
    This test uses the real Validator class and an overridden model.forward() to
    ensure the test environment is as close to production as possible.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imgsz, nc = 640, 3

    H0, W0 = 480, 800
    img = (np.random.rand(H0, W0, 3) * 255).astype(np.uint8)
    gt_xyxy_abs = np.array([[100., 150., 300., 400.]], dtype=np.float32)
    gt_cls = np.array([1], dtype=np.int64)

    transform = ExactLetterboxTransform(img_size=imgsz)
    canvas, _, (oh, ow), r, pad = transform(img)
    padw, padh = pad[0], pad[1]
    gt_yolo_canvas = np.array([
        gt_cls[0],
        ((gt_xyxy_abs[0, 0] + gt_xyxy_abs[0, 2]) / 2 * r + padw) / imgsz,
        ((gt_xyxy_abs[0, 1] + gt_xyxy_abs[0, 3]) / 2 * r + padh) / imgsz,
        ((gt_xyxy_abs[0, 2] - gt_xyxy_abs[0, 0]) * r) / imgsz,
        ((gt_xyxy_abs[0, 3] - gt_xyxy_abs[0, 1]) * r) / imgsz,
    ]).reshape(1, 5)

    model = mock_model_factory(nc=nc, device=device)
    model.eval()

    def set_class_names(names):
        model.names = names

    model.set_class_names = set_class_names

    validator = Validator(model, device=device)
    decoder: DFLDecoder = validator.postprocessor.decoder

    dataset = GoldenDataset(canvas, gt_yolo_canvas, imgsz, r, pad, H0, W0)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_golden)
    model.set_class_names([f"class_{int(c)}" for c in dataloader.dataset.labels[:, 0].tolist()])

    def golden_forward(x):
        feat_shapes = [(imgsz // int(s), imgsz // int(s)) for s in decoder.strides.tolist()]
        anchors, stride_tensor = decoder.get_anchors(feat_shapes)

        gt_xyxy_canvas_t = yolo_to_xyxy(torch.from_numpy(gt_yolo_canvas[:, 1:]),
                                        (imgsz, imgsz)).to(device)

        with torch.no_grad():
            ap = anchors.to(device) * stride_tensor.to(device)
            gt_center = (gt_xyxy_canvas_t[0, :2] + gt_xyxy_canvas_t[0, 2:]) / 2
            idx = torch.argmin(((ap - gt_center)**2).sum(1))

            reg_max = int(decoder.reg_max)
            stride = stride_tensor[idx]
            d_ltrb = torch.cat(
                (gt_center - gt_xyxy_canvas_t[0, :2], gt_xyxy_canvas_t[0, 2:] - gt_center)
            ).squeeze()
            t = (d_ltrb / stride).clamp(0, reg_max - 1 - 1e-3)

            def soft_onehot(v):
                lo = torch.clamp(v.floor(), 0, reg_max - 1).long()
                hi = torch.clamp(lo + 1, 0, reg_max - 1).long()
                w_hi = (v - lo).clamp(0, 1)
                w_lo = 1.0 - w_hi
                oh = torch.full((reg_max, ), -100.0, device=device)
                oh[lo] = torch.log(w_lo + 1e-9) + 100.0
                if hi != lo:
                    oh[hi] = torch.log(w_hi + 1e-9) + 100.0
                return oh

            sides = [soft_onehot(t[i]) for i in range(4)]
            golden_reg = torch.cat(sides)
            golden_cls = torch.full((nc, ), -100.0, device=device)
            golden_cls[gt_cls[0]] = 100.0
            golden_pred = torch.cat([golden_reg, golden_cls])

            outputs = []
            anchor_offset = 0
            for i, shape in enumerate(feat_shapes):
                H, W = shape
                level_output = torch.full((1, nc + 4 * reg_max, H, W), -100.0, device=device)
                count = H * W
                if idx < anchor_offset + count:
                    idx_in_level = idx - anchor_offset
                    y, x = idx_in_level // W, idx_in_level % W
                    level_output[0, :, y, x] = golden_pred
                    outputs.append(level_output)
                    anchor_offset += count
                    break  # Found the correct level, stop searching
                outputs.append(level_output)
                anchor_offset += count

            while len(outputs) < len(feat_shapes):
                shape = feat_shapes[len(outputs)]
                H, W = shape
                level_output = torch.full((1, nc + 4 * reg_max, H, W), -100.0, device=device)
                outputs.append(level_output)

            return outputs

    model.forward = golden_forward

    validator.validate(dataloader)

    metrics = validator.metrics
    map50_95 = metrics.get('mAP50-95', 0.0)
    assert map50_95 == 1.0, f"E2E golden path test failed! Expected mAP=1.0, got {map50_95}"
