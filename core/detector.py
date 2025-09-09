import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Optional, Dict, TYPE_CHECKING
import cv2

if TYPE_CHECKING:
    from PIL.Image import Image

from .base import BaseRunner
from utils.boxes import scale_boxes_from_canvas_to_original, clip_boxes_
from utils.augment import ExactLetterboxTransform

def _annotate_image(
    image: np.ndarray, detections: Dict, class_names: Optional[List[str]] = None
) -> np.ndarray:
    """Helper function to draw boxes and labels on an image."""
    annotated_image = image.copy()
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    class_ids = detections['class_ids'].cpu().numpy().astype(int)

    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

        if class_names:
            label = f"{class_names[class_id]} {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            cv2.putText(
                annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

    return annotated_image

class Detector(BaseRunner):
    """Single image inference for YOLO models."""
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.model.eval()

        state = self.model.get_detection_state()
        self.postprocessor = self.model.get_postprocessor(device=self.device)
        strides_val = state.get('strides', None)
        try:
            if isinstance(strides_val, torch.Tensor):
                strides_list = [int(x) for x in strides_val.flatten().tolist()]
            elif strides_val is None:
                strides_list = []
            else:
                strides_list = [int(x) for x in list(strides_val)]
        except Exception:
            strides_list = []
        self.max_stride = int(max(strides_list)) if len(strides_list) > 0 else 32
        self.letterbox_center = state['letterbox_center']
        self.pad_value = int(state['pad_value'])  # new

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        conf_thresh: float = None,
        iou_thresh: float = None,
        img_size: int = 640
    ) -> Dict:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (path, numpy array, PIL Image)
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
            img_size: Image size for inference
        
        Returns:
            Dictionary with detection tensors {'boxes':, 'scores':, 'class_ids':}
        """
        img_tensor, _, pad, (orig_h, orig_w) = self.preprocess_image(image, img_size)

        outputs, feat_shapes = self.model(img_tensor)

        if conf_thresh is not None:
            self.postprocessor.conf_thresh = conf_thresh
        if iou_thresh is not None:
            self.postprocessor.iou_thresh = iou_thresh

        detections_batch = self.postprocessor(outputs, img_size, feat_shapes=feat_shapes)

        detections = detections_batch[0]

        descaled_boxes = scale_boxes_from_canvas_to_original(
            detections['boxes'], (img_size, img_size), (orig_h, orig_w), (pad[0], pad[1]),
            (pad[2], pad[3])
        )

        x1y1x2y2 = descaled_boxes
        clip_boxes_(x1y1x2y2, (orig_h, orig_w), pixel_edges=True)
        detections['boxes'] = x1y1x2y2

        return detections

    def detect_and_annotate(
        self,
        image: Union[str, Path, np.ndarray, "Image.Image"],
        class_names: Optional[List[str]] = None,
        conf_thresh: float = None,
        iou_thresh: float = None,
        img_size: int = 640
    ) -> np.ndarray:
        """
        Run detection on a single image and return an annotated image.
        
        Args:
            image: Input image (path, numpy array, PIL Image)
            class_names: List of class names for annotation
            conf_thresh: Confidence threshold for detections
            iou_thresh: IoU threshold for NMS
            img_size: Image size for inference
            
        Returns:
            Annotated image as a numpy array.
        """
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        names = class_names or getattr(self.model, 'names', None)
        detections = self.detect(image.copy(), conf_thresh, iou_thresh, img_size)
        annotated_image = _annotate_image(image, detections, names)

        return annotated_image

    def preprocess_image(self, image, img_size):
        """Preprocess image for inference."""
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        transform = ExactLetterboxTransform(
            img_size=img_size,
            center=self.letterbox_center,
            pad_value=self.pad_value,
            pad_to_stride=self.max_stride,
            scaleup=False  # align with val letterboxing for consistent padding in debug
        )
        image, _, (orig_h, orig_w), scale_ratio, pad = transform(image)

        image = torch.from_numpy(image).float().to(self.device)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0) / 255.0

        return image, scale_ratio, pad, (orig_h, orig_w)
