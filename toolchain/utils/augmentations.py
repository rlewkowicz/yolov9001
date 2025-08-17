# augmentations.py

import math
import random

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Kornia is a powerful GPU-accelerated computer vision library
# Install with: pip install kornia
try:
    import kornia
    from kornia.enhance import equalize_clahe
    from kornia.filters import box_blur, median_blur
except ImportError:
    raise ImportError("Kornia is not installed. Please run 'pip install kornia'")

from utils.general import xywh2xyxy, xyxy2xywh


class BatchAugmenter:
    """
    Applies a comprehensive suite of YOLOv5 augmentations to a batch of images and labels on the GPU.
    This class reads hyperparameters from a 'hyp' dictionary and uses Kornia and custom PyTorch
    functions to perform all operations, ensuring maximum performance.
    """
    def __init__(self, hyp, imgsz):
        self.hyp = hyp
        self.imgsz = imgsz

        # Initialize Kornia transforms based on hyperparameters
        # These replace the old Albumentations pixel-level transforms
        self.kornia_transforms = T.RandomApply([
            T.RandomChoice([
                lambda x: box_blur(x, (3, 3)),
                lambda x: median_blur(x, (3, 3))
            ])
        ], p=hyp.get('blur_p', 0.01))

        self.color_jitter = T.ColorJitter(
            brightness=hyp['hsv_v'],
            contrast=hyp.get('contrast', 0.0), # Add contrast if in hyp
            saturation=hyp['hsv_s'],
            hue=hyp['hsv_h']
        )

    def __call__(self, imgs, labels, segments):
        """
        Applies the full augmentation pipeline.
        
        Args:
            imgs (torch.Tensor): Batch of images (B, C, H, W) on GPU.
            labels (torch.Tensor): Batch of labels (N, 6) [batch_idx, cls, x, y, w, h].
            segments (list[torch.Tensor]): List of segment polygons for each image.
        
        Returns:
            (torch.Tensor, torch.Tensor): Augmented images and labels.
        """
        # --- Geometric Augmentations ---
        # These must happen first as they change coordinates
        imgs, labels, segments = self.random_perspective(imgs, labels, segments)

        # --- Instance-level Augmentations ---
        # These add or modify objects in the scene
        if self.hyp.get('copy_paste', 0.0) > 0.0 and len(segments) > 0:
            imgs, labels = self.copy_paste(imgs, labels, segments, p=self.hyp['copy_paste'])
        
        if self.hyp.get('replicate_p', 0.0) > 0.0:
            imgs, labels = self.replicate(imgs, labels, p=self.hyp['replicate_p'])

        # --- Pixel-level Augmentations ---
        # These modify the image content without changing geometry
        if random.random() < self.hyp['mixup']:
            imgs, labels = self.mixup(imgs, labels)
        
        imgs = self.color_jitter(imgs) # Replaces augment_hsv

        if self.hyp.get('clahe_p', 0.01) > 0.0 and random.random() < self.hyp['clahe_p']:
            imgs = (equalize_clahe(imgs) * 255).to(imgs.dtype) / 255.0

        if self.hyp.get('equalize_p', 0.0) > 0.0 and random.random() < self.hyp['equalize_p']:
            imgs = kornia.enhance.equalize(imgs)
            
        imgs = self.kornia_transforms(imgs) # Apply blurs etc.

        # Flips should be last of the geometric transforms
        imgs, labels = self.random_flips(imgs, labels)

        return imgs, labels

    def random_perspective(self, imgs, labels, segments):
        hyp = self.hyp
        height, width = imgs.shape[2:]
        
        # Create a single random transformation matrix for the whole batch
        M = torch.eye(3, device=imgs.device)
        if hyp['perspective'] > 0:
            P = torch.eye(3, device=imgs.device)
            P[2, 0] = random.uniform(-hyp['perspective'], hyp['perspective'])
            P[2, 1] = random.uniform(-hyp['perspective'], hyp['perspective'])
            M = P @ M
        
        R = torch.eye(3, device=imgs.device)
        a = random.uniform(-hyp['degrees'], hyp['degrees'])
        s = random.uniform(1 - hyp['scale'], 1 + hyp['scale'])
        R[:2] = torch.tensor(cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s), device=imgs.device)
        M = R @ M

        S = torch.eye(3, device=imgs.device)
        S[0, 1] = math.tan(random.uniform(-hyp['shear'], hyp['shear']) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-hyp['shear'], hyp['shear']) * math.pi / 180)
        M = S @ M

        T = torch.eye(3, device=imgs.device)
        T[0, 2] = random.uniform(0.5 - hyp['translate'], 0.5 + hyp['translate']) * width
        T[1, 2] = random.uniform(0.5 - hyp['translate'], 0.5 + hyp['translate']) * height
        M = T @ M

        # Apply transformation
        if (M != torch.eye(3, device=imgs.device)).any():
            M_affine = M[:2]
            grid = F.affine_grid(M_affine.unsqueeze(0).expand(imgs.shape[0], -1, -1), imgs.shape, align_corners=False)
            # --- CORRECTED LINE ---
            imgs = F.grid_sample(imgs, grid, padding_mode="zeros", align_corners=False)

        # Transform labels and segments
        if len(labels) > 0:
            n = len(labels)
            xy = torch.ones((n * 4, 3), device=imgs.device)
            
            boxes = xywh2xyxy(labels[:, 2:6])
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 4, 2)
            
            min_xy, _ = xy.min(1); max_xy, _ = xy.max(1)
            new_boxes = torch.cat((min_xy, max_xy), 1)
            
            # Filter boxes
            w = new_boxes[:, 2] - new_boxes[:, 0]
            h = new_boxes[:, 3] - new_boxes[:, 1]
            i = (w > 2) & (h > 2)
            labels = labels[i]
            new_boxes = new_boxes[i]
            
            # Convert back to xywhn
            new_xywh = xyxy2xywh(new_boxes)
            new_xywh[:, [0, 2]] /= width
            new_xywh[:, [1, 3]] /= height
            labels[:, 2:6] = new_xywh

        return imgs, labels, segments # Segments are not transformed in this simple version

    def copy_paste(self, imgs, labels, segments, p=0.5):
        # This is a simplified batch implementation. More complex logic could be added.
        if random.random() > p or len(labels) == 0:
            return imgs, labels

        batch_size = imgs.shape[0]
        instance_labels = []
        instance_imgs = []
        instance_masks = []

        # Collect all instances from the batch
        for i in range(batch_size):
            img_labels = labels[labels[:, 0] == i]
            if len(img_labels) > 0:
                boxes = xywh2xyxy(img_labels[:, 2:6])
                boxes[:, [0, 2]] *= self.imgsz
                boxes[:, [1, 3]] *= self.imgsz
                
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.long()
                    instance_imgs.append(imgs[i, :, y1:y2, x1:x2])
                    
                    mask = torch.zeros_like(imgs[i, 0], device=imgs.device)
                    mask[y1:y2, x1:x2] = 1.0 # Simple box mask
                    instance_masks.append(mask[y1:y2, x1:x2])
                    
                    instance_labels.append(img_labels[j])

        if not instance_imgs:
            return imgs, labels

        # Paste random instances onto random images
        for i in range(batch_size):
            if random.random() < 0.5: # 50% chance to paste on an image
                k = random.randint(0, len(instance_imgs) - 1)
                src_img = instance_imgs[k]
                src_mask = instance_masks[k]
                src_label = instance_labels[k]
                
                h, w = src_img.shape[1:]
                ph, pw = imgs.shape[2:]
                
                if h >= ph or w >= pw: continue

                x_offset = random.randint(0, pw - w)
                y_offset = random.randint(0, ph - h)
                
                # Create pasted mask
                pasted_mask = src_mask.unsqueeze(0)
                
                # Composite
                imgs[i, :, y_offset:y_offset+h, x_offset:x_offset+w] = \
                    torch.where(pasted_mask > 0, src_img, imgs[i, :, y_offset:y_offset+h, x_offset:x_offset+w])
                
                # Add new label
                new_label = src_label.clone().unsqueeze(0)
                new_label[0, 0] = i
                new_label[0, 2] = (x_offset + w / 2) / pw
                new_label[0, 3] = (y_offset + h / 2) / ph
                new_label[0, 4] = w / pw
                new_label[0, 5] = h / ph
                labels = torch.cat([labels, new_label], dim=0)
                
        return imgs, labels

    def replicate(self, imgs, labels, p=0.1):
        if random.random() > p:
            return imgs, labels
            
        for i in range(imgs.shape[0]):
            img_labels = labels[labels[:, 0] == i]
            if len(img_labels) < 2: continue
            
            # Select a random small object to replicate
            boxes = xywh2xyxy(img_labels[:, 2:6])
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            if areas.numel() == 0: continue
            
            j = torch.argmin(areas)
            src_label = img_labels[j]
            box = boxes[j] * self.imgsz
            x1, y1, x2, y2 = box.long()
            
            patch = imgs[i, :, y1:y2, x1:x2]
            h, w = patch.shape[1:]
            
            # Paste it somewhere else
            px, py = random.randint(0, self.imgsz - w), random.randint(0, self.imgsz - h)
            imgs[i, :, py:py+h, px:px+w] = patch
            
            # Add new label
            new_label = src_label.clone().unsqueeze(0)
            new_label[0, 2] = (px + w / 2) / self.imgsz
            new_label[0, 3] = (py + h / 2) / self.imgsz
            labels = torch.cat([labels, new_label], dim=0)
            
        return imgs, labels

    def mixup(self, imgs, labels):
        r = np.random.beta(32.0, 32.0)
        i = torch.randperm(imgs.shape[0]).to(imgs.device)
        imgs = r * imgs + (1 - r) * imgs[i]
        
        labels_mix = labels.clone()
        labels_mix[:, 0] = i[labels[:, 0].long()]
        labels = torch.cat((labels, labels_mix), 0)
        return imgs, labels

    def random_flips(self, imgs, labels):
        if random.random() < self.hyp['fliplr']:
            imgs = torch.flip(imgs, [3])
            if len(labels): labels[:, 2] = 1.0 - labels[:, 2]
        if random.random() < self.hyp['flipud']:
            imgs = torch.flip(imgs, [2])
            if len(labels): labels[:, 3] = 1.0 - labels[:, 3]
        return imgs, labels
