"""
CHASE_DB1 Dataset Loader
========================
Provides same API as dataloader_stare.py.
Expects CHASE_DB1 data directory structure:

  chase-db1-DatasetNinja/
    ds0/
      img/        (Image_*.jpg)
      ann/        (Image_*.jpg.json)

CHASE_DB1 has 28 images total. We use a standard train/test split:
  20 train (01L/R to 10L/R), 8 test (11L/R to 14L/R)
"""

import os
import json
import base64
import zlib
import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

class CHASEDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        img_size: int = 512,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment and (split == "train")
        
        img_dir = os.path.join(root_dir, "ds0", "img")
        ann_dir = os.path.join(root_dir, "ds0", "ann")
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"CHASE_DB1 images directory not found: {img_dir}")
            
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        
        # Split: images starting with Image_11, Image_12, Image_13, Image_14 are test set (8 images)
        # Images starting with Image_01 to Image_10 are train set (20 images)
        test_f = [f for f in img_files if any(f.startswith(f"Image_{i}") for i in ["11", "12", "13", "14"])]
        train_f = [f for f in img_files if f not in test_f]
        
        chosen = test_f if split == "test" else train_f
        
        self.samples = []
        for fname in chosen:
            img_path = os.path.join(img_dir, fname)
            ann_path = os.path.join(ann_dir, fname + ".json")
            if os.path.exists(ann_path):
                self.samples.append((img_path, ann_path))
            else:
                print(f"  [WARN] No annotation found for {fname} in {ann_dir}")
                
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _decode_mask(self, ann_path: str) -> Image.Image:
        with open(ann_path) as f:
            ann = json.load(f)
        w, h = ann['size']['width'], ann['size']['height']
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for obj in ann['objects']:
            if obj['classTitle'] == 'vessel':
                bitmap = obj['bitmap']
                origin = bitmap['origin'] # [x, y]
                data = bitmap['data']
                
                png_bytes = zlib.decompress(base64.b64decode(data))
                bitmap_img = Image.open(io.BytesIO(png_bytes))
                bitmap_np = np.array(bitmap_img)
                if len(bitmap_np.shape) == 3:
                    bitmap_np = bitmap_np[:, :, 0]
                    
                y, x = origin[1], origin[0]
                bh, bw = bitmap_np.shape
                
                # Clip coordinates to bounds just in case
                y_start = max(0, y)
                x_start = max(0, x)
                y_end = min(h, y + bh)
                x_end = min(w, x + bw)
                
                by_start = y_start - y
                bx_start = x_start - x
                by_end = by_start + (y_end - y_start)
                bx_end = bx_start + (x_end - x_start)
                
                if (y_end > y_start) and (x_end > x_start):
                    mask[y_start:y_end, x_start:x_end] = np.maximum(
                        mask[y_start:y_end, x_start:x_end],
                        (bitmap_np[by_start:by_end, bx_start:bx_end] > 0).astype(np.uint8) * 255
                    )
        return Image.fromarray(mask)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        gt = self._decode_mask(ann_path)
        
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        gt = gt.resize((self.img_size, self.img_size), Image.NEAREST)
        
        if self.augment:
            import random
            if random.random() > 0.5:
                img = T.functional.hflip(img)
                gt = T.functional.hflip(gt)
            if random.random() > 0.5:
                img = T.functional.vflip(img)
                gt = T.functional.vflip(gt)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                img = T.functional.rotate(img, angle)
                gt = T.functional.rotate(gt, angle)
                
        img_t = self.to_tensor(img)
        gt_t = (self.to_tensor(gt) > 0.5).float()
        
        return {"image": img_t, "mask": gt_t, "path": img_path}

def create_chase_loader(
    root_dir: str,
    batch_size: int = 4,
    img_size: int = 512,
    num_workers: int = 2,
):
    dataset = CHASEDataset(
        root_dir=root_dir,
        split="test",
        img_size=img_size,
        augment=False,
    )
    print(f"  [CHASE_DB1] Loaded {len(dataset)} test images from {root_dir}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
