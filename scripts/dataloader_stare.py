"""
STARE Dataset Loader
====================
Provides same API as dataloader_unetpp.py.
Expects STARE data directory structure:

  data/STARE/
    images/        (*.ppm or *.png fundus images)
    labels-ah/     (ground truth by Adam Hoover, *.ppm or *.png)
    or
    labels-vk/     (ground truth by Valentina Kouznetsova)

STARE has 20 images total. We use an 80/20 split:
  16 train, 4 test.

Usage:
    from scripts.dataloader_stare import create_stare_loader
    loader = create_stare_loader("data/STARE", batch_size=4)
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch


class STAREDataset(Dataset):
    """
    STARE retinal vessel dataset loader.

    Parameters
    ----------
    root_dir    : path to STARE root (containing images/ and labels-ah/)
    split       : 'train' or 'test'
    img_size    : resize all images to this size
    augment     : apply random flip + rotation for train split
    gt_folder   : 'labels-ah' (default) or 'labels-vk'
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        img_size: int = 512,
        augment: bool = False,
        gt_folder: str = "labels-ah",
    ):
        self.img_size  = img_size
        self.augment   = augment and (split == "train")

        img_dir = os.path.join(root_dir, "images")
        gt_dir  = os.path.join(root_dir, gt_folder)

        if not os.path.exists(img_dir):
            raise FileNotFoundError(
                f"STARE images directory not found: {img_dir}\n"
                f"Expected structure: {root_dir}/images/  and  {root_dir}/{gt_folder}/"
            )

        exts = (".ppm", ".png", ".jpg", ".tif")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

        # 80/20 split (16 train, 4 test) reproducibly
        n_test  = max(1, len(img_files) // 5)
        test_f  = img_files[-n_test:]
        train_f = img_files[:-n_test]
        chosen  = test_f if split == "test" else train_f

        self.samples = []
        for fname in chosen:
            img_path = os.path.join(img_dir, fname)
            # GT may have same name or slightly different extension
            gt_name  = os.path.splitext(fname)[0]
            gt_path  = None
            for ext in exts:
                candidate = os.path.join(gt_dir, gt_name + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            if gt_path is not None:
                self.samples.append((img_path, gt_path))
            else:
                print(f"  [WARN] No GT found for {fname} in {gt_dir}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image-GT pairs found in {root_dir}. "
                f"Check that gt_folder='{gt_folder}' exists and contains matching files."
            )

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        gt  = Image.open(gt_path).convert("L")

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        gt  = gt.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            import random
            if random.random() > 0.5:
                img = T.functional.hflip(img)
                gt  = T.functional.hflip(gt)
            if random.random() > 0.5:
                img = T.functional.vflip(img)
                gt  = T.functional.vflip(gt)
            angle = random.choice([0, 90, 180, 270])
            if angle > 0:
                img = T.functional.rotate(img, angle)
                gt  = T.functional.rotate(gt, angle)

        img_t = self.to_tensor(img)                            # (3, H, W) float [0,1]
        gt_t  = (self.to_tensor(gt) > 0.5).float()           # (1, H, W) binary

        return {"image": img_t, "mask": gt_t, "path": img_path}


def create_stare_loader(
    root_dir: str,
    batch_size: int = 4,
    img_size: int = 512,
    gt_folder: str = "labels-ah",
    num_workers: int = 2,
):
    """
    Returns test DataLoader for STARE.
    """
    dataset = STAREDataset(
        root_dir=root_dir,
        split="test",
        img_size=img_size,
        augment=False,
        gt_folder=gt_folder,
    )
    print(f"  [STARE] Loaded {len(dataset)} test images from {root_dir}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_stare_train_loader(
    root_dir: str,
    batch_size: int = 4,
    img_size: int = 512,
    gt_folder: str = "labels-ah",
    num_workers: int = 2,
):
    """
    Returns train DataLoader for STARE.
    """
    dataset = STAREDataset(
        root_dir=root_dir,
        split="train",
        img_size=img_size,
        augment=True,
        gt_folder=gt_folder,
    )
    print(f"  [STARE] Loaded {len(dataset)} train images from {root_dir}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
