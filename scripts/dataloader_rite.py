"""
RITE Dataset Loader
===================
RITE (Retinal Identification for Tracking and Evaluation) dataset.

Shares the EXACT same images as DRIVE (20 train, 20 test), but provides
artery/vein labels instead of binary vessel masks.

This allows testing whether models preserve the clinical distinction between
arteries and veins - a key requirement for Paper 1's "Secondary Dataset" goal.

Usage:
    from scripts.dataloader_rite import create_rite_loaders
    train_loader, test_loader = create_rite_loaders("data/RITE")
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import random


class RITEDataset(Dataset):
    """
    RITE dataset loader - reuses DRIVE image files.

    Expected structure:
      data/RITE/
        training/
          images/  (symlink or copy of DRIVE training images)
          av/      (artery labels - same format as DRIVE masks)
          ov/      (vein labels)
        test/
          images/
          av/
          ov/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        img_size: int = None,
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.augment = augment and (split == "train")

        # RITE uses 'training' for train split
        split_dir = "training" if split == "train" else "test"
        img_dir = os.path.join(root_dir, split_dir, "images")
        av_dir = os.path.join(root_dir, split_dir, "av")
        ov_dir = os.path.join(root_dir, split_dir, "ov")

        exts = (".png", ".tif", ".jpg", ".gif")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

        self.samples = []
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            base = os.path.splitext(fname)[0]

            # Try to find artery/vein masks
            av_path = None
            ov_path = None
            for ext in exts:
                av_cand = os.path.join(av_dir, base + ext)
                ov_cand = os.path.join(ov_dir, base + ext)
                if os.path.exists(av_cand):
                    av_path = av_cand
                if os.path.exists(ov_cand):
                    ov_path = ov_cand
                if av_path and ov_path:
                    break

            if av_path and ov_path:
                self.samples.append((img_path, av_path, ov_path))
            else:
                print(f"  [WARN] No AV/OV masks for {fname}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No RITE samples found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, av_path, ov_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        av = Image.open(av_path).convert("L")
        ov = Image.open(ov_path).convert("L")

        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                av = TF.hflip(av)
                ov = TF.hflip(ov)
            if random.random() > 0.5:
                img = TF.vflip(img)
                av = TF.vflip(av)
                ov = TF.vflip(ov)

        img_t = T.ToTensor()(img)  # (3, H, W)
        av_t = (T.ToTensor()(av) > 0.5).float()  # (1, H, W) artery binary
        ov_t = (T.ToTensor()(ov) > 0.5).float()  # (1, H, W) vein binary

        # Combined vessel mask = artery OR vein
        vessel_mask = (av_t + ov_t > 0).float()

        return {
            "image": img_t,
            "mask": vessel_mask,
            "artery": av_t,
            "vein": ov_t,
            "path": img_path
        }


class RITEPatchDataset(Dataset):
    """
    RITE with 128x128 patch extraction for VRAM efficiency.
    Same patch strategy as DRIVE for fair comparison.
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 128,
        stride: int = 64,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment

        # Find all training images
        img_dir = os.path.join(root_dir, "training", "images")
        av_dir = os.path.join(root_dir, "training", "av")
        ov_dir = os.path.join(root_dir, "training", "ov")

        exts = (".png", ".tif", ".jpg", ".gif")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])[:20]

        self.patches = []
        for fname in img_files:
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)

            # Find masks
            av_path = None
            ov_path = None
            for ext in exts:
                if os.path.exists(os.path.join(av_dir, base + ext)):
                    av_path = os.path.join(av_dir, base + ext)
                if os.path.exists(os.path.join(ov_dir, base + ext)):
                    ov_path = os.path.join(ov_dir, base + ext)
                if av_path and ov_path:
                    break

            if not (av_path and ov_path):
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            av = (np.array(Image.open(av_path).convert("L")) > 0).astype(np.float32)
            ov = (np.array(Image.open(ov_path).convert("L")) > 0).astype(np.float32)
            mask = (av + ov > 0).astype(np.float32)

            H, W = img.shape[:2]
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.patches.append({
                        "image": img[y:y+patch_size, x:x+patch_size],
                        "mask": mask[y:y+patch_size, x:x+patch_size],
                        "artery": av[y:y+patch_size, x:x+patch_size],
                        "vein": ov[y:y+patch_size, x:x+patch_size],
                    })

        print(f"[RITE] Extracted {len(self.patches)} patches from {len(img_files)} images")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        image = patch["image"].copy()
        mask = patch["mask"].copy()

        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            if random.random() > 0.5:
                delta = random.uniform(-0.1, 0.1)
                image = np.clip(image.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask


def create_rite_loaders(
    root_dir: str,
    batch_size: int = 8,
    patch_size: int = 128,
    num_workers: int = 2,
):
    """Create train and test loaders for RITE."""
    train_dataset = RITEPatchDataset(root_dir, patch_size=patch_size, augment=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_dataset = RITEDataset(root_dir, split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  [RITE] Train: {len(train_dataset)} patches | Test: {len(test_dataset)} images")
    return train_loader, test_loader


if __name__ == "__main__":
    print("RITE Dataset Loader - checking implementation...")
    print("[OK] RITE loader module loaded successfully!")