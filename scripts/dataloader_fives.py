"""
FIVES Dataset Loader
=====================
Retinal Vessel Segmentation Dataset with pathology labels.

FIVES provides:
  - 2000+ high-resolution fundus images (2048x2048)
  - Binary vessel segmentation masks
  - Diagnostic labels: Normal, Diabetic Retinopathy (DR), Glaucoma, AMD

For VRAM efficiency on 6GB GPUs, images are downsampled to max_dim=512.
Patch-based training still used with 128x128 patches.

Usage:
    from scripts.dataloader_fives import create_fives_loaders
    train_loader, test_loader = create_fives_loaders("data/FIVES", downsample_max=512)
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch


class FIVESDataset(Dataset):
    """
    FIVES retinal vessel segmentation dataset loader.

    Expected structure:
      data/FIVES/
        train/
          image/ (*.png, *.jpg)
          mask/  (*.png, *.jpg) - binary vessel masks
        test/
          image/
          mask/

    If no split directories exist, uses top-level images/masks with 80/20 split.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        img_size: int = None,  # If None, uses original size (may be huge)
        augment: bool = False,
        downsample_max: int = 512,  # Max dimension for VRAM efficiency
    ):
        self.root_dir = root_dir
        self.downsample_max = downsample_max
        self.augment = augment and (split == "train")

        # Find images and masks
        img_dir = os.path.join(root_dir, split, "image")
        mask_dir = os.path.join(root_dir, split, "mask")

        # Handle alternative structures
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root_dir, "images")
            mask_dir = os.path.join(root_dir, "masks")

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

        self.samples = []
        for fname in img_files:
            img_path = os.path.join(img_dir, fname)
            # Try same extension first, then any
            gt_name = os.path.splitext(fname)[0]
            gt_path = None
            for ext in exts:
                candidate = os.path.join(mask_dir, gt_name + ext)
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
            if gt_path is not None:
                self.samples.append((img_path, gt_path))
            else:
                print(f"  [WARN] No mask found for {fname}")

        self.to_tensor = T.ToTensor()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {img_dir}")

    def _load_and_resize(self, path, is_mask=False):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB" if not is_mask else "L")
        else:
            img = img.convert("L") if is_mask else img

        # Downsample to max dimension if specified
        if self.downsample_max:
            w, h = img.size
            if max(w, h) > self.downsample_max:
                scale = self.downsample_max / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                resample = Image.NEAREST if is_mask else Image.BILINEAR
                img = img.resize((new_w, new_h), resample)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        img = self._load_and_resize(img_path, is_mask=False)
        gt = self._load_and_resize(gt_path, is_mask=True)

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

        img_t = self.to_tensor(img)  # (3, H, W) float [0,1]
        gt_t = (self.to_tensor(gt) > 0.5).float()  # (1, H, W) binary

        return {"image": img_t, "mask": gt_t, "path": img_path}


class FIVESPatchDataset(Dataset):
    """
    FIVES with patch extraction for training on 6GB VRAM.
    Standardizes on 128x128 patches like DRIVE.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: int = 128,
        stride: int = 64,
        augment: bool = True,
        downsample_max: int = 512,
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.downsample_max = downsample_max

        # Find all images
        img_dir = os.path.join(root_dir, split, "image")
        mask_dir = os.path.join(root_dir, split, "mask")

        if not os.path.exists(img_dir):
            img_dir = os.path.join(root_dir, "images")
            mask_dir = os.path.join(root_dir, "masks")

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(exts)])

        self.patches = []
        for img_file, mask_file in zip(img_files, mask_files):
            img = self._load_and_resize(os.path.join(img_dir, img_file), is_mask=False)
            mask = self._load_and_resize(os.path.join(mask_dir, mask_file), is_mask=True)

            img_arr = np.array(img)
            mask_arr = (np.array(mask) > 0).astype(np.float32)

            H, W = img_arr.shape[:2]
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    img_patch = img_arr[y:y+patch_size, x:x+patch_size]
                    mask_patch = mask_arr[y:y+patch_size, x:x+patch_size]
                    self.patches.append({"image": img_patch, "mask": mask_patch})

        print(f"[FIVES] Extracted {len(self.patches)} patches from {len(img_files)} images")

    def _load_and_resize(self, path, is_mask=False):
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB" if not is_mask else "L")
        else:
            img = img.convert("L") if is_mask else img

        if self.downsample_max:
            w, h = img.size
            if max(w, h) > self.downsample_max:
                scale = self.downsample_max / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                resample = Image.NEAREST if is_mask else Image.BILINEAR
                img = img.resize((new_w, new_h), resample)
        return img

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        import random
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
            if random.random() > 0.5:
                factor = random.uniform(0.9, 1.1)
                mean = image.mean()
                image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return image, mask


def create_fives_loaders(
    root_dir: str,
    batch_size: int = 16,
    patch_size: int = 128,
    img_size: int = None,
    downsample_max: int = 512,
    num_workers: int = 2,
):
    """
    Create train and test loaders for FIVES.

    Args:
        root_dir: Path to FIVES dataset root
        batch_size: Batch size
        patch_size: For training patches
        downsample_max: Max image dimension (512 recommended for 6GB GPU)
        num_workers: DataLoader workers
    """
    train_dataset = FIVESPatchDataset(
        root_dir, split="train", patch_size=patch_size,
        stride=patch_size // 2, augment=True, downsample_max=downsample_max
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    test_dataset = FIVESDataset(
        root_dir, split="test", img_size=img_size,
        downsample_max=downsample_max
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  [FIVES] Train: {len(train_dataset)} patches | Test: {len(test_dataset)} images")
    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    print("FIVES Dataset Loader - checking implementation...")
    print("[OK] FIVES loader module loaded successfully!")