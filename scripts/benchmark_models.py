"""
Structural Benchmark Runner — Paper 1
======================================
Paper: "Beyond Dice: Structural Evaluation and Failure Analysis
        of Retinal Vessel Segmentation"

Runs three model configurations on DRIVE and STARE,
evaluates with full 10-metric structural suite,
prints paper-ready Table 1 + Table 2.

Usage:
    python scripts/benchmark_models.py --dataset DRIVE
    python scripts/benchmark_models.py --dataset STARE
    python scripts/benchmark_models.py --dataset both   # runs all

Models evaluated:
    1. U-Net        (models/unet_baseline.py)
    2. UNet++       (models/unetpp.py)
    3. Retina-UNet  (best checkpoint from your E9 experiments)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.structural_metrics import StructuralMetrics
from evaluation.graph_metrics import VesselGraphMetrics

# -----------------------------------------------------------------------
# CONFIG — edit these paths to match your local setup
# -----------------------------------------------------------------------
MODEL_CONFIGS = {
    "UNet": {
        "checkpoint": "results/ablation_results/exp_a_baseline/checkpoints/best.pth",
        "arch":       "unet",
    },
    "UNetPP": {
        "checkpoint": "results/ablation_results/exp_b_curriculum/checkpoints/best.pth",
        "arch":       "unetpp",
    },
    "RetinaUNet": {
        "checkpoint": "results/ablation_results/exp_d_ours/checkpoints/best.pth",
        "arch":       "unetpp",   # Retina-UNet uses UNet++ backbone
    },
}

DATASET_ROOTS = {
    "DRIVE": "Retina",
    "STARE": "data/STARE",
}

THRESHOLD   = 0.5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "results/paper1_benchmark"
MIN_COMP_PX = 10


# -----------------------------------------------------------------------
# Model loader
# -----------------------------------------------------------------------
def load_model(name: str, cfg: dict):
    arch = cfg["arch"]
    if arch == "unet":
        try:
            from models.unet_baseline import UNet
            model = UNet(in_channels=3, out_channels=1)
        except ImportError:
            # fallback: use UNet++ as U-Net proxy if unet_baseline not present
            print(f"  [WARN] models/unet_baseline.py not found; using UNetPlusPlus for {name}")
            from models.unet_plus_plus import UNetPlusPlus
            model = UNetPlusPlus(in_channels=3, out_channels=1)
    elif arch == "unetpp":
        from models.unet_plus_plus import UNetPlusPlus
        model = UNetPlusPlus(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ckpt_path = cfg["checkpoint"]
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  [OK] Loaded {name} from {ckpt_path}")
    return model


# -----------------------------------------------------------------------
# Dataset loader dispatcher
# -----------------------------------------------------------------------
def get_dataloader(dataset_name: str, batch_size: int = 1):
    root = DATASET_ROOTS[dataset_name]
    if dataset_name == "DRIVE":
        from scripts.dataloader_unetpp import FullImageDataset
        img_dir = os.path.join(root, "test", "image")
        mask_dir = os.path.join(root, "test", "mask")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(root, "images")
            mask_dir = os.path.join(root, "1st_manual")
        dataset = FullImageDataset(img_dir, mask_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "STARE":
        from scripts.dataloader_stare import create_stare_loader
        loader = create_stare_loader(root, batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loader


# -----------------------------------------------------------------------
# Evaluate one model on one dataset
# -----------------------------------------------------------------------
def evaluate_model(model, loader, model_name: str, dataset_name: str) -> dict:
    from PIL import Image
    sm = StructuralMetrics(min_component_px=MIN_COMP_PX, sparse_threshold=0.02, patch_size=32)
    gm = VesselGraphMetrics(min_branch_px=5)

    per_image = []
    all_probs, all_gts = [], []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if isinstance(batch, dict):
                imgs  = batch["image"].to(DEVICE)
                masks = batch["mask"].cpu().numpy().squeeze()
                img_path = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
                fname = os.path.basename(img_path)
            else:
                imgs, masks = batch
                imgs  = imgs.to(DEVICE)
                masks = masks.cpu().numpy().squeeze()
                fname = loader.dataset.image_files[idx]

            out   = model(imgs)
            if isinstance(out, (list, tuple)):
                out = out[0]

            prob = torch.sigmoid(out).cpu().numpy().squeeze()
            pred = (prob >= THRESHOLD).astype(np.uint8)
            gt   = (masks > 0).astype(np.uint8)

            # Load or auto-generate FOV mask
            fov = None
            if dataset_name == "DRIVE":
                fov_dir = os.path.join(DATASET_ROOTS["DRIVE"], "test", "fov")
                fov_path = os.path.join(fov_dir, fname)
                if os.path.exists(fov_path):
                    fov = np.array(Image.open(fov_path).convert("L")) > 127
            
            if fov is None:
                # Green-channel thresholding fallback
                img_np = imgs.cpu().numpy().squeeze()  # shape (3, H, W)
                green_channel = img_np[1]
                fov = green_channel > (10.0 / 255.0)

            # Apply FOV mask
            pred = (pred * fov).astype(np.uint8)
            gt   = (gt * fov).astype(np.uint8)

            # Save predictions as PNGs for failure taxonomy
            pred_dir = f"results/predictions/{model_name}/{dataset_name}"
            os.makedirs(pred_dir, exist_ok=True)
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            pred_img.save(os.path.join(pred_dir, fname))

            if fov is not None:
                all_probs.append(prob[fov].ravel())
                all_gts.append(gt[fov].ravel())

                pred_fov = pred[fov]
                gt_fov = gt[fov]
                tp = np.logical_and(pred_fov, gt_fov).sum()
                tn = np.logical_and(~pred_fov.astype(bool), ~gt_fov.astype(bool)).sum()
                fp = np.logical_and(pred_fov.astype(bool), ~gt_fov.astype(bool)).sum()
                fn = np.logical_and(~pred_fov.astype(bool), gt_fov.astype(bool)).sum()
            else:
                all_probs.append(prob.ravel())
                all_gts.append(gt.ravel())

                tp = np.logical_and(pred, gt).sum()
                tn = np.logical_and(~pred.astype(bool), ~gt.astype(bool)).sum()
                fp = np.logical_and(pred.astype(bool), ~gt.astype(bool)).sum()
                fn = np.logical_and(~pred.astype(bool), gt.astype(bool)).sum()

            std = {
                "Dice":        2*tp / (2*tp + fp + fn + 1e-8),
                "Accuracy":    (tp+tn) / (tp+tn+fp+fn + 1e-8),
                "Sensitivity": tp / (tp+fn + 1e-8),
                "Specificity": tn / (tn+fp + 1e-8),
            }
            str_m = sm.compute_all(pred, gt)
            grp_m = gm.compute_all(pred, gt)
            per_image.append({**std, **str_m, **grp_m})

    auc = roc_auc_score(
        np.concatenate(all_gts), np.concatenate(all_probs)
    )

    # Save per-image metrics to JSON
    per_image_path = os.path.join(RESULTS_DIR, f"per_image_metrics_{model_name}_{dataset_name}.json")
    with open(per_image_path, "w") as f:
        json.dump(per_image, f, indent=2)
    print(f"  Per-image metrics saved to {per_image_path}")

    keys   = list(per_image[0].keys())
    means  = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    stds   = {k + "_std": float(np.std([m[k] for m in per_image])) for k in keys}
    means["AUC_ROC"] = float(auc)
    means["model"]   = model_name
    means["dataset"] = dataset_name
    means["n_images"] = len(per_image)
    return {**means, **stds}


# -----------------------------------------------------------------------
# Pretty-print paper table
# -----------------------------------------------------------------------
def print_paper_table(results: list):
    print("\n" + "=" * 90)
    print("  TABLE 1 — Standard Metrics")
    print("=" * 90)
    header = f"  {'Model':<16} {'Dataset':<8} {'Dice':>7} {'Acc':>7} {'Sens':>7} {'Spec':>7} {'AUC':>7}"
    print(header)
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['model']:<16} {r['dataset']:<8} "
              f"{r['Dice']*100:>6.2f}% {r['Accuracy']*100:>6.2f}% "
              f"{r['Sensitivity']*100:>6.2f}% {r['Specificity']*100:>6.2f}% "
              f"{r['AUC_ROC']*100:>6.2f}%")

    print("\n" + "=" * 90)
    print("  TABLE 2 — Structural Metrics (Novel Contribution)")
    print("=" * 90)
    header2 = f"  {'Model':<16} {'Dataset':<8} {'CCA':>7} {'BPR':>7} {'JPR':>7} {'GED':>8} {'SkelD':>7} {'SkelHD':>8}"
    print(header2)
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['model']:<16} {r['dataset']:<8} "
              f"{r['CCA']:>7.3f} {r['BPR']:>7.3f} {r['JPR']:>7.3f} "
              f"{r['GED_approx']:>8.1f} {r['SkelDice']:>7.3f} {r['SkelHD']:>7.2f}px")
    print("=" * 90)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both",
                        choices=["DRIVE", "STARE", "both"],
                        help="Which dataset(s) to evaluate on")
    args = parser.parse_args()

    datasets = ["DRIVE", "STARE"] if args.dataset == "both" else [args.dataset]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for ds_name in datasets:
        ds_root = DATASET_ROOTS[ds_name]
        if not os.path.exists(ds_root):
            print(f"[SKIP] Dataset root not found: {ds_root}")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")
        loader = get_dataloader(ds_name)

        for model_name, model_cfg in MODEL_CONFIGS.items():
            print(f"\n  Model: {model_name}")
            model = load_model(model_name, model_cfg)
            if model is None:
                continue
            result = evaluate_model(model, loader, model_name, ds_name)
            all_results.append(result)
            print(f"  Dice={result['Dice']*100:.2f}%  "
                  f"CCA={result['CCA']:.3f}  "
                  f"JPR={result['JPR']:.3f}  "
                  f"GED={result['GED_approx']:.1f}")

    if all_results:
        print_paper_table(all_results)
        out_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
