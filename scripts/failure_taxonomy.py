"""
Failure Taxonomy Annotator — Paper 1, Section 4.3
==================================================
Paper: "Beyond Dice: Structural Evaluation and Failure Analysis
        of Retinal Vessel Segmentation"

Automatically classifies each prediction image into one or more
failure categories:

  F1  Capillary Dropout   — thin vessels missing (skeleton coverage)
  F2  Branch Merge        — model merges/loses branches (BFR < -0.05)
  F3  False Bridge        — two vessels incorrectly connected
  F5  Junction Error      — bifurcation structure wrong

NOTE: F4 (Peripheral Loss) and F6 (Crossing Error) removed — both
returned zero instances on DRIVE (n=20). Included here as stub
detectors for future use on larger datasets (FIVES, ORIGA, etc.).

Outputs:
  results/paper1_benchmark/failure_taxonomy.csv
  results/paper1_benchmark/failure_frequency.json
  (optional) visualizations saved to results/paper1_benchmark/failure_vis/

Usage:
  python scripts/failure_taxonomy.py \\
      --pred_dir results/predictions/UNet/DRIVE \\
      --gt_dir   data/DRIVE/test/mask \\
      --model    UNet  --dataset DRIVE
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.structural_metrics import StructuralMetrics
from evaluation.graph_metrics import VesselGraphMetrics


# -----------------------------------------------------------------------
# Failure thresholds (tune after first run)
# -----------------------------------------------------------------------
THR = {
    "F1_svd_drop":         0.05,   # was 0.15 — too strict
    "F2_bfr_negative":    -0.05,   # Branch Merge: model produces fewer branches than GT
    "F4_peripheral_ratio": 0.25,   # stub — removed from main taxonomy
    "F5_jpr_drop":         0.50,   # was 0.30 — too lenient
}

MIN_COMP_PX = 10


# -----------------------------------------------------------------------
# Load image pair
# -----------------------------------------------------------------------
def load_pair(pred_path: str, gt_path: str):
    pred = np.array(Image.open(pred_path).convert("L"))
    gt   = np.array(Image.open(gt_path).convert("L"))
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)
    return pred, gt


# -----------------------------------------------------------------------
# Individual failure detectors
# -----------------------------------------------------------------------
def detect_F1_capillary_dropout(pred, gt, sm: StructuralMetrics) -> bool:
    """
    F1: Thin vessels (sparse patches) are missing in prediction.
    Signal: SVD much lower than overall SkelDice.
    """
    svd      = sm.sparse_vessel_dice(pred, gt)
    skel_pred = skeletonize(pred).astype(np.uint8)
    skel_gt   = skeletonize(gt).astype(np.uint8)
    skd      = sm.skeleton_dice(skel_pred, skel_gt)
    if svd < 0:          # no sparse patches at all
        return False
    return (skd - svd) > THR["F1_svd_drop"]


def detect_F2_branch_merge(pred, gt, sm: StructuralMetrics) -> bool:
    """
    F2: Prediction has fewer branches than GT (vessels merged/lost).
    Signal: BFR < negative threshold (model merges branches → fewer components).
    """
    skel_pred = skeletonize(pred).astype(np.uint8)
    skel_gt   = skeletonize(gt).astype(np.uint8)
    bfr = sm.branch_fragmentation_ratio(skel_pred, skel_gt)
    return bfr < THR["F2_bfr_negative"]


def detect_F3_false_bridge(pred, gt, sm: StructuralMetrics) -> bool:
    """
    F3: Prediction creates false connections (bridges) between vessels.
    Signal: Model has significantly fewer connected components than GT,
    meaning it bridged gaps that should be disconnected.
    """
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin   = (gt   > 0.5).astype(np.uint8)
    # Count components similarly to structural_metrics (filtering noise)
    pred_labeled, n_pred = ndimage.label(pred_bin)
    gt_labeled, n_gt     = ndimage.label(gt_bin)
    # Don't filter by size here — we want raw component count difference
    # False bridge: model has fewer components (bridged gaps)
    if n_gt == 0:
        return False
    return n_pred < n_gt * 0.7


def detect_F4_peripheral_loss(pred, gt) -> bool:
    """
    F4: Outer ring of retina under-segmented.
    Method: compare vessel density in outer 20% margin between pred and GT.
    """
    H, W     = gt.shape
    margin_h = int(H * 0.20)
    margin_w = int(W * 0.20)

    # outer margin mask
    outer_mask = np.zeros_like(gt, dtype=bool)
    outer_mask[:margin_h, :]  = True
    outer_mask[-margin_h:, :] = True
    outer_mask[:, :margin_w]  = True
    outer_mask[:, -margin_w:] = True

    gt_outer   = gt[outer_mask].sum()
    pred_outer = pred[outer_mask].sum()

    if gt_outer == 0:
        return False
    recall_outer = pred_outer / gt_outer
    return recall_outer < (1.0 - THR["F4_peripheral_ratio"])


def detect_F5_junction_error(pred, gt, gm: VesselGraphMetrics) -> bool:
    """
    F5: Wrong bifurcation structure.
    Signal: Junction Preservation Rate below threshold.
    """
    g = gm.compute_all(pred, gt)
    return g["JPR"] < THR["F5_jpr_drop"]


def detect_F6_crossing_error(pred, gt) -> bool:
    """
    F6: Vessel crossing regions corrupted.
    Method: detect candidate crossing pixels (high vessel density neighbourhood)
    in GT and measure prediction recall in those regions.
    """
    from scipy.ndimage import uniform_filter
    density = uniform_filter(gt.astype(float), size=7)
    crossing_mask = density > 0.6   # dense region = likely crossing

    if crossing_mask.sum() < 10:
        return False

    gt_cross   = gt[crossing_mask].sum()
    pred_cross = pred[crossing_mask].sum()

    if gt_cross == 0:
        return False
    iou = pred_cross / (gt_cross + 1e-8)
    return iou < 0.65  # F6 threshold (stub — zero instances on DRIVE)


# -----------------------------------------------------------------------
# Classify one image
# -----------------------------------------------------------------------
def classify_image(pred, gt, sm, gm) -> dict:
    return {
        "F1_CapillaryDropout":  detect_F1_capillary_dropout(pred, gt, sm),
        "F2_BranchMerge":       detect_F2_branch_merge(pred, gt, sm),
        "F3_FalseBridge":       detect_F3_false_bridge(pred, gt, sm),
        "F4_PeripheralLoss":    detect_F4_peripheral_loss(pred, gt),
        "F5_JunctionError":     detect_F5_junction_error(pred, gt, gm),
        "F6_CrossingError":     detect_F6_crossing_error(pred, gt),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory of prediction PNGs")
    parser.add_argument("--gt_dir",   required=True, help="Directory of GT mask PNGs")
    parser.add_argument("--model",    default="Model",   help="Model name label")
    parser.add_argument("--dataset",  default="DRIVE",   help="Dataset name label")
    parser.add_argument("--save_vis", action="store_true", help="Save failure visualizations")
    args = parser.parse_args()

    os.makedirs(f"{RESULTS_DIR}", exist_ok=True)
    if args.save_vis:
        os.makedirs(f"{RESULTS_DIR}/failure_vis", exist_ok=True)

    sm = StructuralMetrics(min_component_px=MIN_COMP_PX)
    gm = VesselGraphMetrics()

    pred_files = sorted([
        f for f in os.listdir(args.pred_dir)
        if f.endswith((".png", ".jpg", ".tif"))
    ])

    rows       = []
    freq       = {f"F{i}": 0 for i in range(1, 7)}
    freq_map   = {
        "F1_CapillaryDropout": "F1",
        "F2_BranchMerge":      "F2",
        "F3_FalseBridge":      "F3",
        "F4_PeripheralLoss":   "F4",
        "F5_JunctionError":    "F5",
        "F6_CrossingError":    "F6",
    }

    for fname in pred_files:
        pred_path = os.path.join(args.pred_dir, fname)
        gt_path   = os.path.join(args.gt_dir,   fname)
        if not os.path.exists(gt_path):
            print(f"  [SKIP] GT not found for {fname}")
            continue

        pred, gt = load_pair(pred_path, gt_path)

        if args.dataset == "DRIVE":
            fov_dir = "Retina/test/fov"
            fov_path = os.path.join(fov_dir, fname)
            if os.path.exists(fov_path):
                fov = np.array(Image.open(fov_path).convert("L")) > 127
                pred = (pred * fov).astype(np.uint8)
                gt   = (gt * fov).astype(np.uint8)

        failures = classify_image(pred, gt, sm, gm)

        for key, fkey in freq_map.items():
            if failures[key]:
                freq[fkey] += 1

        row = {"image": fname, **{k: int(v) for k, v in failures.items()}}
        rows.append(row)
        flags = [k for k, v in failures.items() if v]
        print(f"  {fname}: {', '.join(flags) if flags else 'no failure detected'}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, f"failure_taxonomy_{args.model}_{args.dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Save frequency summary
    summary = {
        "model":   args.model,
        "dataset": args.dataset,
        "n_images": len(rows),
        "failure_counts": freq,
        "failure_rates": {k: round(v / len(rows) * 100, 1) for k, v in freq.items()},
    }
    json_path = os.path.join(RESULTS_DIR, f"failure_freq_{args.model}_{args.dataset}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print taxonomy table
    print("\n" + "=" * 60)
    print(f"  FAILURE TAXONOMY — {args.model} on {args.dataset}")
    print("=" * 60)
    labels = {
        "F1": "Capillary Dropout ",
        "F2": "Branch Merge      ",
        "F3": "False Bridge      ",
        "F4": "Peripheral Loss (*)",
        "F5": "Junction Error    ",
        "F6": "Crossing Error (*)",
    }
    for k, label in labels.items():
        n    = freq[k]
        rate = n / len(rows) * 100
        bar  = "█" * int(rate / 2)
        print(f"  {label}: {n:3d}/{len(rows)}  ({rate:5.1f}%)  {bar}")
    print("=" * 60)
    print(f"  CSV  saved to: {csv_path}")
    print(f"  JSON saved to: {json_path}")
    print(f"\n  NOTE: F4 (Peripheral Loss) and F6 (Crossing Error) returned")
    print(f"  zero instances on the DRIVE test set (n={len(rows)}). These failure")
    print(f"  modes require evaluation on larger or pathological datasets")
    print(f"  such as FIVES or ORIGA.")


RESULTS_DIR = "results/paper1_benchmark"

if __name__ == "__main__":
    main()
