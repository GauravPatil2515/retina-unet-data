"""
Dice vs Structural Metrics Correlation Analysis
================================================
Paper: "Beyond Dice: Structural Evaluation and Failure Analysis"
Section: 4.4 (or Table 3)

This is one of the most important experiments in Paper 1.

If Dice and structural metrics are WEAKLY correlated (|r| < 0.5),
that proves: high Dice does NOT guarantee structural correctness.
That is the paper's central claim.

Output:
  - Pearson r and p-value for: Dice vs CCA, BPR, JPR, GED, SkelDice
  - Scatter plots saved as PNGs (optional)
  - Paper-ready Table 3 printed to stdout

Usage:
  python scripts/correlation_analysis.py \\
      --pred_dir results/predictions/UNetPP/DRIVE \\
      --gt_dir   data/DRIVE/test/mask \\
      --model    UNetPP  --dataset DRIVE

Requires:
  scipy (pip install scipy)
  predictions saved as PNG in --pred_dir (binarised at threshold 0.5)
  If you have probability maps (.npy), pass --prob_dir instead.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.structural_metrics import StructuralMetrics
from evaluation.graph_metrics import VesselGraphMetrics

RESULTS_DIR = "results/paper1_benchmark"
MIN_COMP_PX = 10


def load_mask(path: str) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def compute_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)


def run_correlation(pred_dir, gt_dir, model, dataset, save_plots=False):
    sm  = StructuralMetrics(min_component_px=MIN_COMP_PX)
    gm  = VesselGraphMetrics()

    records = []
    exts    = (".png", ".jpg", ".tif")
    files   = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(exts)])

    print(f"\nComputing per-image metrics for {len(files)} images ...")
    for fname in files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir,   fname)
        if not os.path.exists(gt_path):
            print(f"  [SKIP] {fname}")
            continue
        pred = load_mask(pred_path)
        gt   = load_mask(gt_path)

        str_m = sm.compute_all(pred, gt)
        grp_m = gm.compute_all(pred, gt)
        dice  = compute_dice(pred, gt)

        records.append({"Dice": dice, **str_m, **grp_m, "image": fname})

    if len(records) < 5:
        print("[ERROR] Too few images for correlation. Need at least 5.")
        return

    dice_vals = np.array([r["Dice"] for r in records])
    
    # Structural metrics to correlate against Dice
    targets = {
        "CCA":       np.array([r["CCA"]        for r in records]),
        "BPR":       np.array([r["BPR"]        for r in records]),
        "JPR":       np.array([r["JPR"]        for r in records]),
        "GED":       np.array([r["GED_approx"] for r in records]),
        "SkelDice":  np.array([r["SkelDice"]   for r in records]),
        "SkelHD":    np.array([r["SkelHD"]     for r in records]),
    }

    # -----------------------------------------------------------------------
    # Correlation table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  TABLE 3 — Dice vs Structural Metrics Correlation ({model} / {dataset})")
    print("=" * 70)
    print(f"  {'Metric':<14} {'Pearson r':>10} {'p-value':>10}  {'Spearman r':>12}  Interpretation")
    print("  " + "-" * 62)

    corr_results = {}
    for metric, vals in targets.items():
        pr, pp = pearsonr(dice_vals, vals)
        sr, sp = spearmanr(dice_vals, vals)
        if abs(pr) < 0.3:
            interp = "WEAK   ★ supports paper claim"
        elif abs(pr) < 0.6:
            interp = "MODERATE"
        else:
            interp = "STRONG  (Dice tracks this metric)"
        sign = "♦" if pp < 0.05 else " "
        print(f"  {metric:<14} {pr:>+9.3f}{sign} {pp:>10.4f}  {sr:>+11.3f}   {interp}")
        corr_results[metric] = {"pearson_r": pr, "pearson_p": pp,
                                 "spearman_r": sr, "spearman_p": sp}

    print("  " + "-" * 62)
    print("  ♦ = significant at p < 0.05")
    print("  ★ = weak correlation: Dice does NOT capture this metric")
    print("="*70)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"correlation_{model}_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":   model,
            "dataset": dataset,
            "n":       len(records),
            "correlations": corr_results
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # Optional scatter plots
    if save_plots:
        try:
            import matplotlib.pyplot as plt
            plot_dir = os.path.join(RESULTS_DIR, "correlation_plots")
            os.makedirs(plot_dir, exist_ok=True)
            for metric, vals in targets.items():
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(dice_vals * 100, vals, alpha=0.7, edgecolors="k", s=50)
                r, _ = pearsonr(dice_vals, vals)
                ax.set_xlabel("Dice (%)")
                ax.set_ylabel(metric)
                ax.set_title(f"{model} / {dataset}: Dice vs {metric}  (r={r:.3f})")
                plt.tight_layout()
                fig.savefig(os.path.join(plot_dir,
                            f"dice_vs_{metric}_{model}_{dataset}.png"), dpi=150)
                plt.close()
            print(f"  Scatter plots saved to: {plot_dir}")
        except ImportError:
            print("  [SKIP] matplotlib not installed; no plots saved")

    return corr_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir",   required=True)
    parser.add_argument("--gt_dir",     required=True)
    parser.add_argument("--model",      default="Model")
    parser.add_argument("--dataset",    default="DRIVE")
    parser.add_argument("--save_plots", action="store_true")
    args = parser.parse_args()
    run_correlation(args.pred_dir, args.gt_dir,
                    args.model, args.dataset, args.save_plots)


if __name__ == "__main__":
    main()
