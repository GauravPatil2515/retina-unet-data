"""
Dice vs BPR Correlation Scatter — Figure 3 (Money Figure)
===========================================================
Paper: "Dice Considered Harmful: Structural Failure Analysis of Retinal
        Vessel Segmentation"
Figure: 3 — Per-image Dice vs Branch Preservation Rate scatter

Generates the paper's key figure: regression scatter showing the
Dice-topology anti-incentive (r = -0.708, p < 0.001).

Usage:
    python scripts/plot_correlation.py
    # Uses per_image_metrics_*.json from results/paper1_benchmark/
    # Falls back to synthetic demo data if real files are missing

Output:
    results/figures/fig3_dice_bpr_correlation.pdf
    results/figures/fig3_dice_bpr_correlation.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


RESULTS_DIR = "results/paper1_benchmark"
FIGURE_DIR  = "results/figures"

COLORS  = {"U-Net": "#2196F3", "UNet++": "#FF9800", "Retina-UNet": "#4CAF50"}
MARKERS = {"U-Net": "o",       "UNet++": "s",       "Retina-UNet": "^"}

def load_model_data(model_name, dataset="DRIVE"):
    """Load dice and bpr arrays for a given model."""
    if dataset == "both":
        d_drive, b_drive = load_model_data(model_name, "DRIVE")
        d_stare, b_stare = load_model_data(model_name, "STARE")
        if d_drive is not None and d_stare is not None:
            return np.concatenate([d_drive, d_stare]), np.concatenate([b_drive, b_stare])
        elif d_drive is not None:
            return d_drive, b_drive
        else:
            return d_stare, b_stare

    # Map model names to file patterns
    name_map = {
        "U-Net": "UNet",
        "UNet++": "UNetPP",
        "Retina-UNet": "RetinaUNet"
    }
    fname = f"per_image_metrics_{name_map[model_name]}_{dataset}.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    dice = np.array([img["Dice"] for img in data])
    bpr  = np.array([img["BPR"]  for img in data])
    return dice, bpr

def add_regression(ax, x, y):
    """Add linear regression line with 95% confidence band."""
    slope, intercept, r_value, p_value, se = stats.linregress(x, y)
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r--", linewidth=2, zorder=2,
            label=f"R² = {r_value**2:.3f}")
    # 95% CI
    n = len(x)
    if n > 2:
        t_crit = stats.t.ppf(0.975, df=n-2)
        x_mean = np.mean(x)
        se_band = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
        ax.fill_between(x_line, y_line - t_crit * se_band, y_line + t_crit * se_band,
                        alpha=0.15, color="red", label="95% CI")
    return r_value, p_value

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DRIVE", choices=["DRIVE", "STARE", "both"])
    args = parser.parse_args()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Determine number of models with available data
    model_names = []
    data_dict = {}
    for name in ["U-Net", "UNet++", "Retina-UNet"]:
        dice, bpr = load_model_data(name, dataset=args.dataset)
        if dice is not None and len(dice) >= 4:
            model_names.append(name)
            data_dict[name] = (dice, bpr)
        else:
            print(f"  [WARN] Insufficient data (or file missing) for {name} on {args.dataset}")
    
    if not model_names:
        print("  [ERROR] No valid model data found; cannot generate plot.")
        return
    
    # Create subplots: one column per model
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]  # now axes is 1D list
    
    all_results = {}
    for idx, (name, ax) in enumerate(zip(model_names, axes)):
        dice, bpr = data_dict[name]
        ax.scatter(dice, bpr, c=COLORS[name], marker=MARKERS[name],
                   s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
        r, p = add_regression(ax, dice, bpr)
        all_results[name] = {'r': r, 'p': p, 'n': len(dice)}
        ax.set_xlabel("Dice Coefficient", fontsize=12)
        ax.set_ylabel("Branch Preservation Rate (BPR)", fontsize=12)
        ax.set_title(f"{name} ({args.dataset})", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Optionally add correlation text
        ax.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.3f}",
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle(f"Dice vs Branch Preservation Rate (BPR) per Model ({args.dataset})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # make room for suptitle
    
    suffix = f"_{args.dataset}" if args.dataset != "DRIVE" else ""
    out_pdf = os.path.join(FIGURE_DIR, f"fig3_dice_bpr_correlation{suffix}.pdf")
    out_png = os.path.join(FIGURE_DIR, f"fig3_dice_bpr_correlation{suffix}.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")
    for name in model_names:
        res = all_results[name]
        print(f"  {name}: r = {res['r']:.3f}, p = {res['p']:.3f}, n = {res['n']}")
    plt.close(fig)

if __name__ == "__main__":
    main()
