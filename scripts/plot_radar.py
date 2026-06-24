"""
Radar Chart — Figure 4 (Multi-Metric Comparison)
==================================================
Paper: "Dice Considered Harmful: Structural Failure Analysis of Retinal
        Vessel Segmentation"
Figure: 4 — Radar chart comparing 3 models on 6 structural metrics

All metrics normalized to [0, 1] with 1 = best.
GED is inverted (lower = better → maps to 1 = best).

Usage:
    python scripts/plot_radar.py

Output:
    results/figures/fig4_radar_chart.pdf
    results/figures/fig4_radar_chart.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIGURE_DIR = "results/figures"

# Categories for the radar chart
CATEGORIES = ["Dice", "CCA", "BPR", "JPR", "SkelDice", "GED\n(inverted)"]
N = len(CATEGORIES)

# Raw values from benchmark_results.json
MODELS = {
    "U-Net":        [0.7994, 0.205, 0.811, 0.645, 0.476, 1031.8],
    "UNet++":       [0.8004, 0.224, 0.814, 0.633, 0.476, 1019.4],
    "Retina-UNet":  [0.8048, 0.228, 0.828, 0.682, 0.483,  956.0],
}

# Min/max for normalization (per-metric bounds from your data)
MINS = [0.799, 0.205, 0.811, 0.633, 0.476,  956.0]
MAXS = [0.805, 0.228, 0.828, 0.682, 0.483, 1031.8]

COLORS = {"U-Net": "#2196F3", "UNet++": "#FF9800", "Retina-UNet": "#4CAF50"}


def normalize(vals: list) -> list:
    """Min-max normalize to [0, 1]. GED (index 5) is inverted."""
    normed = []
    for i, v in enumerate(vals):
        if MAXS[i] == MINS[i]:
            normed.append(0.5)
        elif i == 5:  # GED: invert (lower is better → 1.0)
            normed.append(1 - (v - MINS[i]) / (MAXS[i] - MINS[i]))
        else:
            normed.append((v - MINS[i]) / (MAXS[i] - MINS[i]))
    return normed


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model_name, raw_vals in MODELS.items():
        normed = normalize(raw_vals)
        normed += normed[:1]  # close the loop
        ax.plot(angles, normed, "o-", linewidth=2,
                label=model_name, color=COLORS[model_name], markersize=6)
        ax.fill(angles, normed, alpha=0.08, color=COLORS[model_name])

    # Labels and grid
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(True, alpha=0.4)

    ax.set_title(
        "Multi-Metric Comparison\n(normalized, outer = better)",
        fontsize=13, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "fig4_radar_chart.png"),
                dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURE_DIR, "fig4_radar_chart.pdf"),
                dpi=300, bbox_inches="tight")
    print(f"  Saved: {FIGURE_DIR}/fig4_radar_chart.png")
    print(f"  Saved: {FIGURE_DIR}/fig4_radar_chart.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()