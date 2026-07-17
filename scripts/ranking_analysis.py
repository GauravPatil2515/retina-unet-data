"""
Ranking Inversion Analysis
=========================
Computes Dice vs Structural rank comparisons across models.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Data from paper1_status_report.md
models = ["U-Net", "UNet++", "Retina-UNet"]
dice_scores = [0.7989, 0.7999, 0.8042]  # Dice

# Structural scores (mean of z-normalized CCA, JPR, GED)
cca = [0.188, 0.198, 0.204]
bpr = [0.816, 0.818, 0.831]
jpr = [0.647, 0.634, 0.686]
ged = [1016.6, 1001.7, 939.0]

# Z-normalize structural metrics (higher = better, so invert GED)
cca_z = (np.array(cca) - np.mean(cca)) / np.std(cca)
bpr_z = (np.array(bpr) - np.mean(bpr)) / np.std(bpr)
jpr_z = (np.array(jpr) - np.mean(jpr)) / np.std(jpr)
ged_z = -(np.array(ged) - np.mean(ged)) / np.std(ged)  # Inverted

structural_score = (cca_z + bpr_z + jpr_z + ged_z) / 4

# Compute ranks (1 = best)
dice_rank = [3, 2, 1]  # Lower index = better dice (Retina-UNet best)
structural_rank = [3, 2, 1] if np.argmax(structural_score) == 2 else None

# Actually compute proper ranks
dice_rank = list(np.argsort(np.argsort(dice_scores)) + 1)  # 1=lowest, 3=highest
structural_rank = list(np.argsort(np.argsort(-structural_score)) + 1)  # Higher score = better

ranking_data = {
    "models": models,
    "dice_scores": dice_scores,
    "dice_ranks": dice_rank,
    "structural_scores": structural_score.tolist(),
    "structural_ranks": structural_rank,
    "structural_components": {
        "CCA_z": cca_z.tolist(),
        "BPR_z": bpr_z.tolist(),
        "JPR_z": jpr_z.tolist(),
        "GED_z": ged_z.tolist()
    }
}

# Save JSON
Path("results/paper1_benchmark").mkdir(parents=True, exist_ok=True)
with open("results/paper1_benchmark/ranking_inversion.json", "w") as f:
    json.dump(ranking_data, f, indent=2)

# Generate slope chart
fig, ax = plt.subplots(figsize=(8, 5))

y_pos = [0.8, 0.5, 0.2]
x_dice = [0.2] * 3
x_struct = [0.8] * 3

for i, (model, dr, sr, y) in enumerate(zip(models, dice_rank, structural_rank, y_pos)):
    ax.plot([x_dice[i], x_struct[i]], [y, y_pos[sr-1]], 'o-', linewidth=2, markersize=10)
    ax.text(0.05, y, f"{model}", fontsize=11, va='center')
    ax.text(0.95, y_pos[sr-1], f"{model}", fontsize=11, va='center', ha='right')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['Dice\nRank', 'Structural\nRank'])
ax.set_title('Ranking Inversion: Dice vs Structural Metrics', fontsize=14)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('results/figures/fig_ranking_inversion.png', dpi=150, bbox_inches='tight')
print("[OK] Ranking inversion saved")

if __name__ == "__main__":
    print(f"Results: {ranking_data}")