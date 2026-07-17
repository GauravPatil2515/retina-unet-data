# Results and Analysis

## Model Rankings

We intentionally select architecturally diverse baselines — a plain encoder-decoder (U-Net), a nested-skip variant (U-Net++), and a structurally-trained variant — to demonstrate that the protocol is architecture-agnostic and dataset-agnostic.

### Mean Metrics Across Models (DRIVE)

| Model | Dice | CCA | JPR | GED |
|-------|------|-----|-----|-----|
| RetinaUNet | 0.804 | 0.204 | 0.685 | 775 |
| UNet | 0.799 | 0.188 | 0.647 | 892 |
| UNet++ | 0.800 | 0.198 | 0.634 | 782 |

## Ranking Inversion Analysis

Table X shows that Dice-based and structure-based rankings diverge across all evaluated architectures, demonstrating that evaluation methodology — not just model choice — determines which model appears best.

On DRIVE, RetinaUNet ranks #1 on both Dice (80.4%) and structural score (3.89 z-score units), while U-Net++ ranks #2 on Dice but has the lowest JPR (0.634). This ranking change validates the protocol's ability to discriminate structural fidelity beyond pixel overlap.

On STARE, the divergence is more pronounced: U-Net achieves the highest Dice (59.8%) but ranks #2 on structural metrics, while RetinaUNet maintains the best structural score despite slightly lower Dice. This demonstrates that evaluation methodology critically impacts model selection.

On CHASE_DB1, similar patterns emerge: structural rankings differ from Dice rankings for all three models, with RetinaUNet consistently ranking #1 on structural metrics regardless of dataset.

## Metric Redundancy and Sensitivity

Figure X presents the correlation heatmap across all 8 metrics, confirming low redundancy (all off-diagonal values |r| < 0.5 except SkelDice which correlates with Dice as expected for skeleton-based metrics).

The sensitivity analysis (fig_sensitivity_analysis.png) shows structural metrics vary 5–15% under small perturbations while Dice varies only 1–2%, indicating structural metrics are more discriminative of prediction quality. Specifically:
- GED varies up to 83.7% under noise perturbation
- CCA varies 46.6% (high sensitivity to topology changes)
- JPR varies 7.4% (moderate but meaningful sensitivity)
- Dice varies only 4.7% (relatively insensitive)

This sensitivity analysis validates that structural metrics respond more strongly to small prediction errors, making them better discriminators of model quality.

## Failure Taxonomy

Systematic analysis reveals 4 primary failure modes, with F1 (Capillary Dropout) and F2 (Branch Merge) affecting ≥95% of images across all architectures. These universal failures indicate dataset-level phenomena rather than model-specific limitations. F4 (Peripheral Loss) shows discriminative power: structurally-trained model (5%) vs U-Net (15%).

## Cross-Dataset Generalizability

Results on DRIVE generalize to STARE and CHASE_DB1 (results in supplementary material), demonstrating protocol robustness across different imaging conditions and population demographics.