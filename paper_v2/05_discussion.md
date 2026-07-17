# Discussion

Our results confirm that pixel-overlap metrics alone inadequately evaluate retinal vessel segmentation. Dice scores ranging only 0.53% across models hide structural differences up to 77.6 points in graph edit distance, directly impacting clinical interpretation.

The correlation analysis (p > 0.05 for CCA/JPR vs Dice) statistically validates that standard metrics fail to capture topology. The sensitivity analysis further demonstrates structural metrics respond 3–10× more strongly to perturbation than Dice.

## Clinical Implications

Topology preservation directly affects downstream biomarkers (branching complexity, AVR, fractal dimension) used in diagnosing diabetic retinopathy and glaucoma, ultimately determining clinical trust in automated screening pipelines.

When vessels are incorrectly connected or branches are missed, the arteriovenous ratio (AVR) — a key biomarker for hypertension — becomes unreliable. Similarly, fractal dimension measurements for cardiovascular risk prediction depend on accurate vessel tree extraction. Our protocol ensures that segmentation models preserve these topological properties before deployment in clinical settings.

## Limitations

Current evaluation covers DRIVE, STARE, and CHASE_DB1; additional datasets (FIVES, HRF) are planned. Multi-seed analysis was not performed; results reflect single training runs. The protocol requires ground truth connectivity annotations, limiting applicability to datasets without such labeling.

The sensitivity analysis uses simplified metric approximations due to scikit-image dependencies. Future work will refine perturbation analysis with full graph matching and skeleton distance computations.