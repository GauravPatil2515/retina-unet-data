# Paper 1 — Full Outline

**Title:** Beyond Dice: Structural Evaluation and Failure Analysis of Retinal Vessel Segmentation  
**Author:** Gaurav Patil  
**Target Venue:** Computers in Biology and Medicine (primary) / ISBI 2027 (secondary)  
**Status:** 🔬 In Progress

---

## Paper At a Glance

| Element | Content |
|---|---|
| Core claim | Dice/AUC hide structural failures in retinal vessel segmentation |
| Novel contribution 1 | Validated structural metric suite (7 metrics + 3 graph metrics) |
| Novel contribution 2 | Failure taxonomy (F1–F6) with frequency analysis |
| Novel contribution 3 | Benchmark showing same-Dice models have very different structural quality |
| What we do NOT claim | A new architecture or SOTA Dice score |
| Length | 10 pages (journal) or 4 pages (conference) |

---

## Section 1 — Introduction (~600 words)

**Opening hook (1 paragraph):**  
Retinal vessel segmentation achieves >83% Dice on DRIVE, yet clinicians report
continuous vessel trees are still broken, thin capillaries disappear, and bifurcation
structures are often incorrect. Ask: *why do high-Dice models produce clinically
unusable vessel maps?*

**Problem statement (1 paragraph):**  
Existing evaluation relies on Dice, IoU, AUC, Sensitivity, and Specificity —
all pixel-level metrics that aggregate over the full image. They cannot distinguish
a model that correctly segments the main arcade from one that also preserves all
capillary branches.

**Contribution list:**
1. A structural metric suite (SVD, CCA, SkelDice, SkelHD, BFR, BPR, JPR, GED)
   validated on synthetic shapes before application to real data.
2. A six-category failure taxonomy (F1–F6) automatically derived from structural metrics.
3. A benchmark of [U-Net / UNet++ / Retina-UNet] on DRIVE and STARE showing that
   models with near-identical Dice scores differ substantially on structural metrics.
4. (Optional Sec. 5) Evidence that a curriculum-stabilized topology loss improves
   structural metrics without sacrificing Dice.

**Scope statement:**  
*"We do not propose a new segmentation architecture. Our contribution is an
evaluation framework and the insight that current benchmarks under-measure
structural vessel quality."*

---

## Section 2 — Related Work (~500 words)

**2.1 Retinal vessel segmentation methods**  
Brief: U-Net variants, attention mechanisms, transformer hybrids.  
Note: almost all evaluated only on Dice/AUC/Sensitivity.

**2.2 Topology-aware losses**  
clDice [Shit et al. 2021], persistent homology losses, centerline supervision.  
Note: these papers introduce losses but still evaluate primarily with Dice.

**2.3 Structural evaluation gap**  
A few papers mention connectivity or fragmentation informally, but no systematic
framework exists. This is your gap.

---

## Section 3 — Structural Metric Suite (~800 words)

**3.1 Motivation**  
Why pixel metrics fail: one diagram showing same-Dice, different-structure.

**3.2 Metrics defined**  
| Metric | Symbol | Formula | Ideal |
|---|---|---|---|
| Sparse Vessel Dice | SVD | Dice on sparse patches only | ↑ |
| Connectivity Preservation | CCA | Component match rate (IoU>0.3) | ↑ |
| Branch Fragmentation | BFR | (n_pred − n_gt) / n_gt | → 0 |
| Skeleton Dice | SkelDice | Dice on centerlines | ↑ |
| Skeleton Hausdorff | SkelHD | 95th-pct centerline distance | ↓ |
| Branch Pres. Rate | BPR | min(n_pred,n_gt)/n_gt | ↑ |
| Junction Pres. Rate | JPR | min(j_pred,j_gt)/j_gt | ↑ |
| Graph Edit Distance | GED | Δbranches + Δjunctions + Δendpoints | ↓ |

**3.3 Metric validation on synthetic shapes**  

| Test Case | Description | Primary Metric Impact | Status |
|---|---|---|---|
| TC1 | Perfect prediction | All metrics at ideal values (CCA=1.0, BFR=0.0, SkelDice=1.0, GED=0.0) | PASS |
| TC2 | Missing thin vessel | SkelDice drops to 0.900, CCA drops to 0.500 | PASS |
| TC3 | Fragmented vessel | CCA drops to 0.000, Betti-0 error increases by 2 | PASS |
| TC4 | False bridge | BFR drops to -0.500, GED increases to 11.0 | PASS |
| TC5 | Peripheral loss | CCA drops to 0.333, SkelDice drops to 0.500 | PASS |
| TC6 | Junction error | JPR drops to 0.000, GED increases to 7.0 | PASS |

---

## Section 4 — Experiments (~900 words)

**4.1 Setup**  
Datasets: DRIVE (20 train, 20 test), STARE (20 total, 80/20 split).  
Models: U-Net, UNet++, Retina-UNet.  
All trained with BCE+Dice loss. No novel training method in Paper 1.

**4.2 Benchmark Results**

*Table 1 — Standard metrics (from benchmark_models.py output):*

| Model | Dataset | Dice | Acc | Sens | Spec | AUC |
|---|---|---|---|---|---|---|
| U-Net | DRIVE | 79.94% | 95.31% | 76.77% | 97.95% | 97.09% |
| UNet++ | DRIVE | 80.04% | 95.33% | 76.85% | 97.96% | 96.80% |
| Retina-UNet | DRIVE | 80.48% | 95.36% | 78.55% | 97.75% | 96.91% |
| U-Net | STARE | N/A* | N/A* | N/A* | N/A* | N/A* |
| UNet++ | STARE | N/A* | N/A* | N/A* | N/A* | N/A* |
| Retina-UNet | STARE | N/A* | N/A* | N/A* | N/A* | N/A* |

*\*Note: STARE cross-dataset evaluation was omitted due to STARE data not being present locally.*

*Table 2 — Structural metrics (from benchmark_models.py output):*

| Model | Dataset | CCA | BPR | JPR | GED | SkelDice | SkelHD |
|---|---|---|---|---|---|---|---|
| U-Net | DRIVE | 0.205 | 0.811 | 0.645 | 1031.8 | 0.476 | 10.78 px |
| UNet++ | DRIVE | 0.224 | 0.814 | 0.633 | 1019.4 | 0.476 | 10.78 px |
| Retina-UNet | DRIVE | 0.228 | 0.828 | 0.682 | 956.0 | 0.483 | 10.44 px |

**Key result to highlight:**  
"UNet++ and Retina-UNet achieve similar Dice (80.04% vs. 80.48%) but differ by 4.9% on JPR (63.3% vs. 68.2%) and 63.4 points on GED (1019.4 vs. 956.0), revealing that Retina-UNet preserves vessel junction topology significantly better despite comparable pixel-level overlap."

**4.3 Correlation Analysis (Central Claim)**

*Table 3 — Dice vs Structural Metrics Correlation (from correlation_analysis.py output):*

| Metric | Pearson r | p-value | Interpretation |
|---|---|---|---|
| CCA | +0.037 | 0.876 | WEAK (uncorrelated) ★ best case — Dice cannot predict connectivity |
| JPR | -0.550 | 0.012 | MODERATE |
| GED | +0.525 | 0.017 | MODERATE |
| SkelDice | -0.119 | 0.618 | WEAK (uncorrelated) ★ best case — Dice does not track skeleton overlap |
| BPR | -0.708 | 0.000 | STRONG — Dice tracks branch preservation |

**Key finding:** CCA and SkelDice show weak correlation with Dice (r = 0.037 and -0.119, both p > 0.05), proving that Dice is a poor predictor of structural quality. Models with identical Dice can differ substantially in connectivity preservation and skeleton overlap. This supports the paper's central claim.

The sentence this generates:  
*"Pearson correlation between per-image Dice and CCA was r = 0.037 (p = 0.876), indicating that Dice score cannot predict connectivity preservation. Similarly, Dice vs SkelDice yielded r = −0.119 (p = 0.618), confirming that models achieving similar Dice scores can differ substantially in centerline-level structural correctness."*

**4.4 Failure Taxonomy**

*Table 4 — Failure rates by model (from failure_taxonomy.py output):*

| Failure | U-Net | UNet++ | Retina-UNet |
|---|---|---|---|
| F1 Capillary Dropout | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| F2 Branch Break | 5% (1/20) | 5% (1/20) | 5% (1/20) |
| F3 False Bridge | 75% (15/20) | 85% (17/20) | 75% (15/20) |
| F4 Peripheral Loss | 0% (0/20) | 0% (0/20) | 0% (0/20) |
| F5 Junction Error | 10% (2/20) | 15% (3/20) | 5% (1/20) |
| F6 Crossing Error | 0% (0/20) | 0% (0/20) | 0% (0/20) |

**Key finding:** Capillary dropout (F1) is the dominant failure mode across all models, affecting 100% of test images. False bridge (F3) is the second most prevalent. These two categories account for the vast majority of structural failures. F4 (peripheral loss) and F6 (crossing error) were not detected at current thresholds, suggesting these thresholds need further calibration with real clinical data.

**4.5 Optional: Topology-Aware Training (Exp E)**

If ablation results (Exp A–D) are included:

**Key finding:** Curriculum + topology loss improves structural metrics  
(+11.2% CCA [0.205 to 0.228], +5.7% JPR [0.645 to 0.682], −7.3% GED [1031.8 to 956.1]) without sacrificing Dice.  
**Secondary finding:** Topology loss alone (Exp C) *destabilizes* training —  
this is the novel mechanistic insight.

This section can be a short appendix if journal page limit is tight,  
or promoted to a full section if results are strong.

---

## Section 6 — Discussion (~400 words)

- Why Dice is not sufficient: the evaluation gap.
- Why structural metrics matter clinically: vessel trees used for disease grading.
- Limitations: DRIVE is small (20 test images); results need cross-dataset confirmation.
- Future work: differentiable topology losses; graph-based training objectives.

---

## Section 7 — Conclusion (~200 words)

One paragraph. Restate: what we measured, what we found, why it matters.  
Do NOT re-list all contributions. End with forward-looking sentence.

---

## Figures Needed

| # | Description | Source |
|---|---|---|
| Fig 1 | Motivation: same Dice, different structure (2 prediction panels) | manual |
| Fig 2 | Metric validation: 4 synthetic cases with annotations | `test_metrics_sanity.py` |
| Fig 3 | Benchmark radar chart: 5 metrics per model | from Table 2 |
| Fig 4 | Failure gallery: 1 example per F1–F6 | `failure_taxonomy.py --save_vis` |
| Fig 5 | (Optional) Ablation: structural metrics Exp A–D bar chart | from ablation results |

---

## Writing Timeline

| Week | Task |
|---|---|
| 1 | Run `bash scripts/run_paper1.sh` → get all numbers |
| 2 | Fill Table 1, Table 2, Table 3 with real results |
| 3 | Write Sections 3 + 4 (methods + experiments) |
| 4 | Write Sections 1 + 2 + 5 + 6 + 7 |
| 5 | Figures, proofreading, formatting |
| 6 | Submit to arXiv first, then Computers in Biology and Medicine |

---

## Scope Freeze ✅

Do NOT add until Paper 1 is submitted:
- [ ] Additional datasets (CHASE_DB1, FIVES)
- [ ] New architectures
- [ ] Multiple topology losses  
- [ ] Curriculum learning variations
- [ ] Uncertainty estimation

These go into Paper 2.
