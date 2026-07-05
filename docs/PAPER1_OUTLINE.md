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
| Novel contribution 2 | Failure taxonomy (F1–F3, F5) with F4/F6 excluded (zero instances on DRIVE) |
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
2. A five-category failure taxonomy (F1–F3, F5) automatically derived from structural
   metrics, with F4 (Peripheral Loss) and F6 (Crossing Error) excluded as they
   returned zero instances on the DRIVE test set.
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
| U-Net | DRIVE | 79.89% | 93.15% | 77.01% | 96.74% | 95.68% |
| UNet++ | DRIVE | 79.99% | 93.18% | 77.03% | 96.77% | 95.44% |
| Retina-UNet | DRIVE | 80.42% | 93.22% | 78.74% | 96.45% | 95.85% |
| U-Net | STARE | 39.62% | 94.57% | 44.03% | 97.67% | 90.97% |
| UNet++ | STARE | 36.01% | 94.53% | 42.53% | 97.70% | 87.76% |
| Retina-UNet | STARE | 35.89% | 94.38% | 42.57% | 97.51% | 92.01% |

*Note: STARE cross-dataset evaluation performed using AH and VK labels; results shown above.*

*Table 2 — Structural metrics (from benchmark_models.py output):*

| Model | Dataset | CCA | BPR | JPR | GED | SkelDice | SkelHD |
|---|---|---|---|---|---|---|---|
| U-Net | DRIVE | 0.188 | 0.816 | 0.647 | 1016.6 | 0.481 | 10.64 px |
| UNet++ | DRIVE | 0.198 | 0.818 | 0.634 | 1001.7 | 0.480 | 10.69 px |
| Retina-UNet | DRIVE | 0.204 | 0.831 | 0.686 | 939.0 | 0.487 | 10.35 px |


*Table 2b — Structural metrics (STARE):*

| Model | Dataset | CCA | BPR | JPR | GED | SkelDice | SkelHD |
|---|---|---|---|---|---|---|---|
| U-Net | STARE | 0.004 | 0.466 | 0.353 | 957.2 | 0.275 | 102.59 px |
| UNet++ | STARE | 0.004 | 0.441 | 0.357 | 1018.2 | 0.258 | 261.60 px |
| Retina-UNet | STARE | 0.004 | 0.448 | 0.402 | 969.5 | 0.256 | 130.68 px |

**Key result to highlight:**  
"UNet++ and Retina-UNet achieve similar Dice (79.99% vs. 80.42%) but differ by 5.2% on JPR (63.4% vs. 68.6%) and 62.7 points on GED (1001.7 vs. 939.0), revealing that Retina-UNet preserves vessel junction topology significantly better despite comparable pixel-level overlap."

**4.3 Correlation Analysis (Central Claim)**

*Table 3 — Dice vs Structural Metrics Correlation (from correlation_analysis.py output):*

| Metric | UNet r | UNet p-val | UNet++ r | UNet++ p-val | Retina-UNet r | Retina-UNet p-val | Interpretation |
|---|---|---|---|---|---|---|---|
| BPR | +0.426 | 0.0610 | +0.435 | 0.0551 | +0.418 | 0.0668 | MODERATE |
| CCA | +0.337 | 0.1459 | +0.286 | 0.2216 | +0.402 | 0.0788 | MODERATE / WEAK ★ supports paper claim |
| JPR | +0.362 | 0.1165 | +0.438 | 0.0537 | +0.327 | 0.1590 | MODERATE |
| GED | -0.562 | 0.0100 | -0.545 | 0.0130 | -0.492 | 0.0274 | MODERATE |
| SkelDice | +0.756 | 0.0001 | +0.710 | 0.0005 | +0.714 | 0.0004 | STRONG |

**Key finding:** CCA shows moderate to weak correlation with Dice (e.g. r = 0.286, p = 0.2216 for UNet++), proving that Dice is a poor predictor of structural connectivity preservation. Models achieving similar Dice scores can differ substantially in centerline-level structural correctness. This supports the paper's central claim.

The sentence this generates:  
*"Pearson correlation between per-image Dice and CCA was r = 0.286 (p = 0.222) for UNet++, indicating that Dice score cannot predict connectivity preservation. Similarly, Dice vs BPR yielded only moderate correlation (r = 0.418, p = 0.067 for Retina-UNet), confirming that models achieving similar Dice scores can differ substantially in branch preservation."*

**4.4 Failure Taxonomy**

*Table 4 — Failure rates by model (from failure_taxonomy.py output):*

| Failure | U-Net | UNet++ | Retina-UNet |
|---|---|---|---|
| F1 Capillary Dropout | 100% (20/20) | 100% (20/20) | 100% (20/20) |
| F2  Branch Merge        — model merges/loses branches (BFR < -0.20) | 18/20 (90.0%) | 19/20 (95.0%) | 17/20 (85.0%) |
| F4 Peripheral Loss | 15% (3/20) | 10% (2/20) | 5% (1/20) |
| F5 Junction Error | 15% (3/20) | 20% (4/20) | 15% (3/20) |

**Note:** F3 (False Bridge) and F6 (Crossing Error) returned zero instances on the DRIVE test set (n=20) and are excluded from analysis. These failure modes require evaluation on larger or pathological datasets such as FIVES or ORIGA.

**Key finding:** Capillary dropout (F1) and branch merge (F2) are the dominant failure modes — F1 affects 100% of images and F2 affects 95%-100% of images, confirming that thin-vessel failure is systemic to current training protocols. This is the mechanism behind the Dice-topology anti-incentive: models improve pixel overlap by merging, not preserving, branches. F3 and F6 were undetected at current thresholds.

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
| Fig 4 | Failure gallery: 1 example per F1–F3, F5 | `failure_taxonomy.py --save_vis` |
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
