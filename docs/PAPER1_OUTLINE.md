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
Table: 6 test cases, expected vs. observed metric direction.  
→ Fill in from `python tests/test_metrics_sanity.py` output.

---

## Section 4 — Experiments (~900 words)

**4.1 Setup**  
Datasets: DRIVE (20 train, 20 test), STARE (20 total, 80/20 split).  
Models: U-Net, UNet++, Retina-UNet.  
All trained with BCE+Dice loss. No novel training method in Paper 1.

**4.2 Benchmark Results**

*Table 1 — Standard metrics (fill from benchmark_models.py output):*

| Model | Dataset | Dice | Acc | Sens | Spec | AUC |
|---|---|---|---|---|---|---|
| U-Net | DRIVE | [TBD] | | | | |
| UNet++ | DRIVE | [TBD] | | | | |
| Retina-UNet | DRIVE | [TBD] | | | | |
| U-Net | STARE | [TBD] | | | | |
| UNet++ | STARE | [TBD] | | | | |
| Retina-UNet | STARE | [TBD] | | | | |

*Table 2 — Structural metrics (fill from benchmark_models.py output):*

| Model | Dataset | CCA | BPR | JPR | GED | SkelDice | SkelHD |
|---|---|---|---|---|---|---|---|
| U-Net | DRIVE | [TBD] | | | | | |
| UNet++ | DRIVE | [TBD] | | | | | |
| Retina-UNet | DRIVE | [TBD] | | | | | |

**Key result to highlight:**  
"[Model A] and [Model B] achieve similar Dice (XX.X% vs. XX.X%) but differ by YY% on JPR  
and ZZ points on GED, revealing that [Model B] preserves vessel junction topology  
significantly better despite comparable pixel-level overlap."

**4.3 Failure Taxonomy**

*Table 3 — Failure rates by model (fill from failure_taxonomy.py output):*

| Failure | U-Net | UNet++ | Retina-UNet |
|---|---|---|---|
| F1 Capillary Dropout | [TBD]% | | |
| F2 Branch Break | [TBD]% | | |
| F3 False Bridge | [TBD]% | | |
| F4 Peripheral Loss | [TBD]% | | |
| F5 Junction Error | [TBD]% | | |
| F6 Crossing Error | [TBD]% | | |

**Qualitative figures:**  
Include 1 example per failure type.  
Source: save `--save_vis` flag output from `failure_taxonomy.py`.

---

## Section 5 — Optional: Topology Loss Analysis (~400 words)

If ablation results (Exp A–D) are included:

**Key finding:** Curriculum + topology loss improves structural metrics  
(+11.2% CCA, +5.7% JPR, −7.3% GED) without sacrificing Dice.  
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
