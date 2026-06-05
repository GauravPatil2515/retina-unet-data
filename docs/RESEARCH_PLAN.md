# Research Plan: Curriculum + Topology-Preserving Retinal Vessel Segmentation

**Project**: RetinaAI — Beating SOTA on Retinal Blood Vessel Segmentation  
**Author**: Gaurav Patil  
**Branch**: `research/curriculum-topo-loss`  
**Status**: 🔬 Active Research

---

## 🎯 Core Research Claims

1. **Novel Finding**: Standard patch-filtering (hard cutoff `vessel_ratio < 0.01`) creates a systematic training-test distribution mismatch, inflating reported Dice scores.
2. **Fix 1**: Curriculum Learning — 3-stage progressive patch inclusion heals this bias.
3. **Fix 2**: Topological Loss — persistent homology penalty prevents vessel connectivity breaks not captured by Dice/BCE.
4. **New Metrics**: Sparse Vessel Dice (SVD) and Connected Component Accuracy (CCA) expose failures hidden by standard Dice.

---

## 📊 Ablation Table (Target)

| Exp | Curriculum | Topo Loss | Dice (DRIVE) | SVD | CCA | Betti-0 Err |
|-----|-----------|-----------|-------------|-----|-----|-------------|
| A — Baseline | ❌ | ❌ | 83.82% | ~TBD | ~TBD | ~TBD |
| B — +Curriculum | ✅ | ❌ | ~85-86% | ↑ | ↑ | ↓ |
| C — +Topo only | ❌ | ✅ | ~84-85% | ↑ | ↑↑ | ↓↓ |
| **D — Ours** | ✅ | ✅ | **~86-87%** | **↑↑** | **↑↑↑** | **↓↓↓** |

---

## 📁 New Files Added

| File | Purpose |
|------|---------|
| `scripts/dataloader_unetpp.py` | `CurriculumPatchDataset` — 3-stage progressive training |
| `models/losses_unetpp.py` | `TopologicalLoss` (gudhi) + updated `DeepSupervisionLoss` |
| `scripts/train_unetpp.py` | Curriculum stage switching + topo activation mid-training |
| `scripts/evaluate_unetpp.py` | `calculate_sparse_vessel_dice()` + `calculate_connected_component_accuracy()` |
| `requirements_unetpp.txt` | Added `gudhi`, `scikit-image`, `scipy` |

---

## 🗓️ Timeline

| Week | Task |
|------|------|
| 1-2  | Implement & test CurriculumPatchDataset; run Exp A & B |
| 3-5  | Implement & validate TopologicalLoss; run Exp C & D |
| 6    | New metric evaluation (SVD, CCA) on all 4 experiments |
| 7    | Cross-dataset eval: STARE, CHASE_DB1 |
| 8    | Write paper draft |

---

## 🔬 Key Insight (Novel Finding)

The existing `PatchDataset` discards patches where `vessel_ratio < 0.01`.
This means the model:
- Has **never been trained** on the foveal avascular zone
- Has **never been trained** on thin capillary boundary regions
- Has **never been trained** on peripheral retina with sparse vessels

Yet at test time, the model is evaluated on **ALL pixels** of full retinal images.
The 83.82% Dice is therefore an *optimistic* metric measured on an out-of-distribution region.

This single finding + the curriculum fix constitutes a novel methodological contribution.

---

## 📝 Target Venues

- **MICCAI 2026** (Medical Image Computing and Computer Assisted Intervention)
- **ISBI 2026** (IEEE International Symposium on Biomedical Imaging)
- **Computers in Biology and Medicine** (journal)
- **Biomedical Signal Processing and Control** (journal)
