# 🔬 RetinaAI — Retinal Blood Vessel Segmentation with U-Net++

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)](docs/RESEARCH_PLAN.md)

> **Topology-Preserving Retinal Blood Vessel Segmentation using Curriculum-Trained U-Net++ with Persistent Homology Loss.**

An advanced deep learning system for automatic segmentation of blood vessels in retinal fundus images. This project implements **U-Net++ with Curriculum Learning and Topological Loss**, achieving **83.82% Dice** on DRIVE (baseline) and targeting SOTA performance through novel training methodology.

---

## 🎯 Research Contributions

This project is an **active research effort** targeting publication at MICCAI / ISBI 2026. See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the full plan.

### Novel Findings & Methods
1. **Patch Filter Bias** — Standard hard patch filtering (`vessel_ratio < 0.01`) creates a training-test mismatch that inflates Dice scores. We quantify this bias for the first time.
2. **Curriculum Patch Learning** — 3-stage progressive training from vessel-rich → sparse patches, healing the distribution gap.
3. **Topological Loss** — Persistent homology (Betti-0/1 Wasserstein distance) penalises vessel connectivity breaks, a clinically critical failure mode ignored by Dice/BCE.
4. **New Metrics** — Sparse Vessel Dice (SVD) and Connected Component Accuracy (CCA) expose model failures hidden by standard Dice.

---

## 📊 Current Performance (Baseline)

| Metric | Score |
|--------|-------|
| **Dice Coefficient** | **83.82%** |
| **Accuracy** | **96.08%** |
| **Sensitivity** | **82.91%** |
| **Specificity** | **97.97%** |
| **AUC-ROC** | **97.82%** |
| Sparse Vessel Dice (SVD) | TBD (new metric) |
| Connected Component Acc (CCA) | TBD (new metric) |

---

## 📁 Project Structure

```
Retina-Unet/
├── models/
│   ├── unet_plus_plus.py          # U-Net++ architecture (9M params)
│   ├── losses_unetpp.py           # DiceLoss, BCEDiceLoss, TopologicalLoss [NEW]
│   └── __init__.py
├── scripts/
│   ├── dataloader_unetpp.py       # PatchDataset + CurriculumPatchDataset [NEW]
│   ├── train_unetpp.py            # Training pipeline (curriculum-aware) [UPDATED]
│   ├── evaluate_unetpp.py         # Evaluation + SVD + CCA metrics [UPDATED]
│   ├── inference.py               # Single image inference
│   ├── test_model.py              # Quick model test
│   └── dataloader_unetpp.py
├── dashboard/
│   ├── app.py                     # FastAPI web app
│   ├── templates/
│   └── static/
├── Retina/
│   ├── train/ (image/ mask/)
│   └── test/  (image/ mask/)
├── results/
│   ├── checkpoints_unetpp/        # best.pth, latest.pth, metrics.json
│   └── evaluation_results_unetpp/ # test_metrics.json, prediction_*.png
├── docs/
│   ├── RESEARCH_PLAN.md           # Full research plan [NEW]
│   ├── FINAL_RESULTS.md
│   └── README_UNETPP.md
├── requirements_unetpp.txt        # Updated with gudhi, scipy, scikit-image
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/GauravPatil2515/Retina-Unet.git
cd Retina-Unet
pip install -r requirements_unetpp.txt
```

### 2. Download DRIVE Dataset

```bash
python download_datasets.py
```

### 3. Verify Dataset Distribution (Novel Finding)

```bash
python scripts/dataloader_unetpp.py
```

This prints the patch distribution — shows exactly how many patches the baseline hard-filter discards.

### 4. Run Baseline Training (Experiment A)

```bash
# In scripts/train_unetpp.py, set:
#   USE_CURRICULUM = False
#   USE_TOPO_LOSS  = False
python scripts/train_unetpp.py
```

### 5. Run Research Training (Experiment D — Full Method)

```bash
# Default config already has USE_CURRICULUM=True, USE_TOPO_LOSS=True
python scripts/train_unetpp.py
```

### 6. Evaluate with New Metrics

```bash
python scripts/evaluate_unetpp.py
```

Outputs: Dice, Accuracy, AUC **+ Sparse Vessel Dice (SVD) + CCA + Betti-0 Error**

### 7. Run Web Dashboard

```bash
cd dashboard
uvicorn app:app --reload --host localhost --port 8000
# Open: http://localhost:8000
```

---

## 🧠 Architecture & New Components

### U-Net++ (unchanged)
- 5-level encoder-decoder with nested skip connections
- Deep supervision: 4 output heads with weights `[0.25, 0.25, 0.25, 1.0]`
- 9,047,873 trainable parameters
- Input: 512×512×3 → Output: 512×512×1

### [NEW] CurriculumPatchDataset

```python
from scripts.dataloader_unetpp import CurriculumPatchDataset

dataset = CurriculumPatchDataset(train_img_dir, train_mask_dir)
# In epoch loop:
dataset.set_curriculum_stage(1)  # epochs 1-20:  vessel_ratio >= 0.05
dataset.set_curriculum_stage(2)  # epochs 21-40: vessel_ratio >= 0.02
dataset.set_curriculum_stage(3)  # epochs 41-60: vessel_ratio >= 0.001
```

### [NEW] TopologicalLoss

```python
from models.losses_unetpp import TopologicalLoss, DeepSupervisionLoss

# Standalone
topo = TopologicalLoss(w0=1.0, w1=0.5)
loss = topo(pred_logits, target_mask)

# Integrated into training
criterion = DeepSupervisionLoss(use_topo=True, topo_weight=0.1)
```

### [NEW] Evaluation Metrics

```python
from scripts.evaluate_unetpp import (
    calculate_sparse_vessel_dice,
    calculate_connected_component_accuracy
)

svd, n_sparse = calculate_sparse_vessel_dice(predictions)
cca, betti_err = calculate_connected_component_accuracy(predictions)
```

---

## 🗺️ Ablation Experiments

To reproduce the 4 ablation experiments for the paper:

```bash
# Exp A — Baseline (no curriculum, no topo)
# Edit Config: USE_CURRICULUM=False, USE_TOPO_LOSS=False
python scripts/train_unetpp.py
python scripts/evaluate_unetpp.py

# Exp B — Curriculum only
# Edit Config: USE_CURRICULUM=True, USE_TOPO_LOSS=False
python scripts/train_unetpp.py
python scripts/evaluate_unetpp.py

# Exp C — Topo Loss only
# Edit Config: USE_CURRICULUM=False, USE_TOPO_LOSS=True
python scripts/train_unetpp.py
python scripts/evaluate_unetpp.py

# Exp D — Full method (default)
# Config: USE_CURRICULUM=True, USE_TOPO_LOSS=True (default)
python scripts/train_unetpp.py
python scripts/evaluate_unetpp.py
```

---

## 📚 Dataset: DRIVE

- 40 high-resolution fundus images (565×584 px, RGB)
- Split: 20 train + 20 test
- Canon CR5 non-mydriatic camera, 45° FOV
- Expert vessel annotations
- Source: Dutch diabetic retinopathy screening program
- Download: https://drive.grand-challenge.org/

---

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | U-Net++ (Nested U-Net) |
| Parameters | 9.0M |
| Optimizer | Adam (lr=1e-4, wd=1e-5) |
| Loss | BCE+Dice + Topo (λ=0.1, from epoch 21) |
| Curriculum | Stage1: ep1-20, Stage2: ep21-40, Stage3: ep41-60 |
| Batch Size | 8 (effective 16 with grad accum) |
| Patch Size | 128×128, stride 64 |
| GPU | NVIDIA RTX 3050 6GB |
| Mixed Precision | FP16 |

---

## 🏥 Clinical Applications

- Diabetic retinopathy screening
- Hypertensive retinopathy: arterial narrowing detection
- Retinal vein/artery occlusion
- Vessel morphology analysis: tortuosity, fractal dimension, AVR

---

## 📝 Citation

```bibtex
@software{retinaai_curriculum_topo,
  title   = {Topology-Preserving Retinal Vessel Segmentation with Curriculum U-Net++},
  author  = {Gaurav Patil},
  year    = {2026},
  url     = {https://github.com/GauravPatil2515/Retina-Unet}
}
```

```bibtex
@inproceedings{zhou2018unetpp,
  title  = {UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author = {Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  booktitle = {DLMIA/ML-CDS},
  year   = {2018}
}
```

---

## 🤝 Contributing

See [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) for the active research roadmap.

**Author**: Gaurav Patil · [@GauravPatil2515](https://github.com/GauravPatil2515)
