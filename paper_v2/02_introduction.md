# Introduction

## Background
Retinal vessel segmentation enables automated screening for diabetic retinopathy and other vascular pathologies. While deep learning models achieve high accuracy on benchmark datasets, standard evaluation metrics fail to capture topological correctness of vessel centerlines and branching structures.

## Problem Statement
How should retinal vessel segmentation models be evaluated to reflect clinically meaningful vascular topology rather than pixel overlap alone?

Current evaluation practice relies on Dice coefficient and pixel-level metrics, but clinical interpretation requires accurate vessel connectivity, bifurcation preservation, and graph-structure fidelity. This evaluation gap obscures model differences that could impact diagnostic reliability.

## Related Work

### A) Pixel-based Evaluation
Early work on retinal vessel segmentation adopted standard segmentation metrics including Dice, Jaccard, and AUC-ROC. These measures aggregate performance across all pixels without distinguishing topology-critical regions (bifurcations, vessel endpoints) from background or large vessels. Recent studies continue to report these metrics as primary evaluation criteria.

### B) Topology-aware Training Losses  
Recent works propose auxiliary losses encouraging skeleton connectivity, including clDice, connectivity loss, and graph-based objectives. However, these methods still report Dice as the primary metric, despite training with structure-aware objectives.

### C) Clinical Retinal Morphometry
Ophthalmology literature quantifies vessel topology through branching coefficients, arteriovenous ratios, and fractal dimensions. These measures predict cardiovascular risk and diagnose hypertensive retinopathy. Translating these clinical measures to segmentation evaluation requires structural analysis pipelines.

### D) The Evaluation Gap
Despite advances in topology-aware training, no standardized structural evaluation protocol exists for retinal vessel segmentation. Existing works report isolated skeleton or connectivity measures without comprehensive metric suites.