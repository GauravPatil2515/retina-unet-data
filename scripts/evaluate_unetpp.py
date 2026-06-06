"""
Evaluation Script — UNet++ with Full Structural Metrics
=======================================================
Outputs 10-metric table per experiment:
  Standard:   Dice, Accuracy, Sensitivity, Specificity, AUC-ROC
  Structural: CCA, BFR, SVD, SkelDice, SkelHD,
              BranchDiff, Betti0Err, BPR, JPR, GED_approx
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.structural_metrics import StructuralMetrics
from evaluation.graph_metrics import VesselGraphMetrics


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
class EvalConfig:
    MODEL_PATH   = "results/checkpoints_unetpp/best_model.pth"
    DATA_ROOT    = "data/DRIVE"
    BATCH_SIZE   = 1
    THRESHOLD    = 0.5
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR  = "results"
    # Structural metric params
    MIN_COMP_PX     = 10
    SPARSE_THRESH   = 0.02
    PATCH_SIZE      = 32
    MIN_BRANCH_PX   = 5


# ------------------------------------------------------------------
# Standard metrics (pixel-level)
# ------------------------------------------------------------------
def standard_metrics(pred_bin, gt_bin):
    tp = np.logical_and(pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()

    dice     = 2*tp / (2*tp + fp + fn + 1e-8)
    acc      = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens     = tp / (tp + fn + 1e-8)
    spec     = tn / (tn + fp + 1e-8)
    return dict(Dice=dice, Accuracy=acc, Sensitivity=sens, Specificity=spec)


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------
def evaluate(model, dataloader, cfg: EvalConfig):
    sm = StructuralMetrics(
        min_component_px=cfg.MIN_COMP_PX,
        sparse_threshold=cfg.SPARSE_THRESH,
        patch_size=cfg.PATCH_SIZE,
    )
    gm = VesselGraphMetrics(min_branch_px=cfg.MIN_BRANCH_PX)

    all_metrics = []
    all_probs, all_gts = [], []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            imgs   = batch["image"].to(cfg.DEVICE)
            masks  = batch["mask"].cpu().numpy().squeeze()   # (H, W)

            output = model(imgs)
            if isinstance(output, (list, tuple)):
                output = output[0]  # deep supervision: take main output

            prob = torch.sigmoid(output).cpu().numpy().squeeze()  # (H, W)
            pred = (prob >= cfg.THRESHOLD).astype(np.uint8)
            gt   = (masks > 0).astype(np.uint8)

            all_probs.append(prob.ravel())
            all_gts.append(gt.ravel())

            std = standard_metrics(pred.astype(bool), gt.astype(bool))
            str_m = sm.compute_all(pred, gt)
            grp_m = gm.compute_all(pred, gt)

            row = {**std, **str_m, **grp_m}
            all_metrics.append(row)

    # AUC-ROC over full test set
    probs_flat = np.concatenate(all_probs)
    gts_flat   = np.concatenate(all_gts)
    try:
        auc = roc_auc_score(gts_flat, probs_flat)
    except Exception:
        auc = float('nan')

    # Aggregate means
    keys = list(all_metrics[0].keys())
    means = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    means["AUC_ROC"] = auc

    return means


# ------------------------------------------------------------------
# Pretty-print results
# ------------------------------------------------------------------
def print_results(results: dict, label: str = "Experiment"):
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  {'STANDARD METRICS':}")
    print(f"  {'─'*40}")
    for k in ["Dice", "Accuracy", "Sensitivity", "Specificity", "AUC_ROC"]:
        v = results.get(k, float('nan'))
        print(f"  {k:<22}: {v*100:6.2f}%")
    print(f"\n  {'STRUCTURAL METRICS (Paper Table 2)':}")
    print(f"  {'─'*40}")
    for k, fmt in [
        ("CCA",         "{:6.3f} (↑ higher=better)"),
        ("BFR",         "{:+6.3f} (→ 0 is ideal)"),
        ("SVD",         "{:6.3f} (↑ higher=better)"),
        ("SkelDice",    "{:6.3f} (↑ higher=better)"),
        ("SkelHD",      "{:6.2f} px (↓ lower=better)"),
        ("BranchDiff",  "{:6.0f} (↓ lower=better)"),
        ("Betti0Err",   "{:6.0f} (↓ lower=better)"),
        ("BPR",         "{:6.3f} (↑ higher=better)"),
        ("JPR",         "{:6.3f} (↑ higher=better)"),
        ("GED_approx",  "{:6.1f} (↓ lower=better)"),
    ]:
        v = results.get(k, float('nan'))
        try:
            print(f"  {k:<22}: " + fmt.format(v))
        except Exception:
            print(f"  {k:<22}: {v}")
    print(sep)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from scripts.dataloader_unetpp import create_data_loaders
    from models.unetpp import UNetPlusPlus

    cfg = EvalConfig()

    print(f"Loading model from {cfg.MODEL_PATH} ...")
    model = UNetPlusPlus(in_channels=3, out_channels=1).to(cfg.DEVICE)
    ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print("Model loaded.")

    _, test_loader = create_data_loaders(cfg.DATA_ROOT, cfg.BATCH_SIZE)

    print("Running evaluation ...")
    results = evaluate(model, test_loader, cfg)
    print_results(results, label="Evaluation Results")

    # Save to JSON for later comparison across experiments
    out_path = os.path.join(cfg.RESULTS_DIR, "structural_eval.json")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
