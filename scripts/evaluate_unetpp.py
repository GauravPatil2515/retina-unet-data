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
    DATA_ROOT    = "Retina"
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
def evaluate(model, dataloader, cfg: EvalConfig, save_preds: bool = False, model_name: str = "UNetPP", dataset: str = "DRIVE"):
    from PIL import Image
    sm = StructuralMetrics(
        min_component_px=cfg.MIN_COMP_PX,
        sparse_threshold=cfg.SPARSE_THRESH,
        patch_size=cfg.PATCH_SIZE,
    )
    # Corrected instantiation: min_branch_px should be min_branch_px=cfg.MIN_BRANCH_PX
    gm = VesselGraphMetrics(min_branch_px=cfg.MIN_BRANCH_PX)

    all_metrics = []
    all_probs, all_gts = [], []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                imgs   = batch["image"].to(cfg.DEVICE)
                masks  = batch["mask"].cpu().numpy().squeeze()   # (H, W)
                img_path = batch["path"][0] if isinstance(batch["path"], list) else batch["path"]
                fname = os.path.basename(img_path)
            else:
                imgs, masks = batch
                imgs   = imgs.to(cfg.DEVICE)
                masks  = masks.cpu().numpy().squeeze()   # (H, W)
                fname  = dataloader.dataset.image_files[idx]
            fov_dir = os.path.join(cfg.DATA_ROOT.rstrip("/"), "test", "fov")
            fov_path = os.path.join(fov_dir, fname)
            if os.path.exists(fov_path):
                fov = np.array(Image.open(fov_path).convert("L")) > 127
            else:
                fov = None

            output = model(imgs)
            if isinstance(output, (list, tuple)):
                output = output[0]  # deep supervision: take main output

            prob = torch.sigmoid(output).cpu().numpy().squeeze()  # (H, W)
            pred = (prob >= cfg.THRESHOLD).astype(np.uint8)
            gt   = (masks > 0).astype(np.uint8)
            if fov is not None:
                pred = pred * fov
                gt   = gt   * fov

            if save_preds:
                pred_dir = f"results/predictions/{model_name}/{dataset}"
                os.makedirs(pred_dir, exist_ok=True)
                pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                pred_img.save(os.path.join(pred_dir, fname))

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
    import argparse
    from scripts.dataloader_unetpp import create_data_loaders
    from models.unet_plus_plus import UNetPlusPlus

    parser = argparse.ArgumentParser(description="Evaluate UNet++ model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--save_preds", action="store_true", help="Save prediction PNGs")
    parser.add_argument("--model_name", type=str, default="UNetPP", help="Model name for saving predictions")
    parser.add_argument("--dataset", type=str, default="DRIVE", help="Dataset name for saving predictions")
    args = parser.parse_args()

    cfg = EvalConfig()
    if args.model_path:
        cfg.MODEL_PATH = args.model_path
    if args.results_dir:
        cfg.RESULTS_DIR = args.results_dir

    print(f"Loading model from {cfg.MODEL_PATH} ...")
    model = UNetPlusPlus(in_channels=3, out_channels=1).to(cfg.DEVICE)
    ckpt = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print("Model loaded.")

    from scripts.dataloader_unetpp import FullImageDataset
    test_dataset = FullImageDataset(
        os.path.join(cfg.DATA_ROOT, "test", "image"),
        os.path.join(cfg.DATA_ROOT, "test", "mask")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Running evaluation ...")
    results = evaluate(
        model, test_loader, cfg,
        save_preds=args.save_preds,
        model_name=args.model_name,
        dataset=args.dataset
    )
    print_results(results, label=f"Evaluation Results ({os.path.basename(os.path.dirname(os.path.dirname(cfg.MODEL_PATH)))})")

    # Save to JSON for later comparison across experiments
    out_path = os.path.join(cfg.RESULTS_DIR, "structural_eval.json")
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
