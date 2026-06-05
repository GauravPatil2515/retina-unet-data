"""
Evaluation Script for U-Net++ — Research Edition

New metrics added:
  - Sparse Vessel Dice (SVD): Dice computed only on image sub-regions
    where vessel density < sparse_threshold (default 0.02).
    Exposes model failure on thin capillaries and peripheral vessels.

  - Connected Component Accuracy (CCA): fraction of predicted vessel
    components that have IoU > 0.5 with a ground-truth component.
    Topological metric — directly measures vessel connectivity preservation.

  - Mean Betti-0 Error: |n_components_pred - n_components_gt| averaged
    over test images. Lower = better topology.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plus_plus import UNetPlusPlus
from scripts.dataloader_unetpp import FullImageDataset
from models.losses_unetpp import calculate_metrics


# =============================================================================
# PATCH-RECONSTRUCT (unchanged)
# =============================================================================

def reconstruct_from_patches(image, model, device, patch_size=128, stride=64):
    C, H, W = image.shape
    recon   = torch.zeros(1, H, W)
    count   = torch.zeros(1, H, W)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(patch)
                pred = torch.sigmoid(out[-1] if isinstance(out, tuple) else out)
            pred = pred.squeeze(0).cpu()
            recon[:, y:y+patch_size, x:x+patch_size] += pred
            count[:, y:y+patch_size, x:x+patch_size] += 1
    return recon / (count + 1e-6)


# =============================================================================
# [NEW] SPARSE VESSEL DICE (SVD)
# =============================================================================

def calculate_sparse_vessel_dice(predictions, patch_size=128, stride=64,
                                  sparse_threshold=0.02):
    """
    Sparse Vessel Dice (SVD) — Novel metric.

    Computes Dice ONLY over patches where ground-truth vessel_ratio < sparse_threshold.
    This measures model performance on thin capillaries and peripheral retina —
    the regions discarded by standard hard patch-filtering during training.

    Args:
        predictions:       list of dicts with keys 'image', 'mask', 'pred'
        patch_size:        patch size for sub-region evaluation
        stride:            stride for sub-region tiling
        sparse_threshold:  vessel_ratio upper bound for 'sparse' classification

    Returns:
        svd:             float [0,1] — higher is better
        n_sparse_patches: int — number of sparse patches evaluated
    """
    sparse_tp = sparse_fp = sparse_fn = 0
    n_sparse  = 0

    for item in predictions:
        mask = item['mask'][0]   # [H, W] float, ground truth
        pred = item['pred'][0]   # [H, W] float, probabilities
        H, W = mask.shape

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                m_patch = mask[y:y+patch_size, x:x+patch_size]
                p_patch = pred[y:y+patch_size, x:x+patch_size]

                vessel_ratio = m_patch.sum() / (patch_size ** 2)
                if vessel_ratio >= sparse_threshold:
                    continue  # skip dense patches

                p_bin = (p_patch > 0.5).astype(np.float32)
                sparse_tp += int(np.sum((p_bin == 1) & (m_patch == 1)))
                sparse_fp += int(np.sum((p_bin == 1) & (m_patch == 0)))
                sparse_fn += int(np.sum((p_bin == 0) & (m_patch == 1)))
                n_sparse  += 1

    svd = (2 * sparse_tp) / (2 * sparse_tp + sparse_fp + sparse_fn + 1e-6)
    return float(svd), n_sparse


# =============================================================================
# [NEW] CONNECTED COMPONENT ACCURACY (CCA)
# =============================================================================

def calculate_connected_component_accuracy(predictions, threshold=0.5, iou_thresh=0.5):
    """
    Connected Component Accuracy (CCA) — Novel topological metric.

    For each predicted vessel component, checks if it matches a ground-truth
    component with IoU >= iou_thresh. Reports fraction of predicted components
    that are 'correct'. Also reports mean Betti-0 error.

    A topological break (severed vessel) appears as a False Positive component
    in the prediction with NO matching GT component — directly measurable here.

    Args:
        predictions:  list of dicts with 'mask' and 'pred'
        threshold:    probability threshold for binarising predictions
        iou_thresh:   IoU above which a predicted component counts as correct

    Returns:
        cca:           float [0,1] — fraction of correct predicted components
        betti0_error:  float — mean |n_pred_components - n_gt_components|
    """
    cca_scores    = []
    betti_errors  = []

    for item in predictions:
        mask = item['mask'][0].astype(np.uint8)  # [H, W]
        pred = (item['pred'][0] > threshold).astype(np.uint8)

        pred_labeled, n_pred = ndimage.label(pred)
        gt_labeled,   n_gt   = ndimage.label(mask)

        betti_errors.append(abs(n_pred - n_gt))

        if n_pred == 0:
            cca_scores.append(1.0 if n_gt == 0 else 0.0)
            continue

        matched = 0
        for comp_id in range(1, n_pred + 1):
            comp_mask     = (pred_labeled == comp_id)
            overlap_vals  = gt_labeled[comp_mask]
            fg_vals       = overlap_vals[overlap_vals > 0]
            if len(fg_vals) == 0:
                continue  # no overlap with any GT component
            best_gt_id = np.bincount(fg_vals).argmax()
            gt_comp    = (gt_labeled == best_gt_id)
            iou        = (comp_mask & gt_comp).sum() / ((comp_mask | gt_comp).sum() + 1e-6)
            if iou >= iou_thresh:
                matched += 1

        cca_scores.append(matched / n_pred)

    return float(np.mean(cca_scores)), float(np.mean(betti_errors))


# =============================================================================
# STANDARD METRICS (unchanged from original)
# =============================================================================

def calculate_metrics_from_probs(pred, target, threshold=0.5):
    pred_np   = pred.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    pred_bin  = (pred_np > threshold).astype(np.float32)
    TP = np.sum((pred_bin == 1) & (target_np == 1))
    TN = np.sum((pred_bin == 0) & (target_np == 0))
    FP = np.sum((pred_bin == 1) & (target_np == 0))
    FN = np.sum((pred_bin == 0) & (target_np == 1))
    dice        = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(target_np, pred_np)
    except Exception:
        auc = 0.0
    return {'dice': dice, 'accuracy': accuracy, 'sensitivity': sensitivity,
            'specificity': specificity, 'auc': auc,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_model(model_path, test_img_dir, test_mask_dir, device,
                   save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    print('\n[LOAD] Loading trained model...')
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[OK] Loaded epoch {ckpt['epoch']}  val_dice={ckpt['metrics'].get('dice','N/A')}")

    # Load test set
    test_dataset = FullImageDataset(test_img_dir, test_mask_dir)
    print(f'[OK] Test images: {len(test_dataset)}')

    all_metrics  = []
    predictions  = []

    print('\n[EVAL] Running evaluation...')
    for idx in tqdm(range(len(test_dataset))):
        image, mask = test_dataset[idx]
        pred        = reconstruct_from_patches(image, model, device)

        pred_t = pred.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        m      = calculate_metrics_from_probs(pred_t, mask_t)
        all_metrics.append(m)

        predictions.append({
            'image': image.cpu().numpy(),
            'mask':  mask.cpu().numpy(),
            'pred':  pred.cpu().numpy()
        })

    # Standard metrics
    avg = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in ['dice','accuracy','sensitivity','specificity','auc']
    }

    # [NEW] Sparse Vessel Dice
    svd, n_sparse = calculate_sparse_vessel_dice(predictions, sparse_threshold=0.02)

    # [NEW] Connected Component Accuracy
    cca, betti_err = calculate_connected_component_accuracy(predictions)

    # Print results
    print('\n' + '='*80)
    print('EVALUATION RESULTS')
    print('='*80)
    print(f"  Dice Coefficient : {avg['dice']*100:.2f}%")
    print(f"  Accuracy         : {avg['accuracy']*100:.2f}%")
    print(f"  Sensitivity      : {avg['sensitivity']*100:.2f}%")
    print(f"  Specificity      : {avg['specificity']*100:.2f}%")
    print(f"  AUC-ROC          : {avg['auc']*100:.2f}%")
    print(f"  --- Novel Metrics ---")
    print(f"  Sparse Vessel Dice (SVD)  : {svd*100:.2f}%  (over {n_sparse} sparse patches)")
    print(f"  Connected Component Acc   : {cca*100:.2f}%")
    print(f"  Mean Betti-0 Error        : {betti_err:.2f}")
    print('='*80)

    # Save
    import json
    def to_py(obj):
        if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [to_py(i) for i in obj]
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out = {
        'average': avg,
        'sparse_vessel_dice': svd,
        'n_sparse_patches': n_sparse,
        'connected_component_accuracy': cca,
        'mean_betti0_error': betti_err,
        'per_image': all_metrics
    }
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(to_py(out), f, indent=4)

    # Visualise first 5
    visualize_predictions(predictions[:5], save_dir)
    print(f'[OK] Results saved to: {save_dir}')
    return avg, svd, cca, betti_err


def visualize_predictions(predictions, save_dir):
    for idx, d in enumerate(predictions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(d['image'].transpose(1,2,0));  axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(d['mask'][0], cmap='gray');    axes[1].set_title('Ground Truth'); axes[1].axis('off')
        axes[2].imshow(d['pred'][0], cmap='gray');    axes[2].set_title('Prediction'); axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    print(f'[OK] Saved {len(predictions)} visualizations')


if __name__ == '__main__':
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CKPT_PATH  = os.path.join(BASE_DIR, 'results', 'checkpoints_unetpp', 'best.pth')
    DATA_ROOT  = os.path.join(BASE_DIR, 'Retina')
    DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {DEVICE}')
    if os.path.exists(CKPT_PATH):
        evaluate_model(
            CKPT_PATH,
            os.path.join(DATA_ROOT, 'test', 'image'),
            os.path.join(DATA_ROOT, 'test', 'mask'),
            DEVICE,
            save_dir=os.path.join(BASE_DIR, 'results', 'evaluation_results_unetpp')
        )
    else:
        print(f'[ERROR] Checkpoint not found: {CKPT_PATH}')
        print('[INFO] Train first: python scripts/train_unetpp.py')
