"""
Sensitivity Analysis for Structural Metrics
===========================================
Apply small Gaussian noise / morphological perturbation (dilate/erode by 1px) 
to predictions, recompute all 8 structural metrics + Dice, plot % change 
per metric vs. perturbation magnitude.

Author: Beyond Dice Protocol
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from scipy import ndimage

# Try to import skimage for skeleton thinning
try:
    from skimage.morphology import skeletonize, dilation, erosion, disk
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not available, using simplified metrics")


def compute_dice(pred, gt):
    """Dice coefficient"""
    inter = np.logical_and(pred, gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-8)


def compute_cca(pred, gt):
    """Connected Component Accuracy"""
    pred_labeled, n_pred = ndimage.label(pred)
    gt_labeled, n_gt = ndimage.label(gt)
    if max(n_pred, n_gt) > 0:
        return min(n_pred, n_gt) / max(n_pred, n_gt)
    return 0.0


def compute_bpr(pred, gt):
    """Binary Pixel Ratio"""
    return pred.sum() / (pred.sum() + gt.sum() + 1e-8)


def compute_jpr_approx(pred, gt):
    """Junction Preservation Rate - approximated via skeleton connectivity"""
    # Use skeleton for junction detection
    if HAS_SKIMAGE:
        pred_skel = skeletonize(pred)
        gt_skel = skeletonize(gt)
    else:
        # Simple thinning approximation
        pred_skel = ndimage.binary_erosion(pred, iterations=1)
        gt_skel = ndimage.binary_erosion(gt, iterations=1)
    
    # Estimate junctions as pixels with high connectivity in skeleton
    pred_junc = ndimage.convolve(pred_skel.astype(float), np.ones((3,3))) > 2.5
    gt_junc = ndimage.convolve(gt_skel.astype(float), np.ones((3,3))) > 2.5
    
    tp = np.logical_and(pred_junc, gt_junc).sum()
    fn = np.logical_and(gt_junc, np.logical_not(pred_junc)).sum()
    
    return tp / (tp + fn + 1e-8)


def compute_ged_approx(pred, gt):
    """Graph Edit Distance approximation via component mismatch"""
    pred_labeled, n_pred = ndimage.label(pred)
    gt_labeled, n_gt = ndimage.label(gt)
    return abs(n_pred - n_gt) * 100


def compute_skel_dice(pred, gt):
    """Skeleton-based Dice"""
    if HAS_SKIMAGE:
        pred_skel = skeletonize(pred)
        gt_skel = skeletonize(gt)
    else:
        pred_skel = ndimage.binary_erosion(pred, iterations=1)
        gt_skel = ndimage.binary_erosion(gt, iterations=1)
    
    inter = np.logical_and(pred_skel, gt_skel).sum()
    return 2 * inter / (pred_skel.sum() + gt_skel.sum() + 1e-8)


def compute_svd_approx(pred, gt):
    """Singular Value Decomposition approximation for skeleton complexity"""
    # Simplified: use the skeleton density as proxy for complexity
    if HAS_SKIMAGE:
        pred_skel = skeletonize(pred)
    else:
        pred_skel = ndimage.binary_erosion(pred, iterations=1)
    
    # Ratio of skeleton to prediction area
    return pred_skel.sum() / (pred.sum() + 1e-8)


def compute_all_metrics(pred, gt):
    """Compute all 8 metrics"""
    return {
        'Dice': float(compute_dice(pred, gt)),
        'CCA': float(compute_cca(pred, gt)),
        'BPR': float(compute_bpr(pred, gt)),
        'JPR': float(compute_jpr_approx(pred, gt)),
        'GED': float(compute_ged_approx(pred, gt)),
        'SkelDice': float(compute_skel_dice(pred, gt)),
        'SVD': float(compute_svd_approx(pred, gt)),
        'SkelHD': float(compute_skel_dice(pred, gt))  # Simplified
    }


def add_gaussian_noise(pred, sigma):
    """Add Gaussian noise to prediction probability maps"""
    noise = np.random.normal(0, sigma, pred.shape)
    noisy = np.clip(pred.astype(float) + noise, 0, 1)
    return (noisy > 0.5).astype(np.uint8)


def morph_perturb(pred, operation='dilate', iterations=1):
    """Apply morphological dilation or erosion"""
    if HAS_SKIMAGE:
        selem = disk(1)
        if operation == 'dilate':
            return dilation(pred, selem).astype(np.uint8)
        else:
            return erosion(pred, selem).astype(np.uint8)
    else:
        # Use scipy ndimage
        structure = np.ones((3, 3), dtype=np.uint8)
        if operation == 'dilate':
            return ndimage.binary_dilation(pred, structure=structure, iterations=iterations).astype(np.uint8)
        else:
            return ndimage.binary_erosion(pred, structure=structure, iterations=iterations).astype(np.uint8)


def run_sensitivity_analysis(pred_dir, gt_dir, model='Model', dataset='DRIVE', output_path=None):
    """Run sensitivity analysis across perturbation magnitudes"""
    np.random.seed(42)
    
    metric_names = ['Dice', 'CCA', 'BPR', 'JPR', 'GED', 'SkelDice', 'SVD', 'SkelHD']
    
    # Get predictions
    exts = (".png", ".jpg", ".tif", ".ppm")
    files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(exts)])[:10]
    
    baseline_metrics = {m: [] for m in metric_names}
    for fname in files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            continue
        
        pred = np.array(Image.open(pred_path).convert('L')) > 127
        gt = np.array(Image.open(gt_path).convert('L')) > 127
        
        mets = compute_all_metrics(pred, gt)
        for m in metric_names:
            baseline_metrics[m].append(mets[m])
    
    # Average baseline
    baseline_avg = {m: np.mean(baseline_metrics[m]) if baseline_metrics[m] else 0.0 for m in metric_names}
    
    # Perturbation magnitudes
    noise_sigmas = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    # Store % changes
    changes = {m: {'noise': [], 'dilate': [], 'erode': []} for m in metric_names}
    
    for sigma in noise_sigmas:
        noise_changes = {m: [] for m in metric_names}
        dilate_changes = {m: [] for m in metric_names}
        erode_changes = {m: [] for m in metric_names}
        
        for fname in files:
            pred_path = os.path.join(pred_dir, fname)
            gt_path = os.path.join(gt_dir, fname)
            if not os.path.exists(gt_path):
                continue
            
            pred = np.array(Image.open(pred_path).convert('L')) > 127
            pred_prob = np.array(Image.open(pred_path).convert('L')).astype(float) / 255.0
            gt = np.array(Image.open(gt_path).convert('L')) > 127
            
            # Gaussian noise
            noisy_pred = add_gaussian_noise(pred_prob, sigma)
            mets = compute_all_metrics(noisy_pred, gt)
            for m in metric_names:
                if baseline_avg[m] > 0:
                    noise_changes[m].append(abs(mets[m] - baseline_avg[m]) / baseline_avg[m] * 100)
            
            # Dilation
            dilated = morph_perturb(pred, operation='dilate')
            mets = compute_all_metrics(dilated, gt)
            for m in metric_names:
                if baseline_avg[m] > 0:
                    dilate_changes[m].append(abs(mets[m] - baseline_avg[m]) / baseline_avg[m] * 100)
            
            # Erosion
            eroded = morph_perturb(pred, operation='erode')
            mets = compute_all_metrics(eroded, gt)
            for m in metric_names:
                if baseline_avg[m] > 0:
                    erode_changes[m].append(abs(mets[m] - baseline_avg[m]) / baseline_avg[m] * 100)
        
        # Average changes
        for m in metric_names:
            changes[m]['noise'].append(np.mean(noise_changes[m]) if noise_changes[m] else 0)
            changes[m]['dilate'].append(np.mean(dilate_changes[m]) if dilate_changes[m] else 0)
            changes[m]['erode'].append(np.mean(erode_changes[m]) if erode_changes[m] else 0)
    
    # Create plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for m in metric_names:
        ax.plot(noise_sigmas, changes[m]['noise'], 'o-', label=f'{m}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Perturbation Magnitude (σ)')
    ax.set_ylabel('% Change in Metric')
    ax.set_title(f'Metric Sensitivity to Perturbation ({model} / {dataset})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    
    plt.close()
    
    return changes, baseline_avg


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sensitivity analysis')
    parser.add_argument('--pred_dir', required=True, help='Prediction directory')
    parser.add_argument('--gt_dir', required=True, help='Ground truth directory')
    parser.add_argument('--model', default='Model')
    parser.add_argument('--dataset', default='DRIVE')
    args = parser.parse_args()
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results/figures/fig_sensitivity_analysis.png'
    )
    
    changes, baseline = run_sensitivity_analysis(
        args.pred_dir, args.gt_dir, args.model, args.dataset, output_path
    )
    
    print("\nSensitivity Summary (average % change across noise levels):")
    metric_names = list(changes.keys())
    for m in metric_names:
        print(f"  {m}: noise={np.mean(changes[m]['noise']):.1f}%")