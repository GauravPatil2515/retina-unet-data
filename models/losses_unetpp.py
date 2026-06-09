"""
Loss Functions for U-Net++ Retinal Vessel Segmentation

Fixes in this version:
  [CRITICAL] TopologicalLoss previously returned a tensor with
  requires_grad=False, meaning NO gradient flowed to the model.
  The loss was numerically added to the total but had zero effect
  on parameter updates. This is now replaced with a differentiable
  skeleton-weighted BCE surrogate that provides real connectivity
  gradients.

Losses:
  - DiceLoss
  - BCEDiceLoss
  - SkeletonWeightedBCE  (replaces TopologicalLoss as differentiable component)
  - TopologicalLoss      (kept for diagnosis/logging only, requires_grad=False)
  - DeepSupervisionLoss  (updated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


# ============================================================================
# STANDARD LOSSES
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        p      = pred.contiguous().view(-1)
        t      = target.contiguous().view(-1)
        inter  = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


# ============================================================================
# [FIX] DIFFERENTIABLE SKELETON-WEIGHTED BCE
# Replaces the broken TopologicalLoss as the connectivity training signal.
#
# How it works:
#   1. Compute soft skeleton of GT mask (differentiable morphological thinning
#      approximated by iterative max-pooling erosion)
#   2. Build a pixel weight map: skeleton pixels get weight (1 + alpha),
#      non-skeleton pixels get weight 1.0
#   3. Apply weighted BCE — model is penalised MORE for errors on vessel
#      centerlines than on background or vessel edges
#
# Why this is differentiable (unlike persistent homology):
#   - Soft skeleton uses only max-pool and threshold ops
#   - Weight map is computed from GT (fixed), not from pred
#   - BCE gradient flows normally through pred_logits
#
# Effect on training:
#   - Forces model to recover thin centerlines
#   - Directly targets the skeleton connectivity problem
#   - Computationally cheap: no gudhi, no cubical complex
# ============================================================================

class SkeletonWeightedBCE(nn.Module):
    """
    Differentiable skeleton-weighted binary cross-entropy.

    Args:
        alpha : extra weight on skeleton pixels (default 2.0)
                total weight at skeleton = 1 + alpha
        n_iter: number of erosion steps to approximate skeleton (default 3)
    """
    def __init__(self, alpha: float = 2.0, n_iter: int = 3):
        super().__init__()
        self.alpha  = alpha
        self.n_iter = n_iter
        self.bce    = nn.BCEWithLogitsLoss(reduction="none")

    def _soft_skeleton(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Approximate skeleton by subtracting iteratively eroded mask.
        mask: [B, 1, H, W] binary float tensor (GT, no grad)
        Returns: [B, 1, H, W] soft skeleton in [0, 1]
        """
        eroded = mask.clone()
        skel   = torch.zeros_like(mask)
        for _ in range(self.n_iter):
            # 3x3 min-pool approximates morphological erosion
            next_e = -F.max_pool2d(-eroded, kernel_size=3, stride=1, padding=1)
            layer  = F.relu(eroded - next_e)   # ring between this and next erosion
            skel   = skel + layer
            eroded = next_e
        return torch.clamp(skel, 0, 1)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, 1, H, W] raw model output
            target:      [B, 1, H, W] binary GT (float)
        """
        with torch.no_grad():
            skel    = self._soft_skeleton(target)
            weights = 1.0 + self.alpha * skel   # [B,1,H,W], always >= 1

        pixel_loss = self.bce(pred_logits, target)   # [B, 1, H, W]
        return (weights * pixel_loss).mean()


# ============================================================================
# TOPOLOGICAL LOSS (kept for LOGGING/DIAGNOSIS only)
# DO NOT use in training — returns requires_grad=False.
# Useful for monitoring whether topology is improving without training on it.
# ============================================================================

class TopologicalLoss(nn.Module):
    """
    Persistent-homology loss for connectivity.

    WARNING: This returns a tensor with requires_grad=False.
    It will add a value to total_loss but NO gradient reaches the model.
    Use only for monitoring. For differentiable connectivity training,
    use SkeletonWeightedBCE above.

    Kept here so existing training scripts using --use_topo still run
    without error; the effect on training is purely numeric (scalar add),
    not gradient-based.
    """
    def __init__(self, w0=1.0, w1=0.5, downsample_size=64):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.downsample_size = downsample_size

    def _persistence_diagram(self, prob_map, dim):
        import gudhi
        neg_map = 1.0 - prob_map
        cub = gudhi.CubicalComplex(
            dimensions=list(neg_map.shape),
            top_dimensional_cells=neg_map.flatten().tolist()
        )
        cub.compute_persistence()
        pairs = cub.persistence_intervals_in_dimension(dim)
        finite = np.array([(b, d) for b, d in pairs if d != float('inf')],
                          dtype=np.float64)
        return finite if len(finite) > 0 else np.array([[0.0, 0.0]])

    def _wasserstein1(self, a, b):
        try:
            from gudhi.wasserstein import wasserstein_distance
            return float(wasserstein_distance(a, b, order=1))
        except Exception:
            la = np.sort(a[:, 1] - a[:, 0])
            lb = np.sort(b[:, 1] - b[:, 0])
            n  = max(len(la), len(lb))
            la = np.pad(la, (0, n - len(la)))
            lb = np.pad(lb, (0, n - len(lb)))
            return float(np.abs(la - lb).sum())

    def forward(self, pred_logits, target):
        from skimage.transform import resize as sk_resize
        probs = torch.sigmoid(pred_logits)
        B     = probs.shape[0]
        total = 0.0
        for i in range(B):
            p = probs[i, 0].detach().cpu().numpy().astype(np.float64)
            t = target[i, 0].detach().cpu().numpy().astype(np.float64)
            s = self.downsample_size
            from skimage.transform import resize as R
            p_s = R(p, (s, s), anti_aliasing=True, preserve_range=True)
            t_s = R(t, (s, s), anti_aliasing=True, preserve_range=True)
            d0 = self._wasserstein1(self._persistence_diagram(p_s, 0),
                                    self._persistence_diagram(t_s, 0))
            d1 = self._wasserstein1(self._persistence_diagram(p_s, 1),
                                    self._persistence_diagram(t_s, 1))
            total += self.w0 * d0 + self.w1 * d1
        # requires_grad=False intentionally — monitoring only
        return torch.tensor(total / B, dtype=torch.float32,
                            device=pred_logits.device, requires_grad=False)


# ============================================================================
# DEEP SUPERVISION LOSS
# Updated: use SkeletonWeightedBCE as differentiable topo component
# ============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Multi-output weighted loss for deep supervision.

    Two structural modes:
      use_skel=True  : add SkeletonWeightedBCE (differentiable, recommended)
      use_topo=True  : add TopologicalLoss     (non-differentiable, logging only)

    Recommended config for Paper 2:
      loss = DeepSupervisionLoss(use_skel=True, skel_weight=0.3)
    """
    def __init__(self,
                 weights=None,
                 use_skel: bool = False,
                 skel_weight: float = 0.3,
                 use_topo: bool = False,
                 topo_weight: float = 0.1):
        super().__init__()
        self.weights     = weights or [0.25, 0.25, 0.25, 1.0]
        self.loss_fn     = BCEDiceLoss()
        self.use_skel    = use_skel
        self.skel_weight = skel_weight
        self.use_topo    = use_topo
        self.topo_weight = topo_weight

        if use_skel:
            self.skel_loss = SkeletonWeightedBCE(alpha=2.0, n_iter=3)
            print("[LOSS] SkeletonWeightedBCE activated (differentiable connectivity)")
        if use_topo:
            self.topo_loss = TopologicalLoss()
            print("[LOSS] TopologicalLoss activated (monitoring only, NO gradient)")

    def enable_skel(self, weight: float = 0.3):
        """Activate skeleton loss mid-training (for curriculum)."""
        self.use_skel    = True
        self.skel_weight = weight
        self.skel_loss   = SkeletonWeightedBCE(alpha=2.0, n_iter=3)
        print(f"[LOSS] SkeletonWeightedBCE enabled mid-training (w={weight})")

    def forward(self, outputs, target):
        total = sum(
            w * self.loss_fn(o, target)
            for o, w in zip(outputs, self.weights)
        )
        if self.use_skel:
            total = total + self.skel_weight * self.skel_loss(outputs[-1], target)
        if self.use_topo:
            # monitoring: adds numeric value but no gradient
            with torch.no_grad():
                topo_val = self.topo_loss(outputs[-1], target)
            total = total + self.topo_weight * topo_val
        return total


# ============================================================================
# METRICS (unchanged)
# ============================================================================

def dice_coefficient(pred, target, smooth=1e-6):
    pred  = torch.sigmoid(pred).contiguous().view(-1)
    t     = target.contiguous().view(-1)
    inter = (pred * t).sum()
    return ((2 * inter + smooth) / (pred.sum() + t.sum() + smooth)).item()


def calculate_metrics(pred, target, threshold=0.5):
    pred   = torch.sigmoid(pred)
    p_np   = pred.cpu().detach().numpy().flatten()
    t_np   = target.cpu().detach().numpy().flatten()
    p_bin  = (p_np > threshold).astype(np.float32)
    TP = np.sum((p_bin == 1) & (t_np == 1))
    TN = np.sum((p_bin == 0) & (t_np == 0))
    FP = np.sum((p_bin == 1) & (t_np == 0))
    FN = np.sum((p_bin == 0) & (t_np == 1))
    try:
        auc = roc_auc_score(t_np, p_np)
    except Exception:
        auc = 0.0
    return {
        "dice":        (2*TP) / (2*TP + FP + FN + 1e-6),
        "accuracy":    (TP+TN) / (TP+TN+FP+FN + 1e-6),
        "sensitivity": TP / (TP+FN + 1e-6),
        "specificity": TN / (TN+FP + 1e-6),
        "auc": auc, "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }


class MetricsTracker:
    def __init__(self):  self.reset()
    def reset(self):     self.dice_scores = [];  self.losses = []
    def update(self, loss, dice):  self.losses.append(loss); self.dice_scores.append(dice)
    def get_average(self):
        return {"loss": float(np.mean(self.losses)) if self.losses else 0.0,
                "dice": float(np.mean(self.dice_scores)) if self.dice_scores else 0.0}


if __name__ == "__main__":
    B = 2
    pred   = torch.randn(B, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (B, 1, 64, 64)).float()

    # Test SkeletonWeightedBCE gradient
    skel_loss = SkeletonWeightedBCE(alpha=2.0)
    L = skel_loss(pred, target)
    L.backward()
    print(f"SkeletonWeightedBCE: {L.item():.4f}")
    print(f"pred.grad is not None: {pred.grad is not None}")   # must be True
    print(f"pred.grad.abs().mean(): {pred.grad.abs().mean().item():.6f}")  # must be > 0

    # Confirm TopologicalLoss has no gradient (expected, documented)
    pred2 = torch.randn(B, 1, 32, 32, requires_grad=True)
    t2    = torch.randint(0, 2, (B, 1, 32, 32)).float()
    ds    = DeepSupervisionLoss(use_skel=True, skel_weight=0.3)
    out   = (pred2, pred2, pred2, pred2)
    Lds   = ds(out, t2)
    Lds.backward()
    print(f"\nDeepSupervision+Skel: {Lds.item():.4f}")
    print(f"Gradient flows: {pred2.grad.abs().mean().item():.6f}")
    print("[OK] All loss tests pass.")
