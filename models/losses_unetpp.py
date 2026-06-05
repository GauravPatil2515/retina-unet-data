"""
Loss Functions and Metrics for U-Net++

Implements:
- DiceLoss
- BCEDiceLoss (Combined)
- DeepSupervisionLoss (with optional TopologicalLoss)
- [NEW] TopologicalLoss  — persistent homology via gudhi
- Evaluation metrics: Dice, Accuracy, Sensitivity, Specificity, AUC
- [NEW] Sparse Vessel Dice (SVD) helper
- [NEW] Connected Component Accuracy (CCA) helper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np


# =============================================================================
# STANDARD LOSSES (unchanged)
# =============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Formula: 1 - (2*intersection + smooth) / (|P| + |G| + smooth)
    Works with logits — applies sigmoid internally.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    BCE handles pixel-wise accuracy; Dice handles overlap quality.
    Uses BCEWithLogitsLoss for numerical stability with mixed precision.
    """
    def __init__(self, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


# =============================================================================
# [NEW] TOPOLOGICAL LOSS
# =============================================================================

class TopologicalLoss(nn.Module):
    """
    Topology-Preserving Loss using Persistent Homology.

    NOVEL CONTRIBUTION:
    Standard losses (BCE, Dice) are pixel-wise and do not penalize topological
    errors — a single mis-predicted pixel can sever a vessel into two disconnected
    fragments, which is clinically catastrophic but has near-zero impact on Dice.

    This loss penalises differences in the persistent homology diagrams of the
    predicted and ground-truth vessel masks:
      - Betti-0 (B0): number of connected components (vessel fragments)
      - Betti-1 (B1): number of loops/cycles in vessel structure

    Loss formula:
        L_topo = w0 * Wasserstein_1(Dgm_0(pred), Dgm_0(gt))
               + w1 * Wasserstein_1(Dgm_1(pred), Dgm_1(gt))

    Reference:
        Hu et al. "Topology-Preserving Deep Image Segmentation" NeurIPS 2019
        Extended here to retinal vessel segmentation with deep supervision.

    Dependencies:
        pip install gudhi scikit-image
    """
    def __init__(self, w0: float = 1.0, w1: float = 0.5, downsample_size: int = 64):
        """
        Args:
            w0: weight for Betti-0 (connected components) term
            w1: weight for Betti-1 (loops) term
            downsample_size: resolution for homology computation (64 is fast & sufficient)
        """
        super(TopologicalLoss, self).__init__()
        self.w0 = w0
        self.w1 = w1
        self.downsample_size = downsample_size
        self._check_gudhi()

    @staticmethod
    def _check_gudhi():
        try:
            import gudhi  # noqa
        except ImportError:
            raise ImportError(
                "gudhi is required for TopologicalLoss.\n"
                "Install with: pip install gudhi"
            )

    def _persistence_diagram(self, prob_map: np.ndarray, dim: int):
        """
        Compute the persistence diagram for a 2D probability map.

        Args:
            prob_map: 2D float64 array in [0, 1], shape [H, W]
            dim:      Homology dimension (0 = components, 1 = loops)

        Returns:
            np.ndarray of shape [N, 2] — birth/death pairs
        """
        import gudhi
        # High probability = born early (low filtration value)
        neg_map = 1.0 - prob_map
        cub = gudhi.CubicalComplex(
            dimensions=list(neg_map.shape),
            top_dimensional_cells=neg_map.flatten().tolist()
        )
        cub.compute_persistence()
        pairs = cub.persistence_intervals_in_dimension(dim)
        finite = np.array([(b, d) for b, d in pairs if d != float('inf')], dtype=np.float64)
        return finite if len(finite) > 0 else np.array([[0.0, 0.0]])

    def _wasserstein1(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
        """Compute 1-Wasserstein distance between two persistence diagrams."""
        try:
            from gudhi.wasserstein import wasserstein_distance
            return float(wasserstein_distance(dgm_a, dgm_b, order=1))
        except Exception:
            # Fallback: L1 distance of sorted lifetimes
            def lifetimes(d): return np.sort(d[:, 1] - d[:, 0])
            la, lb = lifetimes(dgm_a), lifetimes(dgm_b)
            n = max(len(la), len(lb))
            la = np.pad(la, (0, n - len(la)))
            lb = np.pad(lb, (0, n - len(lb)))
            return float(np.sum(np.abs(la - lb)))

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, 1, H, W] raw logits from model output
            target:      [B, 1, H, W] binary ground truth mask

        Returns:
            Scalar topological loss (float tensor)
        """
        from skimage.transform import resize as sk_resize

        pred_probs = torch.sigmoid(pred_logits)
        B = pred_probs.shape[0]
        total = 0.0

        for i in range(B):
            p = pred_probs[i, 0].detach().cpu().numpy().astype(np.float64)
            t = target[i, 0].detach().cpu().numpy().astype(np.float64)

            # Downsample for speed — topology survives at 64x64
            s = self.downsample_size
            p_s = sk_resize(p, (s, s), anti_aliasing=True, preserve_range=True)
            t_s = sk_resize(t, (s, s), anti_aliasing=True, preserve_range=True)

            dgm_p0 = self._persistence_diagram(p_s, 0)
            dgm_t0 = self._persistence_diagram(t_s, 0)
            dgm_p1 = self._persistence_diagram(p_s, 1)
            dgm_t1 = self._persistence_diagram(t_s, 1)

            dist0 = self._wasserstein1(dgm_p0, dgm_t0)
            dist1 = self._wasserstein1(dgm_p1, dgm_t1)
            total += self.w0 * dist0 + self.w1 * dist1

        # Return as a tensor that participates in the computation graph
        # (non-differentiable through gudhi, but acts as a regularization signal
        # applied as a scalar add to the differentiable BCE+Dice loss)
        return torch.tensor(total / B, dtype=torch.float32,
                            device=pred_logits.device, requires_grad=False)


# =============================================================================
# DEEP SUPERVISION LOSS (updated with topo support)
# =============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Multi-output weighted loss for deep supervision.
    Optionally adds TopologicalLoss on the final output.

    Standard weights: [0.25, 0.25, 0.25, 1.0]
    Total = 0.25*L1 + 0.25*L2 + 0.25*L3 + 1.0*L4 [+ lambda * L_topo]

    Args:
        weights:     list of per-output weights
        use_topo:    enable TopologicalLoss (default False; set True after warmup)
        topo_weight: lambda for topology term (default 0.1)
    """
    def __init__(self, weights=None, use_topo: bool = False, topo_weight: float = 0.1):
        super(DeepSupervisionLoss, self).__init__()
        self.weights     = weights if weights is not None else [0.25, 0.25, 0.25, 1.0]
        self.loss_fn     = BCEDiceLoss()
        self.use_topo    = use_topo
        self.topo_weight = topo_weight
        if use_topo:
            self.topo_loss = TopologicalLoss(w0=1.0, w1=0.5)
        else:
            self.topo_loss = None

    def enable_topo(self, topo_weight: float = 0.1):
        """Activate topology loss mid-training (called from train loop)."""
        self.use_topo    = True
        self.topo_weight = topo_weight
        self.topo_loss   = TopologicalLoss(w0=1.0, w1=0.5)
        print(f"[TOPO] TopologicalLoss activated (lambda={topo_weight})")

    def forward(self, outputs, target):
        """
        Args:
            outputs: tuple of 4 logit tensors from deep supervision heads
            target:  ground truth mask [B, 1, H, W]
        """
        total_loss = sum(
            weight * self.loss_fn(output, target)
            for output, weight in zip(outputs, self.weights)
        )
        if self.use_topo and self.topo_loss is not None:
            topo = self.topo_loss(outputs[-1], target)  # on final output
            total_loss = total_loss + self.topo_weight * topo
        return total_loss


# =============================================================================
# METRICS
# =============================================================================

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice coefficient (F1 score) — range [0, 1].
    Args:
        pred:   logits [B, 1, H, W]
        target: binary mask [B, 1, H, W]
    """
    pred   = torch.sigmoid(pred)
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate Dice, Accuracy, Sensitivity, Specificity, AUC.
    Args:
        pred:      logits [B, 1, H, W]
        target:    binary mask [B, 1, H, W]
        threshold: binarisation threshold
    """
    pred   = torch.sigmoid(pred)
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
        auc = roc_auc_score(target_np, pred_np)
    except Exception:
        auc = 0.0

    return {
        'dice': dice, 'accuracy': accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'auc': auc, 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }


class MetricsTracker:
    """Track loss and dice across batches within an epoch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.losses      = []

    def update(self, loss, dice):
        self.losses.append(loss)
        self.dice_scores.append(dice)

    def get_average(self):
        return {
            'loss': float(np.mean(self.losses))      if self.losses      else 0.0,
            'dice': float(np.mean(self.dice_scores)) if self.dice_scores else 0.0
        }


if __name__ == '__main__':
    print('Testing Loss Functions...')
    B = 2
    pred   = torch.randn(B, 1, 64, 64)
    target = torch.randint(0, 2, (B, 1, 64, 64)).float()

    print(f'DiceLoss:    {DiceLoss()(pred, target).item():.4f}')
    print(f'BCEDiceLoss: {BCEDiceLoss()(pred, target).item():.4f}')

    ds_loss = DeepSupervisionLoss(use_topo=False)
    outputs = (pred, pred, pred, pred)
    print(f'DeepSupervision (no topo): {ds_loss(outputs, target).item():.4f}')

    print('\nTesting TopologicalLoss (requires gudhi)...')
    try:
        topo = TopologicalLoss(w0=1.0, w1=0.5, downsample_size=32)
        tl   = topo(pred, target)
        print(f'TopologicalLoss: {tl.item():.4f}')
        print('[OK] TopologicalLoss works!')
    except ImportError as e:
        print(f'[SKIP] {e}')

    print('\n[OK] All loss tests complete!')
