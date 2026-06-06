"""
Structural Metrics for Retinal Vessel Segmentation
===================================================
Paper: "Beyond Dice: Structural Evaluation and Failure Mode Analysis
        for Retinal Vessel Segmentation"

Metrics implemented:
  1. CCA  - Connectivity Preservation Score   (FIXED: size-filtered)
  2. BFR  - Branch Fragmentation Ratio
  3. SVD  - Sparse Vessel Dice
  4. SkelDice - Skeleton Dice
  5. SkelHD   - Skeleton Hausdorff Distance
  6. BranchDiff - Branch Count Difference
  7. Betti0Err  - Betti-0 Error (connected components delta)

All metrics work on binary numpy arrays (H x W), dtype uint8 / bool.
All metrics are PARAMETER-SAFE: sensible defaults, no silent failures.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
from typing import Tuple, Dict


class StructuralMetrics:
    """
    Compute all structural metrics for one (pred, gt) pair.

    Parameters
    ----------
    min_component_px : int
        Components smaller than this are treated as noise and ignored.
        Default 10 prevents denominator explosion from 1-pixel blobs.
    sparse_threshold : float
        Vessel ratio below which a patch is considered 'sparse'.
        Used only by SVD. Default 0.02 (2% vessel pixels).
    patch_size : int
        Patch size used to compute SVD. Default 32.
    """

    def __init__(
        self,
        min_component_px: int = 10,
        sparse_threshold: float = 0.02,
        patch_size: int = 32,
    ):
        self.min_component_px = min_component_px
        self.sparse_threshold = sparse_threshold
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Compute all structural metrics.

        Parameters
        ----------
        pred : np.ndarray  (H, W)  binary uint8 or bool
        gt   : np.ndarray  (H, W)  binary uint8 or bool

        Returns
        -------
        dict with keys: CCA, BFR, SVD, SkelDice, SkelHD, BranchDiff, Betti0Err
        """
        pred = self._to_binary(pred)
        gt   = self._to_binary(gt)

        skel_pred = skeletonize(pred).astype(np.uint8)
        skel_gt   = skeletonize(gt).astype(np.uint8)

        return {
            "CCA":        self.connectivity_preservation_score(pred, gt),
            "BFR":        self.branch_fragmentation_ratio(skel_pred, skel_gt),
            "SVD":        self.sparse_vessel_dice(pred, gt),
            "SkelDice":   self.skeleton_dice(skel_pred, skel_gt),
            "SkelHD":     self.skeleton_hausdorff(skel_pred, skel_gt),
            "BranchDiff": self.branch_count_diff(skel_pred, skel_gt),
            "Betti0Err":  self.betti0_error(pred, gt),
        }

    # ------------------------------------------------------------------
    # Metric 1 — CCA (Connectivity Preservation Score)
    # ------------------------------------------------------------------

    def connectivity_preservation_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Fraction of GT connected components that have a matching
        predicted component (IoU > 0.3).

        Fix vs original:
          - Components < min_component_px are filtered BEFORE counting.
          - This prevents 1-pixel noise from inflating denominator.

        Returns value in [0, 1]. Higher is better.
        """
        pred = self._to_binary(pred)
        gt   = self._to_binary(gt)

        gt_labeled, n_gt = ndimage.label(gt)
        pred_labeled, _  = ndimage.label(pred)

        # filter small gt components
        gt_sizes = ndimage.sum(gt, gt_labeled, range(1, n_gt + 1))
        valid_gt = [i + 1 for i, s in enumerate(gt_sizes)
                    if s >= self.min_component_px]

        if len(valid_gt) == 0:
            return 1.0  # no meaningful GT components → trivially preserved

        matched = 0
        for gid in valid_gt:
            gt_mask = (gt_labeled == gid)
            # find all pred labels that overlap with this GT component
            overlap_pred_ids = np.unique(pred_labeled[gt_mask])
            overlap_pred_ids = overlap_pred_ids[overlap_pred_ids != 0]

            best_iou = 0.0
            for pid in overlap_pred_ids:
                pred_mask = (pred_labeled == pid)
                # filter small predicted noise
                if pred_mask.sum() < self.min_component_px:
                    continue
                inter = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = inter / (union + 1e-8)
                best_iou = max(best_iou, iou)

            if best_iou > 0.3:
                matched += 1

        return matched / len(valid_gt)

    # ------------------------------------------------------------------
    # Metric 2 — BFR (Branch Fragmentation Ratio)
    # ------------------------------------------------------------------

    def branch_fragmentation_ratio(self, skel_pred: np.ndarray, skel_gt: np.ndarray) -> float:
        """
        Ratio of fragmented skeleton branches.

        BFR = (n_components_pred - n_components_gt) / n_components_gt

        Positive → prediction is more fragmented than GT.
        Negative → prediction merges branches (false bridges).
        Clipped to [-1, 10] to avoid extreme outliers.

        Returns float. Lower absolute value is better; 0 is ideal.
        """
        n_pred = self._count_components(skel_pred)
        n_gt   = self._count_components(skel_gt)
        if n_gt == 0:
            return 0.0
        return float(np.clip((n_pred - n_gt) / n_gt, -1.0, 10.0))

    # ------------------------------------------------------------------
    # Metric 3 — SVD (Sparse Vessel Dice)
    # ------------------------------------------------------------------

    def sparse_vessel_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Dice computed ONLY on patches where GT vessel ratio < sparse_threshold.
        These are thin-capillary / peripheral regions that standard Dice hides.

        Returns Dice in [0, 1]. Higher is better.
        Returns -1.0 if no sparse patches exist (caller should handle).
        """
        H, W = gt.shape
        P = self.patch_size
        tp = fp = fn = 0

        for r in range(0, H - P + 1, P):
            for c in range(0, W - P + 1, P):
                gt_patch   = gt[r:r+P, c:c+P]
                pred_patch = pred[r:r+P, c:c+P]
                vessel_ratio = gt_patch.sum() / (P * P)
                if vessel_ratio < self.sparse_threshold:
                    tp += np.logical_and(pred_patch, gt_patch).sum()
                    fp += np.logical_and(pred_patch, ~gt_patch).sum()
                    fn += np.logical_and(~pred_patch, gt_patch).sum()

        denom = 2 * tp + fp + fn
        if denom == 0:
            return -1.0  # no sparse patches found
        return float(2 * tp / denom)

    # ------------------------------------------------------------------
    # Metric 4 — Skeleton Dice
    # ------------------------------------------------------------------

    def skeleton_dice(self, skel_pred: np.ndarray, skel_gt: np.ndarray) -> float:
        """
        Dice coefficient computed on skeletonized masks.
        Measures centerline-level overlap.

        Returns value in [0, 1]. Higher is better.
        """
        tp = np.logical_and(skel_pred, skel_gt).sum()
        fp = np.logical_and(skel_pred, ~skel_gt.astype(bool)).sum()
        fn = np.logical_and(~skel_pred.astype(bool), skel_gt).sum()
        denom = 2 * tp + fp + fn
        if denom == 0:
            return 1.0  # both skeletons are empty → trivially identical
        return float(2 * tp / denom)

    # ------------------------------------------------------------------
    # Metric 5 — Skeleton Hausdorff Distance
    # ------------------------------------------------------------------

    def skeleton_hausdorff(self, skel_pred: np.ndarray, skel_gt: np.ndarray) -> float:
        """
        95th-percentile Hausdorff distance between skeleton centerlines.
        Uses distance transform for efficiency (avoids O(N^2) pairwise).

        Returns distance in pixels. Lower is better.
        """
        pts_pred = np.argwhere(skel_pred > 0)
        pts_gt   = np.argwhere(skel_gt   > 0)

        if len(pts_pred) == 0 or len(pts_gt) == 0:
            return float(max(skel_pred.shape))  # worst case if one is empty

        # distance from every GT skeleton pixel to nearest pred skeleton pixel
        dist_gt_to_pred = ndimage.distance_transform_edt(1 - skel_pred)
        d_gt = dist_gt_to_pred[skel_gt > 0]

        # distance from every pred skeleton pixel to nearest GT skeleton pixel
        dist_pred_to_gt = ndimage.distance_transform_edt(1 - skel_gt)
        d_pred = dist_pred_to_gt[skel_pred > 0]

        # 95th percentile (robust to outliers)
        return float(max(np.percentile(d_gt, 95), np.percentile(d_pred, 95)))

    # ------------------------------------------------------------------
    # Metric 6 — Branch Count Difference
    # ------------------------------------------------------------------

    def branch_count_diff(self, skel_pred: np.ndarray, skel_gt: np.ndarray) -> int:
        """
        Absolute difference in number of skeleton branch endpoints.
        Uses 8-connectivity neighbourhood to detect endpoints
        (pixels with exactly 1 skeleton neighbour).

        Returns non-negative int. Lower is better; 0 is perfect.
        """
        endpoints_pred = self._count_endpoints(skel_pred)
        endpoints_gt   = self._count_endpoints(skel_gt)
        return abs(endpoints_pred - endpoints_gt)

    # ------------------------------------------------------------------
    # Metric 7 — Betti-0 Error
    # ------------------------------------------------------------------

    def betti0_error(self, pred: np.ndarray, gt: np.ndarray) -> int:
        """
        Absolute difference in number of connected components (Betti-0).
        Components smaller than min_component_px are excluded.

        Returns non-negative int. Lower is better; 0 is perfect.
        """
        n_pred = self._count_components(pred)
        n_gt   = self._count_components(gt)
        return abs(n_pred - n_gt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_binary(arr: np.ndarray) -> np.ndarray:
        return (arr > 0).astype(np.uint8)

    def _count_components(self, mask: np.ndarray) -> int:
        """Count connected components, filtering noise blobs."""
        labeled, n = ndimage.label(mask)
        if n == 0:
            return 0
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        return int(sum(1 for s in sizes if s >= self.min_component_px))

    @staticmethod
    def _count_endpoints(skel: np.ndarray) -> int:
        """
        Count skeleton endpoints: pixels with exactly 1 neighbour
        in 8-connectivity.
        """
        if skel.sum() == 0:
            return 0
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)
        neighbour_count = convolve(skel.astype(np.uint8), kernel, mode='constant')
        endpoints = np.logical_and(skel > 0, neighbour_count == 1)
        return int(endpoints.sum())
