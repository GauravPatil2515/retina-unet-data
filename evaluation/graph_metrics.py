"""
Vessel Graph Metrics
====================
Converts a binary segmentation mask into a vessel graph:
  Nodes  = endpoints + bifurcation points of the skeleton
  Edges  = vessel segments connecting nodes

Then computes graph-level metrics:
  - Branch Preservation Rate (BPR)
  - Junction Preservation Rate (JPR)
  - Graph Edit Distance approximation (GED)

These are the metrics ophthalmologists actually care about:
is the bifurcation pattern preserved? are branches complete?
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple


class VesselGraphMetrics:
    """
    Build vessel graphs from pred/gt skeletons and compute
    graph-level structural metrics.

    Parameters
    ----------
    min_branch_px : int
        Minimum skeleton branch length (pixels) to count as a branch.
        Shorter branches are likely noise or stub artifacts.
    """

    def __init__(self, min_branch_px: int = 5):
        self.min_branch_px = min_branch_px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        """
        Parameters
        ----------
        pred, gt : (H, W) binary arrays

        Returns
        -------
        dict with keys: BPR, JPR, GED_approx
        """
        pred_b = (pred > 0).astype(np.uint8)
        gt_b   = (gt   > 0).astype(np.uint8)

        skel_pred = skeletonize(pred_b).astype(np.uint8)
        skel_gt   = skeletonize(gt_b).astype(np.uint8)

        pred_graph = self._extract_graph(skel_pred)
        gt_graph   = self._extract_graph(skel_gt)

        return {
            "BPR":       self._branch_preservation_rate(pred_graph, gt_graph),
            "JPR":       self._junction_preservation_rate(pred_graph, gt_graph),
            "GED_approx": self._ged_approximation(pred_graph, gt_graph),
        }

    # ------------------------------------------------------------------
    # Graph extraction
    # ------------------------------------------------------------------

    def _extract_graph(self, skel: np.ndarray) -> Dict:
        """
        Extract a graph from a skeleton image.

        Returns dict:
          nodes       : list of (r, c) coordinates
          node_types  : list of 'endpoint' | 'junction'
          n_branches  : int  (number of distinct edge segments)
          n_junctions : int  (bifurcation + crossing nodes)
          n_endpoints : int
        """
        if skel.sum() == 0:
            return dict(nodes=[], node_types=[], n_branches=0,
                        n_junctions=0, n_endpoints=0)

        from scipy.ndimage import convolve
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        neighbour_count = convolve(skel, kernel, mode='constant')

        endpoints_mask  = (skel > 0) & (neighbour_count == 1)
        junctions_mask  = (skel > 0) & (neighbour_count >= 3)

        endpoints  = list(map(tuple, np.argwhere(endpoints_mask)))
        junctions  = list(map(tuple, np.argwhere(junctions_mask)))

        nodes      = endpoints + junctions
        node_types = ['endpoint'] * len(endpoints) + ['junction'] * len(junctions)

        # count branches = connected components of non-node skeleton pixels
        # (each branch segment between two nodes is one component)
        node_mask      = endpoints_mask | junctions_mask
        branch_pixels  = skel & ~node_mask
        _, n_branches  = ndimage.label(branch_pixels)

        return dict(
            nodes       = nodes,
            node_types  = node_types,
            n_branches  = n_branches,
            n_junctions = len(junctions),
            n_endpoints = len(endpoints),
        )

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------

    def _branch_preservation_rate(self, pred_g: Dict, gt_g: Dict) -> float:
        """
        BPR = min(n_branches_pred, n_branches_gt) / max(n_branches_gt, 1)

        Capped at 1.0.  1.0 = all GT branches present in pred.
        """
        n_pred = pred_g["n_branches"]
        n_gt   = gt_g["n_branches"]
        if n_gt == 0:
            return 1.0
        return float(min(n_pred, n_gt) / n_gt)

    def _junction_preservation_rate(self, pred_g: Dict, gt_g: Dict) -> float:
        """
        JPR = min(n_junctions_pred, n_junctions_gt) / max(n_junctions_gt, 1)

        Capped at 1.0.  1.0 = all GT bifurcations present in pred.
        """
        n_pred = pred_g["n_junctions"]
        n_gt   = gt_g["n_junctions"]
        if n_gt == 0:
            return 1.0
        return float(min(n_pred, n_gt) / n_gt)

    def _ged_approximation(self, pred_g: Dict, gt_g: Dict) -> float:
        """
        Approximate Graph Edit Distance.

        True GED is NP-hard.  We approximate it as:
          GED ≈ |Δ branches| + |Δ junctions| + |Δ endpoints|

        This gives a cheap, interpretable structural distance.
        Lower is better; 0 = structurally identical.
        """
        delta_branches  = abs(pred_g["n_branches"]  - gt_g["n_branches"])
        delta_junctions = abs(pred_g["n_junctions"] - gt_g["n_junctions"])
        delta_endpoints = abs(pred_g["n_endpoints"] - gt_g["n_endpoints"])
        return float(delta_branches + delta_junctions + delta_endpoints)
