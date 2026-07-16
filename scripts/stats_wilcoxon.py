"""
Statistical Analysis: Wilcoxon Signed-Rank Test
=============================================
Non-parametric test for paired comparisons between models.
Used to compare metrics image-by-image (not independent samples).

Recommended for Paper 1 Phase 4: Statistical Rigor.
"""

import numpy as np
from scipy.stats import wilcoxon
from typing import Dict, List


def wilcoxon_signed_rank(
    metric_a: List[float],
    metric_b: List[float],
    metric_name: str = "Metric"
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired metric comparison.

    Args:
        metric_a: Per-image metric values for model A
        metric_b: Per-image metric values for model B
        metric_name: Name of metric for reporting

    Returns:
        dict with: statistic, p_value, median_diff, n_greater_a, n_greater_b
    """
    a = np.array(metric_a, dtype=np.float64)
    b = np.array(metric_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal lengths: {len(a)} vs {len(b)}")

    diff = a - b
    stat, p = wilcoxon(diff, zero_method='wilcox')

    n_greater_a = int(np.sum(diff > 0))
    n_greater_b = int(np.sum(diff < 0))

    return {
        "metric": metric_name,
        "statistic": float(stat),
        "p_value": float(p),
        "median_diff": float(np.median(diff)),
        "n_greater_a": n_greater_a,
        "n_greater_b": n_greater_b,
        "significant_005": bool(p < 0.05),
    }


def paired_tests_all_metrics(
    results_a: Dict[str, List[float]],
    results_b: Dict[str, List[float]],
    alpha: float = 0.05
) -> List[Dict]:
    """
    Run Wilcoxon signed-rank test for all metrics.

    Args:
        results_a: {"Dice": [0.8, 0.85, ...], "CCA": [...], ...}
        results_b: Same structure for comparison model

    Returns:
        List of dicts per metric with test statistics
    """
    all_results = []
    for key in results_a:
        if key in results_b:
            result = wilcoxon_signed_rank(results_a[key], results_b[key], key)
            all_results.append(result)
    return all_results


if __name__ == "__main__":
    # Quick test with synthetic data
    a_dice = [0.83, 0.85, 0.82, 0.84, 0.81]
    b_dice = [0.81, 0.83, 0.80, 0.82, 0.79]

    result = wilcoxon_signed_rank(a_dice, b_dice, "Dice")
    print(f"Wilcoxon test result: {result}")
    print("[OK] Wilcoxon signed-rank test module loads successfully!")