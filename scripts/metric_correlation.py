"""
Metric Correlation Analysis
===========================
Compute Pearson/Spearman correlation matrix to identify redundant metrics.

According to the action plan, if two metrics have r > 0.95 across datasets,
one is redundant. This script analyzes all 8+ metrics together.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr


def compute_correlation_matrix(
    metrics_dict: Dict[str, List[float]]
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for all metrics.

    Args:
        metrics_dict: {metric_name: [values_per_image]}

    Returns:
        DataFrame with correlation matrix
    """
    df = pd.DataFrame(metrics_dict)
    return df.corr(method='pearson')


def compute_spearman_matrix(
    metrics_dict: Dict[str, List[float]]
) -> pd.DataFrame:
    """
    Compute Spearman correlation matrix for all metrics.
    More robust to outliers.
    """
    df = pd.DataFrame(metrics_dict)
    return df.corr(method='spearman')


def find_redundant_metrics(
    metrics_dict: Dict[str, List[float]],
    threshold: float = 0.95,
    method: str = 'pearson'
) -> List[Tuple[str, str, float]]:
    """
    Find metric pairs with correlation above threshold.

    Returns:
        List of (metric_a, metric_b, correlation) tuples
    """
    df = pd.DataFrame(metrics_dict)
    redundant = []

    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            if method == 'pearson':
                r, _ = pearsonr(df[col_a], df[col_b])
            else:
                r, _ = spearmanr(df[col_a], df[col_b])

            if abs(r) > threshold:
                redundant.append((col_a, col_b, float(abs(r))))

    return redundant


if __name__ == "__main__":
    print("Metric Correlation Analysis - testing...")

    test_metrics = {
        "Dice": [0.80, 0.82, 0.78, 0.85, 0.79],
        "Accuracy": [0.95, 0.96, 0.94, 0.97, 0.94],
        "Sensitivity": [0.82, 0.84, 0.80, 0.86, 0.81],
    }

    corr = compute_correlation_matrix(test_metrics)
    print(f"\nPearson Correlation Matrix:\n{corr}")

    redundant = find_redundant_metrics(test_metrics, threshold=0.90)
    print(f"\nRedundant pairs (|r| > 0.90): {redundant}")
    print("[OK] Metric correlation analysis module loads successfully!")