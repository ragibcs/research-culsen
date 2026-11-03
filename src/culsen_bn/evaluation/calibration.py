"""
Model confidence calibration metrics and reliability diagram data for CulSen-BN.
Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.
"""

from typing import Any, Dict, List
import numpy as np


def compute_calibration_metrics(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculate ECE, MCE, and Brier Score for predicted positive class probabilities.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability for class 1 (between 0.0 and 1.0).
        n_bins: Number of equal-width confidence bins.
    """
    y_t = np.array(y_true, dtype=np.int32)
    y_p = np.array(y_prob, dtype=np.float64)

    if len(y_t) != len(y_p):
        raise ValueError("y_true and y_prob must have identical length")

    # Brier Score: mean squared error of probability predictions
    brier_score = float(np.mean((y_p - y_t) ** 2))

    # Predicted class and confidence (probability of predicted class)
    y_pred = (y_p >= 0.5).astype(np.int32)
    confidences = np.where(y_pred == 1, y_p, 1.0 - y_p)
    accuracies = (y_pred == y_t).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins_data = []

    total_samples = len(y_t)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Identify samples falling in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_gap = abs(bin_acc - bin_conf)

            ece += (bin_size / total_samples) * bin_gap
            mce = max(mce, bin_gap)

            bins_data.append({
                "bin_index": i,
                "bin_range": [float(bin_lower), float(bin_upper)],
                "sample_count": bin_size,
                "accuracy": bin_acc,
                "confidence": bin_conf,
                "gap": bin_gap,
            })
        else:
            bins_data.append({
                "bin_index": i,
                "bin_range": [float(bin_lower), float(bin_upper)],
                "sample_count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
                "gap": 0.0,
            })

    return {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": float(brier_score),
        "bins": bins_data,
    }
