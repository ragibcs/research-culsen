"""
Statistical significance testing and paired bootstrap confidence intervals for CulSen-BN.
Provides McNemar's test for paired classification outcomes and bootstrap estimation
for metric difference intervals.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from .metrics import compute_classification_metrics


def paired_bootstrap_ci(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired bootstrap confidence intervals for performance differences
    between Model A and Model B on the identical test set.
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    y_t = np.array(y_true)
    y_a = np.array(y_pred_a)
    y_b = np.array(y_pred_b)

    diff_macro_f1 = []
    diff_accuracy = []

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        boot_true = y_t[boot_idx].tolist()
        boot_a = y_a[boot_idx].tolist()
        boot_b = y_b[boot_idx].tolist()

        try:
            m_a = compute_classification_metrics(boot_true, boot_a)
            m_b = compute_classification_metrics(boot_true, boot_b)

            diff_macro_f1.append(m_a["macro_f1"] - m_b["macro_f1"])
            diff_accuracy.append(m_a["accuracy"] - m_b["accuracy"])
        except Exception:
            continue

    alpha = (1.0 - confidence_level) / 2.0
    f1_ci_lower = float(np.percentile(diff_macro_f1, alpha * 100))
    f1_ci_upper = float(np.percentile(diff_macro_f1, (1.0 - alpha) * 100))

    acc_ci_lower = float(np.percentile(diff_accuracy, alpha * 100))
    acc_ci_upper = float(np.percentile(diff_accuracy, (1.0 - alpha) * 100))

    return {
        "confidence_level": confidence_level,
        "n_bootstraps": len(diff_macro_f1),
        "macro_f1_diff_mean": float(np.mean(diff_macro_f1)),
        "macro_f1_diff_ci": [f1_ci_lower, f1_ci_upper],
        "accuracy_diff_mean": float(np.mean(diff_accuracy)),
        "accuracy_diff_ci": [acc_ci_lower, acc_ci_upper],
        "f1_statistically_significant": bool(f1_ci_lower > 0 or f1_ci_upper < 0),
    }


def mcnemar_test(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
) -> Dict[str, Any]:
    """
    McNemar's test for paired classification predictions with continuity correction.
    Table:
                Model B Correct    Model B Incorrect
    Model A Correct     n00               n01 (b)
    Model A Incorrect   n10 (c)           n11
    """
    b = sum(1 for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b) if ya == yt and yb != yt)
    c = sum(1 for yt, ya, yb in zip(y_true, y_pred_a, y_pred_b) if ya != yt and yb == yt)

    # Continuity corrected chi-squared statistic
    if (b + c) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
        from scipy.stats import chi2
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    return {
        "b_model_a_correct_model_b_wrong": b,
        "c_model_a_wrong_model_b_correct": c,
        "mcnemar_statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
    }
