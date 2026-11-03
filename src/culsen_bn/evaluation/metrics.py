"""
Comprehensive classification metrics calculation for CulSen-BN.
Computes Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Binary F1,
Matthews Correlation Coefficient (MCC), and AUROC/AUPRC.
"""

from typing import Any, Dict, List, Optional
import math


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics suite for binary sentiment classification (0=negative, 1=positive).
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        raise ValueError("y_true and y_pred must be non-empty and of identical length.")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n

    # Positive class (1) metrics
    pos_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pos_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pos_f1 = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0.0

    # Negative class (0) metrics
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    neg_f1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0.0

    # Macro-F1: unweighted average of class F1s
    macro_f1 = (pos_f1 + neg_f1) / 2.0
    balanced_accuracy = (pos_recall + neg_recall) / 2.0

    # Matthews Correlation Coefficient (MCC)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    metrics = {
        "n_samples": n,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "positive_precision": float(pos_precision),
        "positive_recall": float(pos_recall),
        "positive_f1": float(pos_f1),
        "negative_precision": float(neg_precision),
        "negative_recall": float(neg_recall),
        "negative_f1": float(neg_f1),
        "mcc": float(mcc),
    }

    # Area Under ROC and PR Curve if probabilities are supplied
    if y_prob is not None and len(y_prob) == n:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = None
            metrics["auprc"] = None

    return metrics
