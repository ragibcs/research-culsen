"""
Unit tests for classification and calibration metric functions.
Verifies Macro-F1, Balanced Accuracy, MCC, ECE, and Brier Score calculations.
"""

import pytest
from culsen_bn.evaluation.metrics import compute_classification_metrics
from culsen_bn.evaluation.calibration import compute_calibration_metrics


def test_perfect_classification_metrics():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 0]
    y_prob = [0.1, 0.2, 0.9, 0.8, 0.95, 0.05]

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["positive_f1"] == 1.0
    assert m["negative_f1"] == 1.0
    assert m["mcc"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["confusion_matrix"]["fp"] == 0
    assert m["confusion_matrix"]["fn"] == 0


def test_inverse_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    m = compute_classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["macro_f1"] == 0.0
    assert m["mcc"] == -1.0


def test_calibration_metrics_calculation():
    y_true = [0, 0, 1, 1]
    # Well-calibrated probabilities
    y_prob = [0.1, 0.2, 0.8, 0.9]

    cal = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert "expected_calibration_error" in cal
    assert "maximum_calibration_error" in cal
    assert "brier_score" in cal
    assert 0.0 <= cal["expected_calibration_error"] <= 1.0
    assert 0.0 <= cal["brier_score"] <= 1.0
