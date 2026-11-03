"""
Cross-season domain generalization evaluation for CulSen-BN.
Analyzes out-of-domain transfer performance across held-out Coke Studio Bangla seasons.
"""

from typing import Any, Dict, List
from .metrics import compute_classification_metrics


def evaluate_cross_season_transfer(
    season_results: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarize performance across leave-one-season-out evaluations (Protocol B).

    Args:
        season_results: Dict mapping test_season -> evaluation metrics dict.
    """
    seasons = sorted(list(season_results.keys()))

    macro_f1s = [season_results[s]["macro_f1"] for s in seasons if "macro_f1" in season_results[s]]
    accuracies = [season_results[s]["accuracy"] for s in seasons if "accuracy" in season_results[s]]
    mccs = [season_results[s]["mcc"] for s in seasons if "mcc" in season_results[s]]

    import numpy as np

    summary = {
        "evaluated_seasons": seasons,
        "per_season_results": season_results,
        "cross_season_macro_f1_mean": float(np.mean(macro_f1s)) if macro_f1s else 0.0,
        "cross_season_macro_f1_std": float(np.std(macro_f1s)) if macro_f1s else 0.0,
        "cross_season_accuracy_mean": float(np.mean(accuracies)) if accuracies else 0.0,
        "cross_season_accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
        "cross_season_mcc_mean": float(np.mean(mccs)) if mccs else 0.0,
        "cross_season_mcc_std": float(np.std(mccs)) if mccs else 0.0,
    }

    return summary
