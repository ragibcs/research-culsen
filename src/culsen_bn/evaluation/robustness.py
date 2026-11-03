"""
Subgroup robustness evaluation module for CulSen-BN.
Measures performance disparities across Bengali script, Banglish, mixed code,
emoji usage, and negation.
"""

from typing import Any, Dict, List
from .metrics import compute_classification_metrics
from ..data.normalize import detect_script, detect_negation, extract_emojis


def evaluate_subgroup_robustness(
    test_df: Any,
    y_pred: List[int],
    y_prob: List[float] = None,
    text_col: str = "text_clean",
    label_col: str = "sentiment",
) -> Dict[str, Any]:
    """
    Compute classification metrics broken down by linguistic subgroups.
    """
    import pandas as pd

    df = test_df.copy().reset_index(drop=True)
    df["y_pred"] = y_pred
    if y_prob is not None:
        df["y_prob"] = y_prob

    # Binary ground truth
    df["y_true"] = (df[label_col] == "positive").astype(int)

    # Subgroup flags
    if "script" not in df.columns:
        df["script"] = df[text_col].astype(str).apply(detect_script)
    df["has_emoji"] = df[text_col].astype(str).apply(lambda t: len(extract_emojis(t)) > 0)
    df["has_negation"] = df[text_col].astype(str).apply(detect_negation)

    subgroups = {
        "overall": df,
        "script_bengali": df[df["script"].isin(["bengali", "bangla"])],
        "script_banglish": df[df["script"].isin(["banglish_latin", "latin"])],
        "script_mixed": df[df["script"] == "mixed"],
        "with_emojis": df[df["has_emoji"] == True],
        "without_emojis": df[df["has_emoji"] == False],
        "with_negation": df[df["has_negation"] == True],
        "without_negation": df[df["has_negation"] == False],
    }

    if "confidence" in df.columns:
        subgroups["high_confidence_ge_80"] = df[df["confidence"] >= 0.80]
        subgroups["low_confidence_lt_65"] = df[df["confidence"] < 0.65]

    results = {}
    for name, sub in subgroups.items():
        if len(sub) >= 5 and sub["y_true"].nunique() > 1:
            prob = sub["y_prob"].tolist() if "y_prob" in sub.columns else None
            m = compute_classification_metrics(sub["y_true"].tolist(), sub["y_pred"].tolist(), prob)
            results[name] = m
        else:
            results[name] = {
                "n_samples": len(sub),
                "note": "Insufficient samples or single class representation for full evaluation.",
            }

    return results
