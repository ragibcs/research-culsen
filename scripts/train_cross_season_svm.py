#!/usr/bin/env python3
"""
Cross-Season Generalization (Protocol B: Leave-One-Season-Out) for Classical SVM Baseline.
Evaluates model on each held-out season (Seasons 1 to 4) on CPU.
"""

from pathlib import Path
import pandas as pd
from culsen_bn.models.svm import BaselineSVMClassifier
from culsen_bn.evaluation.metrics import compute_classification_metrics
from culsen_bn.evaluation.calibration import compute_calibration_metrics
from culsen_bn.utils.logging import get_logger, save_json
from culsen_bn.utils.reproducibility import set_seed


def run_cross_season_svm():
    logger = get_logger("cross_season_svm")
    set_seed(42)

    seasons = [1, 2, 3, 4]
    results = {}

    print("\n" + "=" * 70)
    print("PROTOCOL B: LEAVE-ONE-SEASON-OUT CROSS-SEASON GENERALIZATION (SVM)")
    print("=" * 70)

    for season in seasons:
        split_path = f"data/processed/splits_protocol_b_season_{season}.parquet"
        logger.info(f"Loading Protocol B split for Held-Out Season {season} from {split_path}...")
        df = pd.read_parquet(split_path)

        # Binary classes: positive and negative
        target_classes = ["negative", "positive"]
        df_binary = df[df["sentiment"].isin(target_classes)].copy()

        train_df = df_binary[df_binary["split"] == "train"]
        val_df = df_binary[df_binary["split"] == "val"]
        test_df = df_binary[df_binary["split"] == "test"]

        logger.info(
            f"Season {season} Held-Out: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
        )

        X_train = train_df["text_clean"].astype(str).tolist()
        y_train = (train_df["sentiment"] == "positive").astype(int).tolist()

        X_test = test_df["text_clean"].astype(str).tolist()
        y_test = (test_df["sentiment"] == "positive").astype(int).tolist()

        # Initialize and fit model strictly on train split
        model = BaselineSVMClassifier(
            ngram_range=(2, 5),
            min_df=2,
            max_features=300000,
            sublinear_tf=True,
            C=1.0,
            class_weight="balanced",
            random_state=42,
            max_iter=3000,
        )

        model.fit(X_train, y_train)

        # Evaluate on held-out target season
        test_pred = model.predict(X_test)
        test_prob = [p[1] for p in model.predict_proba(X_test)]

        metrics = compute_classification_metrics(y_test, test_pred, test_prob)
        calib = compute_calibration_metrics(y_test, test_prob)

        print(
            f"Held-Out Season {season:02d}: Macro-F1 = {metrics['macro_f1']:.4f} | "
            f"Acc = {metrics['accuracy']:.4f} | Pos-F1 = {metrics['positive_f1']:.4f} | "
            f"Neg-F1 = {metrics['negative_f1']:.4f} | ECE = {calib['expected_calibration_error']:.4f} | "
            f"Train Time: {model.training_time_sec:.2f}s"
        )

        results[f"season_{season}"] = {
            "held_out_season": season,
            "train_seasons": [s for s in seasons if s != season],
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "training_time_seconds": model.training_time_sec,
            "metrics": metrics,
            "calibration": calib,
        }

    # Compute cross-season mean and standard deviation
    macro_f1s = [results[f"season_{s}"]["metrics"]["macro_f1"] for s in seasons]
    accuracies = [results[f"season_{s}"]["metrics"]["accuracy"] for s in seasons]

    mean_macro_f1 = sum(macro_f1s) / len(macro_f1s)
    mean_acc = sum(accuracies) / len(accuracies)

    summary = {
        "per_season": results,
        "mean_macro_f1": mean_macro_f1,
        "mean_accuracy": mean_acc,
        "macro_f1_scores": macro_f1s,
    }

    print("-" * 70)
    print(f"Mean Cross-Season Macro-F1: {mean_macro_f1:.4f}")
    print(f"Mean Cross-Season Accuracy: {mean_acc:.4f}")
    print("=" * 70 + "\n")

    out_file = "reports/tables/cross_season_svm_metrics.json"
    save_json(summary, out_file)
    logger.info(f"Saved cross-season SVM metrics to {out_file}")
    return summary


if __name__ == "__main__":
    run_cross_season_svm()
