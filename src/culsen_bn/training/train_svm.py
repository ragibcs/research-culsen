"""
SVM training pipeline for CulSen-BN.
Fits Char TF-IDF + LinearSVC on training partition, evaluates on validation and test,
and saves model artifacts, metrics, and test predictions.
"""

from pathlib import Path
from typing import Any, Dict
import yaml
from ..models.svm import BaselineSVMClassifier
from ..evaluation.metrics import compute_classification_metrics
from ..evaluation.calibration import compute_calibration_metrics
from ..evaluation.robustness import evaluate_subgroup_robustness
from ..utils.logging import get_logger, save_json
from ..utils.reproducibility import set_seed


def run_svm_experiment(config_path: str = "configs/baseline_svm.yaml") -> Dict[str, Any]:
    """
    Execute full SVM baseline training and evaluation workflow.
    """
    logger = get_logger("svm_experiment")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("model", {}).get("classifier", {}).get("random_state", 42))

    # Load splits
    split_file = cfg["data"]["split_file"]
    logger.info(f"Loading data splits from {split_file}...")

    import pandas as pd
    splits_df = pd.read_parquet(split_file)

    train_df = splits_df[splits_df["split"] == "train"].copy()
    val_df = splits_df[splits_df["split"] == "val"].copy()
    test_df = splits_df[splits_df["split"] == "test"].copy()

    # Filter to binary sentiment classes (positive=1, negative=0)
    target_classes = cfg["data"].get("target_classes", ["negative", "positive"])
    text_col = cfg["data"].get("text_column", "text_clean")
    label_col = cfg["data"].get("label_column", "sentiment")

    train_df = train_df[train_df[label_col].isin(target_classes)]
    val_df = val_df[val_df[label_col].isin(target_classes)]
    test_df = test_df[test_df[label_col].isin(target_classes)]

    logger.info(
        f"Filtered binary dataset sizes: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}"
    )

    # Encode labels: negative -> 0, positive -> 1
    y_train = (train_df[label_col] == "positive").astype(int).tolist()
    y_val = (val_df[label_col] == "positive").astype(int).tolist()
    y_test = (test_df[label_col] == "positive").astype(int).tolist()

    X_train = train_df[text_col].astype(str).tolist()
    X_val = val_df[text_col].astype(str).tolist()
    X_test = test_df[text_col].astype(str).tolist()

    # Initialize model
    model_cfg = cfg.get("model", {})
    vec_cfg = model_cfg.get("vectorizer", {})
    clf_cfg = model_cfg.get("classifier", {})

    model = BaselineSVMClassifier(
        ngram_range=tuple(vec_cfg.get("ngram_range", [2, 5])),
        min_df=vec_cfg.get("min_df", 2),
        max_features=vec_cfg.get("max_features", 300000),
        sublinear_tf=vec_cfg.get("sublinear_tf", True),
        C=clf_cfg.get("C", 1.0),
        class_weight=clf_cfg.get("class_weight", "balanced"),
        random_state=clf_cfg.get("random_state", 42),
        max_iter=clf_cfg.get("max_iter", 3000),
    )

    logger.info("Fitting Char TF-IDF + LinearSVC strictly on training partition...")
    model.fit(X_train, y_train)
    logger.info(f"Training completed in {model.training_time_sec:.2f} seconds.")

    # Evaluate on Validation
    val_pred = model.predict(X_val)
    val_prob = [p[1] for p in model.predict_proba(X_val)]
    val_metrics = compute_classification_metrics(y_val, val_pred, val_prob)
    logger.info(
        f"Validation Results: Macro-F1 = {val_metrics['macro_f1']:.4f}, Accuracy = {val_metrics['accuracy']:.4f}, MCC = {val_metrics['mcc']:.4f}"
    )

    # Evaluate on Test
    test_pred = model.predict(X_test)
    test_prob = [p[1] for p in model.predict_proba(X_test)]
    test_metrics = compute_classification_metrics(y_test, test_pred, test_prob)
    logger.info(
        f"Test Results: Macro-F1 = {test_metrics['macro_f1']:.4f}, Accuracy = {test_metrics['accuracy']:.4f}, MCC = {test_metrics['mcc']:.4f}"
    )

    # Calibration & Robustness
    calib_metrics = compute_calibration_metrics(y_test, test_prob)
    logger.info(f"Test Expected Calibration Error (ECE): {calib_metrics['expected_calibration_error']:.4f}")

    subgroup_metrics = evaluate_subgroup_robustness(test_df, test_pred, test_prob, text_col=text_col, label_col=label_col)

    # Save outputs
    out_dir = Path(cfg["output"].get("artifacts_dir", "artifacts/baseline_svm"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model.pkl"))

    results = {
        "model_type": "char_tfidf_svm",
        "training_time_seconds": model.training_time_sec,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "calibration": calib_metrics,
        "subgroups": subgroup_metrics,
    }

    metrics_file = cfg["output"].get("metrics_file", "reports/tables/baseline_svm_metrics.json")
    save_json(results, metrics_file)

    # Save test predictions with original metadata
    test_df["svm_pred"] = test_pred
    test_df["svm_prob_positive"] = test_prob
    pred_path = out_dir / "test_predictions.parquet"
    test_df.to_parquet(pred_path)
    logger.info(f"Saved artifacts and predictions to {out_dir}")

    return results
