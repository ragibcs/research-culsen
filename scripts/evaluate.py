#!/usr/bin/env python3
"""
CLI script to evaluate saved models, run statistical tests, or evaluate against human-annotated test sets.
"""

import argparse
from pathlib import Path
import json
from culsen_bn.data.annotation import import_human_annotations
from culsen_bn.evaluation.statistical_tests import paired_bootstrap_ci, mcnemar_test
from culsen_bn.evaluation.metrics import compute_classification_metrics
from culsen_bn.utils.logging import get_logger, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions and conduct statistical tests.")
    parser.add_argument("--human-annotations", type=str, default=None, help="Path to researcher-verified CSV.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model artifact (e.g. artifacts/baseline_svm/model.pkl).")
    parser.add_argument("--model-a-metrics", type=str, default=None, help="Path to JSON metrics for Model A.")
    parser.add_argument("--model-b-metrics", type=str, default=None, help="Path to JSON metrics for Model B.")
    parser.add_argument("--output-metrics", type=str, default="reports/tables/human_verification_evaluation.json", help="Path to save evaluation output JSON.")
    args = parser.parse_args()

    logger = get_logger("evaluate_script")

    if args.human_annotations:
        logger.info(f"Importing and validating human annotations from {args.human_annotations}...")
        report = import_human_annotations(args.human_annotations, original_df=None)
        print("\n" + "=" * 65)
        print("HUMAN ANNOTATION VERIFICATION REPORT")
        print("=" * 65)
        print(f"Total Samples in Batch:     {report['total_samples']}")
        print(f"Researcher Reviewed:       {report['reviewed_count']} ({report['completion_rate_pct']:.1f}%)")
        print(f"Remaining Unreviewed:      {report['unreviewed_count']}")
        if report['reviewed_count'] > 0:
            print(f"Disagreement vs Pseudo-Labels: {report['disagreement_rate_pct']:.2f}%")
            print(f"Agreement vs Pseudo-Labels:    {100.0 - report['disagreement_rate_pct']:.2f}%")
        print("=" * 65)

        ann_df = report["data"]
        reviewed_mask = ann_df["verification_status"].isin(["verified", "adjudicated"]) & ann_df["annotator_label"].isin(["positive", "negative"])
        reviewed_df = ann_df[reviewed_mask].copy()

        eval_summary = {
            "annotation_progress": {
                "total_samples": report["total_samples"],
                "reviewed_count": report["reviewed_count"],
                "unreviewed_count": report["unreviewed_count"],
                "completion_rate_pct": report["completion_rate_pct"],
                "disagreement_rate_pct": report["disagreement_rate_pct"],
            }
        }

        # If a trained model is provided, evaluate model against human ground truth
        if args.model_path and len(reviewed_df) > 0:
            logger.info(f"Evaluating model {args.model_path} against human ground truth ({len(reviewed_df)} samples)...")
            from culsen_bn.models.svm import BaselineSVMClassifier
            from culsen_bn.evaluation.calibration import compute_calibration_metrics

            model = BaselineSVMClassifier.load(args.model_path)
            texts = reviewed_df["comment_text"].astype(str).tolist()
            y_true = (reviewed_df["annotator_label"] == "positive").astype(int).tolist()
            y_pred = model.predict(texts)
            y_prob = [p[1] for p in model.predict_proba(texts)]

            human_eval_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
            calib = compute_calibration_metrics(y_true, y_prob)

            print("\n" + "=" * 65)
            print("MODEL PERFORMANCE ON HUMAN GROUND TRUTH")
            print("=" * 65)
            print(f"Evaluated Samples: {len(y_true)}")
            print(f"Macro-F1:          {human_eval_metrics['macro_f1']:.4f}")
            print(f"Accuracy:          {human_eval_metrics['accuracy']:.4f}")
            print(f"Balanced Acc:      {human_eval_metrics['balanced_accuracy']:.4f}")
            print(f"MCC:               {human_eval_metrics['mcc']:.4f}")
            print(f"Positive F1:       {human_eval_metrics['positive_f1']:.4f} (Prec: {human_eval_metrics['positive_precision']:.4f}, Rec: {human_eval_metrics['positive_recall']:.4f})")
            print(f"Negative F1:       {human_eval_metrics['negative_f1']:.4f} (Prec: {human_eval_metrics['negative_precision']:.4f}, Rec: {human_eval_metrics['negative_recall']:.4f})")
            print(f"Calibration (ECE): {calib['expected_calibration_error']:.4f}")
            print("=" * 65)

            eval_summary["human_ground_truth_metrics"] = human_eval_metrics
            eval_summary["calibration"] = calib

        out_path = Path(args.output_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(eval_summary, str(out_path))
        logger.info(f"Saved evaluation summary to {out_path}")

    if args.model_a_metrics and args.model_b_metrics:
        m_a = load_json(args.model_a_metrics)
        m_b = load_json(args.model_b_metrics)
        logger.info("Loaded metrics for Model A and Model B.")


if __name__ == "__main__":
    main()
