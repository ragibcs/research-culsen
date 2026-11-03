#!/usr/bin/env python3
"""
CLI script to train and evaluate the classical Char TF-IDF + Linear SVM baseline.
Runs on CPU.
"""

import argparse
from pathlib import Path
from culsen_bn.training.train_svm import run_svm_experiment


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Char TF-IDF + Linear SVM baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline_svm.yaml", help="Path to SVM config YAML.")
    args = parser.parse_args()

    results = run_svm_experiment(config_path=args.config)
    test_m = results["test_metrics"]
    print("\n" + "=" * 60)
    print("SVM BASELINE TRAINING AND EVALUATION FINISHED")
    print("=" * 60)
    print(f"Training Time:  {results['training_time_seconds']:.2f} seconds")
    print(f"Macro-F1:       {test_m['macro_f1']:.4f}")
    print(f"Accuracy:       {test_m['accuracy']:.4f}")
    print(f"Balanced Acc:   {test_m['balanced_accuracy']:.4f}")
    print(f"MCC:            {test_m['mcc']:.4f}")
    print(f"Positive F1:    {test_m['positive_f1']:.4f} (Prec: {test_m['positive_precision']:.4f}, Rec: {test_m['positive_recall']:.4f})")
    print(f"Negative F1:    {test_m['negative_f1']:.4f} (Prec: {test_m['negative_precision']:.4f}, Rec: {test_m['negative_recall']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
