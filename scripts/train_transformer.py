#!/usr/bin/env python3
"""
CLI script to train and evaluate transformer models (BanglaBERT / XLM-RoBERTa / CulSen-BN).
Supports Colab GPU, Apple MPS, or CPU fallback.
"""

import argparse
from pathlib import Path
from culsen_bn.training.train_transformer import train_transformer_baseline
from culsen_bn.training.train_culsen import train_culsen_model


def main():
    parser = argparse.ArgumentParser(description="Train transformer models for Bengali sentiment analysis.")
    parser.add_argument("--config", type=str, default="configs/baseline_transformer.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="banglabert", choices=["banglabert", "xlm_roberta", "culsen"], help="Model type to train.")
    parser.add_argument("--variant", type=str, default="B5_full", help="Ablation variant for CulSen model.")
    args = parser.parse_args()

    if args.model == "culsen":
        results = train_culsen_model(
            experiment_config="configs/experiment.yaml",
            variant_name=args.variant,
        )
    else:
        results = train_transformer_baseline(
            config_path=args.config,
            model_override=args.model,
        )

    test_m = results["test_metrics"]
    print("\n" + "=" * 60)
    print(f"{args.model.upper()} TRAINING AND EVALUATION FINISHED")
    print("=" * 60)
    print(f"Model:          {results.get('model_name', args.model)}")
    print(f"Training Time:  {results.get('training_time_seconds', 0.0):.1f}s")
    print(f"Best Epoch:     {results.get('best_epoch', 1)}")
    print(f"Validation F1:  {results.get('validation_macro_f1', 0.0):.4f}")
    print(f"Test Macro-F1:  {test_m['macro_f1']:.4f}")
    print(f"Test Accuracy:  {test_m['accuracy']:.4f}")
    print(f"Test MCC:       {test_m['mcc']:.4f}")
    print(f"Test ECE:       {results.get('calibration', {}).get('expected_calibration_error', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
