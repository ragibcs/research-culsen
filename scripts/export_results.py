#!/usr/bin/env python3
"""
CLI script to aggregate experiment metrics into publication-ready Markdown and LaTeX tables.
"""

from pathlib import Path
import json
from culsen_bn.utils.logging import get_logger, load_json, save_json


def main():
    logger = get_logger("export_results")
    tables_dir = Path("reports/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    svm_metrics_path = tables_dir / "baseline_svm_metrics.json"
    audit_path = tables_dir / "dataset_audit.json"

    logger.info("Aggregating available experiment outputs into publication tables...")

    # Table 1: Dataset Statistics
    if audit_path.exists():
        audit = load_json(str(audit_path))
        lines_t1 = [
            "# Table 1: Coke Studio Bangla Dataset Characteristics",
            "",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| Total Comments | {audit['total_records']:,} | Scraped YouTube comments (Seasons 1–4) |",
            f"| Unique Seasons | {len(audit['unique_seasons'])} | Seasons 1, 2, 3, 4 |",
            f"| Unique Songs / Videos | {audit['unique_videos_count']} | Distinct Coke Studio Bangla tracks |",
            f"| Positive Comments | {audit['sentiment_distribution']['counts'].get('positive', 0):,} ({audit['sentiment_distribution']['percentages'].get('positive', 0.0):.1f}%) | Machine-labeled positive |",
            f"| Neutral Comments | {audit['sentiment_distribution']['counts'].get('neutral', 0):,} ({audit['sentiment_distribution']['percentages'].get('neutral', 0.0):.1f}%) | Machine-labeled neutral |",
            f"| Negative Comments | {audit['sentiment_distribution']['counts'].get('negative', 0):,} ({audit['sentiment_distribution']['percentages'].get('negative', 0.0):.1f}%) | Machine-labeled negative |",
            f"| Bengali Script Ratio | {audit['script_distribution']['percentages'].get('bengali', 0.0):.1f}% | Comments predominantly in Bengali script |",
            f"| Banglish / Latin Ratio | {audit['script_distribution']['percentages'].get('banglish_latin', 0.0):.1f}% | Comments in Romanized Bengali or English |",
            f"| Code-Mixed Ratio | {audit['script_distribution']['percentages'].get('mixed', 0.0):.1f}% | Intra-comment Bengali-Latin script mixing |",
            f"| Comments with Emojis | {audit['linguistic_features']['comments_with_emojis']:,} ({audit['linguistic_features']['comments_with_emojis_pct']:.1f}%) | Comments containing Unicode emojis |",
            f"| Comments with Negation | {audit['linguistic_features']['comments_with_negation']:,} ({audit['linguistic_features']['comments_with_negation_pct']:.1f}%) | Contains Bengali or Banglish negation markers |",
        ]
        with open(tables_dir / "table1_dataset_characteristics.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines_t1))
        logger.info(f"Generated Table 1 at {tables_dir / 'table1_dataset_characteristics.md'}")

    # Table 2: Model Comparison Table
    lines_t2 = [
        "# Table 2: Primary Benchmark Results (In-Domain Video-Grouped Test Set)",
        "",
        "| Architecture | Supervision / Loss | Macro-F1 | Accuracy | Pos F1 | Neg F1 | MCC | ECE (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    if svm_metrics_path.exists():
        svm_m = load_json(str(svm_metrics_path))
        tm = svm_m["test_metrics"]
        cal = svm_m.get("calibration", {})
        lines_t2.append(
            f"| Char TF-IDF + Linear SVM | Standard Hinge Loss | {tm['macro_f1']:.4f} | {tm['accuracy']:.4f} | "
            f"{tm['positive_f1']:.4f} | {tm['negative_f1']:.4f} | {tm['mcc']:.4f} | {cal.get('expected_calibration_error', 0.0)*100:.2f}% |"
        )
    else:
        lines_t2.append("| Char TF-IDF + Linear SVM | Standard Hinge Loss | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |")

    lines_t2.extend([
        "| BanglaBERT Baseline | Cross-Entropy | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* |",
        "| XLM-RoBERTa Baseline | Cross-Entropy | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* |",
        "| **CulSen-BN (Proposed)** | Integrated Multi-Task Loss | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* | *[Pending GPU]* |",
    ])

    with open(tables_dir / "table2_model_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines_t2))
    logger.info(f"Generated Table 2 at {tables_dir / 'table2_model_comparison.md'}")

    print("Tables exported successfully in reports/tables/")


if __name__ == "__main__":
    main()
