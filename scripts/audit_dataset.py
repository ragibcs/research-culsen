#!/usr/bin/env python3
"""
CLI script to audit the Coke Studio Bangla Sentiment Dataset.
Outputs:
1. JSON machine-readable audit report in reports/tables/dataset_audit.json
2. Markdown summary report in reports/dataset_audit.md
3. Exported summary CSV tables in reports/tables/
"""

import argparse
import sys
from pathlib import Path
import yaml

from culsen_bn.data.load_dataset import load_coke_studio_dataset
from culsen_bn.data.audit import audit_dataset, generate_markdown_audit_report
from culsen_bn.utils.logging import get_logger, save_json


def main():
    parser = argparse.ArgumentParser(description="Audit Coke Studio Bangla Sentiment Dataset.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument("--source", type=str, default=None, help="Optional direct parquet path or HF repo ID.")
    parser.add_argument("--config-name", type=str, default="full", choices=["full", "curated"], help="Dataset configuration.")
    args = parser.parse_args()

    logger = get_logger("audit_script")

    # Load config
    cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    logger.info("Loading dataset for audit...")
    df = load_coke_studio_dataset(source=args.source, config_name=args.config_name)
    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns.")

    logger.info("Computing comprehensive dataset statistics...")
    report = audit_dataset(df)

    # Save outputs
    reports_dir = Path("reports")
    tables_dir = Path("reports/tables")
    figures_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    json_path = tables_dir / "dataset_audit.json"
    save_json(report, str(json_path))
    logger.info(f"Saved machine-readable audit to {json_path}")

    md_content = generate_markdown_audit_report(report)
    md_path = reports_dir / "dataset_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved Markdown audit report to {md_path}")

    # Generate figures if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. Sentiment by Season Figure
        plt.figure(figsize=(8, 5))
        season_data = report["season_distribution"]
        seasons = sorted(season_data.keys())
        pos_counts = [season_data[s]["sentiment_counts"].get("positive", 0) for s in seasons]
        neu_counts = [season_data[s]["sentiment_counts"].get("neutral", 0) for s in seasons]
        neg_counts = [season_data[s]["sentiment_counts"].get("negative", 0) for s in seasons]

        import numpy as np
        x = np.arange(len(seasons))
        width = 0.25

        plt.bar(x - width, pos_counts, width, label="Positive", color="#2ca02c")
        plt.bar(x, neu_counts, width, label="Neutral", color="#7f7f7f")
        plt.bar(x + width, neg_counts, width, label="Negative", color="#d62728")

        plt.xlabel("Coke Studio Bangla Season", fontsize=11, fontweight="bold")
        plt.ylabel("Number of Comments", fontsize=11, fontweight="bold")
        plt.title("Sentiment Distribution across Coke Studio Bangla Seasons 1–4", fontsize=12, fontweight="bold")
        plt.xticks(x, [f"Season {s}" for s in seasons])
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "sentiment_by_season.png", dpi=300)
        plt.close()

        # 2. Linguistic Script Distribution Figure
        plt.figure(figsize=(7, 4.5))
        scripts = report["script_distribution"]["counts"]
        script_labels = list(scripts.keys())
        script_vals = list(scripts.values())
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(scripts)]

        plt.pie(script_vals, labels=script_labels, autopct="%1.1f%%", startangle=140, colors=colors)
        plt.title("Distribution of Scripts (Bengali, Banglish/Latin, Mixed)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_dir / "script_distribution.png", dpi=300)
        plt.close()
        logger.info(f"Generated publication-quality figures in {figures_dir}")
    except Exception as e:
        logger.warning(f"Could not generate matplotlib figures (missing library or GUI): {e}")

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Comments:     {report['total_records']:,}")
    print(f"Unique Seasons:     {report['unique_seasons']}")
    print(f"Unique Videos:      {report['unique_videos_count']}")
    print(f"Sentiment Counts:   {report['sentiment_distribution']['counts']}")
    print(f"Exact Duplicates:   {report['duplication']['exact_duplicate_comments']:,}")
    print(f"Multi-Video Dups:   {report['duplication']['texts_spanning_multiple_videos']:,}")
    print(f"Script Breakdown:   {report['script_distribution']['percentages']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
