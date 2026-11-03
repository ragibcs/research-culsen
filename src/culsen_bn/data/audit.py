"""
Comprehensive dataset audit module for Coke Studio Bangla Sentiment Dataset.
Calculates schema, distribution, duplication, linguistic script, emoji, and negation statistics.
Generates machine-readable JSON and human-readable Markdown reports.
"""

from typing import Any, Dict, List
import json
from pathlib import Path
from .normalize import normalize_bengali_text, detect_script, detect_negation, extract_emojis


def audit_dataset(df: Any) -> Dict[str, Any]:
    """
    Perform a complete audit on the loaded dataset DataFrame.
    """
    import pandas as pd
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    total_rows = len(df)
    columns_info = {col: str(df[col].dtype) for col in df.columns}
    missing_values = {col: int(df[col].isna().sum()) for col in df.columns}

    # Sentiment distribution
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    sentiment_pct = {k: float(v / total_rows * 100) for k, v in sentiment_counts.items()}

    # Confidence statistics
    conf_stats = {}
    if "confidence" in df.columns:
        conf_series = df["confidence"].dropna()
        conf_stats = {
            "mean": float(conf_series.mean()),
            "std": float(conf_series.std()),
            "min": float(conf_series.min()),
            "median": float(conf_series.median()),
            "max": float(conf_series.max()),
            "below_0_60": int((conf_series < 0.60).sum()),
            "below_0_70": int((conf_series < 0.70).sum()),
        }

    # Season and Video statistics
    seasons = sorted([int(s) for s in df["season"].dropna().unique()])
    season_distribution = {}
    for s in seasons:
        s_df = df[df["season"] == s]
        season_distribution[str(s)] = {
            "total_comments": len(s_df),
            "sentiment_counts": {k: int(v) for k, v in s_df["sentiment"].value_counts().items()},
            "unique_videos": int(s_df["video_id"].nunique()) if "video_id" in df.columns else 0,
        }

    unique_videos_total = int(df["video_id"].nunique()) if "video_id" in df.columns else 0

    # Duplication analysis
    text_col = "text_clean" if "text_clean" in df.columns else df.columns[0]
    raw_texts = df[text_col].astype(str)
    exact_unique_texts = int(raw_texts.nunique())
    exact_duplicates = int(total_rows - exact_unique_texts)

    normalized_texts = raw_texts.apply(normalize_bengali_text)
    norm_unique_texts = int(normalized_texts.nunique())
    norm_duplicates = int(total_rows - norm_unique_texts)

    # Cross-video duplicates
    cross_video_duplicates = 0
    if "video_id" in df.columns:
        text_video_pairs = pd.DataFrame({"norm_text": normalized_texts, "video_id": df["video_id"]})
        text_video_counts = text_video_pairs.groupby("norm_text")["video_id"].nunique()
        cross_video_duplicates = int((text_video_counts > 1).sum())

    # Length distribution (character and word counts)
    char_lengths = raw_texts.apply(len)
    word_lengths = raw_texts.apply(lambda t: len(t.split()))

    length_stats = {
        "char_length": {
            "mean": float(char_lengths.mean()),
            "std": float(char_lengths.std()),
            "median": float(char_lengths.median()),
            "min": int(char_lengths.min()),
            "max": int(char_lengths.max()),
            "p25": float(np.percentile(char_lengths, 25)),
            "p75": float(np.percentile(char_lengths, 75)),
            "p95": float(np.percentile(char_lengths, 95)),
        },
        "word_length": {
            "mean": float(word_lengths.mean()),
            "std": float(word_lengths.std()),
            "median": float(word_lengths.median()),
            "min": int(word_lengths.min()),
            "max": int(word_lengths.max()),
            "p25": float(np.percentile(word_lengths, 25)),
            "p75": float(np.percentile(word_lengths, 75)),
            "p95": float(np.percentile(word_lengths, 95)),
        },
    }

    # Linguistic Script breakdown
    if "script" in df.columns and df["script"].notna().all():
        script_counts = df["script"].value_counts().to_dict()
    else:
        computed_scripts = raw_texts.apply(detect_script)
        script_counts = computed_scripts.value_counts().to_dict()

    script_pct = {k: float(v / total_rows * 100) for k, v in script_counts.items()}

    # Emoji and Negation frequencies
    has_emoji_count = int(raw_texts.apply(lambda t: len(extract_emojis(t)) > 0).sum())
    has_negation_count = int(raw_texts.apply(detect_negation).sum())

    # Needs review flag if available
    needs_review_count = int(df["needs_review"].sum()) if "needs_review" in df.columns else 0

    audit_report = {
        "dataset_name": "Coke Studio Bangla Sentiment Dataset",
        "total_records": total_rows,
        "total_columns": len(df.columns),
        "columns": columns_info,
        "missing_values": missing_values,
        "sentiment_distribution": {
            "counts": {k: int(v) for k, v in sentiment_counts.items()},
            "percentages": sentiment_pct,
        },
        "confidence_statistics": conf_stats,
        "unique_seasons": seasons,
        "unique_videos_count": unique_videos_total,
        "season_distribution": season_distribution,
        "duplication": {
            "exact_duplicate_comments": exact_duplicates,
            "normalized_duplicate_comments": norm_duplicates,
            "distinct_normalized_texts": norm_unique_texts,
            "texts_spanning_multiple_videos": cross_video_duplicates,
        },
        "length_statistics": length_stats,
        "script_distribution": {
            "counts": {k: int(v) for k, v in script_counts.items()},
            "percentages": script_pct,
        },
        "linguistic_features": {
            "comments_with_emojis": has_emoji_count,
            "comments_with_emojis_pct": float(has_emoji_count / total_rows * 100),
            "comments_with_negation": has_negation_count,
            "comments_with_negation_pct": float(has_negation_count / total_rows * 100),
            "flagged_needs_review": needs_review_count,
            "flagged_needs_review_pct": float(needs_review_count / total_rows * 100),
        },
    }

    return audit_report


def generate_markdown_audit_report(report: Dict[str, Any]) -> str:
    """
    Format the audit dictionary into a publication-ready Markdown report.
    """
    lines = [
        "# CulSen-BN: Dataset Audit & Quality Report",
        "",
        f"**Dataset Name:** {report.get('dataset_name')}",
        f"**Total Records:** {report.get('total_records'):,}",
        f"**Total Columns:** {report.get('total_columns')}",
        f"**Unique Videos:** {report.get('unique_videos_count')}",
        f"**Seasons Covered:** {', '.join(str(s) for s in report.get('unique_seasons', []))}",
        "",
        "## 1. Sentiment Class Distribution (Pseudo-Labels)",
        "| Sentiment Class | Count | Percentage (%) |",
        "| :--- | :--- | :--- |",
    ]

    sent_counts = report.get("sentiment_distribution", {}).get("counts", {})
    sent_pcts = report.get("sentiment_distribution", {}).get("percentages", {})
    for cls, cnt in sent_counts.items():
        pct = sent_pcts.get(cls, 0.0)
        lines.append(f"| `{cls}` | {cnt:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "## 2. Season-wise Breakdown",
        "| Season | Total Comments | Unique Videos | Positive | Neutral | Negative |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    season_dist = report.get("season_distribution", {})
    for season, data in season_dist.items():
        sc = data.get("sentiment_counts", {})
        pos = sc.get("positive", 0)
        neu = sc.get("neutral", 0)
        neg = sc.get("negative", 0)
        lines.append(
            f"| Season {season} | {data.get('total_comments'):,} | {data.get('unique_videos')} | "
            f"{pos:,} | {neu:,} | {neg:,} |"
        )

    lines.extend([
        "",
        "## 3. Duplication and Leakage Hazards",
        f"- **Exact Duplicates:** {report.get('duplication', {}).get('exact_duplicate_comments'):,} comments",
        f"- **Normalized Duplicates:** {report.get('duplication', {}).get('normalized_duplicate_comments'):,} comments",
        f"- **Texts Spanning Multiple Videos:** {report.get('duplication', {}).get('texts_spanning_multiple_videos'):,} texts (must be isolated in splits)",
        "",
        "## 4. Script and Linguistic Characteristics",
        "| Script / Category | Count | Percentage (%) |",
        "| :--- | :--- | :--- |",
    ])

    script_counts = report.get("script_distribution", {}).get("counts", {})
    script_pcts = report.get("script_distribution", {}).get("percentages", {})
    for scr, cnt in script_counts.items():
        pct = script_pcts.get(scr, 0.0)
        lines.append(f"| `{scr}` | {cnt:,} | {pct:.2f}% |")

    ling = report.get("linguistic_features", {})
    lines.extend([
        "",
        "## 5. Salient Linguistic Features",
        f"- **Comments with Emojis:** {ling.get('comments_with_emojis', 0):,} ({ling.get('comments_with_emojis_pct', 0.0):.2f}%)",
        f"- **Comments with Negation Markers:** {ling.get('comments_with_negation', 0):,} ({ling.get('comments_with_negation_pct', 0.0):.2f}%)",
        f"- **Machine-Flagged for Review (`needs_review`):** {ling.get('flagged_needs_review', 0):,} ({ling.get('flagged_needs_review_pct', 0.0):.2f}%)",
        "",
        "## 6. Length Distribution Summary",
        f"- **Character Length:** Mean = {report.get('length_statistics', {}).get('char_length', {}).get('mean', 0.0):.1f}, "
        f"Median = {report.get('length_statistics', {}).get('char_length', {}).get('median', 0.0):.1f}, "
        f"Max = {report.get('length_statistics', {}).get('char_length', {}).get('max', 0)}",
        f"- **Word Length:** Mean = {report.get('length_statistics', {}).get('word_length', {}).get('mean', 0.0):.1f}, "
        f"Median = {report.get('length_statistics', {}).get('word_length', {}).get('median', 0.0):.1f}, "
        f"Max = {report.get('length_statistics', {}).get('word_length', {}).get('max', 0)}",
        "",
        "---",
        "*Note: Original sentiment labels are machine-generated pseudo-labels (`cardiffnlp/twitter-xlm-roberta-base-sentiment`). "
        "Ground truth validation must be conducted via manual human adjudication (Stage 2).*",
    ])

    return "\n".join(lines)
