"""
Deduplication and cross-video contamination analysis for Coke Studio Bangla Sentiment.
"""

from typing import Any, Dict, List, Set, Tuple
from .normalize import normalize_bengali_text


def analyze_duplicates(df: Any, text_col: str = "text_clean", video_col: str = "video_id") -> Dict[str, Any]:
    """
    Analyze exact duplicates, normalized duplicates, and cross-video text duplicates.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    norm_texts = df[text_col].astype(str).apply(normalize_bengali_text)

    # Cross-video duplicates: identical text posted under multiple songs/videos
    cross_video_texts: Set[str] = set()
    if video_col in df.columns:
        video_counts = df.groupby(norm_texts)[video_col].nunique()
        cross_video_texts = set(video_counts[video_counts > 1].index)

    return {
        "total_records": len(df),
        "unique_exact_texts": int(df[text_col].nunique()),
        "unique_normalized_texts": int(norm_texts.nunique()),
        "cross_video_duplicate_count": len(cross_video_texts),
        "cross_video_duplicate_texts": list(cross_video_texts)[:100],  # sample
    }
