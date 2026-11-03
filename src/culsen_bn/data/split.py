"""
Leakage-free dataset splitting module for CulSen-BN.
Implements:
1. Protocol A: Video-grouped in-domain split (~70/15/15) with cross-partition deduplication.
2. Protocol B: Held-out season splits (Leave-One-Season-Out).
3. Automated leakage assertions for video IDs, comment IDs, and normalized text.
"""

from typing import Any, Dict, List, Set, Tuple
from pathlib import Path
from .normalize import normalize_bengali_text


def verify_no_leakage(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    group_col: str = "video_id",
    text_col: str = "text_clean",
    id_col: str = "comment_id",
    check_groups: bool = True,
) -> Dict[str, Any]:
    """
    Verify that no data leakage exists between train, val, and test splits.
    Checks:
    - ID overlap (comment_id)
    - Group overlap (video_id) if check_groups=True
    - Normalized text overlap
    """
    import pandas as pd

    train_ids = set(train_df[id_col].astype(str))
    val_ids = set(val_df[id_col].astype(str))
    test_ids = set(test_df[id_col].astype(str))

    id_overlap_train_val = train_ids.intersection(val_ids)
    id_overlap_train_test = train_ids.intersection(test_ids)
    id_overlap_val_test = val_ids.intersection(test_ids)

    total_id_overlap = len(id_overlap_train_val) + len(id_overlap_train_test) + len(id_overlap_val_test)
    if total_id_overlap > 0:
        raise AssertionError(
            f"ID Leakage detected! Train-Val: {len(id_overlap_train_val)}, "
            f"Train-Test: {len(id_overlap_train_test)}, Val-Test: {len(id_overlap_val_test)}"
        )

    # Group overlap (video_id)
    group_overlap_report = {}
    if check_groups and group_col in train_df.columns:
        train_groups = set(train_df[group_col].dropna().astype(str))
        val_groups = set(val_df[group_col].dropna().astype(str))
        test_groups = set(test_df[group_col].dropna().astype(str))

        g_tv = train_groups.intersection(val_groups)
        g_tt = train_groups.intersection(test_groups)
        g_vt = val_groups.intersection(test_groups)

        total_group_overlap = len(g_tv) + len(g_tt) + len(g_vt)
        if total_group_overlap > 0:
            raise AssertionError(
                f"Video Group Leakage detected! Train-Val: {g_tv}, Train-Test: {g_tt}, Val-Test: {g_vt}"
            )
        group_overlap_report = {
            "train_unique_videos": len(train_groups),
            "val_unique_videos": len(val_groups),
            "test_unique_videos": len(test_groups),
        }

    # Normalized text overlap
    norm_train_text = set(train_df[text_col].astype(str).apply(normalize_bengali_text))
    norm_val_text = set(val_df[text_col].astype(str).apply(normalize_bengali_text))
    norm_test_text = set(test_df[text_col].astype(str).apply(normalize_bengali_text))

    txt_tv = norm_train_text.intersection(norm_val_text)
    txt_tt = norm_train_text.intersection(norm_test_text)
    txt_vt = norm_val_text.intersection(norm_test_text)

    total_text_overlap = len(txt_tv) + len(txt_tt) + len(txt_vt)
    if total_text_overlap > 0:
        raise AssertionError(
            f"Text Content Leakage detected! Identical normalized text appears across partitions: "
            f"Train-Val: {len(txt_tv)}, Train-Test: {len(txt_tt)}, Val-Test: {len(txt_vt)}"
        )

    return {
        "status": "PASSED",
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "group_check": group_overlap_report,
        "text_leakage_count": 0,
        "id_leakage_count": 0,
    }


def create_protocol_a_splits(
    df: Any,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_col: str = "video_id",
    text_col: str = "text_clean",
    id_col: str = "comment_id",
    random_seed: int = 42,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Generate video-grouped in-domain train, validation, and test splits.
    Ensures that:
    1. All comments for any video appear in exactly one partition.
    2. Normalized duplicate comments spanning multiple videos are isolated to prevent cross-partition text leakage.
    """
    import pandas as pd
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Video-grouped partitioning stratified across seasons
    work_df = df.copy()
    work_df["norm_text"] = work_df[text_col].astype(str).apply(normalize_bengali_text)

    rng = np.random.RandomState(random_seed)

    # Partition videos season by season to preserve temporal and artistic balance
    train_videos: List[str] = []
    val_videos: List[str] = []
    test_videos: List[str] = []

    seasons = sorted(work_df["season"].unique()) if "season" in work_df.columns else [1]
    for s in seasons:
        s_df = work_df[work_df["season"] == s] if "season" in work_df.columns else work_df
        vids = sorted(list(s_df[group_col].dropna().unique()))
        rng.shuffle(vids)
        n = len(vids)
        if n >= 3:
            n_test = max(1, int(round(n * test_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            test_v = vids[:n_test]
            val_v = vids[n_test:n_test + n_val]
            train_v = vids[n_test + n_val:]
        elif n == 2:
            test_v = [vids[0]]
            val_v = []
            train_v = [vids[1]]
        else:
            test_v, val_v, train_v = [], [], vids

        train_videos.extend(train_v)
        val_videos.extend(val_v)
        test_videos.extend(test_v)

    # 1. Base candidates grouped strictly by video
    train_cand = work_df[work_df[group_col].isin(train_videos)].copy()
    val_cand = work_df[work_df[group_col].isin(val_videos)].copy()
    test_cand = work_df[work_df[group_col].isin(test_videos)].copy()

    # 2. Strict cross-partition text deduplication:
    # Ensure zero text leakage into validation and test sets
    train_texts = set(train_cand["norm_text"])
    val_df = val_cand[~val_cand["norm_text"].isin(train_texts)].copy()

    val_texts = set(val_df["norm_text"])
    forbidden_for_test = train_texts.union(val_texts)
    test_df = test_cand[~test_cand["norm_text"].isin(forbidden_for_test)].copy()
    train_df = train_cand.copy()

    # Assign split labels
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Run strict verification
    leakage_report = verify_no_leakage(
        train_df, val_df, test_df,
        group_col=group_col,
        text_col=text_col,
        id_col=id_col,
        check_groups=True,
    )

    total_assigned = len(train_df) + len(val_df) + len(test_df)
    manifest = {
        "protocol": "Protocol A (Video-Grouped In-Domain)",
        "random_seed": random_seed,
        "total_samples": total_assigned,
        "train_samples": len(train_df),
        "train_pct": float(len(train_df) / total_assigned * 100),
        "val_samples": len(val_df),
        "val_pct": float(len(val_df) / total_assigned * 100),
        "test_samples": len(test_df),
        "test_pct": float(len(test_df) / total_assigned * 100),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "leakage_verification": leakage_report,
    }

    return train_df, val_df, test_df, manifest


def create_protocol_b_splits(
    df: Any,
    test_season: int,
    val_ratio: float = 0.15,
    group_col: str = "video_id",
    text_col: str = "text_clean",
    id_col: str = "comment_id",
    random_seed: int = 42,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Generate Protocol B: Held-out season generalization split.
    All data from `test_season` is held out strictly for final testing.
    Remaining seasons are partitioned into video-grouped train and val.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    work_df = df.copy()
    work_df["norm_text"] = work_df[text_col].astype(str).apply(normalize_bengali_text)

    test_season_df = work_df[work_df["season"] == test_season].copy()
    source_seasons_df = work_df[work_df["season"] != test_season].copy()

    # Drop any text in source seasons that exactly matches test season text to prevent leakage
    test_norm_texts = set(test_season_df["norm_text"])
    clean_source_df = source_seasons_df[~source_seasons_df["norm_text"].isin(test_norm_texts)].copy()

    # Video-grouped train/val split on source seasons
    source_videos = sorted(list(clean_source_df[group_col].dropna().unique()))
    import numpy as np
    rng = np.random.RandomState(random_seed)
    rng.shuffle(source_videos)

    total_source = len(clean_source_df)
    target_val = int(total_source * val_ratio)

    val_videos: List[str] = []
    val_cnt = 0
    for v in source_videos:
        cnt = (clean_source_df[group_col] == v).sum()
        if val_cnt + cnt <= target_val * 1.1 or val_cnt == 0:
            val_videos.append(v)
            val_cnt += cnt
        else:
            break

    train_videos = [v for v in source_videos if v not in val_videos]

    train_cand = clean_source_df[clean_source_df[group_col].isin(train_videos)].copy()
    val_cand = clean_source_df[clean_source_df[group_col].isin(val_videos)].copy()

    # Deduplicate normalized texts between train and val to prevent in-domain leakage
    train_texts = set(train_cand["norm_text"])
    val_df = val_cand[~val_cand["norm_text"].isin(train_texts)].copy()
    train_df = train_cand.copy()
    test_df = test_season_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Verify no overlap
    leakage_report = verify_no_leakage(
        train_df, val_df, test_df,
        group_col=group_col,
        text_col=text_col,
        id_col=id_col,
        check_groups=True,
    )

    manifest = {
        "protocol": f"Protocol B (Held-Out Season {test_season})",
        "held_out_test_season": test_season,
        "source_seasons": sorted(list(source_seasons_df["season"].unique())),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "leakage_verification": leakage_report,
    }

    return train_df, val_df, test_df, manifest
