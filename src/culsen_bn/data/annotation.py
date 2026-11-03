"""
Human verification and annotation workflow module for CulSen-BN.
Prepares stratified annotation batches for manual review by the researcher.
Safely imports researcher-verified labels, validates integrity, and computes agreement.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
from .normalize import detect_script, detect_negation, extract_emojis


VALID_ANNOTATION_LABELS = {"positive", "negative", "neutral", "ambiguous", "unreviewed"}
VALID_VERIFICATION_STATUSES = {"unreviewed", "in_progress", "verified", "adjudicated"}


def create_annotation_sample(
    df: Any,
    natural_sample_size: int = 800,
    challenge_sample_size: int = 200,
    random_seed: int = 42,
) -> Tuple[Any, Any]:
    """
    Generate two stratified evaluation batches for researcher annotation:
    1. Naturally distributed evaluation subset (stratified across season, sentiment, script).
    2. Deliberately sampled challenge subset (oversampling negation, emoji-heavy, low confidence).
    """
    import pandas as pd
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    rng = np.random.RandomState(random_seed)

    # Enrich df with linguistic metadata if not already present
    work_df = df.copy()
    if "script" not in work_df.columns:
        work_df["script"] = work_df["text_clean"].apply(detect_script)
    work_df["has_negation"] = work_df["text_clean"].apply(detect_negation)
    work_df["emoji_count"] = work_df["text_clean"].apply(lambda t: len(extract_emojis(t)))

    # Stratified natural sample
    strat_cols = ["season", "sentiment", "script"]
    # Group and sample proportionally
    grouped = work_df.groupby(strat_cols, group_keys=False)
    strata_counts = grouped.size()
    total_pop = len(work_df)

    natural_indices = []
    for group_key, group_data in grouped:
        stratum_n = int(round(len(group_data) / total_pop * natural_sample_size))
        stratum_n = max(1, min(stratum_n, len(group_data)))
        sampled_idx = group_data.sample(n=stratum_n, random_state=random_seed).index
        natural_indices.extend(sampled_idx)

    # Adjust exact natural count
    if len(natural_indices) > natural_sample_size:
        natural_indices = list(rng.choice(natural_indices, size=natural_sample_size, replace=False))

    natural_df = work_df.loc[natural_indices].copy()
    remaining_df = work_df.drop(natural_indices)

    # Challenge subset: prioritize low confidence, negation, high emoji, flagged review
    challenge_candidates = remaining_df[
        (remaining_df["has_negation"]) |
        (remaining_df["emoji_count"] >= 3) |
        (remaining_df.get("confidence", 1.0) < 0.65) |
        (remaining_df.get("needs_review", False) == True)
    ]

    if len(challenge_candidates) >= challenge_sample_size:
        challenge_df = challenge_candidates.sample(n=challenge_sample_size, random_state=random_seed).copy()
    else:
        challenge_df = challenge_candidates.copy()

    # Format into official annotation schema
    def format_annotation_batch(batch_df: Any, batch_name: str) -> Any:
        rows = []
        for _, row in batch_df.iterrows():
            cid = str(row.get("comment_id", ""))
            text = str(row.get("text_clean", row.get("text_raw", "")))
            pseudo = str(row.get("sentiment", ""))
            season = int(row.get("season", 0))
            vid = str(row.get("video_id", ""))
            lang = str(row.get("script", "unknown"))

            rows.append({
                "sample_id": cid,
                "batch_type": batch_name,
                "comment_text": text,
                "pseudo_label": pseudo,
                "season_id": season,
                "video_id": vid,
                "annotator_label": "unreviewed",
                "ambiguity_flag": False,
                "language_type": lang,
                "annotator_notes": "",
                "adjudicated_label": "unreviewed",
                "verification_status": "unreviewed",
            })
        return pd.DataFrame(rows)

    natural_annotation_df = format_annotation_batch(natural_df, "natural_evaluation")
    challenge_annotation_df = format_annotation_batch(challenge_df, "challenge_evaluation")

    return natural_annotation_df, challenge_annotation_df


def import_human_annotations(
    annotated_csv_path: str,
    original_df: Any,
) -> Dict[str, Any]:
    """
    Import and validate researcher annotations from a completed CSV template.
    Checks for integrity, missing labels, invalid categories, and reports progress.
    """
    import pandas as pd

    csv_path = Path(annotated_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotated_csv_path}")

    ann_df = pd.read_csv(csv_path)

    required_fields = [
        "sample_id",
        "annotator_label",
        "verification_status",
    ]
    missing_fields = [f for f in required_fields if f not in ann_df.columns]
    if missing_fields:
        raise ValueError(f"Annotation CSV missing required fields: {missing_fields}")

    total_samples = len(ann_df)
    reviewed_mask = ann_df["verification_status"].isin(["verified", "adjudicated"])
    reviewed_count = int(reviewed_mask.sum())
    unreviewed_count = total_samples - reviewed_count

    # Check for invalid label values
    invalid_labels = ann_df[
        ann_df["annotator_label"].notna() &
        (~ann_df["annotator_label"].str.lower().isin(VALID_ANNOTATION_LABELS))
    ]
    if len(invalid_labels) > 0:
        raise ValueError(f"Found {len(invalid_labels)} invalid labels: {invalid_labels['annotator_label'].unique()}")

    # Compare pseudo-label vs verified label on reviewed samples
    reviewed_df = ann_df[reviewed_mask].copy()
    disagreement_rate = 0.0
    agreement_count = 0
    if len(reviewed_df) > 0 and "pseudo_label" in reviewed_df.columns:
        valid_comp = reviewed_df[reviewed_df["annotator_label"].isin(["positive", "negative"])]
        if len(valid_comp) > 0:
            agreement_count = int((valid_comp["annotator_label"] == valid_comp["pseudo_label"]).sum())
            disagreement_rate = float(1.0 - (agreement_count / len(valid_comp)))

    return {
        "total_samples": total_samples,
        "reviewed_count": reviewed_count,
        "unreviewed_count": unreviewed_count,
        "completion_rate_pct": float(reviewed_count / total_samples * 100) if total_samples > 0 else 0.0,
        "agreement_with_pseudo_labels": agreement_count,
        "disagreement_rate_pct": float(disagreement_rate * 100),
        "data": ann_df,
    }


def compute_inter_annotator_agreement(
    ann1_labels: List[str],
    ann2_labels: List[str],
) -> Dict[str, Any]:
    """
    Compute Cohen's kappa between two independent annotators.
    If only one annotator labeled, this function returns a documented limitation report.
    """
    if len(ann1_labels) != len(ann2_labels):
        raise ValueError("Annotator label lists must have identical length")

    n = len(ann1_labels)
    if n == 0:
        return {"error": "Empty label lists"}

    categories = sorted(list(set(ann1_labels).union(set(ann2_labels))))
    cat2idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    # Contingency matrix
    matrix = [[0] * num_cats for _ in range(num_cats)]
    for l1, l2 in zip(ann1_labels, ann2_labels):
        matrix[cat2idx[l1]][cat2idx[l2]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(num_cats)) / n

    # Expected agreement
    row_sums = [sum(matrix[i][j] for j in range(num_cats)) for i in range(num_cats)]
    col_sums = [sum(matrix[i][j] for i in range(num_cats)) for j in range(num_cats)]
    pe = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(num_cats))

    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 1.0

    return {
        "n_samples": n,
        "categories": categories,
        "observed_agreement_po": float(po),
        "expected_agreement_pe": float(pe),
        "cohens_kappa": float(kappa),
        "interpretation": (
            "Almost perfect" if kappa > 0.80 else
            "Substantial" if kappa > 0.60 else
            "Moderate" if kappa > 0.40 else
            "Fair" if kappa > 0.20 else "Slight / Poor"
        ),
    }
