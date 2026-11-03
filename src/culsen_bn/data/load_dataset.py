"""
Dataset loading and schema verification for Coke Studio Bangla Sentiment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os


REQUIRED_COLUMNS = [
    "comment_id",
    "video_id",
    "season",
    "text_clean",
    "sentiment",
    "confidence",
]


def validate_dataset_schema(columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Check whether all required columns are present in the dataset schema.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    return (len(missing) == 0, missing)


def load_coke_studio_dataset(
    source: Optional[str] = None,
    config_name: str = "full",
    use_hf_fallback: bool = True,
) -> Any:
    """
    Load the Coke Studio Bangla Sentiment dataset.
    Prioritizes local parquet files in data/raw/ before attempting HF download.

    Args:
        source: Explicit path to parquet file or HF repo id.
        config_name: 'full' (26 columns) or 'curated' (16 columns).
        use_hf_fallback: Whether to attempt Hugging Face datasets fallback.

    Returns:
        pandas.DataFrame or Dataset object with verified schema.
    """
    # 1. Check local file paths
    candidate_paths = []
    if source and os.path.exists(source):
        candidate_paths.append(Path(source))

    if config_name == "curated":
        candidate_paths.extend([
            Path("data/raw/curated.parquet"),
            Path("data/curated/train-00000-of-00001.parquet"),
        ])
    else:
        candidate_paths.extend([
            Path("data/raw/full.parquet"),
            Path("data/full/train-00000-of-00001.parquet"),
        ])

    for p in candidate_paths:
        if p.exists() and p.stat().st_size > 1000:  # Avoid LFS pointers (<1000 bytes)
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                is_valid, missing = validate_dataset_schema(list(df.columns))
                if not is_valid:
                    raise ValueError(f"Dataset at {p} missing required columns: {missing}")
                return df
            except ImportError:
                # If pandas is not installed, try pyarrow directly
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(p)
                    columns = table.column_names
                    is_valid, missing = validate_dataset_schema(columns)
                    if not is_valid:
                        raise ValueError(f"Dataset at {p} missing required columns: {missing}")
                    return table
                except ImportError:
                    pass

    # 2. HF Datasets Fallback
    if use_hf_fallback:
        try:
            from datasets import load_dataset
            repo_id = source if source and "/" in source else "ragibhasan/coke_studio_bangla_sentiment"
            hf_config = "curated" if config_name == "curated" else "default"
            ds = load_dataset(repo_id, hf_config, split="train")
            try:
                import pandas as pd
                df = ds.to_pandas()
                is_valid, missing = validate_dataset_schema(list(df.columns))
                if not is_valid:
                    raise ValueError(f"HF Dataset missing required columns: {missing}")
                return df
            except ImportError:
                return ds
        except Exception as e:
            raise RuntimeError(
                f"Could not load dataset from local parquet or Hugging Face. "
                f"Ensure data/raw/ contains real parquet files and pandas/pyarrow are installed. Error: {e}"
            )

    raise FileNotFoundError(
        f"No valid local parquet file found for config '{config_name}'. Checked: {[str(p) for p in candidate_paths]}"
    )
