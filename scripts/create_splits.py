#!/usr/bin/env python3
"""
CLI script to create leakage-free dataset partitions and human annotation batches.
Outputs:
1. Protocol A: Video-grouped in-domain train/val/test splits (data/processed/splits_protocol_a.parquet)
2. Protocol B: Leave-one-season-out splits for Seasons 1, 2, 3, 4 (data/processed/splits_protocol_b_s*.parquet)
3. Annotation templates for researcher manual labeling (data/annotations/)
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd

from culsen_bn.data.load_dataset import load_coke_studio_dataset
from culsen_bn.data.split import create_protocol_a_splits, create_protocol_b_splits, verify_no_leakage
from culsen_bn.data.annotation import create_annotation_sample
from culsen_bn.utils.logging import get_logger, save_json
from culsen_bn.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-free data splits and annotation templates.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data configuration YAML.")
    args = parser.parse_args()

    logger = get_logger("splits_generator")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("splitting", {}).get("random_seed", 42)
    set_seed(seed)

    processed_dir = Path(cfg.get("dataset", {}).get("processed_dir", "data/processed"))
    annotations_dir = Path(cfg.get("dataset", {}).get("annotations_dir", "data/annotations"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    df = load_coke_studio_dataset()
    logger.info(f"Loaded {len(df):,} comments.")

    # 1. Protocol A: Video-Grouped In-Domain Split
    logger.info("Generating Protocol A (Video-Grouped In-Domain) splits...")
    proto_a_cfg = cfg.get("splitting", {}).get("protocol_a", {})
    train_a, val_a, test_a, manifest_a = create_protocol_a_splits(
        df=df,
        train_ratio=proto_a_cfg.get("train_ratio", 0.70),
        val_ratio=proto_a_cfg.get("val_ratio", 0.15),
        test_ratio=proto_a_cfg.get("test_ratio", 0.15),
        random_seed=seed,
    )

    combined_a = pd.concat([train_a, val_a, test_a], ignore_index=True)
    out_a_path = processed_dir / "splits_protocol_a.parquet"
    combined_a.to_parquet(out_a_path)
    save_json(manifest_a, str(processed_dir / "manifest_protocol_a.json"))
    logger.info(
        f"Saved Protocol A splits to {out_a_path}: Train={len(train_a):,} ({manifest_a['train_pct']:.1f}%), "
        f"Val={len(val_a):,} ({manifest_a['val_pct']:.1f}%), Test={len(test_a):,} ({manifest_a['test_pct']:.1f}%)"
    )

    # 2. Protocol B: Held-Out-Season Generalization Splits
    seasons = cfg.get("splitting", {}).get("protocol_b", {}).get("seasons", [1, 2, 3, 4])
    logger.info(f"Generating Protocol B splits for held-out seasons: {seasons}...")
    manifests_b = {}
    for season in seasons:
        train_b, val_b, test_b, manifest_b = create_protocol_b_splits(
            df=df,
            test_season=season,
            val_ratio=0.15,
            random_seed=seed,
        )
        combined_b = pd.concat([train_b, val_b, test_b], ignore_index=True)
        out_b_path = processed_dir / f"splits_protocol_b_season_{season}.parquet"
        combined_b.to_parquet(out_b_path)
        manifests_b[f"season_{season}"] = manifest_b
        logger.info(
            f"Saved Protocol B (Held-Out Season {season}) to {out_b_path}: "
            f"Train={len(train_b):,}, Val={len(val_b):,}, Test={len(test_b):,}"
        )

    save_json(manifests_b, str(processed_dir / "manifests_protocol_b.json"))

    # 3. Create Human Verification Batches
    logger.info("Generating stratified human verification templates for researcher labeling...")
    natural_sample, challenge_sample = create_annotation_sample(
        df=test_a,  # Sample strictly from isolated test partition to prevent leakage
        natural_sample_size=600,
        challenge_sample_size=200,
        random_seed=seed,
    )

    natural_csv_path = annotations_dir / "annotation_sample_natural.csv"
    challenge_csv_path = annotations_dir / "annotation_sample_challenge.csv"
    natural_sample.to_csv(natural_csv_path, index=False, encoding="utf-8")
    challenge_sample.to_csv(challenge_csv_path, index=False, encoding="utf-8")

    # Combined master template
    master_template = pd.concat([natural_sample, challenge_sample], ignore_index=True)
    master_template_path = annotations_dir / "annotation_template.csv"
    master_template.to_csv(master_template_path, index=False, encoding="utf-8")

    logger.info(
        f"Saved human annotation templates:\n"
        f"  - Natural evaluation ({len(natural_sample)} samples): {natural_csv_path}\n"
        f"  - Challenge evaluation ({len(challenge_sample)} samples): {challenge_csv_path}\n"
        f"  - Master template ({len(master_template)} samples): {master_template_path}"
    )

    print("\n" + "=" * 60)
    print("SPLITS AND ANNOTATION TEMPLATES CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Protocol A (In-Domain Grouped): {out_a_path}")
    print(f"Protocol B (Leave-One-Season-Out): Seasons 1, 2, 3, 4")
    print(f"Human Annotation Templates: {master_template_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
