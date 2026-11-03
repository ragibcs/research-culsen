"""
Unit tests for data leakage assertions.
Verifies that overlapping video IDs, duplicate texts, or comment IDs
are strictly caught and trigger assertions.
"""

import pytest
import pandas as pd
from culsen_bn.data.split import verify_no_leakage


@pytest.fixture
def clean_splits():
    train_df = pd.DataFrame({
        "comment_id": ["c1", "c2", "c3"],
        "video_id": ["v1", "v1", "v2"],
        "text_clean": ["গানটি চমৎকার", "দারুণ লিরিক", "অসাধারণ সুর"],
        "sentiment": ["positive", "positive", "positive"],
    })
    val_df = pd.DataFrame({
        "comment_id": ["c4", "c5"],
        "video_id": ["v3", "v3"],
        "text_clean": ["মোটেই ভালো লাগেনি", "বাজে পারফরম্যান্স"],
        "sentiment": ["negative", "negative"],
    })
    test_df = pd.DataFrame({
        "comment_id": ["c6", "c7"],
        "video_id": ["v4", "v4"],
        "text_clean": ["আগের সিজন ভালো ছিল", "নতুন সিঙ্গার ভালো"],
        "sentiment": ["negative", "positive"],
    })
    return train_df, val_df, test_df


def test_clean_splits_pass(clean_splits):
    train_df, val_df, test_df = clean_splits
    report = verify_no_leakage(train_df, val_df, test_df)
    assert report["status"] == "PASSED"
    assert report["text_leakage_count"] == 0
    assert report["id_leakage_count"] == 0


def test_comment_id_leakage_fails(clean_splits):
    train_df, val_df, test_df = clean_splits
    # Inject overlapping comment ID into test split
    leaky_test = test_df.copy()
    leaky_test.loc[0, "comment_id"] = "c1"  # Already in train_df

    with pytest.raises(AssertionError, match="ID Leakage detected"):
        verify_no_leakage(train_df, val_df, leaky_test)


def test_video_group_leakage_fails(clean_splits):
    train_df, val_df, test_df = clean_splits
    # Inject overlapping video ID into val split
    leaky_val = val_df.copy()
    leaky_val.loc[0, "video_id"] = "v1"  # Already in train_df

    with pytest.raises(AssertionError, match="Video Group Leakage detected"):
        verify_no_leakage(train_df, leaky_val, test_df)


def test_text_duplicate_leakage_fails(clean_splits):
    train_df, val_df, test_df = clean_splits
    # Inject identical comment text across train and test
    leaky_test = test_df.copy()
    leaky_test.loc[0, "text_clean"] = "গানটি চমৎকার"  # Exact match with train_df c1

    with pytest.raises(AssertionError, match="Text Content Leakage detected"):
        verify_no_leakage(train_df, val_df, leaky_test)
