"""
Unit tests for label mapping, verification status validation, and Cohen's kappa calculation.
"""

import pytest
from culsen_bn.data.annotation import compute_inter_annotator_agreement, VALID_ANNOTATION_LABELS


def test_valid_annotation_labels():
    assert "positive" in VALID_ANNOTATION_LABELS
    assert "negative" in VALID_ANNOTATION_LABELS
    assert "ambiguous" in VALID_ANNOTATION_LABELS
    assert "unreviewed" in VALID_ANNOTATION_LABELS


def test_cohens_kappa_perfect_agreement():
    ann1 = ["positive", "negative", "positive", "negative"]
    ann2 = ["positive", "negative", "positive", "negative"]
    res = compute_inter_annotator_agreement(ann1, ann2)
    assert res["observed_agreement_po"] == 1.0
    assert res["cohens_kappa"] == 1.0
    assert res["interpretation"] == "Almost perfect"


def test_cohens_kappa_disagreement():
    ann1 = ["positive", "positive", "negative", "negative"]
    ann2 = ["negative", "negative", "positive", "positive"]
    res = compute_inter_annotator_agreement(ann1, ann2)
    assert res["observed_agreement_po"] == 0.0
    assert res["cohens_kappa"] < 0.0  # Worse than chance


def test_cohens_kappa_length_mismatch():
    ann1 = ["positive", "negative"]
    ann2 = ["positive"]
    with pytest.raises(ValueError):
        compute_inter_annotator_agreement(ann1, ann2)
