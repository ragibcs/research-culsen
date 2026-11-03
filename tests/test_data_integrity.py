"""
Unit tests for data normalization, script detection, emoji extraction, and schema integrity.
"""

import pytest
from culsen_bn.data.normalize import (
    normalize_bengali_text,
    detect_script,
    extract_emojis,
    detect_negation,
)
from culsen_bn.data.load_dataset import validate_dataset_schema


def test_schema_validation():
    valid_cols = ["comment_id", "video_id", "season", "text_clean", "sentiment", "confidence", "extra_col"]
    is_valid, missing = validate_dataset_schema(valid_cols)
    assert is_valid is True
    assert len(missing) == 0

    invalid_cols = ["comment_id", "text_clean"]
    is_valid, missing = validate_dataset_schema(invalid_cols)
    assert is_valid is False
    assert "video_id" in missing
    assert "season" in missing


def test_bengali_unicode_normalization():
    # Repetitive characters normalization
    raw_repeated = "অনেক ভালোওওওওও গান"
    norm = normalize_bengali_text(raw_repeated)
    assert "ভালোওওও" in norm
    assert "ভালোওওওও" not in norm

    # Excessive whitespace
    raw_ws = "  গানটা    খুব সুন্দর   "
    assert normalize_bengali_text(raw_ws) == "গানটা খুব সুন্দর"


def test_script_detection():
    bn_text = "গানটা অসাধারণ হয়েছে, বারবার শুনছি"
    assert detect_script(bn_text) == "bengali"

    banglish_text = "Gaan ta onek shundor hoyeche"
    assert detect_script(banglish_text) == "banglish_latin"

    mixed_text = "Ei gaan ta অসাধারণ bro"
    assert detect_script(mixed_text) == "mixed"

    emoji_text = "❤️❤️❤️🔥🔥🔥"
    assert detect_script(emoji_text) == "emoji_or_other"


def test_emoji_extraction():
    text_with_emoji = "অসাধারণ গান ❤️🔥👏"
    emojis = extract_emojis(text_with_emoji)
    assert len(emojis) > 0

    text_no_emoji = "সাধারণ মন্তব্য"
    assert len(extract_emojis(text_no_emoji)) == 0


def test_negation_detection():
    # Bengali negation
    assert detect_negation("গানটা একদম ভালো না") is True
    assert detect_negation("তার কণ্ঠের কোনো তুলনা নেই") is True
    assert detect_negation("খুব সুন্দর গান") is False

    # Banglish negation
    assert detect_negation("eta bhalo na") is True
    assert detect_negation("i dont like this track") is True
    assert detect_negation("awesome performance") is False
