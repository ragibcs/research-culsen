"""
Text normalization and linguistic feature extraction for Bengali social media text.
Preserves sentiment-bearing emojis, punctuation, negation, and culturally salient expressions.
"""

import re
import unicodedata
from typing import List, Tuple

# Bengali Unicode range: U+0980 to U+09FF
BENGALI_CHAR_PATTERN = re.compile(r"[ঀ-৿]")
LATIN_CHAR_PATTERN = re.compile(r"[a-zA-Z]")

# Standard Emoji Unicode Ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols, etc.
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
    "]+",
    flags=re.UNICODE,
)

# Bengali negation tokens and common Banglish negation markers
BENGALI_NEGATION_WORDS = {
    "না", "নাই", "নেই", "নয়", "নয়", "নি", "নহে", "নাহ",
    "কখনো না", "কখনোই না", "একদম না", "মোটেই না", "ভালো না",
}

BANGLISH_NEGATION_WORDS = {
    "na", "nai", "nei", "noy", "ni", "nah", "no", "not", "never",
    "dont", "don't", "didnt", "didn't", "cant", "can't", "wont", "won't",
}


def normalize_bengali_text(text: str) -> str:
    """
    Apply Unicode NFC normalization, clean whitespace, and normalize repeated punctuation
    while preserving Bengali characters, diacritics, and sentiment markers.
    """
    if not isinstance(text, str):
        return ""

    # NFC Normalization
    norm_text = unicodedata.normalize("NFC", text)

    # Normalize excessive repetitive characters (e.g. 'অসাধারণন্নন' -> 'অসাধারণ্ন')
    # Keep at most 3 consecutive duplicate characters to preserve emotional emphasis without distortion
    norm_text = re.sub(r"(.)\1{3,}", r"\1\1\1", norm_text)

    # Normalize excessive whitespace
    norm_text = re.sub(r"\s+", " ", norm_text).strip()

    return norm_text


def detect_script(text: str) -> str:
    """
    Classify the primary script of the comment text:
    - 'bengali': Predominantly Bengali script (>70% of alphabet characters)
    - 'banglish_latin': Predominantly Latin characters (>70% of alphabet characters)
    - 'mixed': Substantial mix of Bengali and Latin scripts
    - 'emoji_or_other': Primarily emojis, numbers, or symbols with few alphabetic characters
    """
    if not isinstance(text, str) or not text.strip():
        return "empty"

    bn_chars = len(BENGALI_CHAR_PATTERN.findall(text))
    lat_chars = len(LATIN_CHAR_PATTERN.findall(text))
    total_alpha = bn_chars + lat_chars

    if total_alpha == 0:
        return "emoji_or_other"

    bn_ratio = bn_chars / total_alpha
    lat_ratio = lat_chars / total_alpha

    if bn_ratio >= 0.70:
        return "bengali"
    elif lat_ratio >= 0.70:
        return "banglish_latin"
    else:
        return "mixed"


def extract_emojis(text: str) -> List[str]:
    """
    Extract all emojis present in the text.
    """
    if not isinstance(text, str):
        return []
    return EMOJI_PATTERN.findall(text)


def detect_negation(text: str) -> bool:
    """
    Detect presence of explicit negation markers in Bengali or Banglish.
    """
    if not isinstance(text, str):
        return False

    clean_lower = text.lower()
    words = set([t.strip() for t in re.split(r"[\s,\.!?;:\'\"|/।()]+", clean_lower) if t.strip()])

    # Check word-level match
    if words.intersection(BENGALI_NEGATION_WORDS) or words.intersection(BANGLISH_NEGATION_WORDS):
        return True

    # Check phrase-level match
    for phrase in BENGALI_NEGATION_WORDS:
        if " " in phrase and phrase in text:
            return True

    return False
