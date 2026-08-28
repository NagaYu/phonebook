"""Katakana character-set utilities.

Claim supported: **output charset conformance** (the precondition for copy fidelity).

Japan's National Tax Agency (NTA) resource definition for the Corporate Number
public site states that the furigana field (item 35) "uses only full-width
katakana and the prolonged sound mark (ー)". Phonebook references that charset
from a single place in three different contexts -- training labels, constrained
decoding, and tests -- so that "the output is katakana only" is guaranteed by
construction rather than learned by accident.
"""

from __future__ import annotations

import unicodedata
from typing import Iterable

# --- Character sets -------------------------------------------------------

#: Full-width katakana, ァ (U+30A1) .. ヺ (U+30FA)
_KATAKANA_RANGE = tuple(chr(c) for c in range(0x30A1, 0x30FA + 1))

#: Prolonged sound mark
PROLONGED_MARK = "ー"  # U+30FC

#: Characters allowed in a reading (per NTA resource definition, item 35)
ALLOWED_OUTPUT_CHARS: frozenset[str] = frozenset(_KATAKANA_RANGE) | {PROLONGED_MARK}

#: Small kana. These never start a reading, so the constrained decoder masks
#: them at the first step.
SMALL_KANA: frozenset[str] = frozenset("ァィゥェォッャュョヮヵヶ")

#: Characters that cannot begin a reading (small kana + prolonged mark)
CANNOT_START: frozenset[str] = SMALL_KANA | {PROLONGED_MARK}

#: Deterministic vocabulary order (used to seed the tokenizer)
KATAKANA_VOCAB: tuple[str, ...] = tuple(sorted(ALLOWED_OUTPUT_CHARS))

_HIRAGANA_START, _HIRAGANA_END = 0x3041, 0x3096
_KATA_OFFSET = 0x30A1 - 0x3041


def is_hiragana(ch: str) -> bool:
    return _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END


def is_katakana(ch: str) -> bool:
    return ch in ALLOWED_OUTPUT_CHARS


def hira_to_kata(text: str) -> str:
    """Map hiragana to full-width katakana (1:1, lossless)."""
    return "".join(
        chr(ord(ch) + _KATA_OFFSET) if is_hiragana(ch) else ch for ch in text
    )


def kata_to_hira(text: str) -> str:
    """Map katakana to hiragana (for comparison / matching)."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - _KATA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


_HALFWIDTH_KANA_MAP = {
    "ｱ": "ア", "ｲ": "イ", "ｳ": "ウ", "ｴ": "エ", "ｵ": "オ",
    "ｶ": "カ", "ｷ": "キ", "ｸ": "ク", "ｹ": "ケ", "ｺ": "コ",
    "ｻ": "サ", "ｼ": "シ", "ｽ": "ス", "ｾ": "セ", "ｿ": "ソ",
    "ﾀ": "タ", "ﾁ": "チ", "ﾂ": "ツ", "ﾃ": "テ", "ﾄ": "ト",
    "ﾅ": "ナ", "ﾆ": "ニ", "ﾇ": "ヌ", "ﾈ": "ネ", "ﾉ": "ノ",
    "ﾊ": "ハ", "ﾋ": "ヒ", "ﾌ": "フ", "ﾍ": "ヘ", "ﾎ": "ホ",
    "ﾏ": "マ", "ﾐ": "ミ", "ﾑ": "ム", "ﾒ": "メ", "ﾓ": "モ",
    "ﾔ": "ヤ", "ﾕ": "ユ", "ﾖ": "ヨ",
    "ﾗ": "ラ", "ﾘ": "リ", "ﾙ": "ル", "ﾚ": "レ", "ﾛ": "ロ",
    "ﾜ": "ワ", "ｦ": "ヲ", "ﾝ": "ン",
    "ｧ": "ァ", "ｨ": "ィ", "ｩ": "ゥ", "ｪ": "ェ", "ｫ": "ォ",
    "ｯ": "ッ", "ｬ": "ャ", "ｭ": "ュ", "ｮ": "ョ",
    "ｰ": "ー", "ﾞ": "゙", "ﾟ": "゚",
}


def to_katakana(text: str) -> str:
    """Normalize any kana spelling to "full-width katakana + prolonged mark".

    Absorbs half-width katakana, hiragana and combining voiced marks, then
    composes with NFC. The copy-mechanism test (katakana input is preserved
    verbatim) relies on this function being the identity on already-normalized
    katakana.
    """
    text = "".join(_HALFWIDTH_KANA_MAP.get(ch, ch) for ch in text)
    text = unicodedata.normalize("NFC", text)
    text = hira_to_kata(text)
    # Fold dash-like characters onto the prolonged mark; drop the middle dot.
    for dash in "‐‑‒–—―−－・ｰ":
        if dash == "・":
            text = text.replace(dash, "")
        else:
            text = text.replace(dash, PROLONGED_MARK)
    return text


def is_kana_text(text: str) -> bool:
    """True when every character is kana (hiragana/katakana/prolonged mark).

    This is the trigger condition for the deterministic copy path.
    """
    if not text:
        return False
    return all(is_hiragana(ch) or is_katakana(ch) for ch in text)


def is_valid_reading(text: str, *, allow_empty: bool = False) -> bool:
    """True when the string is a well-formed furigana.

    Full-width katakana + prolonged mark only, and it must not start with a
    small kana or a prolonged mark.
    """
    if not text:
        return allow_empty
    if any(ch not in ALLOWED_OUTPUT_CHARS for ch in text):
        return False
    if text[0] in CANNOT_START:
        return False
    return True


def invalid_chars(text: str) -> list[str]:
    """List disallowed characters (used by the cleansing audit log)."""
    return sorted({ch for ch in text if ch not in ALLOWED_OUTPUT_CHARS})


def strip_invalid(text: str) -> str:
    """Drop disallowed characters. Always record what was dropped."""
    return "".join(ch for ch in text if ch in ALLOWED_OUTPUT_CHARS)


def char_ngrams(text: str, n: int) -> Iterable[str]:
    """Character n-grams. Used to detect unseen kanji bigrams (the hard set)."""
    for i in range(len(text) - n + 1):
        yield text[i : i + n]


# --- Phonetic normalization (long-vowel spelling variation) ---------------
_VOWEL_ROWS = {
    "a": "アカサタナハマヤラワガザダバパャァヮヷ",
    "i": "イキシチニヒミリヰギジヂビピィヸ",
    "u": "ウクスツヌフムユルグズヅブプュゥッヴ",
    "e": "エケセテネヘメレヱゲゼデベペェヶヹ",
    "o": "オコソトノホモヨロヲゴゾドボポョォヺ",
}
KANA_VOWEL: dict[str, str] = {
    ch: vowel for vowel, chars in _VOWEL_ROWS.items() for ch in chars
}


def kana_vowel(ch: str) -> str | None:
    """The vowel ('a'..'o') of a single kana; None for the prolonged mark."""
    return KANA_VOWEL.get(ch)


def phonetic_normalize(reading: str) -> str:
    """Absorb long-vowel spelling variation (オウ→オー, ユウ→ユー, エイ→エー).

    Claim supported: **fairness of the comparison**. pyopenjtalk and MeCab emit
    a *pronunciation* field that writes long vowels with "ー", whereas the NTA
    furigana writes them with "ウ". That is a difference of notation, not of
    knowledge, so we report a lenient exact-match alongside the strict one.
    A gap that survives the lenient metric is the robust result.
    """
    out: list[str] = []
    prev_vowel: str | None = None
    for ch in reading:
        if ch == "ウ" and prev_vowel in ("o", "u"):
            out.append(PROLONGED_MARK)
        elif ch == "イ" and prev_vowel == "e":
            out.append(PROLONGED_MARK)
        else:
            out.append(ch)
        if ch != PROLONGED_MARK:
            prev_vowel = kana_vowel(ch)
    return "".join(out)
