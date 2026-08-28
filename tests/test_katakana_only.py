"""Claim: **the output is katakana and the prolonged mark, and nothing else.**

This follows from constrained decoding, not from training, so it must hold for a
randomly initialized model too. A failure here means either the constraint
implementation is broken or the vocabulary definition and the decoder mask have
drifted apart.
"""

from __future__ import annotations

import pytest

from phonebook.decode import assert_katakana_only
from phonebook.kana import ALLOWED_OUTPUT_CHARS, CANNOT_START, is_valid_reading
from tests.conftest import SAMPLE_NAMES


def test_allowed_charset_is_katakana_and_prolonged_only():
    assert "ー" in ALLOWED_OUTPUT_CHARS
    assert "ア" in ALLOWED_OUTPUT_CHARS and "ヺ" in ALLOWED_OUTPUT_CHARS
    for ch in "あいうえお漢字ABC123 　、。・ｱ":
        assert ch not in ALLOWED_OUTPUT_CHARS, f"{ch} leaked into the allowed set"


@pytest.mark.parametrize("name", SAMPLE_NAMES)
def test_all_candidates_are_katakana(reader, name):
    result = reader.read(name, nbest=3)
    assert result.candidates, "no candidates returned"
    for cand in result.candidates:
        assert is_valid_reading(cand.reading), f"{name} -> {cand.reading!r} is not katakana"
        assert all(ch in ALLOWED_OUTPUT_CHARS for ch in cand.reading)


def test_batch_outputs_pass_runtime_guard(reader):
    results = reader.read_batch(SAMPLE_NAMES, nbest=3)
    assert_katakana_only(results)


def test_reading_never_starts_with_small_kana_or_prolonged(reader):
    results = reader.read_batch(SAMPLE_NAMES, nbest=5)
    for r in results:
        for cand in r.candidates:
            assert cand.reading[0] not in CANNOT_START, f"{cand.reading!r} starts with a small kana or prolonged mark"


def test_english_module_outputs_katakana_only():
    from phonebook.en2kana import EnglishToKatakana

    converter = EnglishToKatakana()
    for word in ["ABC SYSTEM", "NEC", "3M", "Global-Tech 24", "SAKURA", "x9!"]:
        out = converter.convert(word)
        assert all(ch in ALLOWED_OUTPUT_CHARS for ch in out), f"{word} -> {out!r}"


def test_legal_form_readings_are_katakana_only():
    from phonebook.structure import LEGAL_FORMS

    for form, reading in LEGAL_FORMS.items():
        assert is_valid_reading(reading), f"reading {reading!r} of {form} is invalid"
