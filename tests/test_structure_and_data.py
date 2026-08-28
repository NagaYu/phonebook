"""Claims: **the legal-form separation is correct** -- the precondition for the
unseen-entity evaluation -- and **normalization never loses the original surface.**
"""

from __future__ import annotations

import pytest

from phonebook.data import cleanse, normalize_furigana, normalize_name
from phonebook.structure import (
    LEGAL_FORMS,
    StructuralSplitter,
    canonicalize_legal_reading,
)


@pytest.fixture(scope="module")
def splitter():
    return StructuralSplitter()


@pytest.mark.parametrize(
    "name,prefix,core,suffix",
    [
        ("株式会社日本電気", "株式会社", "日本電気", None),
        ("日本電気株式会社", None, "日本電気", "株式会社"),
        ("有限会社山田商店", "有限会社", "山田商店", None),
        ("合同会社アルファ", "合同会社", "アルファ", None),
        ("一般社団法人地域交流推進機構", "一般社団法人", "地域交流推進機構", None),
        ("医療法人社団つばさ会", "医療法人社団", "つばさ会", None),
        ("特定非営利活動法人みらい", "特定非営利活動法人", "みらい", None),
        ("山田商店", None, "山田商店", None),
    ],
)
def test_split_positions(splitter, name, prefix, core, suffix):
    st = splitter.split(name)
    assert st.prefix_form == prefix
    assert st.core == core
    assert st.suffix_form == suffix


def test_longest_match_wins(splitter):
    """医療法人社団 must win over the shorter 医療法人 (longest match)."""
    assert splitter.split("医療法人社団あおば会").prefix_form == "医療法人社団"
    assert splitter.split("医療法人あおば会").prefix_form == "医療法人"


def test_alias_expansion(splitter):
    for alias in ("㈱山田商店", "(株)山田商店", "（株）山田商店"):
        st = splitter.split(alias)
        assert st.prefix_form == "株式会社"
        assert st.core == "山田商店"
        assert st.original == alias, "the original surface is always retained"


def test_brackets_are_separated(splitter):
    st = splitter.split("株式会社ミドリ（旧:みどり商事）")
    assert st.core == "ミドリ"
    assert st.brackets == ["旧:みどり商事"]


def test_trailing_after_legal_form(splitter):
    st = splitter.split("山田商店株式会社東京支店")
    assert st.suffix_form == "株式会社"
    assert st.core == "山田商店"
    assert st.trailing == "東京支店"


def test_legal_form_only_name_is_not_emptied(splitter):
    st = splitter.split("株式会社")
    assert st.core == "株式会社"
    assert st.prefix_form is None


@pytest.mark.parametrize(
    "name,furigana,expected",
    [
        ("株式会社日本電気", "カブシキガイシャニホンデンキ", "ニホンデンキ"),
        ("日本電気株式会社", "ニホンデンキカブシキカイシャ", "ニホンデンキ"),  # non-rendaku variant
        ("有限会社山田商店", "ユウゲンカイシャヤマダショウテン", "ヤマダショウテン"),
    ],
)
def test_align_reading_strips_legal_form(splitter, name, furigana, expected):
    st = splitter.split(name)
    assert splitter.align_reading(st, furigana) == expected


def test_align_reading_returns_none_on_mismatch(splitter):
    """Rows where furigana and surface disagree structurally return None and are
    excluded from training."""
    st = splitter.split("株式会社日本電気")
    assert splitter.align_reading(st, "ニホンデンキ") is None
    assert splitter.align_reading(st, "") is None


def test_compose_is_inverse_of_align(splitter):
    st = splitter.split("株式会社山田商店")
    core = splitter.align_reading(st, "カブシキガイシャヤマダショウテン")
    assert st.compose(core) == "カブシキガイシャヤマダショウテン"


def test_canonicalize_legal_reading():
    assert canonicalize_legal_reading("ヤマダカブシキカイシャ") == "ヤマダカブシキガイシャ"
    assert canonicalize_legal_reading("ユウゲンカイシャA") == "ユウゲンガイシャA"
    assert canonicalize_legal_reading("ニホンデンキ") == "ニホンデンキ"


def test_normalize_keeps_original_available():
    raw = "㈱髙島屋　ＡＢＣ"
    norm = normalize_name(raw)
    assert norm != raw
    assert "高島屋" in norm and "ABC" in norm
    assert "　" not in norm


def test_normalize_furigana_handles_halfwidth_and_hiragana():
    assert normalize_furigana("ﾔﾏﾀﾞ ｼｮｳﾃﾝ") == "ヤマダショウテン"
    assert normalize_furigana("やまだしょうてん") == "ヤマダショウテン"


def test_cleanse_separates_missing_and_records_bias():
    rows = [
        {"corporate_number": "1", "name": "株式会社山田商店", "furigana": "カブシキガイシャヤマダショウテン",
         "kind": "301", "latest": "1", "hihyoji": "0"},
        {"corporate_number": "2", "name": "有限会社緑川", "furigana": "", "kind": "302",
         "latest": "1", "hihyoji": "0"},
        {"corporate_number": "3", "name": "株式会社古い情報", "furigana": "カブシキガイシャフルイジョウホウ",
         "kind": "301", "latest": "0", "hihyoji": "0"},
        {"corporate_number": "4", "name": "株式会社除外", "furigana": "カブシキガイシャジョガイ",
         "kind": "301", "latest": "1", "hihyoji": "1"},
        {"corporate_number": "5", "name": "株式会社不正文字", "furigana": "カブシキガイシャ★",
         "kind": "301", "latest": "1", "hihyoji": "0"},
    ]
    labeled, missing, report = cleanse(rows)
    assert [r.corporate_number for r in labeled] == ["1"]
    assert [r.corporate_number for r in missing] == ["2", "5"]
    assert report.dropped_not_latest == 1
    assert report.dropped_hihyoji == 1
    assert report.invalid_furigana_chars == 1
    assert "★" in report.invalid_char_counter
    assert report.missing_by_kind["302"] == 1


def test_all_legal_forms_have_readings():
    for form, reading in LEGAL_FORMS.items():
        assert reading, f"{form} has no reading"
        assert len(reading) >= len(form) - 2
