"""Shared pytest fixtures.

Design rule: **do not depend on a trained model.** The tests must pass with a
randomly initialized small model, because that is exactly what demonstrates
that "the output is katakana only" and "katakana input is preserved" are
guarantees of the design rather than products of training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.data import Record  # noqa: E402
from phonebook.decode import PhonebookReader  # noqa: E402
from phonebook.en2kana import EnglishToKatakana  # noqa: E402
from phonebook.model import CharSeq2Seq, ModelConfig  # noqa: E402
from phonebook.tokenizer import CharTokenizer  # noqa: E402

SAMPLE_NAMES = [
    "株式会社山田商店",
    "緑川食品株式会社",
    "合同会社あおぞらデザイン",
    "株式会社アルファ電子",
    "医療法人社団つばさ会",
    "有限会社みらい工務店",
    "株式会社ABCシステム",
    "一般社団法人地域交流推進機構",
]


@pytest.fixture(scope="session")
def tokenizer() -> CharTokenizer:
    corpus = SAMPLE_NAMES + [
        "ヤマダショウテン", "ミドリカワショクヒン", "アオゾラデザイン", "アルファデンシ",
        "ツバサカイ", "ミライコウムテン", "エービーシーシステム", "チイキコウリュウスイシンキコウ",
        "日本電気", "ニホンデンキ", "東京", "トウキョウ",
    ]
    return CharTokenizer.build(corpus)


@pytest.fixture(scope="session")
def model(tokenizer) -> CharSeq2Seq:
    torch.manual_seed(0)
    cfg = ModelConfig.preset("tiny", len(tokenizer))
    m = CharSeq2Seq(cfg)
    m.eval()
    return m


@pytest.fixture
def reader(model, tokenizer) -> PhonebookReader:
    return PhonebookReader(
        model, tokenizer, en2kana=EnglishToKatakana(), beam_size=4, max_len=16
    )


@pytest.fixture
def records() -> list[Record]:
    """A small record set for the split tests."""
    from phonebook.structure import StructuralSplitter

    splitter = StructuralSplitter()
    raw = [
        ("1000000000001", "株式会社山田商店", "カブシキガイシャヤマダショウテン", "301"),
        ("1000000000002", "株式会社山田商店", "カブシキガイシャヤマダショウテン", "301"),
        ("1000000000003", "山田商店株式会社", "ヤマダショウテンカブシキガイシャ", "301"),
        ("1000000000004", "株式会社緑川食品", "カブシキガイシャミドリカワショクヒン", "301"),
        ("1000000000005", "有限会社日本電気", "ユウゲンガイシャニホンデンキ", "302"),
        ("1000000000006", "株式会社日本電気", "カブシキガイシャニッポンデンキ", "301"),
        ("1000000000007", "合同会社あおぞら", "ゴウドウガイシャアオゾラ", "305"),
        ("1000000000008", "株式会社未知漢字連接", "カブシキガイシャミチカンジレンセツ", "301"),
    ]
    out = []
    for cn, name, furigana, kind in raw:
        st = splitter.split(name)
        core_reading = splitter.align_reading(st, furigana)
        out.append(
            Record(
                corporate_number=cn,
                name_raw=name,
                name=name,
                furigana_raw=furigana,
                furigana=furigana,
                kind=kind,
                core=st.core,
                core_reading=core_reading or "",
                prefix_form=st.prefix_form,
                suffix_form=st.suffix_form,
                aligned=core_reading is not None,
            )
        )
    return out
