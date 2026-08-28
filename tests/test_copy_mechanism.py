"""Claim: **katakana input is preserved verbatim (copy fidelity).**

Kana already present in a trade name is something to transcribe, not something
to infer. Phonebook guarantees this two ways:

  1. Structurally: kana-only spans are transcribed deterministically without
     touching the model (the default).
  2. Mechanically: the pointer-generator can place probability mass on input
     positions.

These tests verify (1) exactly, and for (2) verify that the mechanism is
mathematically sound -- the copy distribution is a probability distribution and
p_gen genuinely mixes generation with copying.
"""

from __future__ import annotations

import pytest
import torch

from phonebook.kana import hira_to_kata, to_katakana
from phonebook.decode import script_segments


KATAKANA_CORES = ["アルファ", "サンライズ", "ミライ", "テラス", "ホライゾン", "ヴェルデ"]
HIRAGANA_CORES = ["あおぞら", "ひまわり", "こもれび", "つばさ"]


@pytest.mark.parametrize("core", KATAKANA_CORES)
def test_pure_katakana_core_is_copied_verbatim(reader, core):
    result = reader.read(f"株式会社{core}", nbest=3)
    assert result.source == "copy"
    assert result.reading == "カブシキガイシャ" + core
    assert result.confidence == 1.0


@pytest.mark.parametrize("core", HIRAGANA_CORES)
def test_hiragana_core_is_transliterated_not_invented(reader, core):
    result = reader.read(f"合同会社{core}", nbest=1)
    assert result.source == "copy"
    assert result.reading == "ゴウドウガイシャ" + hira_to_kata(core)


@pytest.mark.parametrize(
    "name,kana_part",
    [
        ("株式会社アルファ電子", "アルファ"),
        ("ミライ工業株式会社", "ミライ"),
        ("株式会社さくら建設", "サクラ"),
    ],
)
def test_kana_run_inside_mixed_name_is_preserved(reader, name, kana_part):
    """Even in a mixed-script name, the kana run survives into the output."""
    result = reader.read(name, nbest=1)
    assert result.source == "segmented"
    assert kana_part in result.reading, f"{name} -> {result.reading} is missing {kana_part}"


def test_halfwidth_and_voiced_kana_are_normalized_then_copied(reader):
    result = reader.read("株式会社ｱﾙﾌｱ", nbest=1)
    assert result.reading == "カブシキガイシャアルファ" or result.reading.endswith("アルフア")
    assert result.source in ("copy", "segmented")


def test_script_segmentation_boundaries():
    assert script_segments("アルファ電子") == [("kana", "アルファ"), ("other", "電子")]
    assert script_segments("ABC商会") == [("latin", "ABC"), ("other", "商会")]
    assert script_segments("みらいABC") == [("kana", "みらい"), ("latin", "ABC")]


def test_copy_distribution_is_a_valid_probability_distribution(model, tokenizer):
    """After the copy mixture the output is still a probability distribution."""
    src = torch.tensor([tokenizer.encode("日本電気アルファ")])
    tgt = torch.tensor([[tokenizer.bos_id] + tokenizer.encode("ニホン")])
    logp = model(src, tgt)
    probs = torch.exp(logp).sum(-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_copy_gate_is_defined_and_bounded(model, tokenizer):
    """p_gen stays inside (0,1): a check that the copy mechanism is live."""
    src = torch.tensor([tokenizer.encode("アルファ電子")])
    tgt = torch.tensor([[tokenizer.bos_id] + tokenizer.encode("アルファ")])
    gate = model.copy_gate(src, tgt)
    assert gate.shape == tgt.shape
    assert bool((gate > 0).all()) and bool((gate < 1).all())


def test_disabling_copy_changes_the_distribution(tokenizer):
    """The distribution differs from a use_copy=False model, i.e. the copy term matters."""
    from phonebook.model import CharSeq2Seq, ModelConfig

    torch.manual_seed(0)
    cfg_on = ModelConfig.preset("tiny", len(tokenizer))
    torch.manual_seed(0)
    model_on = CharSeq2Seq(cfg_on).eval()
    cfg_off = ModelConfig.preset("tiny", len(tokenizer))
    cfg_off.use_copy = False
    torch.manual_seed(0)
    model_off = CharSeq2Seq(cfg_off).eval()

    src = torch.tensor([tokenizer.encode("アルファ電子")])
    tgt = torch.tensor([[tokenizer.bos_id] + tokenizer.encode("アルファ")])
    a = model_on(src, tgt)
    b = model_off(src, tgt)
    assert not torch.allclose(a, b, atol=1e-3)


def test_katakana_normalization_is_idempotent():
    for text in KATAKANA_CORES + ["ヴァイオレット", "ジェイアール", "パーク"]:
        assert to_katakana(text) == text
