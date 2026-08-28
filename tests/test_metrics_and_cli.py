"""Claim: **the metrics measure what they say they measure**, plus a CLI smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phonebook.kana import phonetic_normalize
from phonebook.metrics import cer, evaluate, format_table, levenshtein


def test_levenshtein_and_cer():
    assert levenshtein("ニホン", "ニホン") == 0
    assert levenshtein("ニホン", "ニッポン") == 2
    assert cer("ニホン", "ニホン") == 0.0
    assert cer("ニホン", "") == 1.0
    assert cer("", "") == 0.0


def test_exact_match_and_nbest_coverage():
    refs = ["ニホン", "トウキョウ", "オオサカ"]
    preds = ["ニホン", "トーキョー", None]
    nbest = [["ニホン", "ニッポン"], ["トーキョー", "トウキョウ"], ["オサカ"]]
    m = evaluate(refs, preds, nbest=nbest, ks=(1, 3))
    assert m.exact_match == pytest.approx(1 / 3)
    assert m.nbest_coverage[1] == pytest.approx(1 / 3)
    assert m.nbest_coverage[3] == pytest.approx(2 / 3)


def test_phonetic_normalization_makes_long_vowel_forms_equal():
    assert phonetic_normalize("トウキョウ") == phonetic_normalize("トーキョー")
    assert phonetic_normalize("ユウゲンガイシャ") == phonetic_normalize("ユーゲンガイシャ")
    assert phonetic_normalize("ニホン") != phonetic_normalize("ニッポン")
    m = evaluate(["トウキョウ"], ["トーキョー"])
    assert m.exact_match == 0.0
    assert m.exact_match_phonetic == 1.0


def test_ambiguous_accepted_readings_count_as_hit():
    m = evaluate(
        ["ニホンショウジ"],
        ["ニッポンショウジ"],
        nbest=[["ニッポンショウジ", "ニホンショウジ"]],
        accepted_readings=[["ニホンショウジ", "ニッポンショウジ"]],
        ks=(1,),
    )
    assert m.exact_match == 0.0, "strict match is 0 because the top-1 is not the gold reading"
    assert m.nbest_coverage[1] == 1.0, "hitting any attested reading counts as covered"


def test_rejection_metrics_are_reported_separately():
    m = evaluate(
        ["ア", "イ", "ウ"],
        ["ア", "エ", "ウ"],
        confidences=[0.9, 0.2, 0.8],
        rejected=[False, True, False],
    )
    assert m.exact_match == pytest.approx(2 / 3)
    assert m.coverage == pytest.approx(2 / 3)
    assert m.accuracy_on_accepted == 1.0


def test_format_table_is_markdown():
    md = format_table([{"a": "x", "b": 0.5}], [("a", "A"), ("b", "B")])
    assert md.splitlines()[0] == "| A | B |"
    assert md.splitlines()[1] == "|---|---|"


# --- CLI ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "phonebook.cli", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_cli_version():
    proc = run_cli("--version")
    assert proc.returncode == 0
    assert "phonebook" in proc.stdout


def test_cli_split_outputs_structure():
    proc = run_cli("split", "株式会社日本電気")
    assert proc.returncode == 0
    payload = json.loads(proc.stdout.strip())
    assert payload["prefix_form"] == "株式会社"
    assert payload["core"] == "日本電気"


def test_cli_read_without_model_gives_actionable_error():
    proc = run_cli("read", "株式会社日本電気", "--model", "/nonexistent/model")
    assert proc.returncode == 2
    assert "train.py" in proc.stderr


@pytest.mark.skipif(
    not (ROOT / "artifacts" / "model" / "model.pt").exists(),
    reason="no trained model in this environment",
)
def test_cli_read_with_model():
    proc = run_cli("read", "株式会社日本電気", "--nbest", "3", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["structure"]["core"] == "日本電気"
    assert len(payload[0]["candidates"]) <= 3
