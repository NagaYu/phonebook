"""Comparison conditions (A) general-purpose Japanese G2P and (B) a large LLM.

Claim supported: **unseen-entity performance**.

The claim under test is that Phonebook is comparable to existing methods on
known entities but pulls ahead on unseen and hard ones, so the baselines have
to be strong and correctly used for the comparison to mean anything. Wrapped
here behind one interface:

  (A-1) pyopenjtalk (OpenJTalk dictionary) -- the de-facto standard Japanese
        TTS front-end
  (A-2) MeCab + UniDic (fugashi / unidic-lite) -- the pronunciation feature of
        a morphological analyser
  (B)   Claude, prompting only, no additional training

Every wrapper strictly reports ``available=False`` rather than fabricating
output when it cannot run. Filling a results table with conditions that were
never executed would undermine the entire project.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .kana import strip_invalid, to_katakana
from .structure import canonicalize_legal_reading


class Baseline:
    """The minimal interface shared by every condition."""

    name: str = "baseline"
    available: bool = False
    note: str = ""

    def read_batch(self, names: Sequence[str]) -> list[Optional[str]]:
        raise NotImplementedError

    def read(self, name: str) -> Optional[str]:
        return self.read_batch([name])[0]

    def describe(self) -> dict:
        return {"name": self.name, "available": self.available, "note": self.note}


def _postprocess(text: str) -> Optional[str]:
    """Coerce arbitrary output to katakana + prolonged mark; None if impossible."""
    if not text:
        return None
    kana = strip_invalid(to_katakana(text))
    return canonicalize_legal_reading(kana) or None


class PyOpenJTalkBaseline(Baseline):
    """(A-1) pyopenjtalk: general-purpose G2P backed by the OpenJTalk dictionary."""

    name = "pyopenjtalk"

    def __init__(self) -> None:
        try:
            import pyopenjtalk  # noqa: F401

            self._mod = pyopenjtalk
            self.available = True
            self.note = "general-purpose Japanese G2P based on the OpenJTalk dictionary"
        except Exception as exc:  # pragma: no cover - environment dependent
            self._mod = None
            self.available = False
            self.note = f"unavailable: {exc}"

    def read_batch(self, names: Sequence[str]) -> list[Optional[str]]:
        if not self.available:
            return [None] * len(names)
        out: list[Optional[str]] = []
        for name in names:
            try:
                out.append(_postprocess(self._mod.g2p(name, kana=True)))
            except Exception:
                out.append(None)
        return out


class MeCabUniDicBaseline(Baseline):
    """(A-2) MeCab + UniDic: concatenate the pronunciation feature of each morpheme."""

    name = "mecab-unidic"

    def __init__(self) -> None:
        try:
            import fugashi
            import unidic_lite

            self._tagger = fugashi.Tagger(f"-d {unidic_lite.DICDIR}")
            self.available = True
            self.note = "pronunciation (pron) feature from fugashi + unidic-lite"
        except Exception as exc:  # pragma: no cover - environment dependent
            self._tagger = None
            self.available = False
            self.note = f"unavailable: {exc}"

    def read_batch(self, names: Sequence[str]) -> list[Optional[str]]:
        if not self.available:
            return [None] * len(names)
        out: list[Optional[str]] = []
        for name in names:
            parts: list[str] = []
            for word in self._tagger(name):
                feat = word.feature
                kana = getattr(feat, "pron", None) or getattr(feat, "kana", None)
                parts.append(kana if kana and kana != "*" else word.surface)
            out.append(_postprocess("".join(parts)))
        return out


# --- (B) Large language model, prompting only ------------------------------
# The prompt is written in Japanese on purpose: the task is Japanese, and a
# Japanese prompt gives the baseline its best shot. Weakening the baseline with
# an English prompt would make the comparison flattering rather than honest.
LLM_SYSTEM_PROMPT = (
    "あなたは日本語の固有名詞、特に法人名の読み(フリガナ)の専門家です。\n"
    "与えられた法人名に対し、最も確からしい読みを全角カタカナと長音符「ー」のみで答えてください。\n"
    "規則:\n"
    "- 法人格(株式会社/有限会社/一般社団法人など)も読みに含める。前株・後株の位置は表記どおり。\n"
    "- 出力は全角カタカナと「ー」のみ。ひらがな・記号・空白・英数字は使わない。\n"
    "- 商号中の英字はカタカナ表記に直す(例: SYSTEM → システム)。\n"
    "- 確信が持てない場合も、可能性の高い順に候補を挙げる。\n"
    "例:\n"
    "  株式会社山田商店 → カブシキガイシャヤマダショウテン\n"
    "  緑川運送株式会社 → ミドリカワウンソウカブシキガイシャ\n"
)


@dataclass
class LLMCache:
    """Persist LLM responses to disk, for reproducibility and cost control."""

    path: Optional[Path] = None
    data: dict[str, list[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.data = {}
        if self.path and Path(self.path).exists():
            for line in Path(self.path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["name"]] = rec["readings"]

    def get(self, name: str) -> Optional[list[str]]:
        return self.data.get(name)

    def put(self, name: str, readings: list[str]) -> None:
        self.data[name] = readings
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with Path(self.path).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"name": name, "readings": readings}, ensure_ascii=False) + "\n")


class ClaudeBaseline(Baseline):
    """(B) Claude solving the task from a prompt alone.

    No fine-tuning and no dictionary lookup. Phonebook's claim is that it is
    small, CPU-friendly, calibrated and able to abstain -- not that it is
    smarter than a frontier LLM. This condition exists to measure how far
    general linguistic knowledge alone gets you.

    Cost and reproducibility:
      - Responses are always cached to JSONL; repeat runs make no API calls.
      - For large evaluations the Message Batches API (50% discount) is
        recommended.
      - The system prompt is a fixed string so that prefix caching can apply.
    """

    name = "claude"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        cache_path: str | Path | None = None,
        nbest: int = 3,
        max_tokens: int = 512,
        offline_only: bool = False,
    ) -> None:
        self.model = model
        self.nbest = nbest
        self.max_tokens = max_tokens
        self.cache = LLMCache(path=Path(cache_path) if cache_path else None)
        self._client = None
        if offline_only:
            self.available = bool(self.cache.data)
            self.note = "cache only (no API calls)"
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            self.available = True
            self.note = f"{model} / prompting only"
        except Exception as exc:  # pragma: no cover - depends on credentials
            self._client = None
            self.available = bool(self.cache.data)
            self.note = f"API unavailable ({exc}); cache only, {len(self.cache.data)} entries"

    def _query(self, name: str) -> list[str]:
        from pydantic import BaseModel

        class Readings(BaseModel):
            readings: list[str]

        response = self._client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": LLM_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"法人名: {name}\n"
                        f"読みの候補を確からしい順に最大{self.nbest}件。"
                    ),
                }
            ],
            output_format=Readings,
        )
        parsed = response.parsed_output
        return list(parsed.readings) if parsed else []

    def read_nbest(self, name: str) -> list[str]:
        cached = self.cache.get(name)
        if cached is not None:
            return cached
        if self._client is None:
            return []
        try:
            readings = self._query(name)
        except Exception:
            readings = []
        cleaned = [r for r in (_postprocess(x) for x in readings) if r]
        self.cache.put(name, cleaned)
        return cleaned

    def read_batch(self, names: Sequence[str]) -> list[Optional[str]]:
        out: list[Optional[str]] = []
        for name in names:
            cands = self.read_nbest(name)
            out.append(cands[0] if cands else None)
        return out


def default_baselines(llm_cache: str | Path | None = None, offline_only: bool = False) -> list[Baseline]:
    """Construct comparison conditions (A) and (B) with default settings."""
    return [
        PyOpenJTalkBaseline(),
        MeCabUniDicBaseline(),
        ClaudeBaseline(cache_path=llm_cache, offline_only=offline_only),
    ]
