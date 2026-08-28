"""Character-level tokenizer with a shared source/target vocabulary.

Claim supported: **copy fidelity**.

A pointer-generator (copy) mechanism has to map "the character at input
position i" onto "vocabulary entry v". Separate source and target vocabularies
would require an extra lookup table for that map, and would break on unseen
characters. Phonebook uses a **single shared vocabulary** so probability mass
can be copied directly onto any character present in the input -- including
stretches that are already katakana.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .kana import ALLOWED_OUTPUT_CHARS

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = (PAD, BOS, EOS, UNK)


@dataclass
class CharTokenizer:
    """Character <-> id. Special tokens are pinned to ids 0..3."""

    itos: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    # -- Construction ------------------------------------------------------
    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "CharTokenizer":
        """Build the vocabulary from a corpus.

        Every emittable katakana character and the prolonged mark are **always**
        added, even if some never appear in the training data. The guarantee
        about the output charset must not depend on the data, so the constrained
        decoder can never be in a state where it cannot emit a legal character.
        """
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(text)
        chars = {ch for ch, n in counter.items() if n >= min_freq}
        chars |= set(ALLOWED_OUTPUT_CHARS)
        itos = list(SPECIALS) + sorted(chars)
        return cls(itos=itos)

    # -- Conversion --------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "")
        return "".join(out)

    def output_char_ids(self) -> list[int]:
        """Ids the constrained decoder may emit (katakana + prolonged mark + EOS)."""
        ids = [self.stoi[ch] for ch in ALLOWED_OUTPUT_CHARS if ch in self.stoi]
        return sorted(ids + [self.eos_id])

    # -- Persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(itos=data["itos"])
