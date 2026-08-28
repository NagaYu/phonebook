#!/usr/bin/env python3
"""Auxiliary data: Latin-script trade name -> katakana pairs, mined from the corpus.

Claim supported: **unseen-entity performance** -- by keeping Latin script from
contaminating it.

Japanese G2P tends to drop Latin text or read it as romaji, so failures on
Latin-script trade names easily masquerade as "weak on unseen entities". This
script extracts rows whose trade-name core is entirely Latin from the NTA data
itself and builds both a pair corpus and a lexicon.

Two uses:
  1. Augment the lexicon in ``phonebook/en2kana.py`` with real data.
  2. Train the same CharSeq2Seq on English-to-katakana pairs and use it instead
     of the rule-based converter
     (``train.py --data data/processed/en2kana --preset tiny``).

**The split is inherited from the parent dataset.** Re-splitting here would put
Latin trade names seen in the main training set into the en2kana test set,
which is indirect leakage.

Usage:
    python scripts/build_en2kana_data.py --data data/processed --out data/processed/en2kana
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.data import read_jsonl, write_jsonl  # noqa: E402
from phonebook.kana import is_valid_reading  # noqa: E402

LATIN_ONLY = re.compile(r"^[A-Za-z0-9&'\.\-\+ ]+$")
HAS_LATIN = re.compile(r"[A-Za-z]")

#: Parent split -> en2kana split (inherited to avoid leakage)
SPLIT_MAP = {
    "train": "train",
    "dev": "dev",
    "test_known": "test",
    "test_unseen": "test",
    "test_ambiguous": "test",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="data/processed/en2kana")
    p.add_argument("--min-count", type=int, default=2, help="minimum count for a lexicon entry")
    args = p.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    pair_counts: Counter = Counter()
    n_mixed = 0

    for src_split, dst_split in SPLIT_MAP.items():
        path = data_dir / f"{src_split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            core = row.get("core", "")
            reading = row.get("core_reading", "")
            if not core or not reading or not HAS_LATIN.search(core):
                continue
            if not is_valid_reading(reading):
                continue
            if LATIN_ONLY.match(core):
                buckets[dst_split].append(
                    {"core": core.upper(), "core_reading": reading, "origin_split": src_split}
                )
                pair_counts[(core.upper(), reading)] += 1
            else:
                n_mixed += 1

    total = 0
    for split, rows in buckets.items():
        n = write_jsonl(rows, out_dir / f"{split}.jsonl")
        total += n
        print(f"  {split:6s} {n:>7,} pairs -> {out_dir / (split + '.jsonl')}")
    print(
        f"  mixed-script cores (Latin plus other scripts): {n_mixed:,} "
        "-- handled by segmented transcription"
    )

    # Lexicon: only entries from train that occur repeatedly and whose reading
    # is unambiguous.
    train_pairs: Counter = Counter()
    for row in buckets["train"]:
        train_pairs[(row["core"], row["core_reading"])] += 1
    by_word: dict[str, Counter] = {}
    for (word, reading), count in train_pairs.items():
        by_word.setdefault(word, Counter())[reading] += count

    lexicon: dict[str, str] = {}
    ambiguous = 0
    for word, counter in by_word.items():
        top, count = counter.most_common(1)[0]
        if count < args.min_count:
            continue
        if len(counter) > 1 and counter.most_common(2)[1][1] >= count:
            ambiguous += 1  # tied readings: leave it out of the lexicon
            continue
        lexicon[word] = top

    lex_path = out_dir / "lexicon.json"
    lex_path.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nlexicon: {lex_path} ({len(lexicon):,} words; {ambiguous:,} excluded as ambiguous)")
    print("use   : EnglishToKatakana.from_lexicon_file('%s')" % lex_path)
    print("train : python scripts/train.py --data %s --out artifacts/en2kana --preset tiny" % out_dir)
    if total == 0:
        print("\nNo all-Latin trade-name cores found (they are rare in synthetic data).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
