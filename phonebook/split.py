"""Split design. The single most important module in this project.

Claim supported: **unseen-entity performance**.

Reading proper nouns is a task where a sloppy split silently measures
memorization instead of generalization. Phonebook keeps four sets strictly apart:

  (1) known      Known entities: the trade-name core occurs in train, but the
                 **corporate number never does**. This measures what is
                 effectively dictionary lookup.
  (2) unseen     Unseen entities: the core string never occurs in train at all.
  (3) hard       The subset of (2) containing a **kanji bigram** that never
                 occurs in train. Measures compositional generalization that
                 substring memorization cannot solve.
  (4) ambiguous  Same surface, different readings. For cores where several
                 readings genuinely coexist, only the majority reading is placed
                 in train and the **minority readings are isolated into test**.
                 Exact match is unattainable in principle here, so this set is
                 judged by n-best coverage and calibration.

The claim "this is not just memorization" is established by the gap between (1)
and (2)/(3). The soundness of the split therefore *is* the interpretability of
the results, which is why verify_splits() runs both at dataset build time and
in pytest.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

from .data import Record
from .kana import char_ngrams, is_kana_text

SPLIT_NAMES = ("train", "dev", "test_known", "test_unseen", "test_ambiguous")


def _unit_hash(key: str, salt: str) -> float:
    """Deterministic hash into [0,1). sha1 so that it reproduces without a seed."""
    digest = hashlib.sha1(f"{salt}\x00{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def is_kanji(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or code == 0x3005


def kanji_bigrams(text: str) -> set[str]:
    """Adjacent pairs of kanji. The unit used to decide the hard set."""
    return {bg for bg in char_ngrams(text, 2) if is_kanji(bg[0]) and is_kanji(bg[1])}


@dataclass
class SplitConfig:
    """Split configuration.

    Attributes:
        salt: Hash salt. Changing it changes the whole split, so tie it to the
            dataset version.
        unseen_core_ratio: Fraction of trade-name cores reserved for test only.
        dev_ratio: Fraction of known-side corporations routed to dev.
        known_test_ratio: Fraction of known-side corporations routed to test_known.
        min_ambiguous_variants: Minimum number of distinct readings for a core
            to count as ambiguous.
    """

    salt: str = "phonebook-v1"
    unseen_core_ratio: float = 0.12
    dev_ratio: float = 0.03
    known_test_ratio: float = 0.05
    min_ambiguous_variants: int = 2
    max_ambiguous_heldout_per_core: int = 4


@dataclass
class SplitStats:
    counts: dict[str, int] = field(default_factory=dict)
    unique_cores: dict[str, int] = field(default_factory=dict)
    hard_count: int = 0
    copy_only_count: dict[str, int] = field(default_factory=dict)
    ambiguous_cores: int = 0
    ambiguous_pairs: int = 0
    dropped_ambiguous_overflow: int = 0
    kind_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    def dumps(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class Splits:
    """The split result. Each split is a list of Records.

    ambiguous_readings maps a trade-name core to the set of readings that
    genuinely exist for it, so the ambiguous set can be scored by "does the
    n-best contain any attested reading".
    """

    data: dict[str, list[Record]]
    config: SplitConfig
    stats: SplitStats
    hard_core_set: set[str] = field(default_factory=set)
    ambiguous_readings: dict[str, list[str]] = field(default_factory=dict)

    def __getitem__(self, name: str) -> list[Record]:
        return self.data[name]

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def is_hard(self, rec: Record) -> bool:
        return rec.core in self.hard_core_set


def make_splits(records: Sequence[Record], config: SplitConfig | None = None) -> Splits:
    """Split records into train/dev/test_known/test_unseen/test_ambiguous.

    Claim supported: **unseen-entity performance**. Everything reported for the
    four evaluation sets rests on this function. The split uses a deterministic
    hash rather than a random seed, so the same input and the same salt always
    reproduce the same split.
    """
    cfg = config or SplitConfig()
    records = [r for r in records if r.core and r.core_reading]

    # --- Aggregate readings per core -> detect ambiguity --------------------
    by_core: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        by_core[rec.core].append(rec)

    reading_counts: dict[str, Counter] = {
        core: Counter(r.core_reading for r in recs) for core, recs in by_core.items()
    }
    ambiguous_cores = {
        core
        for core, counter in reading_counts.items()
        if len(counter) >= cfg.min_ambiguous_variants
    }

    data: dict[str, list[Record]] = {name: [] for name in SPLIT_NAMES}
    ambiguous_readings: dict[str, list[str]] = {}
    dropped_ambiguous = 0

    for core, recs in by_core.items():
        if core in ambiguous_cores:
            # Same surface, different readings: majority reading goes to
            # train, minority readings to test_ambiguous. The goal is
            # "a surface you have seen, with a reading you have not".
            counter = reading_counts[core]
            ambiguous_readings[core] = [r for r, _ in counter.most_common()]
            majority = counter.most_common(1)[0][0]
            heldout = 0
            for rec in sorted(recs, key=lambda r: r.corporate_number):
                if rec.core_reading == majority:
                    data["train"].append(rec)
                elif heldout < cfg.max_ambiguous_heldout_per_core:
                    data["test_ambiguous"].append(rec)
                    heldout += 1
                else:
                    # Minority records past the cap are **dropped**. Putting
                    # them back into train would make the (core, reading) pair
                    # appear in both train and test_ambiguous, destroying the
                    # definition of "only the reading is unseen".
                    dropped_ambiguous += 1
            continue

        if _unit_hash(core, cfg.salt + ":core") < cfg.unseen_core_ratio:
            # Unseen entity: this core never appears in train at all.
            data["test_unseen"].extend(recs)
            continue

        # Known entity: distribute by corporate number into train/dev/test_known.
        assigned: list[tuple[str, Record]] = []
        for rec in recs:
            u = _unit_hash(rec.corporate_number or rec.name_raw, cfg.salt + ":ent")
            if u < cfg.dev_ratio:
                assigned.append(("dev", rec))
            elif u < cfg.dev_ratio + cfg.known_test_ratio:
                assigned.append(("test_known", rec))
            else:
                assigned.append(("train", rec))
        if not any(name == "train" for name, _ in assigned):
            # If every corporation for this core fell into dev/test, the core
            # would no longer be "known". Move the first record back to train
            # to preserve the definition (the core exists in train).
            assigned[0] = ("train", assigned[0][1])
        for name, rec in assigned:
            data[name].append(rec)

    # --- Hard set: unseen entities containing a kanji bigram absent from train
    train_bigrams: set[str] = set()
    for rec in data["train"]:
        train_bigrams |= kanji_bigrams(rec.core)

    hard_core_set: set[str] = set()
    for rec in data["test_unseen"]:
        bigrams = kanji_bigrams(rec.core)
        if bigrams and not (bigrams <= train_bigrams):
            hard_core_set.add(rec.core)

    stats = _compute_stats(data, hard_core_set, ambiguous_readings)
    stats.dropped_ambiguous_overflow = dropped_ambiguous
    return Splits(
        data=data,
        config=cfg,
        stats=stats,
        hard_core_set=hard_core_set,
        ambiguous_readings=ambiguous_readings,
    )


def _compute_stats(
    data: dict[str, list[Record]],
    hard_core_set: set[str],
    ambiguous_readings: dict[str, list[str]],
) -> SplitStats:
    stats = SplitStats()
    for name, recs in data.items():
        stats.counts[name] = len(recs)
        stats.unique_cores[name] = len({r.core for r in recs})
        stats.copy_only_count[name] = sum(1 for r in recs if is_kana_text(r.core))
        kind_counter = Counter(r.kind_label for r in recs)
        stats.kind_distribution[name] = dict(kind_counter.most_common())
    stats.hard_count = sum(1 for r in data["test_unseen"] if r.core in hard_core_set)
    stats.ambiguous_cores = len(ambiguous_readings)
    stats.ambiguous_pairs = sum(
        len(v) * (len(v) - 1) // 2 for v in ambiguous_readings.values()
    )
    return stats


def hard_subset(splits: Splits) -> list[Record]:
    """Extract the hard set (unseen entities with an unseen kanji bigram)."""
    return [r for r in splits["test_unseen"] if r.core in splits.hard_core_set]


def ambiguous_pairs(splits: Splits, limit: int | None = None) -> list[dict]:
    """Enumerate same-surface/different-reading pairs (for cards and figures)."""
    pairs: list[dict] = []
    for core, readings in sorted(splits.ambiguous_readings.items()):
        for i in range(len(readings)):
            for j in range(i + 1, len(readings)):
                pairs.append({"core": core, "reading_a": readings[i], "reading_b": readings[j]})
                if limit and len(pairs) >= limit:
                    return pairs
    return pairs


# --- Leakage checks --------------------------------------------------------
class LeakageError(AssertionError):
    """A split leak. A fatal error that must abort dataset generation."""


def verify_splits(splits: Splits, *, raise_on_error: bool = True) -> dict:
    """Verify that the split is sound.

    Claim supported: **the validity of the unseen-entity result**. Checks:

      1. No corporation (corporate number) spans more than one split.
      2. No test_unseen core occurs in train.
      3. Every test_known core does occur in train (the definition of "known").
      4. The hard set is a subset of test_unseen.
      5. No test_ambiguous (core, reading) pair occurs in train.
    """
    problems: list[str] = []

    seen_entity: dict[str, str] = {}
    for name, recs in splits.items():
        for rec in recs:
            key = rec.corporate_number
            if not key:
                continue
            if key in seen_entity and seen_entity[key] != name:
                problems.append(
                    f"corporation spans {seen_entity[key]} and {name}: {key} ({rec.name})"
                )
            seen_entity[key] = name

    train_cores = {r.core for r in splits["train"]}
    for rec in splits["test_unseen"]:
        if rec.core in train_cores:
            problems.append(f"core marked unseen is present in train: {rec.core}")
    for rec in splits["test_known"]:
        if rec.core not in train_cores:
            problems.append(f"core marked known is absent from train: {rec.core}")

    unseen_cores = {r.core for r in splits["test_unseen"]}
    for core in splits.hard_core_set:
        if core not in unseen_cores:
            problems.append(f"hard core lies outside test_unseen: {core}")

    train_pairs = {(r.core, r.core_reading) for r in splits["train"]}
    for rec in splits["test_ambiguous"]:
        if (rec.core, rec.core_reading) in train_pairs:
            problems.append(
                f"ambiguous (core, reading) pair is present in train: {rec.core}/{rec.core_reading}"
            )

    result = {"ok": not problems, "problems": problems[:50], "n_problems": len(problems)}
    if problems and raise_on_error:
        raise LeakageError("\n".join(problems[:20]))
    return result


def summarize(splits: Splits) -> str:
    """Human-readable split summary (pasted into the README / dataset card)."""
    lines = ["| split | rows | unique cores | note |", "|---|---:|---:|---|"]
    notes = {
        "train": "training",
        "dev": "calibration and threshold selection",
        "test_known": "known entity (core in train, different corporation)",
        "test_unseen": "unseen entity (core absent from train)",
        "test_ambiguous": "same surface, unseen reading",
    }
    for name in SPLIT_NAMES:
        lines.append(
            f"| {name} | {splits.stats.counts.get(name, 0):,} | "
            f"{splits.stats.unique_cores.get(name, 0):,} | {notes[name]} |"
        )
    lines.append(
        f"| \u2514 of which hard | {splits.stats.hard_count:,} | - | contains an unseen kanji bigram |"
    )
    return "\n".join(lines)
