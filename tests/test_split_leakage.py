"""Claim: **the split has no leakage.**

If these tests fail, the reported "unseen-entity performance" is measuring
something that is not unseen, and every claim in the project collapses. They are
written to be the strictest tests in the suite.
"""

from __future__ import annotations

import pytest

from phonebook.split import (
    LeakageError,
    SplitConfig,
    make_splits,
    hard_subset,
    verify_splits,
)


@pytest.fixture
def splits(records):
    return make_splits(records, SplitConfig(unseen_core_ratio=0.3, dev_ratio=0.1, known_test_ratio=0.2))


def test_no_entity_crosses_splits(splits):
    """No corporation (corporate number) may appear in more than one split."""
    seen: dict[str, str] = {}
    for name, recs in splits.items():
        for rec in recs:
            assert rec.corporate_number not in seen or seen[rec.corporate_number] == name, (
                f"{rec.corporate_number} spans {seen.get(rec.corporate_number)} and {name}"
            )
            seen[rec.corporate_number] = name


def test_unseen_cores_absent_from_train(splits):
    train_cores = {r.core for r in splits["train"]}
    for rec in splits["test_unseen"]:
        assert rec.core not in train_cores, f"core {rec.core} marked unseen is present in train"


def test_known_cores_present_in_train(splits):
    train_cores = {r.core for r in splits["train"]}
    for rec in splits["test_known"]:
        assert rec.core in train_cores


def test_hard_subset_is_within_unseen(splits):
    unseen = {r.corporate_number for r in splits["test_unseen"]}
    for rec in hard_subset(splits):
        assert rec.corporate_number in unseen


def test_hard_cores_contain_unseen_kanji_bigram(splits):
    from phonebook.split import kanji_bigrams

    train_bigrams: set[str] = set()
    for rec in splits["train"]:
        train_bigrams |= kanji_bigrams(rec.core)
    for rec in hard_subset(splits):
        assert not (kanji_bigrams(rec.core) <= train_bigrams), (
            f"every kanji bigram of {rec.core} occurs in train, so it is not hard"
        )


def test_ambiguous_pairs_hold_out_minority_reading(splits):
    """Ambiguous set: the (core, reading) pair is absent from train, while the
    core itself is present."""
    train_pairs = {(r.core, r.core_reading) for r in splits["train"]}
    train_cores = {r.core for r in splits["train"]}
    for rec in splits["test_ambiguous"]:
        assert (rec.core, rec.core_reading) not in train_pairs
        assert rec.core in train_cores, "the surface must exist in train; only the reading is unseen"


def test_verify_splits_passes(splits):
    assert verify_splits(splits)["ok"]


def test_verify_splits_detects_injected_leak(splits):
    """Check that the checker actually catches a leak (a test of the test)."""
    leaked = splits["train"][0]
    splits.data["test_unseen"].append(leaked)
    with pytest.raises(LeakageError):
        verify_splits(splits)


def test_splits_are_deterministic(records):
    a = make_splits(records, SplitConfig(salt="fixed"))
    b = make_splits(records, SplitConfig(salt="fixed"))
    for name in a.keys():
        assert [r.corporate_number for r in a[name]] == [r.corporate_number for r in b[name]]


def test_different_salt_changes_assignment(records):
    a = make_splits(records, SplitConfig(salt="one", unseen_core_ratio=0.5))
    b = make_splits(records, SplitConfig(salt="two", unseen_core_ratio=0.5))
    assert {r.core for r in a["test_unseen"]} != {r.core for r in b["test_unseen"]}


def test_ambiguous_overflow_records_are_dropped_not_leaked_to_train():
    """Minority records past the cap must be dropped, not returned to train.

    Returning them would put the (core, reading) pair in both train and
    test_ambiguous, destroying the definition of the ambiguous set: the surface
    is known, only the reading is unseen.
    """
    from phonebook.data import Record
    from phonebook.split import SplitConfig, make_splits, verify_splits

    records = [
        Record(
            corporate_number=f"{i:013d}", name_raw="日本商事", name="日本商事",
            furigana_raw="", furigana="", kind="301", core="日本商事",
            core_reading="ニホンショウジ" if i < 3 else "ニッポンショウジ", aligned=True,
        )
        for i in range(10)
    ]
    splits = make_splits(records, SplitConfig(max_ambiguous_heldout_per_core=2))
    assert len(splits["test_ambiguous"]) == 2
    assert splits.stats.dropped_ambiguous_overflow == 1
    assert all(r.core_reading == "ニッポンショウジ" for r in splits["train"])
    assert verify_splits(splits)["ok"]
