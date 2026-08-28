#!/usr/bin/env python3
"""Dataset build: CSV -> cleansing -> four-way split -> leakage check -> save/publish.

Claim supported: **the measurability of unseen-entity performance**.

Everything reported in the README rests on this script's output, so three
things are enforced:
  1. If verify_splits() finds a leak, the build **aborts**.
  2. An audit log is written (what was dropped, and how furigana missingness is
     distributed across corporation kinds).
  3. Whether the input was synthetic or real is recorded in the metadata.

Usage:
    python scripts/build_dataset.py --csv data/raw/00_zenkoku_all.csv --out data/processed
    python scripts/build_dataset.py --csv ... --out ... --push-to-hub <user>/phonebook-corporate-readings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.data import (  # noqa: E402
    Record,
    cleanse,
    iter_csv_rows,
    write_jsonl,
)
from phonebook.split import (  # noqa: E402
    SPLIT_NAMES,
    SplitConfig,
    ambiguous_pairs,
    hard_subset,
    make_splits,
    summarize,
    verify_splits,
)

LICENSE_NOTE = (
    "Created by processing data from the National Tax Agency Corporate Number "
    "Publication Site (https://www.houjin-bangou.nta.go.jp/). The published "
    "information may be used under terms conforming to the Japanese Government's "
    "Public Data License (Version 1.0); the source must be credited and the fact "
    "that the data has been modified must be stated."
)


def to_row(rec: Record, split: str, is_hard: bool, all_readings: list[str] | None) -> dict:
    """One published row. Keeps both the original and the normalized surface."""
    return {
        "corporate_number": rec.corporate_number,
        "name_raw": rec.name_raw,
        "name": rec.name,
        "furigana_raw": rec.furigana_raw,
        "furigana": rec.furigana,
        "core": rec.core,
        "core_reading": rec.core_reading,
        "legal_form": rec.prefix_form or rec.suffix_form,
        "legal_form_position": rec.position,
        "kind": rec.kind,
        "kind_label": rec.kind_label,
        "prefecture": rec.prefecture,
        "split": split,
        "is_hard": is_hard,
        "all_readings": all_readings or [rec.core_reading],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="NTA CSV or zip (synthetic CSV also works)")
    parser.add_argument("--encoding", default="utf-8", help="use cp932 for the Shift-JIS edition")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--salt", default="phonebook-v1", help="salt for the split hash")
    parser.add_argument("--unseen-ratio", type=float, default=0.12)
    parser.add_argument("--dev-ratio", type=float, default=0.03)
    parser.add_argument("--known-test-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--synthetic", action="store_true", help="record that the input is synthetic")
    parser.add_argument("--push-to-hub", default=None, help="Hugging Face repository id")
    parser.add_argument("--private", action="store_true", help="push as a private repository")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_csv_rows(args.csv, encoding=args.encoding)
    if args.limit:
        import itertools

        rows = itertools.islice(rows, args.limit)

    print("cleansing ...")
    labeled, missing, report = cleanse(rows)
    print(f"  with furigana: {len(labeled):,} / missing: {len(missing):,}")
    print(f"  dropped as unalignable: {report.unaligned:,}")

    print("splitting ...")
    config = SplitConfig(
        salt=args.salt,
        unseen_core_ratio=args.unseen_ratio,
        dev_ratio=args.dev_ratio,
        known_test_ratio=args.known_test_ratio,
    )
    splits = make_splits(labeled, config)

    print("leakage check ...")
    verify_splits(splits)  # raises LeakageError and aborts on failure
    print("  OK: no corporation spans splits; no unseen core occurs in train")

    hard_cores = splits.hard_core_set
    total = 0
    for name in SPLIT_NAMES:
        recs = splits[name]
        rows_out = [
            to_row(
                r,
                name,
                r.core in hard_cores,
                splits.ambiguous_readings.get(r.core),
            )
            for r in recs
        ]
        n = write_jsonl(rows_out, out_dir / f"{name}.jsonl")
        total += n
        print(f"  {name:16s} {n:>8,} rows -> {out_dir / (name + '.jsonl')}")

    write_jsonl(
        [
            {
                "corporate_number": r.corporate_number,
                "name_raw": r.name_raw,
                "name": r.name,
                "core": r.core,
                "legal_form": r.prefix_form or r.suffix_form,
                "legal_form_position": r.position,
                "kind": r.kind,
                "kind_label": r.kind_label,
                "prefecture": r.prefecture,
            }
            for r in missing
        ],
        out_dir / "missing_furigana.jsonl",
    )
    print(f"  missing_furigana {len(missing):>8,} rows (input to scripts/fill_missing.py)")

    pairs = ambiguous_pairs(splits, limit=200)
    (out_dir / "ambiguous_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "source": "National Tax Agency Corporate Number Publication Site",
        "source_url": "https://www.houjin-bangou.nta.go.jp/",
        "license_note": LICENSE_NOTE,
        "synthetic": bool(args.synthetic),
        "input_csv": str(args.csv),
        "split_config": {
            "salt": config.salt,
            "unseen_core_ratio": config.unseen_core_ratio,
            "dev_ratio": config.dev_ratio,
            "known_test_ratio": config.known_test_ratio,
        },
        "counts": splits.stats.counts,
        "unique_cores": splits.stats.unique_cores,
        "hard_count": splits.stats.hard_count,
        "ambiguous_cores": splits.stats.ambiguous_cores,
        "ambiguous_pairs": splits.stats.ambiguous_pairs,
        "dropped_ambiguous_overflow": splits.stats.dropped_ambiguous_overflow,
        "copy_only_count": splits.stats.copy_only_count,
        "kind_distribution": splits.stats.kind_distribution,
        "missing_furigana": len(missing),
        "cleanse_report": report.to_dict(),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "SPLITS.md").write_text(summarize(splits) + "\n", encoding="utf-8")

    print()
    print(summarize(splits))
    print()
    print("furigana missingness by corporation kind (non-random):")
    for label, stat in sorted(
        report.to_dict()["missing_rate_by_kind"].items(), key=lambda kv: -kv[1]["total"]
    )[:8]:
        print(
            f"  {label:34s} missing {stat['missing_rate']:.1%}  "
            f"({stat['missing']:,}/{stat['total']:,})"
        )

    if args.push_to_hub:
        push(out_dir, args.push_to_hub, metadata, private=args.private)
    return 0


def push(out_dir: Path, repo_id: str, metadata: dict, private: bool = False) -> None:
    """Push the built dataset to the Hugging Face Hub."""
    from datasets import Dataset, DatasetDict

    dd = {}
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}.jsonl"
        if path.exists():
            dd[name] = Dataset.from_json(str(path))
    missing_path = out_dir / "missing_furigana.jsonl"
    if missing_path.exists():
        dd["missing_furigana"] = Dataset.from_json(str(missing_path))
    dataset = DatasetDict(dd)
    print(f"pushing to Hugging Face: {repo_id}")
    dataset.push_to_hub(repo_id, private=private)

    card_path = Path(__file__).resolve().parents[1] / "cards" / "dataset_card.md"
    if card_path.exists():
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  uploaded the dataset card")


if __name__ == "__main__":
    raise SystemExit(main())
