#!/usr/bin/env python3
"""Fill in readings for corporations with no registered furigana.

Claims supported: the practical value of **calibration** and **rejection**, and
the public value of the derived artifact.

A substantial number of corporations in the NTA data have no registered
furigana. This script estimates one for each and publishes it **together with a
confidence and an accept/reject decision**. Two things are stated emphatically:

  1. **These are estimates, not official furigana.** They cannot be used for
     registration or any official procedure. Every output row carries
     ``is_estimate: true``, and the dataset card repeats it.
  2. **The missingness is non-random.** Roughly 30-40% of Kabushiki-Kaisha and
     40-50% of Yugen-Kaisha records lack a furigana, against under 10% for other
     kinds. An "overall accuracy" therefore says nothing useful about the
     quality of this derived data. This script **stratifies by (corporation
     kind x whether the trade-name core appeared in training)** and reports the
     expected accuracy reweighted to the composition of the missing set.

Usage:
    python scripts/fill_missing.py --data data/processed --model artifacts/model \
        --out artifacts/derived/furigana_estimates.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.data import read_jsonl, write_jsonl  # noqa: E402
from phonebook.runtime import load_reader  # noqa: E402
from phonebook.structure import canonicalize_legal_reading  # noqa: E402

DISCLAIMER = (
    "This reading is an ESTIMATE produced by the Phonebook model. It is not the "
    "official furigana published by Japan's National Tax Agency. For the "
    "official furigana, consult the NTA Corporate Number Publication Site."
)


def stratified_accuracy(
    data_dir: Path, reader, missing_rows: list[dict], limit: int, nbest: int
) -> dict:
    """Per-stratum accuracy (kind x core seen/unseen) and the expected accuracy
    on the missing set.

    Claim supported: **a correct estimate under non-random missingness**.
    Accuracy is measured per stratum on the labeled test sets, then reweighted
    to the strata composition of the missing set. Carrying the overall average
    across instead would systematically misjudge a missing set that is skewed
    toward Kabushiki-Kaisha.
    """
    train_cores = {r["core"] for r in read_jsonl(data_dir / "train.jsonl")}
    eval_rows: list[dict] = []
    for split in ("test_known", "test_unseen"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            eval_rows += read_jsonl(path)
    eval_rows = eval_rows[:limit]

    def stratum(row: dict) -> tuple[str, str]:
        seen = "core seen" if row["core"] in train_cores else "core unseen"
        return (row.get("kind", "?"), seen)

    names = [r["name"] for r in eval_rows]
    golds = [canonicalize_legal_reading(r["furigana"]) for r in eval_rows]
    preds: list[str] = []
    confs: list[float] = []
    for i in range(0, len(names), 64):
        for res in reader.read_batch(names[i : i + 64], nbest=1):
            preds.append(canonicalize_legal_reading(res.candidates[0].reading) if res.candidates else "")
            confs.append(res.confidence)

    per: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row, gold, pred in zip(eval_rows, golds, preds):
        per[stratum(row)].append(int(pred == gold))

    missing_dist = Counter(
        (r.get("kind", "?"), "core seen" if r["core"] in train_cores else "core unseen")
        for r in missing_rows
    )
    total_missing = sum(missing_dist.values()) or 1

    strata = {}
    expected = 0.0
    covered = 0
    for key, weight in missing_dist.most_common():
        hits = per.get(key)
        acc = sum(hits) / len(hits) if hits else None
        strata["|".join(key)] = {
            "missing_share": weight / total_missing,
            "n_eval": len(hits) if hits else 0,
            "accuracy": acc,
        }
        if acc is not None:
            expected += (weight / total_missing) * acc
            covered += weight
    return {
        "strata": strata,
        "expected_accuracy_on_missing": expected / (covered / total_missing) if covered else None,
        "coverage_of_strata": covered / total_missing,
        "naive_overall_accuracy": sum(sum(v) for v in per.values())
        / max(sum(len(v) for v in per.values()), 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--model", default="artifacts/model")
    p.add_argument("--out", default="artifacts/derived/furigana_estimates.jsonl")
    p.add_argument("--nbest", type=int, default=3)
    p.add_argument("--threshold", type=float, default=None, help="acceptance threshold (default: threshold.json)")
    p.add_argument("--limit", type=int, default=None, help="cap the number of rows processed")
    p.add_argument("--eval-limit", type=int, default=3000, help="items used to estimate per-stratum accuracy")
    p.add_argument("--push-to-hub", default=None)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data)
    rows = read_jsonl(data_dir / "missing_furigana.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    print(f"records missing furigana: {len(rows):,}")

    reader = load_reader(args.model, threshold=args.threshold)
    threshold = reader.threshold
    print(f"acceptance threshold: {threshold}")

    out_rows: list[dict] = []
    accepted = 0
    for i in range(0, len(rows), 64):
        chunk = rows[i : i + 64]
        results = reader.read_batch([r["name"] for r in chunk], nbest=args.nbest)
        for row, res in zip(chunk, results):
            is_accepted = not res.rejected and bool(res.candidates)
            accepted += int(is_accepted)
            out_rows.append(
                {
                    "corporate_number": row["corporate_number"],
                    "name": row["name_raw"],
                    "name_normalized": row["name"],
                    "kind": row.get("kind"),
                    "kind_label": row.get("kind_label"),
                    "prefecture": row.get("prefecture"),
                    "estimated_furigana": (
                        canonicalize_legal_reading(res.candidates[0].reading)
                        if res.candidates
                        else None
                    ),
                    "confidence": round(res.confidence, 4),
                    "accepted": is_accepted,
                    "candidates": [
                        {"furigana": canonicalize_legal_reading(c.reading), "prob": round(c.prob, 4)}
                        for c in res.candidates
                    ],
                    "source": res.source,
                    "is_estimate": True,
                    "disclaimer": DISCLAIMER,
                }
            )
        done = i + len(chunk)
        if (i // 64) % 50 == 0 or done >= len(rows):
            print(f"  {done:,}/{len(rows):,}", flush=True)

    out_path = Path(args.out)
    write_jsonl(out_rows, out_path)
    print(
        f"wrote {out_path} ({len(out_rows):,} rows; accepted {accepted:,} = "
        f"{accepted/max(len(out_rows),1):.1%})"
    )

    print("\nestimating per-stratum accuracy ...")
    strat = stratified_accuracy(data_dir, reader, rows, args.eval_limit, args.nbest)
    kind_counts = Counter(r.get("kind_label", "?") for r in rows)
    summary = {
        "n_missing": len(rows),
        "n_accepted": accepted,
        "acceptance_rate": accepted / max(len(out_rows), 1),
        "threshold": threshold,
        "missing_by_kind": dict(kind_counts.most_common()),
        "stratified_estimate": strat,
        "disclaimer": DISCLAIMER,
        "non_random_missingness_note": (
            "Missingness is heavily concentrated in Kabushiki-Kaisha and "
            "Yugen-Kaisha. Read the per-stratum numbers, not the overall average."
        ),
    }
    (out_path.parent / "fill_missing_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  naive overall accuracy        : {strat['naive_overall_accuracy']:.3f}")
    if strat["expected_accuracy_on_missing"] is not None:
        print(
            f"  expected accuracy on missing  : "
            f"{strat['expected_accuracy_on_missing']:.3f}   <- read this one"
        )
    print(f"  report: {out_path.parent/'fill_missing_report.json'}")
    print(f"\n{DISCLAIMER}")

    if args.push_to_hub:
        from datasets import Dataset
        from huggingface_hub import HfApi

        Dataset.from_json(str(out_path)).push_to_hub(args.push_to_hub, private=args.private)
        card = Path(__file__).resolve().parents[1] / "cards" / "derived_dataset_card.md"
        if card.exists():
            HfApi().upload_file(
                path_or_fileobj=str(card),
                path_in_repo="README.md",
                repo_id=args.push_to_hub,
                repo_type="dataset",
            )
        print(f"pushed: https://huggingface.co/datasets/{args.push_to_hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
