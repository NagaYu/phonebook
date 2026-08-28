#!/usr/bin/env python3
"""Benchmark: (A) existing G2P, (B) LLM, (C) Phonebook, (D) quantized, per split.

Claims supported: **all of them**, and in particular "comparable on known
entities, separating on unseen and hard ones".

Ground rules
------------
1. **Always report per split.** No overall average is published: an average is
   dominated by the abundance of known entities and invites mistaking
   memorization for generalization.
2. **Conditions that were not run stay blank.** If pyopenjtalk is not
   installed, or API credentials are absent, the condition is recorded as
   ``available: false`` and no numbers are invented.
3. **Do not win on notation.** Alongside strict exact match we report a lenient
   exact match with long vowels normalized (exact_match_phonetic). Existing G2P
   emits a pronunciation form, which the strict metric penalizes unfairly.
4. **Rejection is reported separately.** Comparing an accuracy that already
   excludes rejected items against other conditions would be unfair, so the
   main tables are computed without rejection and its effect gets its own table.

Usage:
    python scripts/evaluate.py --data data/processed --model artifacts/model --out benchmarks
    python scripts/evaluate.py --render-only --out benchmarks   # rebuild RESULTS.md from results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.baselines import (  # noqa: E402
    ClaudeBaseline,
    MeCabUniDicBaseline,
    PyOpenJTalkBaseline,
)
from phonebook.calibrate import extract_features, risk_coverage_curve  # noqa: E402
from phonebook.data import read_jsonl  # noqa: E402
from phonebook.decode import PhonebookReader  # noqa: E402
from phonebook.en2kana import EnglishToKatakana  # noqa: E402
from phonebook.metrics import evaluate, format_table, measure_speed  # noqa: E402
from phonebook.model import CharSeq2Seq  # noqa: E402
from phonebook.structure import StructuralSplitter, canonicalize_legal_reading  # noqa: E402

EVAL_SETS = ("test_known", "test_unseen", "hard", "test_ambiguous")
SET_LABELS = {
    "test_known": "(1) known entity",
    "test_unseen": "(2) unseen entity",
    "hard": "(3) hard (unseen kanji bigram)",
    "test_ambiguous": "(4) ambiguous (same surface, other reading)",
}


def load_eval_sets(data_dir: Path, limit: int | None = None) -> dict[str, list[dict]]:
    """Load the four evaluation sets. ``hard`` is a subset of ``test_unseen``."""
    sets: dict[str, list[dict]] = {}
    unseen = read_jsonl(data_dir / "test_unseen.jsonl")
    sets["test_known"] = read_jsonl(data_dir / "test_known.jsonl")
    sets["test_unseen"] = unseen
    sets["hard"] = [r for r in unseen if r.get("is_hard")]
    sets["test_ambiguous"] = read_jsonl(data_dir / "test_ambiguous.jsonl")
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
    return sets


def accepted_full_readings(rows: list[dict], splitter: StructuralSplitter) -> list[list[str]]:
    """For the ambiguous set: every attested reading of a surface, legal form included."""
    out = []
    for row in rows:
        st = splitter.split(row["name"])
        readings = row.get("all_readings") or [row["core_reading"]]
        out.append([canonicalize_legal_reading(st.compose(r)) for r in readings])
    return out


def make_quantized_model(model: CharSeq2Seq, scheme: str):
    """Return a model loaded with quantize-then-dequantize weights, plus size stats."""
    from phonebook.quantize import apply_dequantized, quantize_state_dict

    tensors, stats = quantize_state_dict(model.state_dict(), scheme=scheme)
    import copy

    qmodel = copy.deepcopy(model)
    qmodel.load_state_dict(apply_dequantized(model.state_dict(), tensors))
    qmodel.eval()
    return qmodel, stats


def run_phonebook(reader: PhonebookReader, names: list[str], nbest: int):
    """For a list of corporate names return (top-1, n-best, calibrated confidence, latency)."""
    preds, nbests, confs = [], [], []
    latencies = []
    for i in range(0, len(names), 64):
        chunk = names[i : i + 64]
        t0 = time.perf_counter()
        results = reader.read_batch(chunk, nbest=nbest)
        dt = (time.perf_counter() - t0) * 1000 / len(chunk)
        for r in results:
            preds.append(canonicalize_legal_reading(r.candidates[0].reading) if r.candidates else None)
            nbests.append([canonicalize_legal_reading(c.reading) for c in r.candidates])
            confs.append(r.confidence)
            latencies.append(dt)
    return preds, nbests, confs, latencies


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--model", default="artifacts/model")
    p.add_argument("--out", default="benchmarks")
    p.add_argument("--limit", type=int, default=None, help="cap the size of each set (for smoke tests)")
    p.add_argument("--nbest", type=int, default=3)
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--speed-n", type=int, default=500, help="number of items used for the speed measurement")
    p.add_argument("--llm-cache", default="benchmarks/llm_cache.jsonl")
    p.add_argument("--llm-online", action="store_true", help="actually call the API for condition B")
    p.add_argument("--llm-limit", type=int, default=300, help="items per set for condition B (cost control)")
    p.add_argument("--skip-quantized", action="store_true")
    p.add_argument(
        "--render-only", action="store_true",
        help="rebuild RESULTS.md from an existing results.json without re-running anything",
    )
    args = p.parse_args()

    if args.render_only:
        out_dir = Path(args.out)
        results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        (out_dir / "RESULTS.md").write_text(render_markdown(results), encoding="utf-8")
        print(f"rewrote {out_dir / 'RESULTS.md'}")
        return 0

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splitter = StructuralSplitter()

    sets = load_eval_sets(data_dir, args.limit)
    for name, rows in sets.items():
        print(f"{SET_LABELS[name]}: {len(rows):,} items")

    references = {
        k: [canonicalize_legal_reading(r["furigana"]) for r in rows] for k, rows in sets.items()
    }
    names = {k: [r["name"] for r in rows] for k, rows in sets.items()}
    accepted = {k: accepted_full_readings(rows, splitter) for k, rows in sets.items()}

    results: dict = {"systems": {}, "sets": {k: len(v) for k, v in sets.items()}}

    # --- (A) general-purpose Japanese G2P ---------------------------------
    for baseline in (PyOpenJTalkBaseline(), MeCabUniDicBaseline()):
        key = f"A:{baseline.name}"
        entry = {"condition": "A", **baseline.describe(), "splits": {}}
        if baseline.available:
            print(f"\n[{key}] running ...")
            for set_name, rows in sets.items():
                preds = baseline.read_batch(names[set_name])
                m = evaluate(
                    references[set_name],
                    preds,
                    nbest=[[p] if p else [] for p in preds],
                    accepted_readings=accepted[set_name],
                    ks=(1, 3),
                )
                entry["splits"][set_name] = m.to_dict()
                print(
                    f"  {SET_LABELS[set_name]:44s} exact {m.exact_match:.3f} / "
                    f"lenient {m.exact_match_phonetic:.3f} / CER {m.cer:.3f}"
                )
            speed = measure_speed(baseline.read_batch, names["test_unseen"][: args.speed_n])
            entry["speed"] = speed.to_dict()
        else:
            print(f"\n[{key}] not run (unavailable): {baseline.note}")
        results["systems"][key] = entry

    # --- (B) large language model ------------------------------------------
    llm = ClaudeBaseline(cache_path=args.llm_cache, offline_only=not args.llm_online)
    entry = {"condition": "B", **llm.describe(), "splits": {}, "n_evaluated": args.llm_limit}
    if llm.available:
        print(f"\n[B:{llm.name}] running (up to {args.llm_limit} items per set) ...")
        for set_name, rows in sets.items():
            sub = slice(0, args.llm_limit)
            nb = [llm.read_nbest(n) for n in names[set_name][sub]]
            preds = [c[0] if c else None for c in nb]
            m = evaluate(
                references[set_name][sub],
                preds,
                nbest=nb,
                accepted_readings=accepted[set_name][sub],
                ks=(1, 3),
            )
            entry["splits"][set_name] = m.to_dict()
            print(
                f"  {SET_LABELS[set_name]:44s} exact {m.exact_match:.3f} / "
                f"n-best@3 {m.nbest_coverage.get(3, 0):.3f}"
            )
    else:
        print(f"\n[B:claude] not run: {llm.note}")
    results["systems"]["B:claude"] = entry

    # --- (C)(D) Phonebook, before and after quantization --------------------
    model_dir = Path(args.model)
    if not (model_dir / "model.pt").exists():
        print(f"\nNo trained model at {model_dir}; skipping conditions (C) and (D).", file=sys.stderr)
    else:
        model, tokenizer = CharSeq2Seq.load(model_dir)
        threshold = None
        thr_path = model_dir / "threshold.json"
        if thr_path.exists():
            threshold = json.loads(thr_path.read_text(encoding="utf-8")).get("threshold")
        from phonebook.calibrate import PlattCalibrator

        cal_path = model_dir / "calibrator.json"
        calibrator = PlattCalibrator.load(cal_path) if cal_path.exists() else None

        variants = [("C:phonebook-fp32", model, {"scheme": "fp32"})]
        if not args.skip_quantized:
            for scheme in ("q4_k_m", "q8_0"):
                qmodel, qstats = make_quantized_model(model, scheme)
                label = "D:phonebook-q4_k_m" if scheme == "q4_k_m" else "D:phonebook-q8_0"
                variants.append((label, qmodel, qstats))

        for label, m_, qstats in variants:
            print(f"\n[{label}] running ...")
            reader = PhonebookReader(
                m_,
                tokenizer,
                en2kana=EnglishToKatakana(),
                calibrator=calibrator,
                beam_size=args.beam_size,
            )
            entry = {
                "condition": label.split(":")[0],
                "available": True,
                "note": f"quantization: {qstats.get('scheme', 'fp32')}",
                "quantization": qstats,
                "parameters": m_.num_parameters(),
                "splits": {},
                "rejection": {},
            }
            for set_name, rows in sets.items():
                preds, nbests, confs, lat = run_phonebook(reader, names[set_name], args.nbest)
                m = evaluate(
                    references[set_name],
                    preds,
                    nbest=nbests,
                    confidences=confs,
                    accepted_readings=accepted[set_name],
                    ks=(1, 3),
                    latencies=lat,
                )
                entry["splits"][set_name] = m.to_dict()
                if threshold is not None:
                    rejected = [c < threshold for c in confs]
                    mr = evaluate(
                        references[set_name], preds, confidences=confs, rejected=rejected
                    )
                    entry["rejection"][set_name] = {
                        "threshold": threshold,
                        "coverage": mr.coverage,
                        "accuracy_on_accepted": mr.accuracy_on_accepted,
                        "accuracy_no_reject": m.exact_match,
                    }
                correct = [(preds[i] or "") == references[set_name][i] for i in range(len(preds))]
                entry.setdefault("risk_coverage", {})[set_name] = risk_coverage_curve(confs, correct)
                print(
                    f"  {SET_LABELS[set_name]:44s} exact {m.exact_match:.3f} / "
                    f"n-best@3 {m.nbest_coverage.get(3, 0):.3f} / CER {m.cer:.3f} / ECE {m.ece:.3f}"
                )
            speed = measure_speed(
                lambda batch: reader.read_batch(batch, nbest=1), names["test_unseen"][: args.speed_n]
            )
            entry["speed"] = speed.to_dict()
            print(
                f"  speed: {speed.ms_per_item:.2f} ms/item ({speed.items_per_second:.0f} items/s), "
                f"resident RSS {speed.rss_mb:.0f} MB"
            )
            results["systems"][label] = entry

    results["metadata"] = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8")) if (data_dir / "metadata.json").exists() else {}
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "RESULTS.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"\nwrote {out_dir/'results.json'} and {out_dir/'RESULTS.md'}")
    return 0


def render_markdown(results: dict) -> str:
    """Render the per-split results as Markdown (pasted into the README)."""
    lines = ["# Benchmark results", ""]
    meta = results.get("metadata", {})
    if meta.get("synthetic"):
        lines += [
            "> **Warning: these numbers come from synthetic data.** They validate that the",
            "> pipeline runs; they are not evidence for the claims. Re-run on the real NTA",
            "> bulk data before citing any figure here.",
            "",
        ]
    lines += [
        "Items evaluated: "
        + ", ".join(f"{SET_LABELS[k]} {v:,}" for k, v in results["sets"].items()),
        "",
    ]

    for metric, label in (
        ("exact_match", "Exact match (strict)"),
        ("exact_match_phonetic", "Exact match (long vowels normalized)"),
        ("cer", "Character error rate (lower is better)"),
    ):
        lines += [f"## {label}", ""]
        rows = []
        for sys_name, entry in results["systems"].items():
            if not entry.get("available"):
                rows.append({"system": sys_name, **{k: "not run" for k in EVAL_SETS}})
                continue
            row = {"system": sys_name}
            for k in EVAL_SETS:
                v = entry["splits"].get(k, {}).get(metric)
                row[k] = f"{v:.3f}" if isinstance(v, float) else "-"
            rows.append(row)
        lines += [
            format_table(rows, [("system", "condition")] + [(k, SET_LABELS[k]) for k in EVAL_SETS]),
            "",
        ]

    lines += ["## n-best@3 coverage", ""]
    rows = []
    for sys_name, entry in results["systems"].items():
        if not entry.get("available"):
            continue
        row = {"system": sys_name}
        for k in EVAL_SETS:
            v = entry["splits"].get(k, {}).get("nbest_coverage", {}).get("3")
            row[k] = f"{v:.3f}" if isinstance(v, float) else "-"
        rows.append(row)
    lines += [
        format_table(rows, [("system", "condition")] + [(k, SET_LABELS[k]) for k in EVAL_SETS]),
        "",
    ]

    lines += ["## Calibration and rejection (Phonebook only)", ""]
    rows = []
    for sys_name, entry in results["systems"].items():
        for set_name, rej in (entry.get("rejection") or {}).items():
            rows.append(
                {
                    "system": sys_name,
                    "set": SET_LABELS[set_name],
                    "ece": f"{entry['splits'][set_name].get('ece', float('nan')):.4f}",
                    "no_reject": f"{rej['accuracy_no_reject']:.3f}",
                    "coverage": f"{rej['coverage']:.3f}",
                    "acc_accepted": f"{rej['accuracy_on_accepted']:.3f}",
                }
            )
    lines += [
        format_table(
            rows,
            [
                ("system", "condition"),
                ("set", "set"),
                ("ece", "ECE"),
                ("no_reject", "exact match, no rejection"),
                ("coverage", "coverage"),
                ("acc_accepted", "precision on accepted"),
            ],
        ),
        "",
    ]

    lines += ["## CPU inference speed and memory", ""]
    rows = []
    for sys_name, entry in results["systems"].items():
        speed = entry.get("speed")
        if not speed:
            continue
        rows.append(
            {
                "system": sys_name,
                "ms": f"{speed['ms_per_item']:.2f}",
                "qps": f"{speed['items_per_second']:.0f}",
                "rss": f"{speed.get('rss_mb', float('nan')):.0f}",
                "size": (
                    f"{entry['quantization']['quantized_bytes'] / 1e6:.1f} MB"
                    if entry.get("quantization", {}).get("quantized_bytes")
                    else "-"
                ),
            }
        )
    lines += [
        format_table(
            rows,
            [
                ("system", "condition"),
                ("ms", "ms/item"),
                ("qps", "items/s"),
                ("rss", "resident RSS (MB)"),
                ("size", "weight size"),
            ],
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
