"""Command-line entry point.

    phonebook read "株式会社日本電気" --nbest 3

Claims supported: **speed** and **rejection**. The CLI always prints the
per-item latency and the calibrated confidence. Nothing is hidden, so anyone
running it can check the "fast and calibrated" claim on the spot.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__


def _cmd_read(args: argparse.Namespace) -> int:
    from .runtime import load_reader

    try:
        reader = load_reader(
            args.model,
            beam_size=args.beam,
            threshold=args.threshold,
            segment_kana=not args.no_segment,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    names = list(args.names)
    if args.stdin:
        names += [line.strip() for line in sys.stdin if line.strip()]
    if not names:
        print("Give at least one corporate name.", file=sys.stderr)
        return 2

    results = reader.read_batch(names, nbest=args.nbest)

    baseline_out = None
    if args.compare:
        from .baselines import PyOpenJTalkBaseline

        baseline = PyOpenJTalkBaseline()
        baseline_out = baseline.read_batch(names) if baseline.available else [None] * len(names)

    if args.json:
        payload = []
        for i, r in enumerate(results):
            d = r.to_dict()
            if baseline_out is not None:
                d["pyopenjtalk"] = baseline_out[i]
            payload.append(d)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for i, r in enumerate(results):
        print(f"{r.name}")
        print(f"  reading    : {r.display}   (confidence {r.confidence:.3f}, path {r.source})")
        if r.rejected:
            print("  note       : below the rejection threshold, returned as 'unknown'")
        for rank, c in enumerate(r.candidates[: args.nbest], start=1):
            print(f"  candidate{rank} : {c.reading}  p={c.prob:.3f}")
        if baseline_out is not None:
            print(f"  pyopenjtalk: {baseline_out[i] or '(unavailable)'}")
        print(
            f"  structure  : prefix={r.structured.prefix_form} "
            f"core={r.structured.core} suffix={r.structured.suffix_form}"
        )
        print(f"  latency    : {r.latency_ms:.2f} ms")
        print()
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    from .structure import StructuralSplitter

    splitter = StructuralSplitter()
    for name in args.names:
        st = splitter.split(name)
        print(json.dumps(st.to_dict(), ensure_ascii=False))
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from .runtime import resolve_model_dir

    path = resolve_model_dir(args.model)
    info = {"version": __version__, "model_dir": str(path), "exists": (path / "model.pt").exists()}
    if info["exists"]:
        from .model import CharSeq2Seq

        model, tokenizer = CharSeq2Seq.load(path)
        info["parameters"] = model.num_parameters()
        info["vocab_size"] = len(tokenizer)
        info["config"] = model.cfg.to_dict()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phonebook",
        description=(
            "Predict the katakana reading of a Japanese corporate name "
            "(a small G2P model built for unseen entities)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"phonebook {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_read = sub.add_parser("read", help="predict the reading of a corporate name")
    p_read.add_argument("names", nargs="*", help="corporate name(s)")
    p_read.add_argument("--nbest", type=int, default=3, help="number of candidates (default 3)")
    p_read.add_argument("--beam", type=int, default=8, help="beam width")
    p_read.add_argument("--model", default=None, help="directory of the trained model")
    p_read.add_argument(
        "--threshold", type=float, default=None,
        help="rejection threshold; below this the answer is 'unknown'",
    )
    p_read.add_argument("--json", action="store_true", help="emit JSON")
    p_read.add_argument("--compare", action="store_true", help="also show the pyopenjtalk output")
    p_read.add_argument("--stdin", action="store_true", help="also read names from stdin")
    p_read.add_argument(
        "--no-segment", action="store_true",
        help="disable deterministic transcription of kana runs",
    )
    p_read.set_defaults(func=_cmd_read)

    p_split = sub.add_parser("split", help="show the legal-form / core decomposition")
    p_split.add_argument("names", nargs="+")
    p_split.set_defaults(func=_cmd_split)

    p_info = sub.add_parser("info", help="show model information")
    p_info.add_argument("--model", default=None)
    p_info.set_defaults(func=_cmd_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
