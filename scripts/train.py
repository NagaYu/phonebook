#!/usr/bin/env python3
"""Training and calibration for CharSeq2Seq.

Claims supported: **unseen-entity performance**, **copy fidelity**, **calibration**.

Training targets only "trade-name core -> reading"; the legal form is handled by
StructuralSplitter's rules. That means:

  - the loss concentrates on the part that is actually hard,
  - memorizing カブシキガイシャ cannot inflate apparent accuracy,
  - generation is shorter, so CPU inference is faster.

After training, on the dev set, this script also
  (a) fits the Platt calibrator (calibrator.json) and
  (b) selects the rejection threshold that meets a target precision
      (threshold.json).
Calibration and rejection are central claims of the release, not decoration, so
they always ship in the same directory as the weights.

Usage:
    python scripts/train.py --data data/processed --out artifacts/model --preset small --epochs 10
    # LoRA transfer to personal names (an exploratory side experiment)
    python scripts/train.py --data data/names --out artifacts/model-names --lora --init-from artifacts/model
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.calibrate import (  # noqa: E402
    PlattCalibrator,
    RejectionPolicy,
    extract_features,
    reliability_diagram,
)
from phonebook.data import read_jsonl  # noqa: E402
from phonebook.decode import PhonebookReader  # noqa: E402
from phonebook.en2kana import EnglishToKatakana  # noqa: E402
from phonebook.model import CharSeq2Seq, ModelConfig, apply_lora, merge_lora  # noqa: E402
from phonebook.tokenizer import CharTokenizer  # noqa: E402


def load_pairs(path: Path, src_field: str = "core", tgt_field: str = "core_reading") -> list[tuple[str, str]]:
    return [
        (row[src_field], row[tgt_field])
        for row in read_jsonl(path)
        if row.get(src_field) and row.get(tgt_field)
    ]


def make_batch(pairs, tokenizer: CharTokenizer, max_src: int, max_tgt: int):
    src_ids = [tokenizer.encode(s)[:max_src] for s, _ in pairs]
    tgt_ids = [tokenizer.encode(t, bos=True, eos=True)[: max_tgt + 2] for _, t in pairs]
    bs = len(pairs)
    S = max(len(x) for x in src_ids)
    T = max(len(x) for x in tgt_ids)
    src = torch.zeros(bs, S, dtype=torch.long)
    tgt = torch.zeros(bs, T, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt[:, :-1], tgt[:, 1:]


def evaluate_exact(model, tokenizer, pairs, beam_size=4, limit=800) -> float:
    """Dev exact-match, for progress during training. The real evaluation is
    scripts/evaluate.py."""
    from phonebook.decode import ConstrainedBeamSearch

    subset = pairs[:limit]
    beam = ConstrainedBeamSearch(model, tokenizer, beam_size=beam_size, max_len=32)
    hits = 0
    for i in range(0, len(subset), 64):
        chunk = subset[i : i + 64]
        results = beam.search([s for s, _ in chunk], nbest=1)
        for (src, gold), hyps in zip(chunk, results):
            if hyps and hyps[0].text == gold:
                hits += 1
    return hits / max(len(subset), 1)


def train(args) -> int:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)

    data_dir = Path(args.data)
    train_pairs = load_pairs(data_dir / "train.jsonl", args.src_field, args.tgt_field)
    dev_path = data_dir / "dev.jsonl"
    dev_pairs = load_pairs(dev_path, args.src_field, args.tgt_field) if dev_path.exists() else []
    if not train_pairs:
        print("training data is empty", file=sys.stderr)
        return 2
    print(f"train {len(train_pairs):,} pairs / dev {len(dev_pairs):,} pairs")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.init_from:
        model, tokenizer = CharSeq2Seq.load(args.init_from)
        print(f"initialized from {args.init_from} ({model.num_parameters()/1e6:.1f}M params)")
    else:
        tokenizer = CharTokenizer.build(
            [s for s, _ in train_pairs] + [t for _, t in train_pairs]
        )
        cfg = ModelConfig.preset(args.preset, len(tokenizer))
        cfg.use_copy = not args.no_copy
        model = CharSeq2Seq(cfg)
        print(
            f"new model: preset={args.preset} "
            f"{model.num_parameters()/1e6:.1f}M params, vocab={len(tokenizer)}"
        )

    if args.lora:
        n = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"LoRA: replaced {n} layers, {trainable:,} trainable parameters")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.03 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    step = 0
    best_dev = -1.0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_pairs)
        running = 0.0
        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i : i + args.batch_size]
            src, tgt_in, tgt_out = make_batch(
                batch, tokenizer, model.cfg.max_src_len, model.cfg.max_tgt_len - 2
            )
            logp = model(src, tgt_in)
            loss = F.nll_loss(
                logp.reshape(-1, logp.size(-1)), tgt_out.reshape(-1), ignore_index=0
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            opt.step()
            running += float(loss.detach())
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"  epoch {epoch} step {step}/{total_steps} loss {running/args.log_every:.4f} "
                    f"({elapsed:.0f}s, {step/elapsed:.1f} step/s)"
                )
                running = 0.0
        if args.max_steps and step >= args.max_steps:
            break
        if dev_pairs:
            model.eval()
            acc = evaluate_exact(model, tokenizer, dev_pairs, limit=args.dev_limit)
            print(f"  [epoch {epoch}] dev exact match = {acc:.4f}")
            if acc >= best_dev:
                best_dev = acc
                # Do not save during LoRA training: the update is merged into
                # the base weights exactly once, at the end. Merging mid-run
                # would stop the remaining epochs from learning anything.
                if not args.lora:
                    model.save(out_dir, tokenizer)
        elif not args.lora:
            model.save(out_dir, tokenizer)
    if args.lora:
        merge_lora(model)
        model.save(out_dir, tokenizer)
        print("  merged the LoRA update into the base weights and saved")
    elif not dev_pairs or best_dev < 0:
        model.save(out_dir, tokenizer)

    print(f"saved to {out_dir} (best dev exact match = {best_dev:.4f})")

    if dev_pairs and args.calibrate:
        calibrate(out_dir, data_dir, args)
    return 0


def calibrate(out_dir: Path, data_dir: Path, args) -> None:
    """Fit the calibrator and choose the rejection threshold on dev.

    Claims supported: **calibration quality** and **precision under rejection**.
    The threshold chosen here is written to artifacts/model/threshold.json and
    reused by the CLI, the Gradio app and fill_missing.py, so the operating
    point can never drift between evaluation and deployment.
    """
    print("\ncalibrating on dev ...")
    model, tokenizer = CharSeq2Seq.load(out_dir)
    reader = PhonebookReader(
        model, tokenizer, en2kana=EnglishToKatakana(), beam_size=args.beam_size
    )
    rows = read_jsonl(data_dir / "dev.jsonl")
    rows = rows[: args.calibration_limit]
    cores = [r[args.src_field] for r in rows]
    golds = [r[args.tgt_field] for r in rows]

    features: list[list[float]] = []
    correct: list[bool] = []
    raw_conf: list[float] = []
    for i in range(0, len(cores), 128):
        chunk = cores[i : i + 128]
        cand_lists = reader.read_cores(chunk, nbest=args.nbest)
        for cands, gold in zip(cand_lists, golds[i : i + 128]):
            features.append(extract_features(cands, length_hint=len(gold)))
            top = cands[0].reading if cands else ""
            correct.append(top == gold)
            raw_conf.append(cands[0].prob if cands else 0.0)

    calibrator = PlattCalibrator().fit(features, correct)
    calibrator.save(out_dir / "calibrator.json")
    calibrated = calibrator.predict(features)

    _, ece_raw, _ = reliability_diagram(raw_conf, correct)
    _, ece_cal, _ = reliability_diagram(calibrated, correct)
    threshold, coverage, precision = RejectionPolicy.fit_for_precision(
        calibrated, correct, target_precision=args.target_precision
    )
    (out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "target_precision": args.target_precision,
                "dev_coverage": coverage,
                "dev_precision_on_accepted": precision,
                "dev_ece_raw": ece_raw,
                "dev_ece_calibrated": ece_cal,
                "dev_accuracy": sum(correct) / max(len(correct), 1),
                "n_dev": len(correct),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  ECE: {ece_raw:.4f} before calibration -> {ece_cal:.4f} after")
    print(
        f"  threshold {threshold:.3f}: coverage {coverage:.1%}, "
        f"precision on accepted {precision:.1%} (target {args.target_precision:.0%})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="artifacts/model")
    p.add_argument("--preset", default="small", choices=["tiny", "small", "base"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20240401)
    p.add_argument("--threads", type=int, default=max(1, (__import__("os").cpu_count() or 4) - 1))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--dev-limit", type=int, default=600)
    p.add_argument("--no-copy", action="store_true", help="disable the copy mechanism (ablation)")
    p.add_argument("--src-field", default="core")
    p.add_argument("--tgt-field", default="core_reading")
    p.add_argument("--init-from", default=None, help="initialize from an existing model (LoRA transfer)")
    p.add_argument("--lora", action="store_true", help="LoRA transfer training (personal-name experiment)")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--calibrate", action="store_true", default=True)
    p.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    p.add_argument("--calibration-limit", type=int, default=3000)
    p.add_argument("--target-precision", type=float, default=0.95)
    p.add_argument("--beam-size", type=int, default=8)
    p.add_argument("--nbest", type=int, default=3)
    args = p.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
