#!/usr/bin/env python3
"""Generate figures from benchmarks/results.json into figures/.

Claims supported: **unseen-entity performance** (the headline figures),
**calibration**, **rejection**, **speed**.

The headline pair is figures/known_vs_unseen.png and
figures/generalization_gap.png: they show at a glance whether the methods are
close on known entities and separate on unseen and hard ones, which is the
centre of the claim that this is not merely memorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SET_ORDER = ["test_known", "test_unseen", "hard", "test_ambiguous"]
SET_LABELS = {
    "test_known": "(1) known\nentity",
    "test_unseen": "(2) unseen\nentity",
    "hard": "(3) hard\nunseen kanji bigram",
    "test_ambiguous": "(4) ambiguous\nsame surface",
}
#: One colour per condition; when a condition has several systems (pyopenjtalk
#: and mecab, say) they are separated by lightness. A legend you cannot read
#: halves the value of the figure.
COLORS = {
    "A": ["#9aa0a6", "#5f6368"],
    "B": ["#7e57c2", "#4527a0"],
    "C": ["#1a73e8", "#0b4da2"],
    "D": ["#00897b", "#26a69a", "#004d40"],
}


def setup_japanese_font() -> str | None:
    """Find and select a Japanese-capable font (labels are otherwise ASCII)."""
    candidates = [
        "Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "Noto Sans CJK JP",
        "Noto Sans JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "MS Gothic",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def color_for(system: str, index: int = 0) -> str:
    palette = COLORS.get(system.split(":")[0], ["#888888"])
    return palette[index % len(palette)]


def color_map(results: dict) -> dict[str, str]:
    """Assign colours by order of appearance within each condition."""
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for name in results["systems"]:
        cond = name.split(":")[0]
        idx = seen.get(cond, 0)
        seen[cond] = idx + 1
        out[name] = color_for(name, idx)
    return out


def short_label(system: str) -> str:
    return system.split(":", 1)[-1]


def collect(results: dict, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, entry in results["systems"].items():
        if not entry.get("available"):
            continue
        vals = {}
        for s in SET_ORDER:
            v = entry.get("splits", {}).get(s, {}).get(metric)
            if isinstance(v, (int, float)):
                vals[s] = float(v)
        if vals:
            out[name] = vals
    return out


def fig_known_vs_unseen(results: dict, out_dir: Path, metric: str, fname: str, title: str,
                        subtitle: str = "") -> None:
    data = collect(results, metric)
    if not data:
        return
    colors = color_map(results)
    systems = list(data)
    n = len(systems)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, system in enumerate(systems):
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(SET_ORDER))]
        ys = [data[system].get(s, 0.0) for s in SET_ORDER]
        bars = ax.bar(xs, ys, width=width * 0.92, label=short_label(system), color=colors[system])
        for rect, y in zip(bars, ys):
            ax.text(rect.get_x() + rect.get_width() / 2, y + 0.012, f"{y:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(SET_ORDER)))
    ax.set_xticklabels([SET_LABELS[s] for s in SET_ORDER])
    ax.set_ylabel(title)
    ax.set_ylim(0, min(1.05, max((max(v.values()) for v in data.values()), default=1) * 1.25))
    ax.set_title(f"{title} by evaluation set", fontsize=13)
    if subtitle:
        ax.text(0.5, -0.16, subtitle, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#555555")
    ax.legend(fontsize=9, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    ax.axvline(0.5, color="#bbbbbb", lw=1.2, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(0.44, ymax * 0.955, "solvable by memorization", ha="right", fontsize=8.5, color="#999999")
    ax.text(0.56, ymax * 0.955, "requires generalization", ha="left", fontsize=8.5, color="#999999")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=160)
    plt.close(fig)
    print(f"  {out_dir/fname}")


def fig_generalization_gap(results: dict, out_dir: Path) -> None:
    """Line plot of the drop from known to unseen to hard (headline figure 2).

    Two panels: strict exact match and the long-vowel-normalized version. With
    only the strict metric, existing G2P sits uniformly low because of its
    pronunciation notation and its "drop" looks small; with only the lenient one
    the harder criterion disappears. Side by side, comparing drops is meaningful.
    """
    colors = color_map(results)
    order = ["test_known", "test_unseen", "hard"]
    panels = [
        ("exact_match", "exact match (strict)"),
        ("exact_match_phonetic", "exact match (long vowels normalized)"),
    ]
    if not collect(results, "exact_match"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, (metric, label) in zip(axes, panels):
        data = collect(results, metric)
        for i, (system, vals) in enumerate(data.items()):
            ys = [vals.get(s) for s in order]
            if any(y is None for y in ys):
                continue
            ax.plot(range(len(order)), ys, marker="o", lw=2.2, ms=6,
                    label=short_label(system), color=colors[system], alpha=0.9)
            drop = ys[0] - ys[-1]
            ax.annotate(
                f"drop {drop:+.2f}", (len(order) - 1, ys[-1]),
                textcoords="offset points", xytext=(10, (i % 3 - 1) * 11), fontsize=8.5,
                color=colors[system],
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SET_LABELS[s].replace("\n", " ") for s in order], fontsize=9)
        ax.set_xlim(-0.15, len(order) - 1 + 0.7)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("exact match")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Memorization or generalization: the drop from known to unseen and hard", fontsize=13)
    fig.text(
        0.5, 0.005,
        "Note: a system that is flat because it is uniformly poor has a small drop without generalizing. "
        "Read the level and the drop together.",
        ha="center", fontsize=8.5, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "generalization_gap.png", dpi=160)
    plt.close(fig)
    print(f"  {out_dir/'generalization_gap.png'}")


def fig_reliability(results: dict, out_dir: Path) -> None:
    """Reliability diagram: confidence against empirical accuracy."""
    from phonebook.calibrate import reliability_diagram

    target = None
    for name, entry in results["systems"].items():
        if name.startswith("C:") and entry.get("available"):
            target = (name, entry)
            break
    if not target:
        return
    name, entry = target
    fig, axes = plt.subplots(1, len(SET_ORDER), figsize=(4 * len(SET_ORDER), 4), sharey=True)
    if len(SET_ORDER) == 1:
        axes = [axes]
    for ax, set_name in zip(axes, SET_ORDER):
        curve = entry.get("risk_coverage", {}).get(set_name)
        split = entry.get("splits", {}).get(set_name, {})
        ax.plot([0, 1], [0, 1], ls="--", color="#bbbbbb", lw=1, label="ideal")
        if curve:
            # Use the accuracy at each risk-coverage threshold as a stand-in
            # for confidence bins.
            xs = [pt["threshold"] for pt in curve]
            ys = [pt["accuracy"] for pt in curve]
            ax.plot(xs, ys, marker="o", ms=3, color="#1a73e8", label="accuracy at confidence >= t")
        ax.set_title(f"{SET_LABELS[set_name]}\nECE={split.get('ece', float('nan')):.3f}", fontsize=10)
        ax.set_xlabel("confidence")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("empirical accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle("Calibration: how often it is right when it says it is confident", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=160)
    plt.close(fig)
    print(f"  {out_dir/'reliability.png'}")


def fig_rejection(results: dict, out_dir: Path) -> None:
    """The coverage / precision trade-off produced by rejection."""
    entry = None
    label = ""
    for name, e in results["systems"].items():
        if name.startswith("C:") and e.get("available"):
            entry, label = e, name
            break
    if not entry or "risk_coverage" not in entry:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for set_name in SET_ORDER:
        curve = entry["risk_coverage"].get(set_name)
        if not curve:
            continue
        ax.plot([p["coverage"] for p in curve], [p["accuracy"] for p in curve],
                marker="o", ms=3, lw=2, label=SET_LABELS[set_name].replace("\n", " "))
    rej = entry.get("rejection", {})
    for set_name, info in rej.items():
        ax.scatter([info["coverage"]], [info["accuracy_on_accepted"]], marker="*", s=180, zorder=5,
                   color="#d81b60")
    ax.set_xlabel("coverage (fraction not rejected)")
    ax.set_ylabel("exact match on accepted items")
    ax.set_title("Rejection: answers get more accurate as the model is allowed to say 'unknown'", fontsize=13)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "rejection.png", dpi=160)
    plt.close(fig)
    print(f"  {out_dir/'rejection.png'}")


def fig_speed_size(results: dict, out_dir: Path) -> None:
    colors = color_map(results)
    rows = []
    for name, entry in results["systems"].items():
        speed = entry.get("speed")
        if not speed:
            continue
        size_mb = entry.get("quantization", {}).get("quantized_bytes", 0) / 1e6
        rows.append((short_label(name), speed["ms_per_item"], size_mb, colors[name]))
    if not rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = [r[0] for r in rows]
    ax1.barh(labels, [r[1] for r in rows], color=[r[3] for r in rows])
    ax1.set_xlabel("inference time per item (ms, CPU)")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.25)
    sized = [r for r in rows if r[2] > 0]
    if sized:
        ax2.barh([r[0] for r in sized], [r[2] for r in sized], color=[r[3] for r in sized])
        ax2.set_xlabel("weight size (MB)")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.25)
    else:
        ax2.axis("off")
    fig.suptitle("CPU inference speed and weight size", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "speed_size.png", dpi=160)
    plt.close(fig)
    print(f"  {out_dir/'speed_size.png'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="benchmarks/results.json")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    font = setup_japanese_font()
    if font:
        print(f"Japanese-capable font: {font}")
    else:
        print("warning: no Japanese font found; Japanese glyphs in labels may render as boxes.")

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("generating figures ...")
    fig_known_vs_unseen(
        results, out_dir, "exact_match", "known_vs_unseen.png", "exact match (strict)",
        subtitle="Strict matching counts the pronunciation notation of existing G2P "
                 "(ユーゲン / ショーテン) as a mismatch. "
                 "See known_vs_unseen_phonetic.png for the notation-independent comparison.",
    )
    fig_known_vs_unseen(
        results, out_dir, "exact_match_phonetic", "known_vs_unseen_phonetic.png",
        "exact match (long vowels normalized)",
        subtitle="A lenient criterion that absorbs long-vowel spelling variation (トウ / トー). "
                 "A gap that survives this criterion is the robust result.",
    )
    fig_generalization_gap(results, out_dir)
    fig_reliability(results, out_dir)
    fig_rejection(results, out_dir)
    fig_speed_size(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
