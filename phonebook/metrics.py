"""Evaluation metrics.

Claims supported: **all of them**. This module is where the claims become
numbers, and its only job is to compute the same metrics for every evaluation
set (known / unseen / hard / ambiguous).

The choice of metrics is deliberate:
  - Exact match: a furigana is only usable in practice if it is entirely
    correct, so this is the headline metric.
  - CER: exact match alone cannot distinguish a near miss from a total miss. If
    exact match drops on unseen entities but CER stays low, generalization is
    still happening.
  - n-best@3 coverage: on the ambiguous set, forcing a single answer is
    unfair in principle. This matches how the system is actually used --
    presenting candidates.
  - Calibration and rejection: whether the system knows when it is about to be
    wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional, Sequence

from .kana import phonetic_normalize


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insert/delete/substitute). The numerator of CER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. Falls back sensibly when the reference is empty."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


@dataclass
class MetricResult:
    n: int = 0
    exact_match: float = 0.0
    exact_match_phonetic: float = 0.0
    cer: float = 0.0
    nbest_coverage: dict[int, float] = field(default_factory=dict)
    n_rejected: int = 0
    rejection_rate: float = 0.0
    accuracy_on_accepted: float = 0.0
    coverage: float = 0.0
    ece: Optional[float] = None
    brier: Optional[float] = None
    mean_confidence: Optional[float] = None
    latency_ms_mean: Optional[float] = None
    latency_ms_p95: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["nbest_coverage"] = {str(k): v for k, v in self.nbest_coverage.items()}
        return d


def evaluate(
    references: Sequence[str],
    predictions: Sequence[Optional[str]],
    *,
    nbest: Sequence[Sequence[str]] | None = None,
    confidences: Sequence[float] | None = None,
    rejected: Sequence[bool] | None = None,
    accepted_readings: Sequence[Sequence[str]] | None = None,
    ks: Sequence[int] = (1, 3, 5),
    latencies: Sequence[float] | None = None,
    n_bins: int = 10,
) -> MetricResult:
    """Compute every metric for one evaluation set.

    Args:
        accepted_readings: For the ambiguous set. Pass the set of readings that
            genuinely exist for each surface to also measure the lenient
            criterion "hitting any attested reading counts".
    """
    from .calibrate import brier_score, reliability_diagram

    n = len(references)
    res = MetricResult(n=n)
    if n == 0:
        return res

    rejected = list(rejected) if rejected is not None else [False] * n
    exact = 0
    exact_phon = 0
    cer_sum = 0.0
    for ref, pred in zip(references, predictions):
        hyp = pred or ""
        exact += int(hyp == ref)
        exact_phon += int(phonetic_normalize(hyp) == phonetic_normalize(ref))
        cer_sum += cer(ref, hyp)
    res.exact_match = exact / n
    res.exact_match_phonetic = exact_phon / n
    res.cer = cer_sum / n

    if nbest is not None:
        for k in ks:
            hits = 0
            for i, ref in enumerate(references):
                cands = list(nbest[i])[:k]
                gold = set(accepted_readings[i]) if accepted_readings else {ref}
                hits += int(any(c in gold for c in cands))
            res.nbest_coverage[k] = hits / n

    res.n_rejected = sum(1 for r in rejected if r)
    res.rejection_rate = res.n_rejected / n
    res.coverage = 1.0 - res.rejection_rate
    accepted_idx = [i for i in range(n) if not rejected[i]]
    if accepted_idx:
        res.accuracy_on_accepted = sum(
            int((predictions[i] or "") == references[i]) for i in accepted_idx
        ) / len(accepted_idx)

    if confidences is not None:
        correct = [(predictions[i] or "") == references[i] for i in range(n)]
        _, ece, _ = reliability_diagram(confidences, correct, n_bins=n_bins)
        res.ece = ece
        res.brier = brier_score(confidences, correct)
        res.mean_confidence = sum(confidences) / n

    if latencies:
        ordered = sorted(latencies)
        res.latency_ms_mean = sum(ordered) / len(ordered)
        res.latency_ms_p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return res


# --- Speed and memory ------------------------------------------------------
@dataclass
class SpeedReport:
    """Speed and memory.

    Two memory numbers are reported, because conflating them misleads:
      - ``rss_mb``: resident memory right after the measurement, close to what
        this condition actually uses.
      - ``peak_rss_mb``: the process-wide cumulative peak. When several
        conditions run in one process it includes earlier ones, so it cannot be
        used to compare conditions.
    """

    n: int
    total_seconds: float
    items_per_second: float
    ms_per_item: float
    rss_mb: float
    rss_delta_mb: float
    peak_rss_mb: float
    batch_size: int

    def to_dict(self) -> dict:
        return asdict(self)


def _current_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def measure_speed(fn: Callable[[Sequence[str]], object], items: Sequence[str], batch_size: int = 32) -> SpeedReport:
    """Measure CPU inference throughput and memory.

    Claim supported: **speed**. The same instrument is used before and after
    quantization and for the existing G2P systems. One batch is run as warm-up
    first, since the first call mixes in dictionary loading and lazy
    initialization and would distort the comparison.
    """
    import resource
    import sys as _sys

    if items:
        fn(items[: min(batch_size, len(items))])  # warm-up
    rss_before = _current_rss_mb()

    start = time.perf_counter()
    for i in range(0, len(items), batch_size):
        fn(items[i : i + batch_size])
    elapsed = time.perf_counter() - start

    rss_after = _current_rss_mb()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux
    peak_mb = rss / (1024 * 1024) if _sys.platform == "darwin" else rss / 1024
    n = max(len(items), 1)
    return SpeedReport(
        n=len(items),
        total_seconds=elapsed,
        items_per_second=n / elapsed if elapsed > 0 else float("inf"),
        ms_per_item=elapsed * 1000 / n,
        rss_mb=rss_after if rss_after is not None else float("nan"),
        rss_delta_mb=(rss_after - rss_before) if (rss_after is not None and rss_before is not None) else float("nan"),
        peak_rss_mb=peak_mb,
        batch_size=batch_size,
    )


def format_table(rows: Iterable[dict], columns: Sequence[tuple[str, str]]) -> str:
    """Render a Markdown table for the README / benchmark results."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, sep]
    for row in rows:
        cells = []
        for key, _ in columns:
            v = row.get(key, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
