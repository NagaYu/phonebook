"""Calibration (temperature scaling / Platt) and the rejection mechanism.

Claims supported: **calibration quality** and **precision under rejection**.

For proper nouns the reading is sometimes undetermined in principle (the same
surface really does have several readings). Phonebook therefore aims not only
to be right, but to *know how likely it is to be right*:

  - when it reports confidence 0.9, roughly 90% of those answers are correct
    (calibration);
  - below a threshold it answers "unknown", and the answers it does return are
    accurate enough to act on (rejection).

Together these make the downstream task -- filling in missing furigana --
usable: the machine-produced reading becomes a way to prioritize human review.
That is what gives the derived dataset its public value.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch


# --- Metrics ---------------------------------------------------------------
@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_confidence: float
    accuracy: float


def reliability_diagram(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[list[ReliabilityBin], float, float]:
    """Per-bin (mean confidence, empirical accuracy) plus ECE and MCE.

    ECE = sum_b (n_b/N) |acc_b - conf_b|. Source data for figures/reliability.png.
    """
    bins: list[ReliabilityBin] = []
    n = len(confidences)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (c > lo or (b == 0 and c >= lo)) and c <= hi
        ]
        if not idx:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        bins.append(ReliabilityBin(lo, hi, len(idx), conf, acc))
        gap = abs(acc - conf)
        ece += (len(idx) / max(n, 1)) * gap
        mce = max(mce, gap)
    return bins, ece, mce


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return float("nan")
    return sum((c - float(y)) ** 2 for c, y in zip(confidences, correct)) / len(confidences)


# --- Temperature scaling ---------------------------------------------------
@dataclass
class TemperatureScaler:
    """A scalar temperature T applied to sequence log-probabilities.

    p_i is proportional to exp(logp_i / T), renormalized within the n-best list.
    T > 1 softens confidence, T < 1 sharpens it. Fitted by minimizing negative
    log-likelihood on the dev set.
    """

    temperature: float = 1.0

    def fit(
        self,
        logprob_lists: Sequence[Sequence[float]],
        gold_index: Sequence[int],
        *,
        steps: int = 300,
        lr: float = 0.05,
    ) -> "TemperatureScaler":
        """gold_index[i] is the position of the correct candidate, or -1 when
        the gold reading is not in the n-best list.

        For -1 we treat "outside the n-best" as one virtual class whose logit is
        the residual mass log(1 - sum p). This also helps the model of rejection.
        """
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([log_t], lr=lr)
        rows = [
            (torch.tensor(lp, dtype=torch.float32), gi)
            for lp, gi in zip(logprob_lists, gold_index)
            if len(lp) > 0
        ]
        if not rows:
            return self
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.zeros(())
            for lp, gi in rows:
                residual = torch.log1p(-torch.exp(lp).sum().clamp(max=1 - 1e-6)).reshape(1)
                scores = torch.cat([lp, residual]) / torch.exp(log_t)
                target = gi if gi >= 0 else len(lp)
                loss = loss - torch.log_softmax(scores, dim=0)[target]
            loss = loss / len(rows)
            loss.backward()
            opt.step()
        self.temperature = float(torch.exp(log_t).item())
        return self

    def apply(self, logprobs: Sequence[float]) -> list[float]:
        if not logprobs:
            return []
        scores = torch.tensor(logprobs, dtype=torch.float32) / self.temperature
        return torch.softmax(scores, dim=0).tolist()


# --- Platt scaling (the calibrator actually used for rejection) ------------
FEATURE_NAMES = ("log_p_top", "margin", "log_p_raw_per_char", "n_cands", "length")


def extract_features(candidates: Sequence, length_hint: int | None = None) -> list[float]:
    """Build calibration features from a candidate list.

    - log_p_top: log of the renormalized top-1 probability within the n-best
    - margin: log-probability gap between rank 1 and rank 2 (expected to be
      small on the ambiguous set)
    - log_p_raw_per_char: raw sequence log-probability per character, which
      removes the length effect
    - n_cands / length: correction terms
    """
    if not candidates:
        return [-10.0, 0.0, -10.0, 0.0, 0.0]
    top = candidates[0]
    p_top = max(getattr(top, "prob", 0.0), 1e-9)
    second = candidates[1].prob if len(candidates) > 1 else 1e-9
    length = length_hint if length_hint is not None else max(len(top.reading), 1)
    raw = getattr(top, "raw_logprob", 0.0)
    return [
        math.log(p_top),
        math.log(p_top) - math.log(max(second, 1e-9)),
        raw / max(length, 1),
        float(len(candidates)),
        float(length),
    ]


@dataclass
class PlattCalibrator:
    """Logistic regression from features to P(top-1 is correct).

    Claim supported: **calibration quality**. Using the raw sequence probability
    as a confidence is systematically overconfident because of renormalization
    within the beam. Fitting the mapping against the binary correct/incorrect
    label on dev makes the confidence readable as an actual probability.
    """

    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    bias: float = 0.0
    mean: list[float] = field(default_factory=lambda: [0.0] * len(FEATURE_NAMES))
    std: list[float] = field(default_factory=lambda: [1.0] * len(FEATURE_NAMES))
    fitted: bool = False

    def fit(
        self,
        features: Sequence[Sequence[float]],
        correct: Sequence[bool],
        *,
        steps: int = 800,
        lr: float = 0.05,
        l2: float = 1e-3,
    ) -> "PlattCalibrator":
        x = torch.tensor([list(f) for f in features], dtype=torch.float32)
        y = torch.tensor([float(c) for c in correct], dtype=torch.float32)
        if x.numel() == 0:
            return self
        mean = x.mean(0)
        std = x.std(0).clamp_min(1e-6)
        xn = (x - mean) / std
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logits = xn @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = loss + l2 * (w * w).sum()
            loss.backward()
            opt.step()
        self.weights = w.detach().tolist()
        self.bias = float(b.detach().item())
        self.mean = mean.tolist()
        self.std = std.tolist()
        self.fitted = True
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if not self.fitted:
            # Unfitted: fall back to the renormalized top-1 probability
            return [math.exp(f[0]) for f in features]
        x = torch.tensor([list(f) for f in features], dtype=torch.float32)
        xn = (x - torch.tensor(self.mean)) / torch.tensor(self.std)
        logits = xn @ torch.tensor(self.weights) + self.bias
        return torch.sigmoid(logits).tolist()

    def predict_one(self, candidates: Sequence) -> float:
        return self.predict([extract_features(candidates)])[0]

    # -- Persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "fitted": self.fitted,
            "feature_names": list(FEATURE_NAMES),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=d["weights"], bias=d["bias"], mean=d["mean"], std=d["std"], fitted=d["fitted"]
        )


# --- Rejection -------------------------------------------------------------
@dataclass
class RejectionPolicy:
    """Answer "unknown" when the confidence falls below a threshold.

    Claim supported: **precision under rejection**. Rejection is not a way to
    dodge errors and inflate accuracy; it is the mechanism that decides where
    human review should go in the downstream furigana-completion task. Coverage
    and precision-on-accepted are therefore always reported together.
    """

    threshold: float = 0.5

    def accept(self, confidence: float) -> bool:
        return confidence >= self.threshold

    @staticmethod
    def fit_for_precision(
        confidences: Sequence[float],
        correct: Sequence[bool],
        target_precision: float = 0.95,
        min_coverage: float = 0.0,
    ) -> tuple[float, float, float]:
        """Return the lowest threshold that reaches the target precision.

        Returns: (threshold, coverage, precision). If no threshold reaches the
        target, returns the best-precision operating point instead, in which
        case precision < target.
        """
        order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
        best = (1.0, 0.0, 0.0)
        n_correct = 0
        for rank, i in enumerate(order, start=1):
            n_correct += int(correct[i])
            precision = n_correct / rank
            coverage = rank / max(len(order), 1)
            if precision >= target_precision and coverage >= min_coverage:
                best = (confidences[i], coverage, precision)
        if best == (1.0, 0.0, 0.0) and order:
            # Target unreachable: return the best-precision point
            n_correct = 0
            best_prec = -1.0
            for rank, i in enumerate(order, start=1):
                n_correct += int(correct[i])
                precision = n_correct / rank
                if precision > best_prec and rank >= max(1, int(0.05 * len(order))):
                    best_prec = precision
                    best = (confidences[i], rank / len(order), precision)
        return best


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_points: int = 21
) -> list[dict]:
    """Coverage (fraction not rejected) against accuracy. Source data for
    figures/rejection.png."""
    order = sorted(range(len(confidences)), key=lambda i: -confidences[i])
    total = len(order)
    if total == 0:
        return []
    curve: list[dict] = []
    for p in range(1, n_points + 1):
        cut = max(1, round(total * p / n_points))
        idx = order[:cut]
        acc = sum(1 for i in idx if correct[i]) / cut
        curve.append(
            {
                "coverage": cut / total,
                "accuracy": acc,
                "risk": 1 - acc,
                "threshold": confidences[idx[-1]],
            }
        )
    return curve
