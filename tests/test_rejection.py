"""Claims: **the rejection threshold behaves as intended** and **calibration
turns a score into a probability.**

Rejection trades coverage (how many answers are returned) against precision (how
many of them are right). These tests check that the trade is monotone, that the
threshold really acts as a boundary, and that calibration corrects overconfidence.
"""

from __future__ import annotations

import math
import random

import pytest

from phonebook.calibrate import (
    PlattCalibrator,
    RejectionPolicy,
    TemperatureScaler,
    brier_score,
    extract_features,
    reliability_diagram,
    risk_coverage_curve,
)
from phonebook.decode import Candidate


class _StubCalibrator:
    """A calibrator that returns a fixed confidence, to isolate threshold behaviour."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_one(self, candidates) -> float:
        return self.value


def test_threshold_marks_low_confidence_as_unknown(reader):
    """At confidence 0.30 and threshold 0.50 the answer is rejected as 'unknown'."""
    reader.calibrator = _StubCalibrator(0.30)
    reader.threshold = 0.50
    results = reader.read_batch(["株式会社日本電気", "緑川食品株式会社"], nbest=3)
    for r in results:
        assert r.source == "model"
        assert r.rejected
        assert r.reading is None
        assert r.display == "unknown"
        assert r.candidates, "candidates are kept even when rejected, for reference"


def test_threshold_boundary_is_inclusive(reader):
    """Confidence exactly equal to the threshold is accepted; only conf < threshold rejects."""
    reader.calibrator = _StubCalibrator(0.50)
    reader.threshold = 0.50
    assert not reader.read("株式会社日本電気", nbest=1).rejected
    reader.threshold = 0.5000001
    assert reader.read("株式会社日本電気", nbest=1).rejected


def test_confidence_below_threshold_is_rejected_but_above_is_kept(reader):
    reader.calibrator = _StubCalibrator(0.62)
    reader.threshold = 0.8
    assert reader.read("緑川食品株式会社", nbest=1).rejected
    reader.threshold = 0.6
    assert not reader.read("緑川食品株式会社", nbest=1).rejected


def test_threshold_zero_accepts_everything(reader):
    reader.threshold = 0.0
    results = reader.read_batch(["株式会社日本電気", "緑川食品株式会社"], nbest=3)
    assert all(not r.rejected for r in results)
    assert all(r.reading is not None for r in results)


def test_copy_path_is_never_rejected(reader):
    """The deterministic copy path has confidence 1.0 and is never rejected."""
    reader.threshold = 0.999
    result = reader.read("株式会社アルファ", nbest=1)
    assert result.source == "copy"
    assert not result.rejected


def test_rejection_policy_accept_boundary():
    policy = RejectionPolicy(threshold=0.7)
    assert policy.accept(0.7)
    assert policy.accept(0.71)
    assert not policy.accept(0.6999)


def test_higher_threshold_gives_higher_precision_lower_coverage():
    random.seed(0)
    conf = [random.random() for _ in range(2000)]
    correct = [random.random() < c for c in conf]  # a perfectly calibrated hypothetical
    curve = risk_coverage_curve(conf, correct, n_points=10)
    coverages = [p["coverage"] for p in curve]
    assert coverages == sorted(coverages)
    # Lower coverage means higher accuracy; not strictly monotone, so compare endpoints.
    assert curve[0]["accuracy"] > curve[-1]["accuracy"]


def test_fit_for_precision_reaches_target():
    random.seed(1)
    conf = [random.random() for _ in range(3000)]
    correct = [random.random() < c for c in conf]
    threshold, coverage, precision = RejectionPolicy.fit_for_precision(conf, correct, 0.9)
    assert precision >= 0.9 - 1e-9
    assert 0.0 < coverage <= 1.0
    accepted = [(c, y) for c, y in zip(conf, correct) if c >= threshold]
    achieved = sum(1 for _, y in accepted) / len(accepted)
    assert achieved >= 0.85, "precision after actually applying the threshold is far off target"


def test_calibration_reduces_ece_on_overconfident_scores():
    """Platt-calibrating an overconfident score reduces ECE."""
    random.seed(2)
    raw, correct, feats = [], [], []
    for _ in range(2000):
        true_p = random.random()
        # The raw score is systematically higher than the true accuracy.
        shown = min(0.999, true_p ** 0.4)
        raw.append(shown)
        correct.append(random.random() < true_p)
        feats.append([math.log(shown), 1.0, -0.5, 3.0, 8.0])
    _, ece_raw, _ = reliability_diagram(raw, correct)
    cal = PlattCalibrator().fit(feats, correct, steps=400)
    calibrated = cal.predict(feats)
    _, ece_cal, _ = reliability_diagram(calibrated, correct)
    assert ece_cal < ece_raw, f"calibrated ECE {ece_cal:.3f} did not improve on {ece_raw:.3f}"
    assert brier_score(calibrated, correct) <= brier_score(raw, correct) + 1e-6


def test_temperature_scaler_softens_overconfidence():
    scaler = TemperatureScaler().fit(
        [[-0.01, -6.0]] * 50 + [[-0.02, -5.0]] * 50,
        [0] * 50 + [1] * 50,
        steps=120,
    )
    probs = scaler.apply([-0.01, -6.0])
    assert abs(sum(probs) - 1.0) < 1e-6
    assert scaler.temperature > 0


def test_extract_features_shape_and_margin():
    cands = [Candidate("ニホン", 0.7, -1.2), Candidate("ニッポン", 0.3, -2.4)]
    feats = extract_features(cands)
    assert len(feats) == 5
    assert feats[1] > 0, "margin must be positive when rank 1 beats rank 2"
    single = extract_features([Candidate("ニホン", 1.0, -0.1)])
    assert single[1] > feats[1], "a single candidate should give a larger margin"


def test_unfitted_calibrator_falls_back_to_raw_probability():
    cal = PlattCalibrator()
    feats = extract_features([Candidate("ニホン", 0.42, -1.0)])
    assert abs(cal.predict([feats])[0] - 0.42) < 1e-6


def test_calibrator_roundtrip(tmp_path):
    cal = PlattCalibrator().fit([[0.1, 0.2, 0.3, 3.0, 8.0]] * 40, [True] * 20 + [False] * 20, steps=50)
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = PlattCalibrator.load(path)
    assert loaded.fitted
    assert abs(loaded.predict([[0.1, 0.2, 0.3, 3.0, 8.0]])[0] - cal.predict([[0.1, 0.2, 0.3, 3.0, 8.0]])[0]) < 1e-6
