"""Loading a trained bundle: model, tokenizer, calibrator and threshold.

Claims supported: **speed** and **rejection**. A thin layer whose job is to make
the released artifact usable in three lines. The calibrator and the threshold
live in the same directory as the model and are wired up automatically on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_MODEL_DIR = Path("artifacts/model")


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    """Resolve in order: --model, then $PHONEBOOK_MODEL, then artifacts/model."""
    if model_dir:
        return Path(model_dir)
    env = os.environ.get("PHONEBOOK_MODEL")
    if env:
        return Path(env)
    return DEFAULT_MODEL_DIR


def load_reader(
    model_dir: str | Path | None = None,
    *,
    beam_size: int = 8,
    threshold: Optional[float] = None,
    segment_kana: bool = True,
    use_calibrator: bool = True,
    use_en2kana: bool = True,
):
    """Assemble and return a PhonebookReader.

    Raises:
        FileNotFoundError: when no trained model is found; the message explains
            how to train one.
    """
    from .calibrate import PlattCalibrator
    from .decode import PhonebookReader
    from .en2kana import EnglishToKatakana
    from .model import CharSeq2Seq

    path = resolve_model_dir(model_dir)
    if not (path / "model.pt").exists():
        raise FileNotFoundError(
            f"No trained model found at {path}.\n"
            "Train one, for example:\n"
            "  python scripts/make_synthetic.py --out data/raw/synthetic.csv\n"
            "  python scripts/build_dataset.py --csv data/raw/synthetic.csv --out data/processed\n"
            "  python scripts/train.py --data data/processed --out artifacts/model --epochs 8\n"
            "or point --model at an existing one."
        )
    model, tokenizer = CharSeq2Seq.load(path)
    calibrator = None
    cal_path = path / "calibrator.json"
    if use_calibrator and cal_path.exists():
        calibrator = PlattCalibrator.load(cal_path)
    if threshold is None:
        meta_path = path / "threshold.json"
        if meta_path.exists():
            threshold = json.loads(meta_path.read_text(encoding="utf-8")).get("threshold")
    return PhonebookReader(
        model,
        tokenizer,
        en2kana=EnglishToKatakana() if use_en2kana else None,
        calibrator=calibrator,
        beam_size=beam_size,
        threshold=threshold,
        segment_kana=segment_kana,
    )
