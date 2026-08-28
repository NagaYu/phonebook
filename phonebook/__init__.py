"""Phonebook: Japanese corporate names to katakana readings.

A small Japanese G2P model that is built for unseen entities, reports
calibrated confidence, and answers "unknown" when it does not know. See
README.md and the per-module docstrings for details.

Source: created by processing data from the National Tax Agency Corporate
Number Publication Site (https://www.houjin-bangou.nta.go.jp/).
"""

__version__ = "0.1.0"

from .structure import StructuralSplitter, StructuredName, LEGAL_FORMS  # noqa: F401
from .tokenizer import CharTokenizer  # noqa: F401
from .model import CharSeq2Seq, ModelConfig  # noqa: F401
from .decode import PhonebookReader, ConstrainedBeamSearch, ReadingResult  # noqa: F401
from .calibrate import PlattCalibrator, RejectionPolicy, TemperatureScaler  # noqa: F401
from .en2kana import EnglishToKatakana  # noqa: F401

__all__ = [
    "StructuralSplitter",
    "StructuredName",
    "LEGAL_FORMS",
    "CharTokenizer",
    "CharSeq2Seq",
    "ModelConfig",
    "PhonebookReader",
    "ConstrainedBeamSearch",
    "ReadingResult",
    "PlattCalibrator",
    "RejectionPolicy",
    "TemperatureScaler",
    "EnglishToKatakana",
    "load_reader",
]


def load_reader(model_dir=None, **kwargs):
    """Build a PhonebookReader with default settings (the three-line entry point)."""
    from .runtime import load_reader as _load

    return _load(model_dir, **kwargs)
