"""Claim: **the quantization is real** -- the storage format matches GGML's
definition and the round-trip is exact.

The numbers reported for condition (D), "after Q4_K_M quantization", come from
weights produced by this implementation. If the format were wrong, those numbers
would mean nothing beyond "something was reduced to 4 bits".
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phonebook.gguf import GGUFTensor, read_gguf, write_gguf
from phonebook.quantize import (
    BYTES_PER_BLOCK,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    QK_K,
    dequantize_q4_k,
    dequantize_q6_k,
    plan_quantization,
    quantize_q4_k,
    quantize_q6_k,
    quantize_q8_0,
    quantize_state_dict,
    apply_dequantized,
)


@pytest.fixture
def weights():
    return np.random.default_rng(0).normal(0, 0.02, (64, 256)).astype(np.float32)


def test_block_sizes_match_ggml_definitions():
    assert BYTES_PER_BLOCK[GGML_TYPE_Q8_0] == 34          # f16 + 32 x int8
    assert BYTES_PER_BLOCK[GGML_TYPE_Q4_K] == 144         # 2 + 2 + 12 + 128
    assert BYTES_PER_BLOCK[GGML_TYPE_Q6_K] == 210         # 128 + 64 + 16 + 2


def test_bits_per_weight(weights):
    for fn, expected in ((quantize_q8_0, 8.5), (quantize_q4_k, 4.5), (quantize_q6_k, 6.5625)):
        data, _ = fn(weights)
        assert len(data) * 8 / weights.size == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "quant,dequant", [(quantize_q4_k, dequantize_q4_k), (quantize_q6_k, dequantize_q6_k)]
)
def test_roundtrip_is_bit_exact(weights, quant, dequant):
    """Reading the written bytes back reproduces the dequantized values exactly."""
    data, deq = quant(weights)
    back = dequant(data, weights.size).reshape(weights.shape)
    assert np.array_equal(back, deq)


def test_quantization_error_ordering(weights):
    """Accuracy orders as Q8_0 > Q6_K > Q4_K, consistent with the bit widths."""
    errs = {}
    for name, fn in (("q8", quantize_q8_0), ("q6", quantize_q6_k), ("q4", quantize_q4_k)):
        _, deq = fn(weights)
        errs[name] = float(np.abs(deq - weights).mean())
    assert errs["q8"] < errs["q6"] < errs["q4"]
    assert errs["q4"] < 0.02 * float(np.abs(weights).mean()) * 10  # relative error has not blown up


def test_padding_for_non_multiple_shapes():
    x = np.random.default_rng(1).normal(0, 0.1, (3, 100)).astype(np.float32)  # 300 elements
    data, deq = quantize_q4_k(x)
    assert deq.shape == x.shape
    assert len(data) == 2 * BYTES_PER_BLOCK[GGML_TYPE_Q4_K]  # rounded up to a multiple of 256


def test_q4_k_m_plan_assigns_expected_types():
    from phonebook.model import CharSeq2Seq, ModelConfig

    model = CharSeq2Seq(ModelConfig.preset("tiny", 64))
    plan = plan_quantization(model.state_dict(), "q4_k_m")
    types = {name: plan.type_of(name) for name in model.state_dict()}
    assert types["embed.weight"] == GGML_TYPE_Q6_K, "embeddings stay at higher precision"
    assert all(
        t == GGML_TYPE_F32
        for name, t in types.items()
        if model.state_dict()[name].ndim < 2
    ), "1-D tensors (LayerNorm/bias) stay F32"
    assert GGML_TYPE_Q4_K in types.values(), "most tensors are Q4_K"


def test_quantized_model_stays_usable():
    """Quantize-then-dequantize weights still yield a valid distribution and katakana output."""
    from phonebook.decode import PhonebookReader
    from phonebook.model import CharSeq2Seq, ModelConfig
    from phonebook.tokenizer import CharTokenizer
    from phonebook.kana import is_valid_reading

    tok = CharTokenizer.build(["日本電気商会", "ニホンデンキショウカイ"])
    model = CharSeq2Seq(ModelConfig.preset("tiny", len(tok))).eval()
    tensors, stats = quantize_state_dict(model.state_dict(), "q4_k_m")
    assert stats["compression_ratio"] > 3.0
    model.load_state_dict(apply_dequantized(model.state_dict(), tensors))
    reader = PhonebookReader(model, tok, beam_size=4, max_len=12)
    for res in reader.read_batch(["株式会社日本電気"], nbest=2):
        for cand in res.candidates:
            assert is_valid_reading(cand.reading)


def test_gguf_roundtrip(tmp_path, weights):
    data, deq = quantize_q4_k(weights)
    path = write_gguf(
        tmp_path / "m.gguf",
        [GGUFTensor("w", weights.shape, GGML_TYPE_Q4_K, data)],
        {"general.name": "t", "phonebook.d_model": 384, "tokenizer.ggml.tokens": ["ア", "イ"]},
    )
    parsed = read_gguf(path)
    assert parsed["version"] == 3
    assert parsed["metadata"]["phonebook.d_model"] == 384
    assert parsed["metadata"]["tokenizer.ggml.tokens"] == ["ア", "イ"]
    info = parsed["tensors"][0]
    assert info["shape"] == weights.shape, "GGUF writes ggml (reversed) order and the reader restores it"
    assert np.array_equal(dequantize_q4_k(info["data"], weights.size).reshape(weights.shape), deq)


def test_gguf_header_is_well_formed(tmp_path, weights):
    data, _ = quantize_q8_0(weights)
    path = write_gguf(tmp_path / "m.gguf", [GGUFTensor("w", weights.shape, GGML_TYPE_Q8_0, data)])
    raw = path.read_bytes()
    assert raw[:4] == b"GGUF"
    assert int.from_bytes(raw[4:8], "little") == 3
    assert int.from_bytes(raw[8:16], "little") == 1  # tensor count
