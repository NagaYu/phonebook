"""GGML-compatible quantization (Q8_0 / Q4_K / Q6_K) and the Q4_K_M mixture.

Claims supported: **speed and memory**, and **that accuracy survives quantization**.

Evaluation condition (D) is "after Q4_K_M quantization", so what is implemented
here is llama.cpp's **storage format itself**:

  Q8_0 : 32-element blocks, f16 scale + int8 values        (8.5 bits/weight)
  Q4_K : 256-element super-blocks = 8 x 32. Each sub-block's scale and minimum
         are quantized to 6 bits and reconstructed from f16 d / dmin.
         (2 + 2 + 12 + 128) bytes = 4.5 bits/weight
  Q6_K : 256-element super-blocks with an int8 scale every 16 elements.
         (128 + 64 + 16 + 2) bytes = 6.5625 bits/weight

Two differences worth stating plainly:

1. Scale search: llama.cpp minimizes error with an iterative search
   (make_qkx2_quants); this implementation uses the direct min/max solution.
   The **storage format is identical**, but the quantized values from the same
   weights are not bit-identical to llama.cpp's (error is slightly larger).
2. Blocking: llama.cpp blocks along rows (the last dimension). Phonebook has
   shapes whose row length is not a multiple of 256 (d_model = 384), so this
   implementation flattens the tensor in row-major order, cuts it into
   256-element blocks and zero-pads the tail.

Consequently the GGUF files produced here are valid GGUF containers holding
genuine Q4_K/Q6_K/Q8_0 blocks, but llama.cpp cannot *run* Phonebook's custom
architecture. Inference lives on the PyTorch / ONNX / MLX side. The model card
states this too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

QK8_0 = 32
QK_K = 256

#: GGUF tensor type ids (ggml_type)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q6_K: "Q6_K",
}

BYTES_PER_BLOCK = {
    GGML_TYPE_Q8_0: 2 + QK8_0,          # 34
    GGML_TYPE_Q4_K: 2 + 2 + 12 + 128,   # 144
    GGML_TYPE_Q6_K: 128 + 64 + 16 + 2,  # 210
}
BLOCK_ELEMS = {GGML_TYPE_Q8_0: QK8_0, GGML_TYPE_Q4_K: QK_K, GGML_TYPE_Q6_K: QK_K}


def _pad_to(x: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    n = x.size
    rem = (-n) % multiple
    if rem:
        x = np.concatenate([x, np.zeros(rem, dtype=x.dtype)])
    return x, n


# --- Q8_0 -----------------------------------------------------------------
def quantize_q8_0(x: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Symmetric int8 quantization in 32-element blocks. Returns (bytes, dequantized)."""
    flat, n = _pad_to(np.asarray(x, dtype=np.float32).ravel(), QK8_0)
    blocks = flat.reshape(-1, QK8_0)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    d[d == 0] = 1e-30
    q = np.clip(np.rint(blocks / d[:, None]), -128, 127).astype(np.int8)
    # The scale actually stored is f16, so the dequantized values are computed
    # from the f16-rounded scale to match exactly what a reader would see.
    d16 = d.astype(np.float16)
    deq = (q.astype(np.float32) * d16.astype(np.float32)[:, None]).ravel()[:n]

    out = bytearray()
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes()
        out += q[i].tobytes()
    return bytes(out), deq.reshape(np.asarray(x).shape)


# --- Q4_K -----------------------------------------------------------------
def _pack_scales_k4(sc: np.ndarray, mn: np.ndarray) -> np.ndarray:
    """Pack eight 6-bit scales and eight 6-bit minima into 12 bytes.

    The layout matches exactly what llama.cpp's get_scale_min_k4() reads back.
    """
    q = np.zeros(12, dtype=np.uint8)
    for j in range(4):
        q[j] = sc[j] & 63
        q[j + 4] = mn[j] & 63
    for j in range(4, 8):
        q[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4)
        q[j - 4] |= (sc[j] >> 4) << 6
        q[j] |= (mn[j] >> 4) << 6
    return q


def _unpack_scales_k4(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = np.zeros(8, dtype=np.int32)
    mn = np.zeros(8, dtype=np.int32)
    for j in range(8):
        if j < 4:
            sc[j] = q[j] & 63
            mn[j] = q[j + 4] & 63
        else:
            sc[j] = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
            mn[j] = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return sc, mn


def quantize_q4_k(x: np.ndarray) -> tuple[bytes, np.ndarray]:
    """4-bit k-quant over 256-element super-blocks."""
    flat, n = _pad_to(np.asarray(x, dtype=np.float32).ravel(), QK_K)
    supers = flat.reshape(-1, 8, 32)
    out = bytearray()
    deq = np.empty_like(supers)

    for s in range(supers.shape[0]):
        blk = supers[s]
        mins = np.minimum(blk.min(axis=1), 0.0)      # y = scale*q + min
        maxs = blk.max(axis=1)
        scales = (maxs - mins) / 15.0
        scales[scales <= 0] = 1e-30
        b = -mins                                    # y = scale*q - b, b >= 0

        d_super = float(np.float16(float(scales.max()) / 63.0)) or 1e-30
        dmin_super = float(np.float16(float(b.max()) / 63.0))
        if dmin_super <= 0:
            dmin_super = float(np.float16(1e-30)) or 1e-30
        sc_q = np.clip(np.rint(scales / d_super), 0, 63).astype(np.int32)
        mn_q = np.clip(np.rint(b / dmin_super), 0, 63).astype(np.int32)

        eff_scale = d_super * sc_q
        eff_min = dmin_super * mn_q
        eff_scale_safe = np.where(eff_scale == 0, 1e-30, eff_scale)
        q = np.clip(np.rint((blk + eff_min[:, None]) / eff_scale_safe[:, None]), 0, 15).astype(np.uint8)
        deq[s] = eff_scale[:, None] * q - eff_min[:, None]

        out += np.float16(d_super).tobytes()
        out += np.float16(dmin_super).tobytes()
        out += _pack_scales_k4(sc_q, mn_q).tobytes()
        # qs: each 32-byte chunk packs sub-block 2k in the low nibbles and
        # sub-block 2k+1 in the high nibbles.
        qs = np.zeros(128, dtype=np.uint8)
        for k in range(4):
            lo = q[2 * k]
            hi = q[2 * k + 1]
            qs[k * 32 : (k + 1) * 32] = lo | (hi << 4)
        out += qs.tobytes()

    return bytes(out), deq.reshape(-1)[:n].reshape(np.asarray(x).shape)


def dequantize_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q4_K bytes to float32 (reference implementation for round-trip tests)."""
    nblocks = len(data) // BYTES_PER_BLOCK[GGML_TYPE_Q4_K]
    out = np.empty(nblocks * QK_K, dtype=np.float32)
    pos = 0
    for s in range(nblocks):
        d = np.frombuffer(data, dtype=np.float16, count=1, offset=pos)[0].astype(np.float32)
        dmin = np.frombuffer(data, dtype=np.float16, count=1, offset=pos + 2)[0].astype(np.float32)
        scales = np.frombuffer(data, dtype=np.uint8, count=12, offset=pos + 4)
        qs = np.frombuffer(data, dtype=np.uint8, count=128, offset=pos + 16)
        sc, mn = _unpack_scales_k4(scales)
        for k in range(4):
            chunk = qs[k * 32 : (k + 1) * 32]
            lo = (chunk & 0xF).astype(np.float32)
            hi = (chunk >> 4).astype(np.float32)
            j0, j1 = 2 * k, 2 * k + 1
            base = s * QK_K + j0 * 32
            out[base : base + 32] = d * sc[j0] * lo - dmin * mn[j0]
            out[base + 32 : base + 64] = d * sc[j1] * hi - dmin * mn[j1]
        pos += BYTES_PER_BLOCK[GGML_TYPE_Q4_K]
    return out[:n_elements]


# --- Q6_K -----------------------------------------------------------------
def quantize_q6_k(x: np.ndarray) -> tuple[bytes, np.ndarray]:
    """6-bit k-quant over 256-element super-blocks (int8 scale every 16 elements)."""
    flat, n = _pad_to(np.asarray(x, dtype=np.float32).ravel(), QK_K)
    supers = flat.reshape(-1, 16, 16)
    out = bytearray()
    deq = np.empty_like(supers)

    for s in range(supers.shape[0]):
        blk = supers[s]
        amax = np.abs(blk).max(axis=1)
        scales = amax / 32.0
        scales[scales <= 0] = 1e-30
        d_super = float(np.float16(float(np.abs(scales).max()) / 127.0))
        if d_super <= 0:
            d_super = float(np.float16(1e-30)) or 1e-30
        sc_q = np.clip(np.rint(scales / d_super), -127, 127).astype(np.int8)
        eff = (d_super * sc_q.astype(np.float32))
        eff_safe = np.where(eff == 0, 1e-30, eff)
        q = np.clip(np.rint(blk / eff_safe[:, None]), -32, 31).astype(np.int32)
        deq[s] = eff[:, None] * q

        qv = (q + 32).astype(np.uint8).reshape(QK_K)  # 0..63
        ql = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(64, dtype=np.uint8)
        for half in range(2):  # process QK_K in halves of 128
            base = half * 128
            ql_off = half * 64
            qh_off = half * 32
            for l in range(32):
                v1 = qv[base + l]
                v2 = qv[base + l + 32]
                v3 = qv[base + l + 64]
                v4 = qv[base + l + 96]
                ql[ql_off + l] = (v1 & 0xF) | ((v3 & 0xF) << 4)
                ql[ql_off + l + 32] = (v2 & 0xF) | ((v4 & 0xF) << 4)
                qh[qh_off + l] = (
                    (v1 >> 4) | ((v2 >> 4) << 2) | ((v3 >> 4) << 4) | ((v4 >> 4) << 6)
                )
        out += ql.tobytes()
        out += qh.tobytes()
        out += sc_q.tobytes()
        out += np.float16(d_super).tobytes()

    return bytes(out), deq.reshape(-1)[:n].reshape(np.asarray(x).shape)


def dequantize_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    nblocks = len(data) // BYTES_PER_BLOCK[GGML_TYPE_Q6_K]
    out = np.empty(nblocks * QK_K, dtype=np.float32)
    pos = 0
    for s in range(nblocks):
        ql = np.frombuffer(data, dtype=np.uint8, count=128, offset=pos)
        qh = np.frombuffer(data, dtype=np.uint8, count=64, offset=pos + 128)
        sc = np.frombuffer(data, dtype=np.int8, count=16, offset=pos + 192)
        d = np.frombuffer(data, dtype=np.float16, count=1, offset=pos + 208)[0].astype(np.float32)
        for half in range(2):
            base = s * QK_K + half * 128
            ql_off = half * 64
            qh_off = half * 32
            for l in range(32):
                is_ = l // 16
                h = qh[qh_off + l]
                q1 = int((ql[ql_off + l] & 0xF) | (((h >> 0) & 3) << 4)) - 32
                q2 = int((ql[ql_off + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32
                q3 = int((ql[ql_off + l] >> 4) | (((h >> 4) & 3) << 4)) - 32
                q4 = int((ql[ql_off + l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32
                sc_off = half * 8
                out[base + l] = d * sc[sc_off + is_ + 0] * q1
                out[base + l + 32] = d * sc[sc_off + is_ + 2] * q2
                out[base + l + 64] = d * sc[sc_off + is_ + 4] * q3
                out[base + l + 96] = d * sc[sc_off + is_ + 6] * q4
        pos += BYTES_PER_BLOCK[GGML_TYPE_Q6_K]
    return out[:n_elements]


QUANTIZERS = {
    GGML_TYPE_Q8_0: quantize_q8_0,
    GGML_TYPE_Q4_K: quantize_q4_k,
    GGML_TYPE_Q6_K: quantize_q6_k,
}


# --- Mixture policy --------------------------------------------------------
@dataclass
class QuantPlan:
    """Assignment from tensor name to ggml type."""

    assignment: dict[str, int]
    scheme: str

    def type_of(self, name: str) -> int:
        return self.assignment.get(name, GGML_TYPE_F32)


def plan_quantization(state_dict, scheme: str = "q4_k_m") -> QuantPlan:
    """Assign ggml types following llama.cpp's Q4_K_M / Q8_0 recipes.

    The Q4_K_M policy here is a simplified reproduction of llama.cpp's mixture:
      - 1-D tensors (LayerNorm, bias) stay F32: they are error-sensitive and
        contribute almost nothing to file size.
      - Embeddings and the output projection use Q6_K, since they feed the
        vocabulary distribution directly.
      - linear2 and the cross-attention output projection in the first 1/8 of
        the layers and in the last layer use Q6_K.
      - Every other 2-D tensor uses Q4_K.
    """
    assignment: dict[str, int] = {}
    names = list(state_dict.keys())
    layer_ids = sorted(
        {
            int(n.split(".")[1])
            for n in names
            if n.startswith(("encoder.", "decoder.")) and n.split(".")[1].isdigit()
        }
    )
    n_layers = max(layer_ids) + 1 if layer_ids else 0
    high_precision_layers = set(range(max(1, n_layers // 8))) | ({n_layers - 1} if n_layers else set())

    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        if scheme == "q8_0":
            assignment[name] = GGML_TYPE_Q8_0 if arr.ndim >= 2 else GGML_TYPE_F32
            continue
        if arr.ndim < 2:
            assignment[name] = GGML_TYPE_F32
            continue
        if "embed" in name or "out_proj.weight" in name or name.endswith("pos.weight"):
            assignment[name] = GGML_TYPE_Q6_K
            continue
        parts = name.split(".")
        layer = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
        if layer in high_precision_layers and ("linear2" in name or "cross_attn.out_proj" in name):
            assignment[name] = GGML_TYPE_Q6_K
            continue
        assignment[name] = GGML_TYPE_Q4_K
    return QuantPlan(assignment=assignment, scheme=scheme)


@dataclass
class QuantizedTensor:
    name: str
    ggml_type: int
    shape: tuple[int, ...]
    data: bytes
    dequantized: np.ndarray

    @property
    def n_bytes(self) -> int:
        return len(self.data)


def quantize_state_dict(state_dict, scheme: str = "q4_k_m") -> tuple[list[QuantizedTensor], dict]:
    """Quantize an entire state_dict; returns (tensors, statistics).

    Writing ``dequantized`` back into the original state_dict lets the
    quantized model be evaluated in plain PyTorch, which is how condition (D)
    is measured.
    """
    plan = plan_quantization(state_dict, scheme)
    tensors: list[QuantizedTensor] = []
    orig_bytes = 0
    quant_bytes = 0
    by_type: dict[str, int] = {}

    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        ggml_type = plan.type_of(name)
        orig_bytes += arr.size * 4
        if ggml_type == GGML_TYPE_F32:
            data = arr.tobytes()
            deq = arr
        else:
            data, deq = QUANTIZERS[ggml_type](arr)
        quant_bytes += len(data)
        by_type[TYPE_NAMES[ggml_type]] = by_type.get(TYPE_NAMES[ggml_type], 0) + len(data)
        tensors.append(
            QuantizedTensor(
                name=name, ggml_type=ggml_type, shape=tuple(arr.shape), data=data, dequantized=deq
            )
        )

    stats = {
        "scheme": scheme,
        "float32_bytes": orig_bytes,
        "quantized_bytes": quant_bytes,
        "compression_ratio": round(orig_bytes / max(quant_bytes, 1), 3),
        "bytes_by_type": by_type,
        "n_tensors": len(tensors),
    }
    return tensors, stats


def apply_dequantized(state_dict, tensors: Iterable[QuantizedTensor]):
    """Write quantize-then-dequantize weights back into a state_dict (condition D)."""
    import torch

    new = {}
    lookup = {t.name: t for t in tensors}
    for name, tensor in state_dict.items():
        if name in lookup:
            new[name] = torch.from_numpy(np.ascontiguousarray(lookup[name].dequantized)).to(
                tensor.dtype
            )
        else:
            new[name] = tensor
    return new
