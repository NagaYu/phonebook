"""A minimal GGUF v3 writer and reader.

Claim supported: **distributability**. The quantized weights are placed in
GGUF, a container that many tools can read. The header, metadata key-values,
tensor info and alignment all follow the specification, so generic GGUF tooling
can list the metadata and tensors.

Caveat (also stated on the model card): Phonebook is a custom architecture that
llama.cpp does not know, so these GGUF files **cannot be run by llama.cpp**.
GGUF is used here as a distribution and inspection format; inference happens on
the PyTorch / ONNX / MLX side.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GGUF_MAGIC = 0x46554747  # "GGUF" (little endian)
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# gguf_metadata_value_type
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12


def _w_string(buf: bytearray, s: str) -> None:
    raw = s.encode("utf-8")
    buf += struct.pack("<Q", len(raw))
    buf += raw


def _w_value(buf: bytearray, value: Any) -> None:
    if isinstance(value, bool):
        buf += struct.pack("<I", T_BOOL) + struct.pack("<B", int(value))
    elif isinstance(value, int):
        buf += struct.pack("<I", T_INT64) + struct.pack("<q", value)
    elif isinstance(value, float):
        buf += struct.pack("<I", T_FLOAT32) + struct.pack("<f", value)
    elif isinstance(value, str):
        buf += struct.pack("<I", T_STRING)
        _w_string(buf, value)
    elif isinstance(value, (list, tuple)):
        buf += struct.pack("<I", T_ARRAY)
        if not value:
            buf += struct.pack("<I", T_STRING) + struct.pack("<Q", 0)
            return
        first = value[0]
        if isinstance(first, str):
            buf += struct.pack("<I", T_STRING) + struct.pack("<Q", len(value))
            for v in value:
                _w_string(buf, v)
        elif isinstance(first, float):
            buf += struct.pack("<I", T_FLOAT32) + struct.pack("<Q", len(value))
            for v in value:
                buf += struct.pack("<f", v)
        else:
            buf += struct.pack("<I", T_INT64) + struct.pack("<Q", len(value))
            for v in value:
                buf += struct.pack("<q", int(v))
    else:
        raise TypeError(f"type cannot be written to GGUF: {type(value)}")


@dataclass
class GGUFTensor:
    name: str
    shape: tuple[int, ...]  # numpy order (row-major); reversed to ggml order on write
    ggml_type: int
    data: bytes


def write_gguf(
    path: str | Path,
    tensors: Iterable[GGUFTensor],
    metadata: dict[str, Any] | None = None,
    alignment: int = DEFAULT_ALIGNMENT,
) -> Path:
    """Write a GGUF file."""
    tensors = list(tensors)
    meta = dict(metadata or {})
    meta.setdefault("general.architecture", "phonebook-charseq2seq")
    meta.setdefault("general.name", "phonebook")
    meta["general.alignment"] = int(alignment)

    header = bytearray()
    header += struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))
    header += struct.pack("<Q", len(meta))
    for key, value in meta.items():
        _w_string(header, key)
        _w_value(header, value)

    info = bytearray()
    offset = 0
    for t in tensors:
        _w_string(info, t.name)
        dims = tuple(reversed(t.shape)) or (1,)
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", int(d))
        info += struct.pack("<I", int(t.ggml_type))
        info += struct.pack("<Q", offset)
        size = len(t.data)
        offset += size + ((-size) % alignment)

    body = bytearray(header) + info
    pad = (-len(body)) % alignment
    body += b"\x00" * pad
    for t in tensors:
        body += t.data
        body += b"\x00" * ((-len(t.data)) % alignment)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return path


# --- Reading (for inspection and round-trip tests) -------------------------
class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return out

    def string(self) -> str:
        (n,) = self.take("<Q")
        raw = self.data[self.pos : self.pos + n]
        self.pos += n
        return raw.decode("utf-8")

    def value(self):
        (vtype,) = self.take("<I")
        return self._value_of(vtype)

    def _value_of(self, vtype: int):
        if vtype == T_STRING:
            return self.string()
        if vtype == T_BOOL:
            return bool(self.take("<B")[0])
        if vtype == T_INT64:
            return self.take("<q")[0]
        if vtype == T_UINT64:
            return self.take("<Q")[0]
        if vtype == T_INT32:
            return self.take("<i")[0]
        if vtype == T_UINT32:
            return self.take("<I")[0]
        if vtype == T_FLOAT32:
            return self.take("<f")[0]
        if vtype == T_ARRAY:
            (etype,) = self.take("<I")
            (count,) = self.take("<Q")
            return [self._value_of(etype) for _ in range(count)]
        raise ValueError(f"unsupported GGUF value type: {vtype}")


def read_gguf(path: str | Path) -> dict:
    """Read a GGUF file; returns metadata and tensor info including raw bytes."""
    data = Path(path).read_bytes()
    r = _Reader(data)
    magic, version, n_tensors, n_kv = r.take("<IIQQ")
    if magic != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    meta: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = r.value()
    alignment = int(meta.get("general.alignment", DEFAULT_ALIGNMENT))

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        (ndims,) = r.take("<I")
        dims = [r.take("<Q")[0] for _ in range(ndims)]
        (ggml_type,) = r.take("<I")
        (offset,) = r.take("<Q")
        infos.append(
            {
                "name": name,
                "shape": tuple(reversed(dims)),
                "ggml_type": ggml_type,
                "offset": offset,
            }
        )
    data_start = r.pos + ((-r.pos) % alignment)
    for i, info in enumerate(infos):
        end = (
            data_start + infos[i + 1]["offset"] if i + 1 < len(infos) else len(data)
        )
        info["data"] = data[data_start + info["offset"] : end]
    return {"version": version, "metadata": meta, "tensors": infos}
