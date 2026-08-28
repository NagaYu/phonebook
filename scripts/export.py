#!/usr/bin/env python3
"""Export to GGUF (Q4_K_M / Q8_0), ONNX and MLX, and push to Hugging Face.

Claims supported: **speed and memory**, and **distributability**.

What each format is and is not, stated plainly so it cannot be misread:

  GGUF : holds the quantized weights in llama.cpp's **storage format**. Because
         Phonebook is a custom architecture that llama.cpp does not know, these
         files **cannot be run by llama.cpp**. GGUF is used as a container for
         distributing and inspecting weights. The quantization itself is real
         (genuine Q4_K/Q6_K/Q8_0 block structure), and the accuracy and size
         reported for condition (D) are measured on exactly these weights.
  ONNX : the runtime format for CPU deployment; runs as-is under onnxruntime.
  MLX  : for Apple Silicon. Weights (npz) plus config, with a loading example.

Usage:
    python scripts/export.py --model artifacts/model --out artifacts/export
    python scripts/export.py --model artifacts/model --out artifacts/export --push-to-hub <user>/phonebook
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.gguf import GGUFTensor, read_gguf, write_gguf  # noqa: E402
from phonebook.model import CharSeq2Seq  # noqa: E402
from phonebook.quantize import TYPE_NAMES, quantize_state_dict  # noqa: E402


def export_gguf(model: CharSeq2Seq, tokenizer, out_dir: Path, scheme: str) -> dict:
    tensors, stats = quantize_state_dict(model.state_dict(), scheme=scheme)
    gguf_tensors = [
        GGUFTensor(name=t.name, shape=t.shape, ggml_type=t.ggml_type, data=t.data) for t in tensors
    ]
    metadata = {
        "general.architecture": "phonebook-charseq2seq",
        "general.name": "phonebook",
        "general.description": (
            "Japanese corporate name to katakana reading. Cannot be executed by "
            "llama.cpp: this is a custom architecture."
        ),
        "general.license": "apache-2.0",
        "general.quantization_scheme": scheme,
        "phonebook.d_model": model.cfg.d_model,
        "phonebook.n_head": model.cfg.nhead,
        "phonebook.encoder_layers": model.cfg.num_encoder_layers,
        "phonebook.decoder_layers": model.cfg.num_decoder_layers,
        "phonebook.feed_forward_length": model.cfg.dim_feedforward,
        "phonebook.use_copy": bool(model.cfg.use_copy),
        "phonebook.max_src_len": model.cfg.max_src_len,
        "phonebook.max_tgt_len": model.cfg.max_tgt_len,
        "tokenizer.ggml.model": "char",
        "tokenizer.ggml.tokens": list(tokenizer.itos),
    }
    name = {"q4_k_m": "phonebook-Q4_K_M.gguf", "q8_0": "phonebook-Q8_0.gguf"}[scheme]
    path = write_gguf(out_dir / name, gguf_tensors, metadata)
    check = read_gguf(path)
    by_type: dict[str, int] = {}
    for info in check["tensors"]:
        label = TYPE_NAMES.get(info["ggml_type"], str(info["ggml_type"]))
        by_type[label] = by_type.get(label, 0) + 1
    stats.update(
        {
            "path": str(path),
            "file_bytes": path.stat().st_size,
            "tensor_type_counts": by_type,
            "readback_ok": len(check["tensors"]) == len(gguf_tensors),
        }
    )
    print(
        f"  GGUF {scheme}: {path.name} {path.stat().st_size/1e6:.1f} MB "
        f"(compression {stats['compression_ratio']}x, tensor types {by_type})"
    )
    return stats


def export_onnx(model: CharSeq2Seq, out_dir: Path) -> dict:
    """Export a single graph: (src, tgt_in) -> log probabilities.

    Beam search runs on the host, so one teacher-forcing forward pass is all the
    graph needs. Phonebook generates around 20 characters, which is fast enough
    on CPU even without a KV cache.
    """
    path = out_dir / "phonebook.onnx"
    model.eval()
    src = torch.randint(4, 40, (2, 8), dtype=torch.long)
    tgt = torch.randint(4, 40, (2, 6), dtype=torch.long)
    try:
        torch.onnx.export(
            model,
            (src, tgt),
            str(path),
            input_names=["src", "tgt_in"],
            output_names=["log_probs"],
            dynamic_axes={
                "src": {0: "batch", 1: "src_len"},
                "tgt_in": {0: "batch", 1: "tgt_len"},
                "log_probs": {0: "batch", 1: "tgt_len"},
            },
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  ONNX: export failed ({exc})")
        return {"ok": False, "error": str(exc)}

    info = {"ok": True, "path": str(path), "file_bytes": path.stat().st_size}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"src": src.numpy(), "tgt_in": tgt.numpy()})[0]
        ref = model(src, tgt).detach().numpy()
        info["max_abs_diff_vs_torch"] = float(np.abs(out - ref).max())
        print(
            f"  ONNX: {path.name} {path.stat().st_size/1e6:.1f} MB / "
            f"max abs diff vs torch {info['max_abs_diff_vs_torch']:.2e}"
        )
    except Exception as exc:  # pragma: no cover
        info["runtime_check"] = f"could not verify: {exc}"
        print(f"  ONNX: exported, runtime check skipped ({exc})")
    return info


MLX_README = """# Phonebook weights for MLX

`weights.npz` is the PyTorch state_dict saved as numpy arrays. `config.json`
holds the model configuration and `tokenizer.json` the character vocabulary.

```python
import json, mlx.core as mx, numpy as np
weights = {k: mx.array(v) for k, v in np.load("weights.npz").items()}
config = json.load(open("config.json"))
```

See phonebook/model.py for the reference implementation of the layer stack and
the copy mechanism.
"""


def export_mlx(model: CharSeq2Seq, tokenizer, out_dir: Path) -> dict:
    mlx_dir = out_dir / "mlx"
    mlx_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(mlx_dir / "weights.npz", **arrays)
    (mlx_dir / "config.json").write_text(
        json.dumps(model.cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer.save(mlx_dir / "tokenizer.json")
    (mlx_dir / "README.md").write_text(MLX_README, encoding="utf-8")
    size = sum(f.stat().st_size for f in mlx_dir.glob("*"))
    print(f"  MLX: {mlx_dir} ({size/1e6:.1f} MB)")
    return {"ok": True, "path": str(mlx_dir), "bytes": size}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="artifacts/model")
    p.add_argument("--out", default="artifacts/export")
    p.add_argument("--formats", default="gguf,onnx,mlx")
    p.add_argument("--push-to-hub", default=None, help="Hugging Face model repository id")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = CharSeq2Seq.load(model_dir)
    model.eval()
    print(f"model: {model_dir} ({model.num_parameters()/1e6:.1f}M params)")

    formats = {f.strip() for f in args.formats.split(",") if f.strip()}
    manifest: dict = {"parameters": model.num_parameters(), "config": model.cfg.to_dict()}

    if "gguf" in formats:
        print("exporting GGUF ...")
        manifest["gguf"] = {
            scheme: export_gguf(model, tokenizer, out_dir, scheme) for scheme in ("q4_k_m", "q8_0")
        }
    if "onnx" in formats:
        print("exporting ONNX ...")
        manifest["onnx"] = export_onnx(model, out_dir)
    if "mlx" in formats:
        print("exporting MLX ...")
        manifest["mlx"] = export_mlx(model, tokenizer, out_dir)

    # Ship the PyTorch bundle too: the model is only usable together with the
    # calibrator and the threshold.
    for fname in ("model.pt", "config.json", "tokenizer.json", "calibrator.json", "threshold.json"):
        src = model_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nmanifest: {out_dir/'export_manifest.json'}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.private, exist_ok=True)
        card = Path(__file__).resolve().parents[1] / "cards" / "model_card.md"
        if card.exists():
            shutil.copy(card, out_dir / "README.md")
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push_to_hub)
        print(f"pushed: https://huggingface.co/{args.push_to_hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
