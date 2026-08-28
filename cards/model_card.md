---
language:
  - ja
license: apache-2.0
library_name: phonebook
pipeline_tag: text2text-generation
tags:
  - japanese
  - katakana
  - g2p
  - grapheme-to-phoneme
  - corporate-names
  - copy-mechanism
  - calibration
  - selective-prediction
  - gguf
  - onnx
  - mlx
metrics:
  - exact_match
  - cer
---

# Phonebook — reading Japanese corporate names by structure, not by memory

**Getting the reading of "株式会社◯◯" right is easy. The hard part is the ◯◯ you have never seen.**

A character-level seq2seq model specialised in mapping Japanese corporate names to
katakana readings. About 17M parameters by default, built to run on CPU, with a copy
mechanism, constrained decoding, calibrated confidence and a rejection option.

> ## ⚠️ This checkpoint is trained on synthetic data
>
> The weights published here were trained on the synthetic corpus produced by
> `scripts/make_synthetic.py`, which exists so the full pipeline can be validated
> without the National Tax Agency bulk download. **They are a working reference
> implementation, not a production model.** For real performance, fetch the NTA data and
> retrain — the repository automates every step. The evaluation numbers shipped with the
> repository carry the same warning.

## Usage

```python
from phonebook import load_reader
reader = load_reader("path/to/model")
result = reader.read("株式会社日本電気", nbest=3)
print(result.reading, result.confidence)   # reading and calibrated confidence
print(result.to_dict()["candidates"])      # n-best with probabilities
```

```bash
phonebook read "株式会社日本電気" --nbest 3
```

## Design

1. **StructuralSplitter** — strips the legal form (leading/trailing, parentheses, shop
   names) with rules and assigns it a deterministic reading, so the model only sees the
   trade-name core. This is not only about speed: it removes from the evaluation the
   effect where memorizing カブシキガイシャ inflates apparent accuracy.
2. **CharSeq2Seq with a copy mechanism** — a character-level Transformer. The
   pointer-generator lets it transcribe katakana and hiragana already present in the
   input. Kana-only spans are transcribed deterministically, so copying is a structural
   guarantee rather than a learned habit.
3. **Constrained decoding** — the vocabulary is restricted to full-width katakana, the
   prolonged mark and EOS. The output character set is therefore a property of the search
   space, not of training, and cannot be violated even by untrained weights.
4. **Calibration and rejection** — Platt scaling turns the score into a probability, and
   below a threshold the model answers "unknown".

## Evaluation

**Read the numbers per split.** An overall average is dominated by the abundance of known
entities and invites mistaking memorization for generalization.

| Set | Definition |
|---|---|
| (1) known entity | The trade-name core occurs in training data (the corporate number does not) |
| (2) unseen entity | The core never occurs in training data |
| (3) hard | Subset of (2) containing a kanji bigram absent from training data |
| (4) ambiguous | Same surface, different attested reading; the minority reading is isolated |

Conditions: (A) pyopenjtalk / MeCab+UniDic, (B) a large LLM with prompting only,
(C) Phonebook, (D) Phonebook after Q4_K_M quantization.
Metrics: exact match (strict and long-vowel-normalized), CER, n-best@3 coverage, ECE,
coverage and precision under rejection, ms/item and resident RSS on CPU.

Numbers are in `benchmarks/RESULTS.md` in the repository.

## Export formats (note)

- **GGUF (Q4_K_M / Q8_0)**: real Q4_K / Q6_K / Q8_0 block structure, bit-exact
  round-trip. But **llama.cpp cannot run these files** — Phonebook is a custom
  architecture. GGUF is used for distribution and inspection.
- **ONNX**: runs as-is under onnxruntime; beam search runs on the host.
- **MLX**: weights (npz) plus config; see `mlx/README.md` for a loading example.

## Limitations

- The target is **corporate names**. Reading personal names is not the objective and is
  kept to a LoRA transfer experiment.
- Same-surface/different-reading cases (日本 = ニホン / ニッポン) have no unique answer
  in principle. Use the n-best and the confidence rather than top-1 exact match. On the
  ambiguous set the model is also poorly calibrated: it is confident about the majority
  reading, which is precisely the wrong one there.
- Latin script is resolved by lexicon → acronym → Hepburn romaji → English spelling
  rules. English spelling-to-sound is rule-based and imperfect.
- Output is an **estimate**, not official furigana.

## Training data

**Source: National Tax Agency Corporate Number Publication Site
(https://www.houjin-bangou.nta.go.jp/) — created by processing that data.**
The published information may be used under terms conforming to the Japanese
Government's Public Data License (Version 1.0), which requires crediting the source and
stating that the data has been modified. Check the current terms at
https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/.

(The checkpoint published here is trained on synthetic data modelled on that schema; see
the warning above.)

Model weights are distributed under Apache-2.0.
