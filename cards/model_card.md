---
language:
  - ja
license: apache-2.0
library_name: phonebook
tags:
  - text2text-generation
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
katakana readings. About 17M parameters, built to run on CPU, with a copy mechanism,
constrained decoding, calibrated confidence and a rejection option.

Code, dataset builder, evaluation and figures: **https://github.com/NagaYu/phonebook**

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

### Measured results (synthetic corpus - see the warning above)

Exact match with long vowels normalized, which is the fair comparison against G2P
systems that emit a pronunciation form:

| condition | (1) known | (2) unseen | (3) hard | (4) ambiguous |
|---|---:|---:|---:|---:|
| pyopenjtalk | 0.532 | 0.533 | 0.521 | 0.295 |
| MeCab+UniDic | 0.470 | 0.470 | 0.443 | 0.240 |
| Phonebook fp32 | **0.947** | **0.648** | **0.642** | 0.081 |
| Phonebook Q4_K_M | **0.947** | **0.649** | **0.645** | 0.083 |

Character error rate: Phonebook 0.004 / 0.027 / 0.027 / 0.120 against pyopenjtalk
0.144 / 0.140 / 0.143 / 0.185. CPU: ~54 ms/item at beam 8 without a KV cache.

**Where the hypothesis fails, stated plainly.** The design hypothesis was that the gap
would *widen* on unseen entities. It does not. Phonebook leads pyopenjtalk by +0.415 on
known entities but only +0.115 on unseen and +0.121 on hard, and its own drop from known
to hard is -0.31 against pyopenjtalk's -0.01. It remains clearly ahead in absolute terms
on unseen entities (0.649 vs 0.533, with roughly five times lower CER), but "ahead" and
"the gap widens" are different claims and only the first one holds here.

Calibration behaves the same way: ECE is 0.019 on known entities, 0.274 on unseen and
0.786 on ambiguous. The rejection threshold fitted on dev (0.366) leaves coverage at 0.99
on the test splits and barely lifts precision, because dev is dominated by
known-entity-like cases. **Fit the threshold on data matching your deployment
distribution.**

Quantization is free: Q4_K_M matches fp32 within +/-0.003 exact match on every split, at
10.8 MB of weights and the same latency.

Full tables: `benchmarks/RESULTS.md` in the repository.

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
