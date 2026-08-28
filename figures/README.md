# figures/

Output directory for `scripts/make_figures.py`.

| File | What it shows |
|---|---|
| `known_vs_unseen.png` | **Headline.** Exact match per evaluation set: whether the methods are close on known entities and separate on unseen and hard ones |
| `known_vs_unseen_phonetic.png` | The same with long vowels normalized, i.e. whether the gap survives once notation differences are removed |
| `generalization_gap.png` | The drop from known to unseen to hard. A smaller drop means less reliance on memorization |
| `reliability.png` | Confidence against empirical accuracy (calibration) |
| `rejection.png` | Coverage against precision on accepted items (the rejection mechanism) |
| `speed_size.png` | CPU inference time and weight size, before and after quantization |

Japanese glyphs appear in some labels (example names, kana). `scripts/make_figures.py`
searches for a Japanese-capable font and warns if none is available.
