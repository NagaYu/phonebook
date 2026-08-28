# data/

| Directory | Contents | Tracked by git |
|---|---|---|
| `raw/` | NTA bulk data (CSV/zip), or synthetic data | **no** (.gitignore) |
| `processed/` | Cleansed and split JSONL, metadata, split summary | **no** |
| `processed/en2kana/` | Auxiliary Latin-to-katakana pairs and lexicon | **no** |

## Why the raw data is not redistributed

The NTA Corporate Number Publication Site does permit redistribution under terms
conforming to the Public Data License (Version 1.0). This repository still does
not ship the raw data, for three reasons.

1. The bulk file is hundreds of megabytes and is refreshed monthly. Pinning a
   copy in the repository would guarantee it goes stale.
2. Every user needs to **read the terms themselves and credit the source in
   their own artifacts**. Making them fetch the data is the most reliable way to
   ensure that step is not skipped.
3. To keep "processed" and "original" from being confused. What this repository
   distributes is the *method* for producing the processed data -- the code --
   not the processed data itself.

See `scripts/fetch_houjin.py` (which prints the terms) for how to obtain it. To
exercise the whole pipeline with no data on hand, use
`scripts/make_synthetic.py`. **Numbers obtained on synthetic data are not
evidence for any claim.**

## Source

Source: National Tax Agency Corporate Number Publication Site (NTA),
https://www.houjin-bangou.nta.go.jp/ - created by processing that data.
