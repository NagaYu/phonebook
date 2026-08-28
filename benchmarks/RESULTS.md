# Benchmark results

> **Warning: these numbers come from synthetic data.** They validate that the
> pipeline runs; they are not evidence for the claims. Re-run on the real NTA
> bulk data before citing any figure here.

Items evaluated: (1) known entity 1,227, (2) unseen entity 4,638, (3) hard (unseen kanji bigram) 662, (4) ambiguous (same surface, other reading) 1,652

## Exact match (strict)

| condition | (1) known entity | (2) unseen entity | (3) hard (unseen kanji bigram) | (4) ambiguous (same surface, other reading) |
|---|---|---|---|---|
| A:pyopenjtalk | 0.139 | 0.130 | 0.110 | 0.081 |
| A:mecab-unidic | 0.131 | 0.127 | 0.107 | 0.070 |
| B:claude | not run | not run | not run | not run |
| C:phonebook-fp32 | 0.947 | 0.648 | 0.642 | 0.081 |
| D:phonebook-q4_k_m | 0.947 | 0.649 | 0.645 | 0.083 |
| D:phonebook-q8_0 | 0.948 | 0.647 | 0.642 | 0.081 |

## Exact match (long vowels normalized)

| condition | (1) known entity | (2) unseen entity | (3) hard (unseen kanji bigram) | (4) ambiguous (same surface, other reading) |
|---|---|---|---|---|
| A:pyopenjtalk | 0.532 | 0.533 | 0.521 | 0.295 |
| A:mecab-unidic | 0.470 | 0.470 | 0.443 | 0.240 |
| B:claude | not run | not run | not run | not run |
| C:phonebook-fp32 | 0.947 | 0.648 | 0.642 | 0.081 |
| D:phonebook-q4_k_m | 0.947 | 0.649 | 0.645 | 0.083 |
| D:phonebook-q8_0 | 0.948 | 0.647 | 0.642 | 0.081 |

## Character error rate (lower is better)

| condition | (1) known entity | (2) unseen entity | (3) hard (unseen kanji bigram) | (4) ambiguous (same surface, other reading) |
|---|---|---|---|---|
| A:pyopenjtalk | 0.144 | 0.140 | 0.143 | 0.185 |
| A:mecab-unidic | 0.146 | 0.145 | 0.142 | 0.193 |
| B:claude | not run | not run | not run | not run |
| C:phonebook-fp32 | 0.004 | 0.027 | 0.027 | 0.120 |
| D:phonebook-q4_k_m | 0.004 | 0.027 | 0.026 | 0.120 |
| D:phonebook-q8_0 | 0.004 | 0.027 | 0.027 | 0.120 |

## n-best@3 coverage

| condition | (1) known entity | (2) unseen entity | (3) hard (unseen kanji bigram) | (4) ambiguous (same surface, other reading) |
|---|---|---|---|---|
| A:pyopenjtalk | 0.139 | 0.130 | 0.110 | 0.145 |
| A:mecab-unidic | 0.131 | 0.127 | 0.107 | 0.133 |
| C:phonebook-fp32 | 0.998 | 0.943 | 0.937 | 0.998 |
| D:phonebook-q4_k_m | 0.998 | 0.942 | 0.938 | 0.998 |
| D:phonebook-q8_0 | 0.998 | 0.943 | 0.937 | 0.998 |

## Calibration and rejection (Phonebook only)

| condition | set | ECE | exact match, no rejection | coverage | precision on accepted |
|---|---|---|---|---|---|
| C:phonebook-fp32 | (1) known entity | 0.0202 | 0.947 | 0.996 | 0.949 |
| C:phonebook-fp32 | (2) unseen entity | 0.2744 | 0.648 | 0.990 | 0.651 |
| C:phonebook-fp32 | (3) hard (unseen kanji bigram) | 0.2799 | 0.642 | 0.976 | 0.652 |
| C:phonebook-fp32 | (4) ambiguous (same surface, other reading) | 0.7857 | 0.081 | 0.976 | 0.079 |
| D:phonebook-q4_k_m | (1) known entity | 0.0185 | 0.947 | 0.993 | 0.952 |
| D:phonebook-q4_k_m | (2) unseen entity | 0.2745 | 0.649 | 0.987 | 0.652 |
| D:phonebook-q4_k_m | (3) hard (unseen kanji bigram) | 0.2838 | 0.645 | 0.967 | 0.655 |
| D:phonebook-q4_k_m | (4) ambiguous (same surface, other reading) | 0.7824 | 0.083 | 0.975 | 0.081 |
| D:phonebook-q8_0 | (1) known entity | 0.0214 | 0.948 | 0.996 | 0.950 |
| D:phonebook-q8_0 | (2) unseen entity | 0.2740 | 0.647 | 0.990 | 0.650 |
| D:phonebook-q8_0 | (3) hard (unseen kanji bigram) | 0.2800 | 0.642 | 0.977 | 0.651 |
| D:phonebook-q8_0 | (4) ambiguous (same surface, other reading) | 0.7856 | 0.081 | 0.976 | 0.079 |

## CPU inference speed and memory

| condition | ms/item | items/s | resident RSS (MB) | weight size |
|---|---|---|---|---|
| A:pyopenjtalk | 0.02 | 45563 | 237 | - |
| A:mecab-unidic | 0.01 | 78576 | 256 | - |
| C:phonebook-fp32 | 54.24 | 18 | 1199 | - |
| D:phonebook-q4_k_m | 53.92 | 19 | 1212 | 10.8 MB |
| D:phonebook-q8_0 | 55.34 | 18 | 1220 | 18.1 MB |
