---
language:
  - ja
license: other
license_name: japan-public-data-license-1.0
license_link: https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/
pretty_name: Phonebook corporate-name readings (with known / unseen / hard / ambiguous splits)
task_categories:
  - text2text-generation
tags:
  - japanese
  - katakana
  - g2p
  - named-entity
  - corporate-names
  - generalization
  - data-splits
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: dev
        path: dev.jsonl
      - split: test_known
        path: test_known.jsonl
      - split: test_unseen
        path: test_unseen.jsonl
      - split: test_ambiguous
        path: test_ambiguous.jsonl
      - split: missing_furigana
        path: missing_furigana.jsonl
---

# Phonebook corporate-name readings

**Japanese corporate name → katakana reading. Not a lookup table: a split design that
separates memorization from generalization.**

## Source and terms of use (read this)

**Source: National Tax Agency Corporate Number Publication Site
(https://www.houjin-bangou.nta.go.jp/) — created by processing that data.**

- The published information may be used freely — reproduction, public transmission and
  adaptation included, commercial use permitted — under terms conforming to the Japanese
  Government's Public Data License (Version 1.0).
- **Crediting the source is required.** If you process the data you must **also state
  that you did so**.
- Publishing it in a manner suggesting the government produced your work is prohibited.
- Always check the current terms at https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/.
  The description here reflects 2026-08 and is not the terms themselves.
- The furigana column (item 35 of the resource definition) is specified as "full-width
  katakana and the prolonged sound mark only", blank when unregistered. This dataset
  enforces the same character set.

## Why the split is the main event

Reading proper nouns is trivially "solved" when the same trade-name core sits in both
train and test. That is memorization, not generalization. This dataset keeps four sets
strictly apart.

| split | Definition | What it measures |
|---|---|---|
| `train` | training | — |
| `dev` | calibration and rejection-threshold selection | — |
| `test_known` | The core occurs in train, but **the corporate number does not** | (1) known entity — effectively dictionary lookup |
| `test_unseen` | The core string never occurs in train | (2) unseen entity — genuine generalization |
| `test_unseen` with `is_hard=true` | Subset of (2) containing a **kanji bigram** absent from train | (3) hard — compositional generalization |
| `test_ambiguous` | Several readings attested for one surface; the majority goes to train and the minority is isolated | (4) ambiguity — a problem with no unique answer |
| `missing_furigana` | Corporations with no registered furigana (unlabelled) | the completion task |

The split uses a **salted deterministic hash**, not a random seed, so the same input
always reproduces the same split. A leakage check (`verify_splits`) runs at build time
and aborts the build on failure.

## Fields

| Field | Description |
|---|---|
| `corporate_number` | 13-digit corporate number; the unique key of the split |
| `name_raw` / `name` | Original / normalized surface (**the original is always retained**) |
| `furigana_raw` / `furigana` | Original / normalized furigana (katakana + prolonged mark only) |
| `core` / `core_reading` | Trade-name core and its reading, legal form removed; the unit of training and evaluation |
| `legal_form` / `legal_form_position` | Legal form and its position (`prefix` / `suffix` / `none`) |
| `kind` / `kind_label` | Corporation kind code and label (301 = Kabushiki-Kaisha, etc.) |
| `prefecture` | Prefecture of the head office |
| `split` | Which split the row belongs to |
| `is_hard` | Whether it is in the hard set (contains an unseen kanji bigram) |
| `all_readings` | Every attested reading of that core, used to score the ambiguous set |

## Cleansing policy

- Keep only the current record (item 30, `latest=1`) and drop rows excluded from search
  (item 36, `hihyoji=1`).
- Normalize spelling variation (width, variant kanji, punctuation) but **always retain
  the original**. Over-normalizing drifts away from what users actually type;
  under-normalizing lets the same company span train and test under two surfaces.
- Rows where the furigana and the surface disagree structurally — the legal-form reading
  cannot be stripped — are **excluded from training**. Being lenient here injects label
  noise directly into the claim about unseen entities.
- A furigana containing disallowed characters is treated as missing, and the audit log
  records the per-character counts.

## Known biases

- **Furigana missingness is non-random**, markedly higher for Kabushiki-Kaisha and
  Yugen-Kaisha and low for other kinds. If you use the `missing_furigana` split, an
  overall accuracy will not describe its quality; evaluate per stratum.
- The distribution over corporation kinds is heavily skewed toward Kabushiki-Kaisha
  (this is the real distribution).
- Locations skew toward urban prefectures.

## Out-of-scope uses

- Identifying individuals. This is published corporate information, not a personal-data
  dataset.
- Use as official furigana. `core_reading` is derived by mechanically stripping the legal
  form from the published furigana, and the stripping rules are this project's own.
  Consult the original source.

## Citation

```
Phonebook: katakana readings for Japanese corporate names, with a known/unseen split design
Source: National Tax Agency Corporate Number Publication Site (NTA), processed
```
