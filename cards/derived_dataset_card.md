---
language:
  - ja
license: other
license_name: japan-public-data-license-1.0
license_link: https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/
pretty_name: Estimated furigana for corporations with none registered (with confidence; estimates, not official)
task_categories:
  - text2text-generation
tags:
  - japanese
  - katakana
  - machine-generated
  - estimates
  - confidence
  - corporate-names
---

# Estimated readings for corporations with no registered furigana

> ## ⚠️ These are estimates, not official furigana
>
> `estimated_furigana` is **machine-generated** by the Phonebook model. It is not the
> official furigana published by Japan's National Tax Agency. **It cannot be used for
> registration, official procedures, or identity verification.** For the official
> furigana, consult the NTA Corporate Number Publication Site.
> Every row carries `is_estimate: true` and a disclaimer.

## What this is for

A substantial number of corporations in the NTA Corporate Number data have no registered
furigana. For record linkage, search and text-to-speech, the absence of a reading is
itself the obstacle. This dataset supplies an estimated reading **together with a
confidence and an accept/reject decision**.

Because the confidence is calibrated, it can be used to **prioritize human review**. The
value here is not "read everything automatically" but "have a person look at the low-
confidence portion only".

## Fields

| Field | Description |
|---|---|
| `corporate_number` | Corporate number |
| `name` / `name_normalized` | Original / normalized trade name |
| `estimated_furigana` | **Estimated** reading (katakana + prolonged mark only) |
| `confidence` | Calibrated confidence (estimated probability that the top-1 is correct) |
| `accepted` | Whether it passed the threshold; `false` means "unknown" |
| `candidates` | n-best with probabilities |
| `source` | Inference path (`copy` = kana transcription, `model`, `segmented`, `en2kana`, `rule`) |
| `kind` / `kind_label` | Corporation kind — **required for stratified evaluation** |
| `is_estimate` | Always `true` |

## ⚠️ The missingness is non-random — an overall accuracy says nothing

Furigana absence is strongly concentrated by corporation kind. Typically:

- **Kabushiki-Kaisha: 30–40% missing**
- **Yugen-Kaisha: 40–50% missing**
- Other kinds: under 10%

So this derived dataset is a non-random sample dominated by Kabushiki-Kaisha and
Yugen-Kaisha. Carrying over an "overall accuracy" measured on labelled test sets will
systematically misjudge its quality.

The accompanying report `fill_missing_report.json` therefore
**stratifies by (corporation kind × whether the trade-name core appeared in training)**
and reports the **expected accuracy reweighted to the composition of the missing set**.
Use that number.

Corporations lacking a furigana also tend to be smaller, newer, or registration-only
entities, so the distribution of the trade names themselves may differ from those with a
registered furigana. That residual bias is not measured here and is stated explicitly.

## Recommended use

1. Send only rows with `accepted == true` into automated processing.
2. Treat `accepted == false` as "reading unknown" and queue it for human review.
3. Inspect the acceptance rate and confidence distribution per corporation kind before
   settling on your own threshold.
4. If you publish or redistribute this, **carry the "estimate" status forward**.

## Source

**Source: National Tax Agency Corporate Number Publication Site
(https://www.houjin-bangou.nta.go.jp/) — created by processing that data.**
The trade name, corporate number, address and corporation kind come from that published
information. `estimated_furigana` and `confidence` are **estimates produced by this
project**.
