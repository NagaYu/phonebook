#!/usr/bin/env python3
"""Synthetic corpus generator, for pipeline validation. **Not for headline numbers.**

Why this exists
---------------
The NTA bulk data is hundreds of megabytes and requires reading the terms of use
and downloading it by hand. But whether the chain "split design -> training ->
evaluation -> figures -> export" runs end to end should be verifiable without
that data. This script emits synthetic records in the **same 30-column layout**
as the NTA CSV, so every downstream script treats real and synthetic data
identically.

What it reproduces from the real data (the properties the claims depend on):
  - trade-name cores composed from morphemes, so unseen kanji bigrams arise
    naturally
  - genuine same-surface/different-reading pairs (日本 = ニホン/ニッポン,
    大和 = ヤマト/ダイワ, ...)
  - reading alternations such as rendaku (sequential voicing)
  - **furigana missingness concentrated in Kabushiki-Kaisha and Yugen-Kaisha**
    (non-random missingness)
  - a legal-form distribution close to the real one (mostly Kabushiki-Kaisha)

**Numbers obtained on synthetic data are not evidence for the claims in the
README.** Only results reproduced on the real data belong in the results table;
build_dataset.py records a ``synthetic`` flag in the dataset metadata so the two
can never be confused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phonebook.data import CSV_COLUMNS  # noqa: E402
from phonebook.structure import LEGAL_FORMS  # noqa: E402

# --- Morpheme lexicon: (kanji, candidate readings) --------------------------
PLACES: list[tuple[str, list[str]]] = [
    ("日本", ["ニホン", "ニッポン"]), ("東京", ["トウキョウ"]), ("大阪", ["オオサカ"]),
    ("名古屋", ["ナゴヤ"]), ("横浜", ["ヨコハマ"]), ("神戸", ["コウベ", "カンベ"]),
    ("京都", ["キョウト"]), ("札幌", ["サッポロ"]), ("福岡", ["フクオカ"]),
    ("仙台", ["センダイ"]), ("広島", ["ヒロシマ"]), ("新潟", ["ニイガタ"]),
    ("金沢", ["カナザワ", "カネザワ"]), ("静岡", ["シズオカ"]), ("岡山", ["オカヤマ"]),
    ("熊本", ["クマモト"]), ("長野", ["ナガノ"]), ("山形", ["ヤマガタ"]),
    ("秋田", ["アキタ"]), ("群馬", ["グンマ"]), ("千葉", ["チバ"]), ("埼玉", ["サイタマ"]),
    ("茨城", ["イバラキ"]), ("三重", ["ミエ"]), ("滋賀", ["シガ"]), ("奈良", ["ナラ"]),
    ("鳥取", ["トットリ"]), ("島根", ["シマネ"]), ("山口", ["ヤマグチ"]),
    ("香川", ["カガワ"]), ("愛媛", ["エヒメ"]), ("高知", ["コウチ"]), ("佐賀", ["サガ"]),
    ("長崎", ["ナガサキ"]), ("大分", ["オオイタ"]), ("宮崎", ["ミヤザキ"]),
    ("沖縄", ["オキナワ"]), ("北海", ["ホッカイ"]), ("東海", ["トウカイ"]),
    ("関西", ["カンサイ"]), ("中央", ["チュウオウ"]), ("北陸", ["ホクリク"]),
    ("山陽", ["サンヨウ"]), ("河内", ["カワチ", "コウチ"]), ("大和", ["ヤマト", "ダイワ"]),
    ("春日", ["カスガ", "ハルヒ"]), ("生駒", ["イコマ"]), ("日野", ["ヒノ"]),
]

SURNAMES: list[tuple[str, list[str]]] = [
    ("山田", ["ヤマダ"]), ("田中", ["タナカ"]), ("佐藤", ["サトウ"]), ("鈴木", ["スズキ"]),
    ("高橋", ["タカハシ"]), ("渡辺", ["ワタナベ"]), ("伊藤", ["イトウ"]),
    ("中村", ["ナカムラ"]), ("小林", ["コバヤシ"]), ("加藤", ["カトウ"]),
    ("吉田", ["ヨシダ"]), ("山本", ["ヤマモト"]), ("松本", ["マツモト"]),
    ("井上", ["イノウエ"]), ("木村", ["キムラ"]), ("清水", ["シミズ"]),
    ("斎藤", ["サイトウ"]), ("池田", ["イケダ"]), ("橋本", ["ハシモト"]),
    ("石川", ["イシカワ"]), ("前田", ["マエダ"]), ("藤田", ["フジタ"]),
    ("後藤", ["ゴトウ"]), ("岡田", ["オカダ"]), ("村上", ["ムラカミ"]),
    ("近藤", ["コンドウ"]), ("石井", ["イシイ"]), ("上田", ["ウエダ", "カミタ"]),
    ("中島", ["ナカジマ", "ナカシマ"]), ("新田", ["ニッタ", "シンデン"]),
    ("山崎", ["ヤマザキ", "ヤマサキ"]), ("中田", ["ナカタ", "ナカダ"]),
    ("小谷", ["コタニ", "オタニ"]), ("白鳥", ["シラトリ", "ハクチョウ"]),
    ("柳田", ["ヤナギダ", "ヤナギタ"]), ("三宅", ["ミヤケ"]), ("一色", ["イッシキ"]),
]

BUSINESS: list[tuple[str, list[str]]] = [
    ("商事", ["ショウジ"]), ("産業", ["サンギョウ"]), ("工業", ["コウギョウ"]),
    ("建設", ["ケンセツ"]), ("運輸", ["ウンユ"]), ("運送", ["ウンソウ"]),
    ("物流", ["ブツリュウ"]), ("電機", ["デンキ"]), ("電気", ["デンキ"]),
    ("電子", ["デンシ"]), ("精機", ["セイキ"]), ("精工", ["セイコウ"]),
    ("製作所", ["セイサクショ"]), ("製菓", ["セイカ"]), ("製薬", ["セイヤク"]),
    ("食品", ["ショクヒン"]), ("農園", ["ノウエン"]), ("水産", ["スイサン"]),
    ("木材", ["モクザイ"]), ("鉄工", ["テッコウ"]), ("化学", ["カガク"]),
    ("情報", ["ジョウホウ"]), ("通信", ["ツウシン"]), ("印刷", ["インサツ"]),
    ("出版", ["シュッパン"]), ("広告", ["コウコク"]), ("企画", ["キカク"]),
    ("設計", ["セッケイ"]), ("開発", ["カイハツ"]), ("技研", ["ギケン"]),
    ("総研", ["ソウケン"]), ("保険", ["ホケン"]), ("証券", ["ショウケン"]),
    ("不動産", ["フドウサン"]), ("住宅", ["ジュウタク"]), ("興業", ["コウギョウ"]),
    ("商会", ["ショウカイ"]), ("商店", ["ショウテン"]), ("本舗", ["ホンポ"]),
    ("酒造", ["シュゾウ"]), ("織物", ["オリモノ"]), ("塗装", ["トソウ"]),
    ("電設", ["デンセツ"]), ("設備", ["セツビ"]), ("工務店", ["コウムテン"]),
    ("薬局", ["ヤッキョク"]), ("書店", ["ショテン"]), ("医院", ["イイン"]),
]

MODIFIERS: list[tuple[str, list[str]]] = [
    ("新", ["シン"]), ("大", ["ダイ"]), ("小", ["コ"]), ("東", ["ヒガシ", "アズマ"]),
    ("西", ["ニシ"]), ("南", ["ミナミ"]), ("北", ["キタ"]), ("上", ["カミ"]),
    ("下", ["シモ"]), ("中", ["ナカ"]), ("光", ["ヒカリ"]), ("星", ["ホシ"]),
    ("森", ["モリ"]), ("川", ["カワ"]), ("山", ["ヤマ"]), ("海", ["ウミ"]),
    ("緑", ["ミドリ"]), ("青", ["アオ"]), ("白", ["シロ"]), ("黒", ["クロ"]),
    ("金", ["キン"]), ("銀", ["ギン"]), ("千", ["セン"]), ("百", ["ヒャク"]),
]

KATAKANA_WORDS = [
    "アルファ", "ベータ", "ガンマ", "デルタ", "オメガ", "サンライズ", "ホライゾン",
    "クリエイト", "パートナー", "ネクスト", "フロンティア", "ステップ", "リンク",
    "ブリッジ", "コンパス", "アトリエ", "テラス", "ポルト", "ミライ", "サクラ",
]
HIRAGANA_WORDS = ["あおぞら", "みどり", "ひまわり", "こもれび", "つばさ", "さくら", "あすなろ"]
LATIN_WORDS = [
    "ABC", "NEXT", "GLOBAL", "TECH", "SYSTEM", "DESIGN", "PLUS", "ONE",
    "SAKURA", "MIRAI", "HIKARI", "MIDORI", "AQUA", "STAR", "LINK",
]

PREFECTURES = [
    ("東京都", "13"), ("大阪府", "27"), ("愛知県", "23"), ("神奈川県", "14"),
    ("北海道", "01"), ("福岡県", "40"), ("兵庫県", "28"), ("静岡県", "22"),
    ("広島県", "34"), ("宮城県", "04"), ("新潟県", "15"), ("長野県", "20"),
]

#: (legal form, probability, kind code, probability the form leads the name)
LEGAL_DISTRIBUTION = [
    ("株式会社", 0.60, "301", 0.85),
    ("有限会社", 0.13, "302", 0.80),
    ("合同会社", 0.07, "305", 0.90),
    ("一般社団法人", 0.04, "399", 1.0),
    ("特定非営利活動法人", 0.025, "399", 1.0),
    ("医療法人", 0.02, "399", 1.0),
    ("社会福祉法人", 0.015, "399", 1.0),
    ("学校法人", 0.012, "399", 1.0),
    ("宗教法人", 0.012, "399", 1.0),
    ("一般財団法人", 0.01, "399", 1.0),
    ("合資会社", 0.008, "304", 0.7),
    ("合名会社", 0.005, "303", 0.7),
    ("農業協同組合", 0.005, "399", 0.5),
    ("信用金庫", 0.005, "399", 0.3),
    ("", 0.043, "499", 0.0),
]

#: Furigana missing rate, deliberately **skewed toward Kabushiki-Kaisha and
#: Yugen-Kaisha** to mimic the non-random missingness of the real data.
MISSING_RATE_BY_KIND = {"301": 0.34, "302": 0.46, "305": 0.12, "399": 0.07, "499": 0.05}

VOICED = {"カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
          "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
          "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
          "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ"}
VOICED_CHARS = set(VOICED.values()) | set("ヴ")


def _h(key: str, salt: str = "") -> float:
    return int.from_bytes(hashlib.sha1((salt + key).encode()).digest()[:8], "big") / float(1 << 64)


def apply_rendaku(prev: str, nxt: str, key: str) -> str:
    """Rendaku (sequential voicing), respecting Lyman's Law: it does not occur
    when the second element already contains a voiced obstruent."""
    if not prev or not nxt:
        return nxt
    if nxt[0] not in VOICED:
        return nxt
    if set(nxt) & VOICED_CHARS:
        return nxt
    if _h(key, "rendaku") < 0.35:
        return VOICED[nxt[0]] + nxt[1:]
    return nxt


def build_core_pool(rng: random.Random, n_cores: int) -> list[tuple[str, list[list[str]]]]:
    """Build the pool of trade-name cores as (surface, per-morpheme readings)."""
    pool: list[tuple[str, list[list[str]]]] = []
    seen: set[str] = set()
    groups = [PLACES, SURNAMES, BUSINESS, MODIFIERS]
    while len(pool) < n_cores:
        style = rng.random()
        if style < 0.06:
            word = rng.choice(KATAKANA_WORDS)
            surface, readings = word, [[word]]
        elif style < 0.10:
            word = rng.choice(HIRAGANA_WORDS)
            from phonebook.kana import hira_to_kata

            surface, readings = word, [[hira_to_kata(word)]]
        elif style < 0.14:
            word = rng.choice(LATIN_WORDS)
            surface, readings = word, [[None]]  # reading resolved later via en2kana
        elif style < 0.20:
            kana = rng.choice(KATAKANA_WORDS)
            tail = rng.choice(BUSINESS)
            surface = kana + tail[0]
            readings = [[kana], list(tail[1])]
        else:
            k = rng.choices([1, 2, 3], weights=[0.18, 0.62, 0.20])[0]
            parts = []
            for i in range(k):
                if i == k - 1 and k > 1:
                    parts.append(rng.choice(BUSINESS))
                else:
                    parts.append(rng.choice(rng.choices(groups, weights=[3, 3, 1, 2])[0]))
            surface = "".join(p[0] for p in parts)
            readings = [list(p[1]) for p in parts]
        if surface in seen or not surface:
            continue
        seen.add(surface)
        pool.append((surface, readings))
    return pool


def compose_reading(surface: str, readings: list[list[str]], entity_key: str) -> str | None:
    """Pick a reading per morpheme, apply rendaku, and concatenate.

    Morphemes with several readings are resolved using a **per-corporation
    key**, which is what makes the same surface genuinely carry different
    readings across companies -- the ambiguous set.
    """
    parts: list[str] = []
    for i, cands in enumerate(readings):
        if cands == [None]:
            from phonebook.en2kana import EnglishToKatakana

            parts.append(EnglishToKatakana().convert(surface))
            continue
        if len(cands) == 1:
            reading = cands[0]
        else:
            idx = int(_h(entity_key + surface + str(i), "variant") * len(cands))
            reading = cands[min(idx, len(cands) - 1)]
        if parts:
            reading = apply_rendaku(parts[-1], reading, surface + str(i))
        parts.append(reading)
    out = "".join(parts)
    return out or None


def corporate_number(index: int) -> str:
    """A 13-digit corporate-number-like id. Synthetic; no check digit."""
    return f"{1000000000000 + index * 7919 % 8999999999999:013d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/synthetic.csv")
    parser.add_argument("--n", type=int, default=60000, help="number of records")
    parser.add_argument(
        "--cores", type=int, default=None,
        help="number of distinct trade-name cores (default n/2)",
    )
    parser.add_argument("--seed", type=int, default=20240401)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_cores = args.cores or max(200, args.n // 2)
    pool = build_core_pool(rng, n_cores)

    forms = [f for f, _, _, _ in LEGAL_DISTRIBUTION]
    weights = [w for _, w, _, _ in LEGAL_DISTRIBUTION]
    kinds = {f: k for f, _, k, _ in LEGAL_DISTRIBUTION}
    prefix_rates = {f: p for f, _, _, p in LEGAL_DISTRIBUTION}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for i in range(args.n):
            surface, readings = pool[rng.randrange(len(pool))]
            cn = corporate_number(i)
            core_reading = compose_reading(surface, readings, cn)
            if not core_reading:
                continue
            form = rng.choices(forms, weights=weights)[0]
            kind = kinds[form]
            if form and rng.random() < prefix_rates[form]:
                name = form + surface
                furigana = LEGAL_FORMS[form] + core_reading
            elif form:
                name = surface + form
                furigana = core_reading + LEGAL_FORMS[form]
            else:
                name = surface
                furigana = core_reading

            if rng.random() < MISSING_RATE_BY_KIND.get(kind, 0.1):
                furigana = ""
                n_missing += 1

            pref, pref_code = rng.choice(PREFECTURES)
            row = [""] * len(CSV_COLUMNS)
            row[0] = str(i + 1)
            row[1] = cn
            row[2] = "01"
            row[4] = "2026-08-01"
            row[6] = name
            row[8] = kind
            row[9] = pref
            row[10] = f"{rng.randrange(1, 30)}区"
            row[11] = f"{rng.randrange(1, 9)}丁目{rng.randrange(1, 30)}番"
            row[13] = pref_code
            row[23] = "1"  # latest flag
            row[28] = furigana
            row[29] = "0"  # not excluded from search
            writer.writerow(row)

    print(f"wrote {out_path} ({args.n:,} rows, {n_cores:,} distinct cores)")
    print(
        f"missing furigana: {n_missing:,} rows ({n_missing / args.n:.1%}), "
        "deliberately skewed toward Kabushiki-Kaisha / Yugen-Kaisha"
    )
    print("NOTE: this is synthetic data. Only real-data numbers belong in the results table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
