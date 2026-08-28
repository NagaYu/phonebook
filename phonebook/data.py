"""Loading and cleansing the NTA Corporate Number public data.

Claim supported: **the measurability of unseen-entity performance**.

All this module has to do is produce data in a state where the evaluation can
be trusted. Split the records without normalizing spelling variation and the
same company slips into both train and test under two different surfaces, so
what you call "unseen" is effectively seen. Normalize too aggressively and the
original spelling is lost, so the distribution drifts away from what a user
actually types. Hence the design rule of this module: **keep both the
normalized surface and the original**.

Data source and terms of use
----------------------------
Source: National Tax Agency Corporate Number Publication Site
        (https://www.houjin-bangou.nta.go.jp/)
The information published there may be reproduced, adapted and used
commercially under terms conforming to the Japanese Government's Public Data
License (Version 1.0), provided the source is credited. Because this project
modifies the data, the fact that it has been modified must be stated in
addition to the source (see README and the dataset card).
This repository does **not** redistribute the raw data; fetch it yourself with
scripts/fetch_houjin.py.

CSV layout (Resource Definition for download files / Web-API, v4.1):
  - 30 columns, no header row
  - Column 29 (furigana): "full-width katakana and the prolonged mark only";
    blank when unregistered
  - Column 30 (hihyoji): 1 means excluded from search (address no longer exists)
  - Column 24 (latest): 1 is the current record, 0 is historical
"""

from __future__ import annotations

import csv
import io
import json
import unicodedata
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .kana import ALLOWED_OUTPUT_CHARS, invalid_chars, is_valid_reading, to_katakana
from .structure import StructuralSplitter, StructuredName

# --- CSV schema (Resource Definition v4.1, download file) ------------------
CSV_COLUMNS: tuple[str, ...] = (
    "sequence_number",       # 1  sequence number
    "corporate_number",      # 2  corporate number (13 digits)
    "process",               # 3  process category
    "correct",               # 4  correction category
    "update_date",           # 5  update date
    "change_date",           # 6  change date
    "name",                  # 7  trade name
    "name_image_id",         # 8  trade-name image id
    "kind",                  # 9  corporation kind code
    "prefecture_name",       # 10 address: prefecture
    "city_name",             # 11 address: city/ward
    "street_number",         # 12 address: street
    "address_image_id",      # 13 address image id
    "prefecture_code",       # 14 prefecture code
    "city_code",             # 15 city code
    "post_code",             # 16 postal code
    "address_outside",       # 17 overseas address
    "address_outside_image_id",  # 18
    "close_date",            # 19 registry closure date
    "close_cause",           # 20 registry closure cause
    "successor_corporate_number",  # 21
    "change_cause",          # 22 change details
    "assignment_date",       # 23 corporate-number assignment date
    "latest",                # 24 latest flag (1 = current)
    "en_name",               # 25 trade name (English)
    "en_prefecture_name",    # 26
    "en_city_name",          # 27
    "en_address_outside",    # 28
    "furigana",              # 29 furigana (katakana + prolonged mark; blank if absent)
    "hihyoji",               # 30 excluded-from-search flag (1 = excluded)
)

#: Corporation kind codes (Resource Definition, item 15)
KIND_LABELS: dict[str, str] = {
    "101": "National government organ",
    "201": "Local public entity",
    "301": "Kabushiki-Kaisha (stock company)",
    "302": "Yugen-Kaisha (limited company)",
    "303": "Gomei-Kaisha (general partnership)",
    "304": "Goshi-Kaisha (limited partnership)",
    "305": "Godo-Kaisha (LLC)",
    "399": "Other incorporated entity",
    "401": "Foreign company etc.",
    "499": "Other",
}

# --- Surface normalization ------------------------------------------------
#: Old/variant kanji forms -> standard forms. Deliberately limited to variants
#: that are frequent in trade names. Completeness is *not* the goal: an
#: over-eager mapping merges distinct companies and breaks the split.
ITAIJI_MAP: dict[str, str] = {
    "髙": "高", "﨑": "崎", "濵": "浜", "濱": "浜", "桒": "桑", "德": "徳",
    "瀨": "瀬", "邊": "辺", "邉": "辺", "眞": "真", "齋": "斎", "齊": "斉",
    "國": "国", "學": "学", "澤": "沢", "惠": "恵", "應": "応", "圓": "円",
    "藏": "蔵", "廣": "広", "戀": "恋", "櫻": "桜", "萬": "万", "亞": "亜",
    "壽": "寿", "拂": "払", "晝": "昼", "會": "会", "區": "区", "數": "数",
    "圖": "図", "彌": "弥", "曾": "曽", "驛": "駅", "豐": "豊",
}

#: Punctuation variation common in trade names: dashes and the middle dot.
SYMBOL_MAP: dict[str, str] = {
    "･": "・",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "　": " ",
}


def normalize_name(name: str) -> str:
    """Normalize a trade-name surface. Callers must retain the original.

    Applies NFKC (folding full-width alphanumerics and compatibility ligatures),
    then absorbs variant kanji and punctuation. The prolonged mark survives NFKC
    and is deliberately kept distinct from hyphens.
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", name)
    text = "".join(ITAIJI_MAP.get(ch, ch) for ch in text)
    text = "".join(SYMBOL_MAP.get(ch, ch) for ch in text)
    text = " ".join(text.split())
    return text.strip()


def normalize_furigana(furigana: str) -> str:
    """Normalize a furigana: strip spaces, fold to katakana + prolonged mark."""
    if not furigana:
        return ""
    text = to_katakana(unicodedata.normalize("NFKC", furigana))
    return "".join(text.split())


# --- Record ---------------------------------------------------------------
@dataclass
class Record:
    """One cleansed corporation.

    Attributes:
        corporate_number: 13-digit corporate number; the unique key for the
            known-entity split.
        name_raw / name: original / normalized surface.
        furigana_raw / furigana: original / normalized furigana; empty = missing.
        core / core_reading: trade-name core and its reading, legal form removed.
        kind: corporation kind code, used for stratified evaluation of the
            non-random missingness.
    """

    corporate_number: str
    name_raw: str
    name: str
    furigana_raw: str
    furigana: str
    kind: str
    prefecture: str = ""
    city: str = ""
    en_name: str = ""
    core: str = ""
    core_reading: str = ""
    prefix_form: Optional[str] = None
    suffix_form: Optional[str] = None
    aligned: bool = False

    @property
    def has_furigana(self) -> bool:
        return bool(self.furigana)

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, "Unknown")

    @property
    def position(self) -> str:
        if self.prefix_form:
            return "prefix"
        if self.suffix_form:
            return "suffix"
        return "none"

    def to_dict(self) -> dict:
        return asdict(self)


# --- Reading ---------------------------------------------------------------
def iter_csv_rows(path: str | Path, encoding: str = "utf-8") -> Iterator[dict]:
    """Yield rows of the NTA bulk CSV (or zip) as dicts.

    For a zip, every contained .csv/.asc member is read in order. There is no
    header row.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for member in sorted(zf.namelist()):
                if not member.lower().endswith((".csv", ".asc", ".txt")):
                    continue
                with zf.open(member) as fh:
                    text = io.TextIOWrapper(fh, encoding=encoding, errors="replace", newline="")
                    yield from _iter_reader(text)
    else:
        with path.open("r", encoding=encoding, errors="replace", newline="") as fh:
            yield from _iter_reader(fh)


def _iter_reader(fh) -> Iterator[dict]:
    for row in csv.reader(fh):
        if not row:
            continue
        if len(row) < len(CSV_COLUMNS):
            row = row + [""] * (len(CSV_COLUMNS) - len(row))
        yield dict(zip(CSV_COLUMNS, row[: len(CSV_COLUMNS)]))


@dataclass
class CleanseReport:
    """Cleansing audit log. Always record what was dropped and how much."""

    total_rows: int = 0
    kept: int = 0
    dropped_not_latest: int = 0
    dropped_hihyoji: int = 0
    dropped_empty_name: int = 0
    missing_furigana: int = 0
    invalid_furigana_chars: int = 0
    unaligned: int = 0
    invalid_char_counter: Counter = None  # type: ignore[assignment]
    missing_by_kind: Counter = None  # type: ignore[assignment]
    total_by_kind: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.invalid_char_counter = Counter()
        self.missing_by_kind = Counter()
        self.total_by_kind = Counter()

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not isinstance(v, Counter)}
        d["invalid_chars"] = dict(self.invalid_char_counter.most_common(50))
        d["missing_rate_by_kind"] = {
            KIND_LABELS.get(k, k): {
                "total": self.total_by_kind[k],
                "missing": self.missing_by_kind[k],
                "missing_rate": round(self.missing_by_kind[k] / max(1, self.total_by_kind[k]), 4),
            }
            for k in sorted(self.total_by_kind)
        }
        return d

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def cleanse(
    rows: Iterable[dict],
    *,
    splitter: StructuralSplitter | None = None,
    latest_only: bool = True,
    drop_hihyoji: bool = True,
    strict_charset: bool = True,
) -> tuple[list[Record], list[Record], CleanseReport]:
    """Split raw rows into (labeled, missing-furigana, audit log).

    Rows with an empty furigana are routed to a separate list rather than
    discarded, because they are the target of scripts/fill_missing.py -- the
    derived artifact with public value. The missingness is non-random (heavily
    concentrated in Kabushiki-Kaisha and Yugen-Kaisha), so the audit log always
    records the missing rate per corporation kind.
    """
    splitter = splitter or StructuralSplitter()
    labeled: list[Record] = []
    missing: list[Record] = []
    report = CleanseReport()

    for row in rows:
        report.total_rows += 1
        if latest_only and row.get("latest", "1").strip() not in ("", "1"):
            report.dropped_not_latest += 1
            continue
        if drop_hihyoji and row.get("hihyoji", "0").strip() == "1":
            report.dropped_hihyoji += 1
            continue

        name_raw = (row.get("name") or "").strip()
        if not name_raw:
            report.dropped_empty_name += 1
            continue

        name = normalize_name(name_raw)
        furigana_raw = (row.get("furigana") or "").strip()
        furigana = normalize_furigana(furigana_raw)
        kind = (row.get("kind") or "").strip()
        report.total_by_kind[kind] += 1

        if furigana and strict_charset and not is_valid_reading(furigana):
            report.invalid_char_counter.update(invalid_chars(furigana))
            report.invalid_furigana_chars += 1
            furigana = ""

        structured = splitter.split(name)
        rec = Record(
            corporate_number=(row.get("corporate_number") or "").strip(),
            name_raw=name_raw,
            name=name,
            furigana_raw=furigana_raw,
            furigana=furigana,
            kind=kind,
            prefecture=(row.get("prefecture_name") or "").strip(),
            city=(row.get("city_name") or "").strip(),
            en_name=(row.get("en_name") or "").strip(),
            core=structured.core,
            prefix_form=structured.prefix_form,
            suffix_form=structured.suffix_form,
        )

        if not furigana:
            report.missing_furigana += 1
            report.missing_by_kind[kind] += 1
            missing.append(rec)
            continue

        core_reading = splitter.align_reading(structured, furigana)
        if core_reading is None:
            report.unaligned += 1
            # Unalignable rows are excluded from training, and are not
            # counted as missing-furigana either.
            continue
        rec.core_reading = core_reading
        rec.aligned = True
        labeled.append(rec)
        report.kept += 1

    return labeled, missing, report


# --- Persistence -----------------------------------------------------------
def write_jsonl(records: Iterable[Record | dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            d = rec.to_dict() if isinstance(rec, Record) else rec
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def records_from_jsonl(path: str | Path) -> list[Record]:
    fields = set(Record.__dataclass_fields__)
    return [Record(**{k: v for k, v in d.items() if k in fields}) for d in read_jsonl(path)]
