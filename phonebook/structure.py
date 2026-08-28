"""StructuralSplitter: separating the legal form from the trade-name core.

Claims supported: **unseen-entity performance** and **speed**.

Most of the characters in a Japanese corporate name belong to the legal form
(株式会社 "Co., Ltd.", 一般社団法人 "general incorporated association", ...).
Those have deterministic readings, so making a neural model predict them is
wasted capacity -- and worse, it inflates apparent accuracy: a model that has
merely memorized カブシキガイシャ looks good on any metric computed over the
full name. Stripping the legal form first buys three things at once:

  1. Evaluation measures only the trade-name core, so memorization and
     generalization become distinguishable.
  2. Decoding length shrinks, which makes CPU inference faster.
  3. Errors on the legal form become structurally impossible.

Handles prefixed and suffixed legal forms, parenthesized annotations, and
trailing branch/shop designations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from .kana import to_katakana

# --- Legal-form lexicon ---------------------------------------------------
# (surface form, canonical reading). Matched longest-first, so the table is
# sorted by descending length below. Readings follow the most frequent spelling
# observed in the NTA furigana data.
LEGAL_FORMS: dict[str, str] = {
    # Companies under the Companies Act
    "株式会社": "カブシキガイシャ",
    "有限会社": "ユウゲンガイシャ",
    "合同会社": "ゴウドウガイシャ",
    "合名会社": "ゴウメイガイシャ",
    "合資会社": "ゴウシガイシャ",
    "相互会社": "ソウゴガイシャ",
    "特定目的会社": "トクテイモクテキガイシャ",
    # Associations and foundations
    "一般社団法人": "イッパンシャダンホウジン",
    "一般財団法人": "イッパンザイダンホウジン",
    "公益社団法人": "コウエキシャダンホウジン",
    "公益財団法人": "コウエキザイダンホウジン",
    # Medical, welfare, education, religion
    "医療法人社団": "イリョウホウジンシャダン",
    "医療法人財団": "イリョウホウジンザイダン",
    "社会医療法人": "シャカイイリョウホウジン",
    "医療法人": "イリョウホウジン",
    "社会福祉法人": "シャカイフクシホウジン",
    "学校法人": "ガッコウホウジン",
    "準学校法人": "ジュンガッコウホウジン",
    "宗教法人": "シュウキョウホウジン",
    "更生保護法人": "コウセイホゴホウジン",
    "職業訓練法人": "ショクギョウクンレンホウジン",
    # Public and incorporated administrative agencies
    "独立行政法人": "ドクリツギョウセイホウジン",
    "地方独立行政法人": "チホウドクリツギョウセイホウジン",
    "国立大学法人": "コクリツダイガクホウジン",
    "公立大学法人": "コウリツダイガクホウジン",
    "国立研究開発法人": "コクリツケンキュウカイハツホウジン",
    # Licensed professional corporations
    "弁護士法人": "ベンゴシホウジン",
    "税理士法人": "ゼイリシホウジン",
    "司法書士法人": "シホウショシホウジン",
    "行政書士法人": "ギョウセイショシホウジン",
    "社会保険労務士法人": "シャカイホケンロウムシホウジン",
    "土地家屋調査士法人": "トチカオクチョウサシホウジン",
    "監査法人": "カンサホウジン",
    "特許業務法人": "トッキョギョウムホウジン",
    # Cooperatives and unions
    "特定非営利活動法人": "トクテイヒエイリカツドウホウジン",
    "農事組合法人": "ノウジクミアイホウジン",
    "事業協同組合": "ジギョウキョウドウクミアイ",
    "農業協同組合": "ノウギョウキョウドウクミアイ",
    "漁業協同組合": "ギョギョウキョウドウクミアイ",
    "森林組合": "シンリンクミアイ",
    "生活協同組合": "セイカツキョウドウクミアイ",
    "消費生活協同組合": "ショウヒセイカツキョウドウクミアイ",
    "企業組合": "キギョウクミアイ",
    "協業組合": "キョウギョウクミアイ",
    "商工組合": "ショウコウクミアイ",
    "管理組合法人": "カンリクミアイホウジン",
    "労働組合": "ロウドウクミアイ",
    # Financial institutions
    "信用金庫": "シンヨウキンコ",
    "信用組合": "シンヨウクミアイ",
    "信用協同組合": "シンヨウキョウドウクミアイ",
    "労働金庫": "ロウドウキンコ",
    "投資法人": "トウシホウジン",
}

#: Spelling variants -> canonical form. Absorbs ligatures such as ㈱ and the
#: parenthesized abbreviations such as （株）.
LEGAL_FORM_ALIASES: dict[str, str] = {
    "㈱": "株式会社",
    "(株)": "株式会社",
    "（株）": "株式会社",
    "㈲": "有限会社",
    "(有)": "有限会社",
    "（有）": "有限会社",
    "㈳": "社団法人",
    "(同)": "合同会社",
    "（同）": "合同会社",
    "㈴": "合名会社",
    "㈵": "合資会社",
    "(福)": "社会福祉法人",
    "（福）": "社会福祉法人",
    "(医)": "医療法人",
    "（医）": "医療法人",
    "(学)": "学校法人",
    "（学）": "学校法人",
    "(宗)": "宗教法人",
    "（宗）": "宗教法人",
}

#: Reading variants observed in the real data. Accepted during label alignment.
LEGAL_FORM_READING_VARIANTS: dict[str, tuple[str, ...]] = {
    "株式会社": ("カブシキガイシャ", "カブシキカイシャ"),
    "有限会社": ("ユウゲンガイシャ", "ユウゲンカイシャ"),
    "合同会社": ("ゴウドウガイシャ", "ゴウドウカイシャ"),
    "合名会社": ("ゴウメイガイシャ", "ゴウメイカイシャ"),
    "合資会社": ("ゴウシガイシャ", "ゴウシカイシャ"),
    "相互会社": ("ソウゴガイシャ", "ソウゴカイシャ"),
    "特定目的会社": ("トクテイモクテキガイシャ", "トクテイモクテキカイシャ"),
}

_SORTED_FORMS: list[str] = sorted(LEGAL_FORMS, key=len, reverse=True)

#: Parenthesized annotations (branch names, former names). Split off from the
#: core and handled separately.
_BRACKET_RE = re.compile(r"[（(【〔\[]([^）)】〕\]]*)[）)】〕\]]")


def readings_for(form: str) -> tuple[str, ...]:
    """Accepted readings for a legal form; the first entry is canonical."""
    if form in LEGAL_FORM_READING_VARIANTS:
        return LEGAL_FORM_READING_VARIANTS[form]
    canonical = LEGAL_FORMS.get(form)
    return (canonical,) if canonical else ()


@dataclass
class StructuredName:
    """A corporate name decomposed into prefix legal form + core + suffix.

    Attributes:
        original: The input exactly as given. The original spelling is always kept.
        prefix_form: Leading legal form (maezakabu), or None.
        suffix_form: Trailing legal form (atozakabu), or None.
        core: The trade-name core. Training and evaluation target only this.
        brackets: Text found inside parentheses (branch names, former names).
        trailing: Text left after a mid-string legal form, e.g. the branch part
            of 株式会社山田商店東京支店.
    """

    original: str
    core: str
    prefix_form: Optional[str] = None
    suffix_form: Optional[str] = None
    brackets: list[str] = field(default_factory=list)
    trailing: str = ""

    @property
    def legal_form(self) -> Optional[str]:
        return self.prefix_form or self.suffix_form

    @property
    def position(self) -> str:
        if self.prefix_form:
            return "prefix"  # legal form leads the name
        if self.suffix_form:
            return "suffix"  # legal form trails the name
        return "none"

    @property
    def prefix_reading(self) -> str:
        return LEGAL_FORMS.get(self.prefix_form or "", "")

    @property
    def suffix_reading(self) -> str:
        return LEGAL_FORMS.get(self.suffix_form or "", "")

    def compose(self, core_reading: str, trailing_reading: str = "") -> str:
        """Compose the reading of the whole name from the core reading."""
        return self.prefix_reading + core_reading + trailing_reading + self.suffix_reading

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position"] = self.position
        return d


class StructuralSplitter:
    """Rule-based splitter that hands the model only the trade-name core.

    Claim supported: **unseen-entity performance**. The legal form appears in
    the vast majority of records, so evaluating with it included makes it easy
    to reach the false conclusion that "even unseen entities score highly".
    Separating it is what makes the evaluation honest.
    """

    def __init__(self, forms: dict[str, str] | None = None) -> None:
        self.forms = dict(forms or LEGAL_FORMS)
        self._sorted = sorted(self.forms, key=len, reverse=True)

    # -- Surface preprocessing ---------------------------------------------
    @staticmethod
    def expand_aliases(name: str) -> str:
        """Expand ㈱ / （株） to their canonical form. Callers keep the original."""
        for alias, form in LEGAL_FORM_ALIASES.items():
            if alias in name:
                name = name.replace(alias, form)
        return name

    # -- Decomposition -----------------------------------------------------
    def split(self, name: str) -> StructuredName:
        """Decompose a corporate name.

        A leading legal form is tried first, then a trailing one, then a
        mid-string occurrence whose remainder becomes ``trailing``. If nothing
        matches, the whole input is the core.
        """
        original = name
        work = self.expand_aliases(name).strip()

        brackets = [m.group(1) for m in _BRACKET_RE.finditer(work) if m.group(1)]
        work = _BRACKET_RE.sub("", work).strip()

        prefix_form = None
        suffix_form = None
        trailing = ""

        for form in self._sorted:
            if work.startswith(form) and len(work) > len(form):
                prefix_form = form
                work = work[len(form):]
                break
        if prefix_form is None:
            for form in self._sorted:
                if work.endswith(form) and len(work) > len(form):
                    suffix_form = form
                    work = work[: -len(form)]
                    break
        if prefix_form is None and suffix_form is None:
            # Mid-string form, e.g. "<core>株式会社<branch>"
            for form in self._sorted:
                idx = work.find(form)
                if idx > 0:
                    suffix_form = form
                    trailing = work[idx + len(form):]
                    work = work[:idx]
                    break

        core = work.strip("　 ")
        if not core:
            # Degenerate case such as a name that is only "株式会社": never
            # produce an empty core, fall back to the original string.
            core = original
            prefix_form = suffix_form = None
            trailing = ""
        return StructuredName(
            original=original,
            core=core,
            prefix_form=prefix_form,
            suffix_form=suffix_form,
            brackets=brackets,
            trailing=trailing,
        )

    # -- Label-side alignment ----------------------------------------------
    def align_reading(self, structured: StructuredName, full_reading: str) -> Optional[str]:
        """Strip the legal-form reading off the full furigana to get the core reading.

        Returns None when it cannot be stripped -- i.e. the furigana and the
        surface disagree structurally -- and the caller must then drop that row
        from training. Being lenient here injects label noise straight into the
        claim about unseen entities, which is exactly the claim that has to hold up.
        """
        reading = to_katakana(full_reading).strip()
        if not reading:
            return None

        if structured.prefix_form:
            for cand in readings_for(structured.prefix_form):
                if reading.startswith(cand):
                    reading = reading[len(cand):]
                    break
            else:
                return None
        if structured.suffix_form:
            for cand in readings_for(structured.suffix_form):
                if reading.endswith(cand):
                    reading = reading[: -len(cand)]
                    break
            else:
                return None
        reading = reading.strip()
        return reading or None


#: Legal-form reading variants unified at scoring time (variant -> canonical).
_READING_CANON: dict[str, str] = {
    variant: variants[0]
    for variants in LEGAL_FORM_READING_VARIANTS.values()
    for variant in variants[1:]
}


def canonicalize_legal_reading(reading: str) -> str:
    """Unify legal-form reading variants, e.g. カブシキカイシャ -> カブシキガイシャ.

    Claim supported: **fairness of the comparison**. The NTA data genuinely
    contains both the rendaku and non-rendaku spellings. Every condition
    (existing G2P, LLM, Phonebook) is scored after the same normalization, so
    that this spelling variation cannot decide the comparison.
    """
    if not reading:
        return reading
    for variant, canon in _READING_CANON.items():
        if variant in reading:
            reading = reading.replace(variant, canon)
    return reading
