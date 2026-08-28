"""EnglishToKatakana: converting Latin letters and digits inside trade names.

Claim supported: **unseen-entity performance** -- specifically, closing a hole in it.

Latin script is common in Japanese trade names (株式会社ABC SYSTEM). General
Japanese G2P tends to drop it silently or read it as romaji, which means the
unseen-entity evaluation gets contaminated by failures that have nothing to do
with kanji. This module carves the Latin segments out and handles them on a
separate path, so the claim "robust to unseen kanji bigrams" is not muddied by
English.

Three stages:
  1. Lexicon (EN_LEXICON): for words frequent in company names, a hard-coded
     reading is simply the most accurate option.
  2. Acronym detection: short all-caps strings without vowels are spelled out
     (NEC -> エヌイーシー).
  3. Rule-based conversion: longest-match grapheme rules with a syllable model.

If higher accuracy is needed, the same CharSeq2Seq can be trained on
English-to-katakana pairs and dropped in (`EnglishToKatakana(model_dir=...)`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .kana import PROLONGED_MARK, is_valid_reading, strip_invalid

# --- Lexicon ---------------------------------------------------------------
#: English words frequent in company names. Precision over coverage: restricted
#: to high-frequency items.
EN_LEXICON: dict[str, str] = {
    "SYSTEM": "システム", "SYSTEMS": "システムズ", "SERVICE": "サービス",
    "SERVICES": "サービシズ", "SOLUTION": "ソリューション", "SOLUTIONS": "ソリューションズ",
    "TECH": "テック", "TECHNO": "テクノ", "TECHNOLOGY": "テクノロジー",
    "TECHNOLOGIES": "テクノロジーズ", "HOLDINGS": "ホールディングス", "HOLDING": "ホールディング",
    "GROUP": "グループ", "PARTNERS": "パートナーズ", "PARTNER": "パートナー",
    "CORPORATION": "コーポレーション", "CORP": "コープ", "COMPANY": "カンパニー",
    "INTERNATIONAL": "インターナショナル", "GLOBAL": "グローバル", "JAPAN": "ジャパン",
    "TOKYO": "トウキョウ", "OSAKA": "オオサカ", "NIPPON": "ニッポン", "NIHON": "ニホン",
    "DESIGN": "デザイン", "OFFICE": "オフィス", "STUDIO": "スタジオ", "WORKS": "ワークス",
    "WORK": "ワーク", "LAB": "ラボ", "LABO": "ラボ", "LABS": "ラボ",
    "RESEARCH": "リサーチ", "ENGINEERING": "エンジニアリング", "ENGINEER": "エンジニア",
    "CONSULTING": "コンサルティング", "CONSULTANT": "コンサルタント",
    "TRADING": "トレーディング", "TRADE": "トレード", "PLANNING": "プランニング",
    "PROJECT": "プロジェクト", "NETWORK": "ネットワーク", "NETWORKS": "ネットワークス",
    "DATA": "データ", "DIGITAL": "デジタル", "MEDIA": "メディア", "MOBILE": "モバイル",
    "SOFT": "ソフト", "SOFTWARE": "ソフトウェア", "HOUSE": "ハウス", "HOME": "ホーム",
    "HOMES": "ホームズ", "LIFE": "ライフ", "CARE": "ケア", "MEDICAL": "メディカル",
    "PHARMA": "ファーマ", "FOOD": "フード", "FOODS": "フーズ", "FARM": "ファーム",
    "GREEN": "グリーン", "BLUE": "ブルー", "WHITE": "ホワイト", "BLACK": "ブラック",
    "STAR": "スター", "SUN": "サン", "SKY": "スカイ", "SEA": "シー", "CITY": "シティ",
    "TOWN": "タウン", "PLAZA": "プラザ", "CENTER": "センター", "CENTRE": "センター",
    "FIRST": "ファースト", "NEXT": "ネクスト", "NEW": "ニュー", "ONE": "ワン",
    "PLUS": "プラス", "PRO": "プロ", "MAX": "マックス", "MINI": "ミニ",
    "AUTO": "オート", "MOTOR": "モーター", "MOTORS": "モーターズ", "LOGISTICS": "ロジスティクス",
    "TRANSPORT": "トランスポート", "EXPRESS": "エクスプレス", "LINE": "ライン",
    "STEEL": "スチール", "METAL": "メタル", "PAPER": "ペーパー", "PRINT": "プリント",
    "PRINTING": "プリンティング", "ELECTRIC": "エレクトリック", "ELECTRONICS": "エレクトロニクス",
    "ENERGY": "エナジー", "POWER": "パワー", "WATER": "ウォーター", "AQUA": "アクア",
    "CLEAN": "クリーン", "SAFETY": "セーフティ", "SECURITY": "セキュリティ",
    "BANK": "バンク", "CAPITAL": "キャピタル", "FINANCE": "ファイナンス",
    "INSURANCE": "インシュアランス", "REAL": "リアル", "ESTATE": "エステート",
    "BUILD": "ビルド", "BUILDING": "ビルディング", "CONSTRUCTION": "コンストラクション",
    "ENTERPRISE": "エンタープライズ", "INDUSTRY": "インダストリー", "INDUSTRIES": "インダストリーズ",
    "MANUFACTURING": "マニュファクチャリング", "PRODUCTS": "プロダクツ", "PRODUCT": "プロダクト",
    "MARKET": "マーケット", "MARKETING": "マーケティング", "SHOP": "ショップ",
    "STORE": "ストア", "SALES": "セールス", "SUPPORT": "サポート", "SYSTEMSOLUTION": "システムソリューション",
    "AND": "アンド", "THE": "ザ", "OF": "オブ", "FOR": "フォー",
}

#: Letter-by-letter readings, for acronyms
LETTER_READINGS: dict[str, str] = {
    "A": "エー", "B": "ビー", "C": "シー", "D": "ディー", "E": "イー", "F": "エフ",
    "G": "ジー", "H": "エイチ", "I": "アイ", "J": "ジェー", "K": "ケー", "L": "エル",
    "M": "エム", "N": "エヌ", "O": "オー", "P": "ピー", "Q": "キュー", "R": "アール",
    "S": "エス", "T": "ティー", "U": "ユー", "V": "ブイ", "W": "ダブリュー",
    "X": "エックス", "Y": "ワイ", "Z": "ゼット",
}

#: Digit readings when adjacent to letters (3M -> スリーエム)
DIGIT_EN: dict[str, str] = {
    "0": "ゼロ", "1": "ワン", "2": "ツー", "3": "スリー", "4": "フォー",
    "5": "ファイブ", "6": "シックス", "7": "セブン", "8": "エイト", "9": "ナイン",
}

_JP_DIGITS = ["", "イチ", "ニ", "サン", "ヨン", "ゴ", "ロク", "ナナ", "ハチ", "キュウ"]


def japanese_number(n: int) -> str:
    """Render 0..99,999,999 as a Japanese numeral reading in katakana,
    including the euphonic changes (300 -> サンビャク, 800 -> ハッピャク)."""
    if n == 0:
        return "ゼロ"
    if n >= 100000000:
        return "".join(DIGIT_EN[d] for d in str(n))

    def under_10000(x: int) -> str:
        out = ""
        sen, x = divmod(x, 1000)
        if sen:
            out += {1: "セン", 3: "サンゼン", 8: "ハッセン"}.get(sen, _JP_DIGITS[sen] + "セン")
        hyaku, x = divmod(x, 100)
        if hyaku:
            out += {1: "ヒャク", 3: "サンビャク", 6: "ロッピャク", 8: "ハッピャク"}.get(
                hyaku, _JP_DIGITS[hyaku] + "ヒャク"
            )
        juu, ichi = divmod(x, 10)
        if juu:
            out += "ジュウ" if juu == 1 else _JP_DIGITS[juu] + "ジュウ"
        if ichi:
            out += _JP_DIGITS[ichi]
        return out

    man, rest = divmod(n, 10000)
    out = ""
    if man:
        out += ("イチマン" if man == 1 else under_10000(man) + "マン")
    if rest:
        out += under_10000(rest)
    return out


# --- Rule-based conversion -------------------------------------------------
#: Onset consonant x vowel row -> kana, following loanword conventions.
CV_TABLE: dict[str, tuple[str, str, str, str, str]] = {
    "":   ("ア", "イ", "ウ", "エ", "オ"),
    "k":  ("カ", "キ", "ク", "ケ", "コ"),
    "c":  ("カ", "シ", "ク", "セ", "コ"),
    "g":  ("ガ", "ギ", "グ", "ゲ", "ゴ"),
    "s":  ("サ", "シ", "ス", "セ", "ソ"),
    "z":  ("ザ", "ジ", "ズ", "ゼ", "ゾ"),
    "t":  ("タ", "ティ", "トゥ", "テ", "ト"),
    "d":  ("ダ", "ディ", "ドゥ", "デ", "ド"),
    "n":  ("ナ", "ニ", "ヌ", "ネ", "ノ"),
    "h":  ("ハ", "ヒ", "フ", "ヘ", "ホ"),
    "b":  ("バ", "ビ", "ブ", "ベ", "ボ"),
    "p":  ("パ", "ピ", "プ", "ペ", "ポ"),
    "m":  ("マ", "ミ", "ム", "メ", "モ"),
    "y":  ("ヤ", "イ", "ユ", "イエ", "ヨ"),
    "r":  ("ラ", "リ", "ル", "レ", "ロ"),
    "l":  ("ラ", "リ", "ル", "レ", "ロ"),
    "w":  ("ワ", "ウィ", "ウ", "ウェ", "ウォ"),
    "f":  ("ファ", "フィ", "フ", "フェ", "フォ"),
    "v":  ("ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"),
    "j":  ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "x":  ("クサ", "クシ", "クス", "クセ", "クソ"),
    "q":  ("クア", "クイ", "ク", "クエ", "クオ"),
    "ch": ("チャ", "チ", "チュ", "チェ", "チョ"),
    "sh": ("シャ", "シ", "シュ", "シェ", "ショ"),
    "th": ("サ", "シ", "ス", "セ", "ソ"),
    "ph": ("ファ", "フィ", "フ", "フェ", "フォ"),
    "wh": ("ワ", "ウィ", "フ", "ウェ", "ホ"),
    "gh": ("ガ", "ギ", "グ", "ゲ", "ゴ"),
    "ts": ("ツァ", "ツィ", "ツ", "ツェ", "ツォ"),
    "qu": ("クア", "クイ", "ク", "クエ", "クオ"),
    "ck": ("カ", "キ", "ク", "ケ", "コ"),
    "dge": ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "tch": ("チャ", "チ", "チュ", "チェ", "チョ"),
    "sch": ("シャ", "シ", "シュ", "シェ", "ショ"),
    "ng": ("ンガ", "ンギ", "ング", "ンゲ", "ンゴ"),
}

#: Vowel-letter group -> (row, trailing kana). The row indexes CV_TABLE a/i/u/e/o.
_ROWS = "aiueo"
VOWEL_MAP: dict[str, tuple[str, str]] = {
    "a": ("a", ""), "i": ("i", ""), "u": ("a", ""), "e": ("e", ""), "o": ("o", ""),
    "ee": ("i", "ー"), "ea": ("i", "ー"), "ie": ("i", "ー"), "ei": ("e", "イ"),
    "ey": ("i", "ー"), "ai": ("e", "イ"), "ay": ("e", "イ"), "oo": ("u", "ー"),
    "ou": ("a", "ウ"), "ow": ("o", "ウ"), "oa": ("o", "ー"), "oi": ("o", "イ"),
    "oy": ("o", "イ"), "au": ("o", "ー"), "aw": ("o", "ー"), "eu": ("u", "ー"),
    "ue": ("u", "ー"), "ui": ("u", "ー"), "y": ("i", "ー"), "eau": ("o", "ー"),
    "igh": ("a", "イ"), "ough": ("o", "ー"),
}

#: Syllable-final / word-final consonant -> kana
CODA_MAP: dict[str, str] = {
    "b": "ブ", "c": "ク", "ck": "ック", "d": "ド", "f": "フ", "g": "グ", "h": "",
    "j": "ジ", "k": "ク", "l": "ル", "m": "ム", "n": "ン", "p": "プ", "q": "ク",
    "r": "", "s": "ス", "t": "ト", "v": "ヴ", "w": "", "x": "クス", "y": "イ",
    "z": "ズ", "ch": "チ", "sh": "シュ", "th": "ス", "ph": "フ", "ng": "ング",
    "gh": "", "wh": "ホ", "ts": "ツ", "qu": "ク", "dge": "ッジ", "tch": "ッチ",
    "sch": "シュ",
}

_CONSONANT_DIGRAPHS = ("sch", "tch", "dge", "ch", "sh", "th", "ph", "wh", "ck", "ng", "qu", "ts", "gh")
_VOWEL_GROUPS = sorted(VOWEL_MAP, key=len, reverse=True)
_VOWELS = set("aeiou")

# --- Romaji (very common in Japanese company names) ------------------------
#: Hepburn syllable table. The Kunrei forms ti/tu/si/hu/zi are deliberately
#: **excluded**: allowing them would misread English words (time, tune, site) as
#: romaji, so restricting the table to Hepburn is itself the discriminator
#: between English and romaji.
_ROMAJI_TABLE: dict[str, str] = {
    "a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ",
    "ka": "カ", "ki": "キ", "ku": "ク", "ke": "ケ", "ko": "コ",
    "sa": "サ", "shi": "シ", "su": "ス", "se": "セ", "so": "ソ",
    "ta": "タ", "chi": "チ", "tsu": "ツ", "te": "テ", "to": "ト",
    "na": "ナ", "ni": "ニ", "nu": "ヌ", "ne": "ネ", "no": "ノ",
    "ha": "ハ", "hi": "ヒ", "fu": "フ", "he": "ヘ", "ho": "ホ",
    "ma": "マ", "mi": "ミ", "mu": "ム", "me": "メ", "mo": "モ",
    "ya": "ヤ", "yu": "ユ", "yo": "ヨ",
    "ra": "ラ", "ri": "リ", "ru": "ル", "re": "レ", "ro": "ロ",
    "wa": "ワ", "wo": "ヲ", "n": "ン",
    "ga": "ガ", "gi": "ギ", "gu": "グ", "ge": "ゲ", "go": "ゴ",
    "za": "ザ", "ji": "ジ", "zu": "ズ", "ze": "ゼ", "zo": "ゾ",
    "da": "ダ", "de": "デ", "do": "ド",
    "ba": "バ", "bi": "ビ", "bu": "ブ", "be": "ベ", "bo": "ボ",
    "pa": "パ", "pi": "ピ", "pu": "プ", "pe": "ペ", "po": "ポ",
    "kya": "キャ", "kyu": "キュ", "kyo": "キョ",
    "gya": "ギャ", "gyu": "ギュ", "gyo": "ギョ",
    "sha": "シャ", "shu": "シュ", "sho": "ショ",
    "ja": "ジャ", "ju": "ジュ", "jo": "ジョ",
    "cha": "チャ", "chu": "チュ", "cho": "チョ",
    "nya": "ニャ", "nyu": "ニュ", "nyo": "ニョ",
    "hya": "ヒャ", "hyu": "ヒュ", "hyo": "ヒョ",
    "bya": "ビャ", "byu": "ビュ", "byo": "ビョ",
    "pya": "ピャ", "pyu": "ピュ", "pyo": "ピョ",
    "mya": "ミャ", "myu": "ミュ", "myo": "ミョ",
    "rya": "リャ", "ryu": "リュ", "ryo": "リョ",
}
_ROMAJI_KEYS = sorted(_ROMAJI_TABLE, key=len, reverse=True)
_ROMAJI_RE = re.compile(r"^[a-z]+$")


def looks_like_romaji(word: str) -> bool:
    """Whether a word looks like Japanese written in romaji.

    Claim supported: **unseen-entity performance**. Latin text in Japanese trade
    names is more often romaji than English (MIRAI, SAKURA, MIDORI). Applying
    English rules to those fails systematically, so romaji is tried first.

    The primary condition is that the word parses completely under Hepburn;
    on top of that, spellings characteristic of English are excluded. A false
    positive (treating an English word as romaji) degrades the output more than
    a false negative, so the exclusions are deliberately conservative.
    """
    w = word.lower()
    if len(w) < 4 or not _ROMAJI_RE.match(w):
        return False
    if set(w) & set("lqvx"):
        return False
    if w[-1] not in _VOWELS and w[-1] != "n":
        return False
    # English silent e (make, sunrise, code): consonant + final e is not romaji
    if w.endswith("e") and len(w) >= 2 and w[-2] not in _VOWELS:
        return False
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if ch == "c" and nxt != "h":
            return False
        if ch == "f" and nxt != "u":
            return False
        if ch == "w" and nxt not in ("a", "o"):
            return False
        if ch == "j" and nxt not in ("a", "i", "u", "o"):
            return False
        if ch == "y" and nxt not in ("a", "u", "o"):
            return False
    if "ee" in w or "oo" in w:
        return False
    return romaji_to_kana(w) is not None


def romaji_to_kana(word: str) -> Optional[str]:
    """Convert Hepburn romaji to katakana; None if it does not parse."""
    w = word.lower()
    out: list[str] = []
    i = 0
    while i < len(w):
        if i + 1 < len(w) and w[i] == w[i + 1] and w[i] not in _VOWELS and w[i] != "n":
            out.append("ッ")
            i += 1
            continue
        for key in _ROMAJI_KEYS:
            if w.startswith(key, i):
                if key == "n" and i + 1 < len(w) and (w[i + 1] in _VOWELS or w[i + 1] == "y"):
                    continue
                out.append(_ROMAJI_TABLE[key])
                i += len(key)
                break
        else:
            return None
    return "".join(out)


_VOWEL_LETTERS = set("AEIOU")

_TOKEN_RE = re.compile(r"[A-Za-z]+|[0-9]+|[^A-Za-z0-9]+")


class EnglishToKatakana:
    """Convert strings of Latin letters and digits to katakana.

    Args:
        lexicon: Extra dictionary entries (company-specific readings).
        model_dir: Directory of a CharSeq2Seq trained on English-to-katakana
            pairs. When given, words absent from the lexicon are converted by
            the model instead of by rules.
    """

    def __init__(self, lexicon: dict[str, str] | None = None, model_dir: str | Path | None = None) -> None:
        self.lexicon = dict(EN_LEXICON)
        if lexicon:
            self.lexicon.update({k.upper(): v for k, v in lexicon.items()})
        self._reader = None
        if model_dir is not None:
            self._reader = self._load_model(model_dir)

    @staticmethod
    def _load_model(model_dir: str | Path):
        from .decode import PhonebookReader
        from .model import CharSeq2Seq

        model, tokenizer = CharSeq2Seq.load(model_dir)
        return PhonebookReader(model, tokenizer, segment_kana=False)

    @classmethod
    def from_lexicon_file(cls, path: str | Path) -> "EnglishToKatakana":
        """Build from a JSON dictionary of the form {"ABC": "エービーシー", ...}."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(lexicon=data)

    # -- Detection ---------------------------------------------------------
    @staticmethod
    def looks_like_acronym(word: str) -> bool:
        """Acronym-likeness: all caps with no vowels, or at most three letters."""
        if not word.isupper():
            return False
        if len(word) <= 3:
            return True
        return not (set(word) & _VOWEL_LETTERS)

    # -- Conversion --------------------------------------------------------
    def convert_word(self, word: str) -> str:
        """One word to katakana: lexicon -> acronym -> model -> romaji -> English rules."""
        upper = word.upper()
        if upper in self.lexicon:
            return self.lexicon[upper]
        if self.looks_like_acronym(upper):
            return "".join(LETTER_READINGS.get(ch, "") for ch in upper)
        if self._reader is not None:
            res = self._reader.read(upper, nbest=1)
            if res.reading and is_valid_reading(res.reading):
                return res.reading
        if looks_like_romaji(upper):
            kana = romaji_to_kana(upper)
            if kana:
                return kana
        return self.transliterate(upper)

    @staticmethod
    def transliterate(word: str) -> str:
        """Katakana from English spelling via syllable analysis.

        Splits into onset consonant cluster + vowel group + coda. This fits the
        Japanese syllable structure -- every consonant gets a vowel -- far
        better than plain character substitution. Includes magic-e
        (make -> メイク) and r-colouring (car -> カー).
        """
        w = word.lower()
        out: list[str] = []
        i = 0
        n = len(w)
        last_short = False  # was the previous syllable a short vowel? (gates final gemination)

        def match_consonant(pos: int) -> str:
            for dg in _CONSONANT_DIGRAPHS:
                if w.startswith(dg, pos):
                    return dg
            return w[pos] if pos < n and w[pos] not in _VOWELS else ""

        while i < n:
            ch = w[i]
            if ch not in _VOWELS and ch not in "y":
                cons = match_consonant(i)
                if not cons:
                    i += 1
                    continue
                nxt = i + len(cons)
                # A doubled consonant becomes a geminate (small tsu)
                if nxt < n and w[nxt] == cons and len(cons) == 1 and cons not in "aeiounr":
                    out.append("ッ")
                    i = nxt
                    continue
                vowel, vlen, tail = _read_vowel(w, nxt)
                if vowel is None:
                    coda = CODA_MAP.get(cons, "")
                    # Short vowel + final voiceless stop geminates (planet -> プラネット)
                    if nxt >= n and cons in ("t", "k", "p") and out and last_short:
                        coda = "ッ" + coda
                    out.append(coda)
                    i = nxt
                    continue
                row = _ROWS.index(vowel)
                table = CV_TABLE.get(cons)
                if table is None:
                    out.append(CODA_MAP.get(cons, ""))
                    i = nxt
                    continue
                out.append(table[row] + tail)
                last_short = tail == ""
                i = nxt + vlen
            else:
                vowel, vlen, tail = _read_vowel(w, i)
                if vowel is None:
                    i += 1
                    continue
                out.append(CV_TABLE[""][_ROWS.index(vowel)] + tail)
                last_short = tail == ""
                i += vlen
        text = "".join(out)
        text = text.replace("ーー", "ー")
        return strip_invalid(text)

    def convert(self, text: str) -> str:
        """Convert a whole string of Latin letters, digits and punctuation."""
        if not text:
            return ""
        out: list[str] = []
        tokens = _TOKEN_RE.findall(text)
        for idx, tok in enumerate(tokens):
            if tok.isdigit():
                neighbours = tokens[max(0, idx - 1) : idx] + tokens[idx + 1 : idx + 2]
                alpha_adjacent = any(t.isalpha() for t in neighbours)
                if alpha_adjacent and len(tok) <= 2:
                    out.append("".join(DIGIT_EN[d] for d in tok))
                else:
                    out.append(japanese_number(int(tok)))
            elif tok.isalpha():
                out.append(self.convert_word(tok))
            else:
                # Punctuation and whitespace carry no reading
                continue
        return strip_invalid("".join(out))


def _read_vowel(word: str, pos: int) -> tuple[Optional[str], int, str]:
    """Read a vowel group starting at pos. Returns (row, chars consumed, tail kana).

    Handles magic-e (a final e lengthens or diphthongizes the preceding vowel)
    and r-colouring.
    """
    n = len(word)
    if pos >= n:
        return None, 0, ""
    if word[pos] not in _VOWELS and word[pos] != "y":
        return None, 0, ""

    # r-colouring: vowel + r + (consonant or end of word)
    for group in _VOWEL_GROUPS:
        if not word.startswith(group, pos):
            continue
        end = pos + len(group)
        if len(group) == 1 and end < n and word[end] == "r" and (
            end + 1 >= n or word[end + 1] not in _VOWELS
        ):
            row = "o" if group == "o" else "a"
            return row, len(group) + 1, "ー"
        # magic-e: single vowel + one consonant + word-final e
        if len(group) == 1 and end + 1 < n and word[end] not in _VOWELS and word[end + 1] == "e" and end + 2 == n:
            magic = {"a": ("e", "イ"), "i": ("a", "イ"), "o": ("o", "ー"),
                     "u": ("u", "ー"), "e": ("i", "ー")}
            row, tail = magic[group]
            return row, 1, tail
        # Silent word-final e (words of three letters or more)
        if group == "e" and end == n and n >= 3 and word[pos - 1] not in _VOWELS:
            return None, 1, ""
        row, tail = VOWEL_MAP[group]
        return row, len(group), tail
    return None, 0, ""
