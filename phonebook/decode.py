"""Constrained decoding, n-best, and inference orchestration.

Claims supported: **output charset guarantee**, **copy fidelity**, **speed**.

- Constrained decoding: beam search over a vocabulary restricted to katakana,
  the prolonged mark and EOS. "The output is katakana only" is therefore a
  property of the **search space**, not of training, and cannot be broken even
  by untrained weights. pytest verifies exactly that with a randomly
  initialized model.
- Copy path: when the trade-name core is already all kana, it is transcribed
  deterministically without touching the model. That is what makes "katakana
  input is preserved verbatim" a structural guarantee. The model's own
  pointer-generator does the same job inside cores that mix kanji and kana.
- Speed: decoding starts after the legal form has been stripped, which cuts the
  average generation length substantially.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from .kana import is_kana_text, is_valid_reading, to_katakana, CANNOT_START, PROLONGED_MARK
from .model import CharSeq2Seq
from .structure import StructuralSplitter, StructuredName
from .tokenizer import CharTokenizer

_LATIN_RE = re.compile(r"^[A-Za-z0-9&'\.\-\+ ]+$")


@dataclass
class Hypothesis:
    """One reading hypothesis. ``prob`` is the constrained sequence probability
    before calibration."""

    text: str
    logprob: float

    @property
    def prob(self) -> float:
        return float(torch.exp(torch.tensor(self.logprob)))


@dataclass
class Candidate:
    reading: str
    prob: float
    raw_logprob: float = 0.0


@dataclass
class ReadingResult:
    """One inference result.

    Attributes:
        name: The input, as given.
        candidates: Reading candidates for the whole corporate name, most
            probable first.
        confidence: Calibrated estimate of P(top-1 is correct).
        rejected: Whether the answer was rejected; if True the reading is None
            and the display value is "unknown".
        source: Which path produced it: copy / model / en2kana / segmented / rule.
        latency_ms: Inference time.
    """

    name: str
    structured: StructuredName
    candidates: list[Candidate] = field(default_factory=list)
    confidence: float = 0.0
    rejected: bool = False
    source: str = "model"
    latency_ms: float = 0.0

    @property
    def reading(self) -> Optional[str]:
        if self.rejected or not self.candidates:
            return None
        return self.candidates[0].reading

    @property
    def display(self) -> str:
        return self.reading if self.reading is not None else "unknown"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "reading": self.reading,
            "display": self.display,
            "confidence": round(self.confidence, 6),
            "rejected": self.rejected,
            "source": self.source,
            "latency_ms": round(self.latency_ms, 3),
            "candidates": [
                {"reading": c.reading, "prob": round(c.prob, 6)} for c in self.candidates
            ],
            "structure": {
                "prefix_form": self.structured.prefix_form,
                "core": self.structured.core,
                "suffix_form": self.structured.suffix_form,
                "position": self.structured.position,
            },
        }


class ConstrainedBeamSearch:
    """Batched beam search constrained to katakana and the prolonged mark."""

    def __init__(
        self,
        model: CharSeq2Seq,
        tokenizer: CharTokenizer,
        *,
        beam_size: int = 8,
        max_len: int = 40,
        temperature: float = 1.0,
        length_penalty: float = 0.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_len = max_len
        self.temperature = temperature
        self.length_penalty = length_penalty
        self._allowed = self._build_allowed_mask()

    def _build_allowed_mask(self) -> torch.Tensor:
        """A (2, V) boolean mask: row 0 for the first step, row 1 for the rest."""
        vocab = len(self.tokenizer)
        allowed = torch.zeros(2, vocab, dtype=torch.bool)
        for idx in self.tokenizer.output_char_ids():
            allowed[:, idx] = True
        # Never start with a small kana, the prolonged mark, or EOS (which
        # would allow an empty reading).
        for ch in CANNOT_START:
            if ch in self.tokenizer.stoi:
                allowed[0, self.tokenizer.stoi[ch]] = False
        allowed[0, self.tokenizer.eos_id] = False
        return allowed

    @torch.no_grad()
    def search(self, sources: Sequence[str], nbest: int = 3) -> list[list[Hypothesis]]:
        """Return the n-best readings for each trade-name core in the batch."""
        if not sources:
            return []
        tok, model = self.tokenizer, self.model
        device = next(model.parameters()).device
        k = max(self.beam_size, nbest)
        b = len(sources)

        maxlen_src = min(max(len(s) for s in sources) or 1, model.cfg.max_src_len)
        src = torch.full((b, maxlen_src), tok.pad_id, dtype=torch.long)
        for i, s in enumerate(sources):
            ids = tok.encode(s)[:maxlen_src]
            src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        src = src.to(device)

        memory, src_pad = model.encode(src)
        memory = memory.repeat_interleave(k, dim=0)
        src_pad = src_pad.repeat_interleave(k, dim=0)
        src_k = src.repeat_interleave(k, dim=0)

        tokens = torch.full((b * k, 1), tok.bos_id, dtype=torch.long, device=device)
        scores = torch.full((b, k), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)
        finished = torch.zeros(b * k, dtype=torch.bool, device=device)

        vocab = len(tok)
        allowed = self._allowed.to(device)
        neg_inf = torch.tensor(float("-inf"), device=device)

        for step in range(self.max_len):
            logp = model.decode(memory, src_k, src_pad, tokens)[:, -1, :]
            if self.temperature != 1.0:
                logp = F.log_softmax(logp / self.temperature, dim=-1)
            mask = allowed[0] if step == 0 else allowed[1]
            logp = torch.where(mask[None, :], logp, neg_inf)

            # Finished beams keep emitting PAD with probability 1, freezing
            # their score.
            if finished.any():
                frozen = torch.full((int(finished.sum()), vocab), float("-inf"), device=device)
                frozen[:, tok.pad_id] = 0.0
                logp[finished] = frozen

            cand = scores[:, None] + logp  # (B*K, V)
            cand = cand.view(b, k * vocab)
            top_scores, top_idx = cand.topk(k, dim=-1)
            beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")
            token_idx = top_idx % vocab

            flat_beam = (torch.arange(b, device=device)[:, None] * k + beam_idx).view(-1)
            tokens = torch.cat([tokens[flat_beam], token_idx.view(-1, 1)], dim=1)
            scores = top_scores.view(-1)
            finished = finished[flat_beam] | token_idx.view(-1).eq(tok.eos_id)
            if bool(finished.all()):
                break

        results: list[list[Hypothesis]] = []
        seq = tokens.view(b, k, -1).cpu()
        sc = scores.view(b, k).cpu()
        for i in range(b):
            hyps: list[Hypothesis] = []
            seen: set[str] = set()
            order = torch.argsort(sc[i], descending=True)
            for j in order.tolist():
                score = float(sc[i, j])
                if score == float("-inf"):
                    continue
                text = tok.decode(seq[i, j].tolist())
                if not text or text in seen:
                    continue
                seen.add(text)
                if self.length_penalty:
                    score = score / (max(len(text), 1) ** self.length_penalty)
                hyps.append(Hypothesis(text=text, logprob=score))
                if len(hyps) >= nbest:
                    break
            results.append(hyps)
        return results


def renormalize(hyps: Sequence[Hypothesis]) -> list[float]:
    """Renormalize probabilities within the n-best list (a calibration feature)."""
    if not hyps:
        return []
    scores = torch.tensor([h.logprob for h in hyps])
    return torch.softmax(scores, dim=0).tolist()


def script_segments(text: str) -> list[tuple[str, str]]:
    """Segment by script: a list of ("kana"|"latin"|"other", substring).

    Claim supported: **copy fidelity**. A run of kana is something to transcribe,
    not something to convert. Carving it out here preserves the kana regardless
    of what the model weights happen to be.
    """
    segs: list[tuple[str, str]] = []
    for ch in text:
        if is_kana_text(ch):
            kind = "kana"
        elif _LATIN_RE.match(ch):
            kind = "latin"
        else:
            kind = "other"
        if segs and segs[-1][0] == kind:
            segs[-1] = (kind, segs[-1][1] + ch)
        else:
            segs.append((kind, ch))
    return segs


class PhonebookReader:
    """Ties together StructuralSplitter, CharSeq2Seq, EnglishToKatakana,
    calibration and rejection."""

    def __init__(
        self,
        model: CharSeq2Seq,
        tokenizer: CharTokenizer,
        *,
        splitter: StructuralSplitter | None = None,
        en2kana=None,
        calibrator=None,
        beam_size: int = 8,
        max_len: int = 40,
        temperature: float = 1.0,
        threshold: float | None = None,
        segment_kana: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.splitter = splitter or StructuralSplitter()
        self.en2kana = en2kana
        self.calibrator = calibrator
        self.threshold = threshold
        self.segment_kana = segment_kana
        self.beam = ConstrainedBeamSearch(
            model, tokenizer, beam_size=beam_size, max_len=max_len, temperature=temperature
        )

    # -- Read trade-name cores only ----------------------------------------
    def read_cores(self, cores: Sequence[str], nbest: int = 3) -> list[list[Candidate]]:
        """Cores -> candidate lists. All-kana and all-Latin cores bypass the model."""
        out: list[list[Candidate]] = [[] for _ in cores]
        to_model: list[int] = []
        model_inputs: list[str] = []

        for i, core in enumerate(cores):
            if not core:
                out[i] = []
            elif is_kana_text(core):
                out[i] = [Candidate(reading=to_katakana(core), prob=1.0, raw_logprob=0.0)]
            elif self.en2kana is not None and _LATIN_RE.match(core):
                reading = self.en2kana.convert(core)
                out[i] = [Candidate(reading=reading, prob=1.0, raw_logprob=0.0)] if reading else []
            elif self.segment_kana and self._needs_segmentation(core):
                out[i] = self._read_segmented(core, nbest)
            else:
                to_model.append(i)
                model_inputs.append(core)

        if model_inputs:
            batches = self.beam.search(model_inputs, nbest=nbest)
            for idx, hyps in zip(to_model, batches):
                probs = renormalize(hyps)
                out[idx] = [
                    Candidate(reading=h.text, prob=p, raw_logprob=h.logprob)
                    for h, p in zip(hyps, probs)
                ]
        return out

    @staticmethod
    def _needs_segmentation(core: str) -> bool:
        kinds = {kind for kind, _ in script_segments(core)}
        return "kana" in kinds and len(kinds) > 1

    def _read_segmented(self, core: str, nbest: int) -> list[Candidate]:
        """Transcribe kana runs, route the rest to the model / English module, concatenate."""
        segs = script_segments(core)
        per_seg: list[list[Candidate]] = []
        model_positions: list[int] = []
        model_texts: list[str] = []
        for pos, (kind, text) in enumerate(segs):
            if kind == "kana":
                per_seg.append([Candidate(reading=to_katakana(text), prob=1.0)])
            elif kind == "latin" and self.en2kana is not None:
                per_seg.append([Candidate(reading=self.en2kana.convert(text), prob=1.0)])
            else:
                per_seg.append([])
                model_positions.append(pos)
                model_texts.append(text)
        if model_texts:
            for pos, hyps in zip(model_positions, self.beam.search(model_texts, nbest=nbest)):
                probs = renormalize(hyps)
                per_seg[pos] = [
                    Candidate(reading=h.text, prob=p, raw_logprob=h.logprob)
                    for h, p in zip(hyps, probs)
                ] or [Candidate(reading="", prob=1.0)]

        combos: list[Candidate] = [Candidate(reading="", prob=1.0, raw_logprob=0.0)]
        for cands in per_seg:
            nxt: list[Candidate] = []
            for base in combos:
                for c in cands:
                    nxt.append(
                        Candidate(
                            reading=base.reading + c.reading,
                            prob=base.prob * c.prob,
                            raw_logprob=base.raw_logprob + c.raw_logprob,
                        )
                    )
            combos = sorted(nxt, key=lambda c: -c.prob)[: max(nbest, 4)]
        return combos[:nbest]

    # -- Read the full corporate name --------------------------------------
    def read_batch(self, names: Sequence[str], nbest: int = 3) -> list[ReadingResult]:
        from .data import normalize_name

        start = time.perf_counter()
        # Normalize before splitting (absorbs half-width kana, full-width
        # alphanumerics, variant kanji). ReadingResult.name keeps the original.
        structs = [self.splitter.split(normalize_name(n) or n) for n in names]
        core_cands = self.read_cores([s.core for s in structs], nbest=nbest)
        trailing_cands = self.read_cores([s.trailing for s in structs], nbest=1)

        elapsed_ms = (time.perf_counter() - start) * 1000.0 / max(len(names), 1)
        results: list[ReadingResult] = []
        for name, st, cands, tr in zip(names, structs, core_cands, trailing_cands):
            trailing_reading = tr[0].reading if tr else ""
            full = [
                Candidate(
                    reading=st.compose(c.reading, trailing_reading),
                    prob=c.prob,
                    raw_logprob=c.raw_logprob,
                )
                for c in cands
            ]
            source = self._source_for(st.core)
            conf = self._confidence(cands, source)
            rejected = self.threshold is not None and conf < self.threshold
            results.append(
                ReadingResult(
                    name=name,
                    structured=st,
                    candidates=full,
                    confidence=conf,
                    rejected=rejected,
                    source=source,
                    latency_ms=elapsed_ms,
                )
            )
        return results

    def read(self, name: str, nbest: int = 3) -> ReadingResult:
        return self.read_batch([name], nbest=nbest)[0]

    def _source_for(self, core: str) -> str:
        if not core:
            return "rule"
        if is_kana_text(core):
            return "copy"
        if self.en2kana is not None and _LATIN_RE.match(core):
            return "en2kana"
        if self.segment_kana and self._needs_segmentation(core):
            return "segmented"
        return "model"

    def _confidence(self, cands: Sequence[Candidate], source: str) -> float:
        """Calibrated confidence; falls back to the renormalized n-best probability."""
        if not cands:
            return 0.0
        if source in ("copy", "rule"):
            return 1.0
        top = cands[0]
        if self.calibrator is not None:
            return float(self.calibrator.predict_one(cands))
        return float(top.prob)


def assert_katakana_only(results: Iterable[ReadingResult]) -> None:
    """Defensive runtime check that every output is katakana + prolonged mark."""
    for r in results:
        for c in r.candidates:
            if not is_valid_reading(c.reading):
                raise ValueError(f"non-katakana output detected: {r.name} -> {c.reading!r}")
