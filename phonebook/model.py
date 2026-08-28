"""CharSeq2Seq: a character-level Transformer with a copy mechanism.

Claims supported: **copy fidelity** and **speed**.

Three design decisions:

1. **Character level with a shared vocabulary.** Proper nouns are a mass of
   out-of-vocabulary material; subword tokenization collapses the moment an
   unseen kanji bigram appears. At the character level the input is always
   representable, including on the hard set.
2. **Pointer-generator (copy) mechanism.** Trade names frequently already
   contain katakana or hiragana -- アルファ inside 株式会社アルファ電子 -- and
   that is a *transcription* problem, not an inference problem. Mixing the
   generation distribution with an attention distribution over input positions
   lets the model transcribe even unseen katakana runs verbatim. p_gen going to
   zero on those positions is direct evidence that copying was acquired as a
   mechanism rather than memorized.
3. **Small.** Running on a plain CPU is a precondition for the released
   artifact. The default is about 17M parameters (preset="small");
   preset="base" scales to about 130M.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import CharTokenizer


# --- Configuration ---------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 384
    nhead: int = 6
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_src_len: int = 64
    max_tgt_len: int = 96
    use_copy: bool = True
    tie_embeddings: bool = True

    @classmethod
    def preset(cls, name: str, vocab_size: int) -> "ModelConfig":
        """Three sizes, all intended to run on CPU.

        tiny  : for tests (~1M parameters)
        small : default (~17M parameters)
        base  : ~130M parameters, matching the 0.1-0.3B design target
        """
        presets = {
            "tiny": dict(d_model=128, nhead=4, num_encoder_layers=2,
                         num_decoder_layers=2, dim_feedforward=512),
            "small": dict(d_model=384, nhead=6, num_encoder_layers=4,
                          num_decoder_layers=4, dim_feedforward=1536),
            "base": dict(d_model=768, nhead=12, num_encoder_layers=8,
                         num_decoder_layers=8, dim_feedforward=3072),
        }
        if name not in presets:
            raise ValueError(f"unknown preset: {name} (choose from {sorted(presets)})")
        return cls(vocab_size=vocab_size, **presets[name])

    def to_dict(self) -> dict:
        return asdict(self)


# --- Layers ----------------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn)
        h = self.norm2(x)
        x = x + self.dropout(self.linear2(F.gelu(self.linear1(h))))
        return x


class DecoderLayer(nn.Module):
    """Causal self-attention + cross-attention. The cross-attention weights are
    returned so they can serve as the copy distribution."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        sa, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(sa)

        h = self.norm2(x)
        ca, attn_w = self.cross_attn(
            h, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        x = x + self.dropout(ca)

        h = self.norm3(x)
        x = x + self.dropout(self.linear2(F.gelu(self.linear1(h))))
        return x, attn_w


# --- Model -----------------------------------------------------------------
class CharSeq2Seq(nn.Module):
    """Character-level seq2seq from a trade-name core to its katakana reading."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.src_pos = nn.Embedding(cfg.max_src_len, cfg.d_model)
        self.tgt_pos = nn.Embedding(cfg.max_tgt_len, cfg.d_model)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.norm_enc = nn.LayerNorm(cfg.d_model)
        self.norm_dec = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.out_proj.weight = self.embed.weight
        if cfg.use_copy:
            # p_gen = sigmoid(w . [decoder state ; context ; previous input embedding])
            self.p_gen = nn.Linear(cfg.d_model * 3, 1)
        self.scale = math.sqrt(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- Encoding ----------------------------------------------------------
    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src: (B, S) -> (memory (B,S,D), src_pad_mask (B,S) where True = ignore)."""
        b, s = src.shape
        pad_mask = src.eq(0)
        pos = torch.arange(s, device=src.device).clamp_max(self.cfg.max_src_len - 1)
        x = self.embed(src) * self.scale + self.src_pos(pos)[None, :, :]
        for layer in self.encoder:
            x = layer(x, pad_mask)
        return self.norm_enc(x), pad_mask

    # -- Decoding ----------------------------------------------------------
    def decode(
        self,
        memory: torch.Tensor,
        src: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_in: torch.Tensor,
    ) -> torch.Tensor:
        """Decoding shared by teacher forcing and beam search. Returns **log
        probabilities** of shape (B, T, V).

        Copy mechanism: the final decoder layer's cross-attention (B,T,S) is
        scatter-added onto the input token ids to turn it into a distribution
        over the vocabulary, then mixed with the generation distribution by p_gen.
        """
        b, t = tgt_in.shape
        pos = torch.arange(t, device=tgt_in.device).clamp_max(self.cfg.max_tgt_len - 1)
        x = self.embed(tgt_in) * self.scale + self.tgt_pos(pos)[None, :, :]
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=tgt_in.device), diagonal=1
        )
        attn_w = None
        for layer in self.decoder:
            x, attn_w = layer(x, memory, causal, src_pad_mask)
        x = self.norm_dec(x)

        gen_logits = self.out_proj(x)  # (B,T,V)
        if not self.cfg.use_copy or attn_w is None:
            return F.log_softmax(gen_logits, dim=-1)

        gen_probs = F.softmax(gen_logits, dim=-1)
        attn = attn_w.masked_fill(src_pad_mask[:, None, :], 0.0)
        denom = attn.sum(-1, keepdim=True).clamp_min(1e-9)
        attn = attn / denom

        context = torch.bmm(attn, memory)  # (B,T,D)
        prev_emb = self.embed(tgt_in)
        p_gen = torch.sigmoid(self.p_gen(torch.cat([x, context, prev_emb], dim=-1)))  # (B,T,1)

        copy_probs = gen_probs.new_zeros(gen_probs.shape)
        index = src[:, None, :].expand(b, t, src.size(1))
        copy_probs.scatter_add_(2, index, attn)
        copy_probs[:, :, 0] = 0.0  # never copy onto PAD

        final = p_gen * gen_probs + (1.0 - p_gen) * copy_probs
        return torch.log(final.clamp_min(1e-12))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, pad_mask = self.encode(src)
        return self.decode(memory, src, pad_mask, tgt_in)

    def copy_gate(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Return p_gen, for diagnosing whether the copy mechanism is working.

        Claim supported: **copy fidelity**. A small p_gen (i.e. weight shifted
        to the copy side) at positions where the input is already katakana is
        direct evidence that transcription is implemented as a mechanism rather
        than memorized.
        """
        if not self.cfg.use_copy:
            raise RuntimeError("copy is disabled in this config")
        memory, pad_mask = self.encode(src)
        b, t = tgt_in.shape
        pos = torch.arange(t, device=tgt_in.device).clamp_max(self.cfg.max_tgt_len - 1)
        x = self.embed(tgt_in) * self.scale + self.tgt_pos(pos)[None, :, :]
        causal = torch.triu(torch.full((t, t), float("-inf"), device=tgt_in.device), diagonal=1)
        attn_w = None
        for layer in self.decoder:
            x, attn_w = layer(x, memory, causal, pad_mask)
        x = self.norm_dec(x)
        attn = attn_w.masked_fill(pad_mask[:, None, :], 0.0)
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
        context = torch.bmm(attn, memory)
        prev_emb = self.embed(tgt_in)
        return torch.sigmoid(self.p_gen(torch.cat([x, context, prev_emb], dim=-1))).squeeze(-1)

    # -- Persistence -------------------------------------------------------
    def save(self, directory: str | Path, tokenizer: CharTokenizer | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        (directory / "config.json").write_text(
            json.dumps(self.cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if tokenizer is not None:
            tokenizer.save(directory / "tokenizer.json")

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> tuple["CharSeq2Seq", CharTokenizer]:
        directory = Path(directory)
        cfg = ModelConfig(**json.loads((directory / "config.json").read_text(encoding="utf-8")))
        model = cls(cfg)
        state = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        tokenizer = CharTokenizer.load(directory / "tokenizer.json")
        return model, tokenizer


# --- LoRA (for the personal-name transfer experiment; not the main goal) ---
class LoRALinear(nn.Module):
    """Freeze an nn.Linear and train only a low-rank update.

    This does not support the main claim; it explores **transferability**.
    Reading personal names has prior work of its own (Namelti and others) and is
    not Phonebook's objective, so it is kept to a small LoRA transfer experiment
    rather than full training.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling

    def merge(self) -> nn.Linear:
        with torch.no_grad():
            self.base.weight += (self.lora_b @ self.lora_a) * self.scaling
        return self.base


def apply_lora(
    model: nn.Module, r: int = 8, alpha: int = 16, targets: tuple[str, ...] = ("linear1", "linear2")
) -> int:
    """Replace the targeted nn.Linear modules with LoRALinear; return the count."""
    replaced = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
                replaced += 1
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    return replaced


def merge_lora(model: nn.Module) -> None:
    """Fold the LoRA update back into the base weights (call before exporting)."""
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
