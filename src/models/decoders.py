import math
from abc import ABC, abstractmethod
from typing import Optional, Type

import torch
import torch.nn as nn

from .blocks import VanillaFFN


class BaseDecoder(nn.Module, ABC):
    """
    Common interface shared by LSTMDecoder and TransformerDecoder.

    forward(memory, memory_mask, answer_in) -> logits  (B, T_out, V)
        Teacher Forcing training step.
    generate(memory, memory_mask, bos_id, eos_id, max_len, beam_size=1)
        Auto-regressive inference; greedy when beam_size == 1.
    """

    def __init__(self, vocab_size: int, pad_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    @abstractmethod
    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        answer_in: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def generate(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
        beam_size: int = 1,
    ) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# A1 — LSTM decoder
# ---------------------------------------------------------------------------
class LSTMDecoder(BaseDecoder):
    """
    Multi-layer LSTMCell decoder (stack of `num_layers`).

    At each step, the output of layer ℓ feeds layer ℓ+1; only the top layer's
    hidden state is mapped to vocab. Context is the mean-pooled memory (kept
    simple — same as before, but now compounded across `num_layers` cells).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        memory_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(vocab_size, pad_id)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # First cell takes [embed; ctx]; deeper cells take previous layer's h.
        cells = []
        for layer in range(num_layers):
            in_dim = (embed_dim + memory_dim) if layer == 0 else hidden_dim
            cells.append(nn.LSTMCell(in_dim, hidden_dim))
        self.cells = nn.ModuleList(cells)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, ref: torch.Tensor):
        B = ref.size(0)
        device = ref.device
        h = [torch.zeros(B, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, device=device) for _ in range(self.num_layers)]
        return h, c

    @staticmethod
    def _pooled_context(memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        mask = memory_mask.unsqueeze(-1).to(memory.dtype)
        summed = (memory * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _step(self, x_in, h, c):
        """Run all `num_layers` cells once. Returns updated h/c lists."""
        x = x_in
        for layer in range(self.num_layers):
            h_l, c_l = self.cells[layer](x, (h[layer], c[layer]))
            h[layer] = h_l
            c[layer] = c_l
            x = self.dropout(h_l) if layer < self.num_layers - 1 else h_l
        return h, c

    def forward(self, memory, memory_mask, answer_in):
        ctx = self._pooled_context(memory, memory_mask)
        emb = self.embed(answer_in)
        h, c = self._init_state(ctx)

        logits_steps = []
        for t in range(emb.size(1)):
            x_t = torch.cat([emb[:, t, :], ctx], dim=-1)
            h, c = self._step(x_t, h, c)
            logits_steps.append(self.out(self.dropout(h[-1])))
        return torch.stack(logits_steps, dim=1)

    @torch.no_grad()
    def generate(self, memory, memory_mask, bos_id, eos_id, max_len=64, beam_size=1):
        B = memory.size(0)
        device = memory.device
        ctx = self._pooled_context(memory, memory_mask)
        h, c = self._init_state(ctx)

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embed(tokens[:, -1])
            x_t = torch.cat([emb, ctx], dim=-1)
            h, c = self._step(x_t, h, c)
            logits = self.out(h[-1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, self.pad_id), next_tok)
            tokens = torch.cat([tokens, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break
        return tokens[:, 1:]


# ---------------------------------------------------------------------------
# A2 / A3 — Transformer decoder
# ---------------------------------------------------------------------------
class _MultiheadSelfAttn(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class _MultiheadCrossAttn(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, q, kv, key_padding_mask=None):
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class TransformerDecoderBlock(nn.Module):
    """
    One decoder block:
        x = x + self_attn (norm(x), causal mask)
        x = x + cross_attn(norm(x), memory)
        x = x + ffn(norm(x))

    Pre-Norm style. `norm_cls` and `ffn_cls` are injected so the same block code
    powers both A2 (LayerNorm + VanillaFFN) and A3 (RMSNorm + SwiGLU).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        norm_cls: Type[nn.Module] = nn.LayerNorm,
        ffn_cls: Type[nn.Module] = VanillaFFN,
    ):
        super().__init__()
        self.norm1 = norm_cls(dim)
        self.self_attn = _MultiheadSelfAttn(dim, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = norm_cls(dim)
        self.cross_attn = _MultiheadCrossAttn(dim, n_heads, dropout)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = norm_cls(dim)
        self.ffn = ffn_cls(dim, ffn_dim, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask, memory_key_padding_mask=None):
        x = x + self.drop1(self.self_attn(self.norm1(x), attn_mask=causal_mask))
        x = x + self.drop2(self.cross_attn(self.norm2(x), memory, key_padding_mask=memory_key_padding_mask))
        x = x + self.drop3(self.ffn(self.norm3(x)))
        return x


class _SinusoidalPosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerDecoder(BaseDecoder):
    """
    Stack of TransformerDecoderBlock + output projection.

    Configurations:
        A2 — norm_cls=LayerNorm, ffn_cls=VanillaFFN
        A3 — norm_cls=RMSNorm,   ffn_cls=SwiGLU
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 64,
        norm_cls: Type[nn.Module] = nn.LayerNorm,
        ffn_cls: Type[nn.Module] = VanillaFFN,
    ):
        super().__init__(vocab_size, pad_id)
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos = _SinusoidalPosEnc(dim, max_len=max_len + 8)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    dim=dim,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    norm_cls=norm_cls,
                    ffn_cls=ffn_cls,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = norm_cls(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def _decode_step(self, ans_in, memory, memory_key_padding_mask):
        x = self.drop(self.pos(self.embed(ans_in)))
        T = x.size(1)
        causal = self._causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, memory, causal_mask=causal, memory_key_padding_mask=memory_key_padding_mask)
        x = self.norm_out(x)
        return self.out(x)

    def forward(self, memory, memory_mask, answer_in):
        # nn.MultiheadAttention's key_padding_mask: True == position is padded (ignored)
        kpm = memory_mask == 0 if memory_mask is not None else None
        return self._decode_step(answer_in, memory, kpm)

    @torch.no_grad()
    def generate(self, memory, memory_mask, bos_id, eos_id, max_len=64, beam_size=1):
        B = memory.size(0)
        device = memory.device
        kpm = memory_mask == 0 if memory_mask is not None else None

        if beam_size <= 1:
            tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_len):
                logits = self._decode_step(tokens, memory, kpm)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, self.pad_id), next_tok)
                tokens = torch.cat([tokens, next_tok], dim=1)
                finished = finished | (next_tok.squeeze(1) == eos_id)
                if finished.all():
                    break
            return tokens[:, 1:]

        # Beam search (per-sample, simple implementation)
        return self._beam_search(memory, kpm, bos_id, eos_id, max_len, beam_size)

    @torch.no_grad()
    def _beam_search(self, memory, kpm, bos_id, eos_id, max_len, beam_size):
        B = memory.size(0)
        device = memory.device
        outputs = []

        for i in range(B):
            mem_i = memory[i : i + 1].expand(beam_size, -1, -1).contiguous()
            kpm_i = None if kpm is None else kpm[i : i + 1].expand(beam_size, -1).contiguous()

            seqs = torch.full((beam_size, 1), bos_id, dtype=torch.long, device=device)
            scores = torch.zeros(beam_size, device=device)
            scores[1:] = float("-inf")  # only beam 0 is alive at step 0
            finished = torch.zeros(beam_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                logits = self._decode_step(seqs, mem_i, kpm_i)[:, -1, :]
                logp = torch.log_softmax(logits, dim=-1)         # (beam, V)
                logp = logp.masked_fill(finished.unsqueeze(1), 0.0)
                cand = scores.unsqueeze(1) + logp                # (beam, V)
                flat = cand.view(-1)
                topv, topi = flat.topk(beam_size)
                beam_idx = topi // self.vocab_size
                tok_idx = topi % self.vocab_size

                seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
                scores = topv
                finished = finished[beam_idx] | (tok_idx == eos_id)
                if finished.all():
                    break

            best = scores.argmax().item()
            outputs.append(seqs[best, 1:])

        max_T = max(o.size(0) for o in outputs)
        padded = torch.full((B, max_T), self.pad_id, dtype=torch.long, device=device)
        for i, o in enumerate(outputs):
            padded[i, : o.size(0)] = o
        return padded
