"""
Lite VQA wrapper that consumes precomputed (frozen) backbone features instead
of running SigLIP / PhoBERT each step. Only the trainable parts remain:
  - img_proj (768 -> 512)
  - txt_proj (768 -> 512)
  - cross-attention fusion
  - decoder (LSTM / Transformer)
"""

import torch
import torch.nn as nn

from .fusion import CrossAttentionFusion
from .decoders import BaseDecoder


class CachedVQAModel(nn.Module):
    def __init__(
        self,
        img_proj: nn.Linear,
        txt_proj: nn.Linear,
        fusion: CrossAttentionFusion,
        decoder: BaseDecoder,
    ):
        super().__init__()
        self.img_proj = img_proj
        self.txt_proj = txt_proj
        self.fusion = fusion
        self.decoder = decoder

    def encode(self, img_feat, txt_feat, txt_mask):
        v = self.img_proj(img_feat)                              # (B, 196, D)
        t = self.txt_proj(txt_feat)                              # (B, T_q, D)
        memory, mask = self.fusion(t, txt_mask, v)               # (B, T_q, D)
        return memory, mask

    def forward(self, img_feat, txt_feat, txt_mask, answer_in):
        memory, mask = self.encode(img_feat, txt_feat, txt_mask)
        return self.decoder(memory, mask, answer_in)

    @torch.no_grad()
    def generate(
        self,
        img_feat,
        txt_feat,
        txt_mask,
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
        beam_size: int = 1,
    ):
        memory, mask = self.encode(img_feat, txt_feat, txt_mask)
        return self.decoder.generate(memory, mask, bos_id, eos_id, max_len, beam_size)
