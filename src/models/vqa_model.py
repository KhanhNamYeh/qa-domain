import torch
import torch.nn as nn

from .encoders import ImageEncoder, QuestionEncoder
from .fusion import CrossAttentionFusion
from .decoders import BaseDecoder


class VQAModel(nn.Module):
    """
    End-to-end VQA wrapper.

        pixel_values  ──► ImageEncoder    ──►   visual_feats  (B, 196, D)
        q_ids,q_mask  ──► QuestionEncoder ──►   text_feats    (B, T_q, D), text_mask
        text_feats × visual_feats ──► CrossAttentionFusion ──► fused (B, T_q, D)
        fused (memory) + answer_in ──► BaseDecoder ──► logits (B, T_a, V)

    Decoder is injected via constructor so A1 (LSTM), A2 (Transformer + LN+GELU),
    and A3 (Transformer + RMSNorm+SwiGLU) can be swapped without touching this class.
    """

    def __init__(
        self,
        image_encoder: ImageEncoder,
        question_encoder: QuestionEncoder,
        fusion: CrossAttentionFusion,
        decoder: BaseDecoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.question_encoder = question_encoder
        self.fusion = fusion
        self.decoder = decoder

    def encode(self, pixel_values, question_ids, question_mask):
        visual = self.image_encoder(pixel_values)                       # (B, 196, D)
        text, text_mask = self.question_encoder(question_ids, question_mask)
        memory, memory_mask = self.fusion(text, text_mask, visual)      # (B, T_q, D)
        return memory, memory_mask

    def forward(self, pixel_values, question_ids, question_mask, answer_in):
        memory, memory_mask = self.encode(pixel_values, question_ids, question_mask)
        return self.decoder(memory, memory_mask, answer_in)

    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        question_ids,
        question_mask,
        bos_id: int,
        eos_id: int,
        max_len: int = 64,
        beam_size: int = 1,
    ):
        memory, memory_mask = self.encode(pixel_values, question_ids, question_mask)
        return self.decoder.generate(memory, memory_mask, bos_id, eos_id, max_len, beam_size)
