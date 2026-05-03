import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention: text (Query) attends over visual tokens (Key/Value).
    Input :
        text_feats   (B, T_q, D)
        text_mask    (B, T_q)        1 = keep, 0 = pad
        visual_feats (B, T_v, D)
    Output:
        fused        (B, T_q, D)
        text_mask    (B, T_q)        unchanged, returned for downstream cross-attn
    """

    def __init__(self, dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        visual_feats: torch.Tensor,
    ):
        q = self.norm_q(text_feats)
        kv = self.norm_kv(visual_feats)

        # All visual tokens are valid; key_padding_mask=None.
        attn_out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        fused = self.norm_out(text_feats + self.dropout(attn_out))
        return fused, text_mask
