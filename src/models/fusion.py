import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    """One pre-norm cross-attn block: text(Q) attends visual(KV), then residual."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feats, visual_feats):
        q  = self.norm_q(text_feats)
        kv = self.norm_kv(visual_feats)
        attn_out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        return text_feats + self.dropout(attn_out)


class CrossAttentionFusion(nn.Module):
    """
    Stack of `n_layers` cross-attention blocks: text (Query) attends visual
    (Key/Value). After the stack, a final LayerNorm normalises the fused output.

        text_feats   (B, T_q, D)
        text_mask    (B, T_q)
        visual_feats (B, T_v, D)
    →
        fused        (B, T_q, D)
        text_mask    (B, T_q)
    """

    def __init__(self, dim: int = 512, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CrossAttnBlock(dim=dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, text_feats, text_mask, visual_feats):
        x = text_feats
        for blk in self.blocks:
            x = blk(x, visual_feats)
        return self.norm_out(x), text_mask
