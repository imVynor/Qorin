from typing import Optional, Tuple
import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import SwiGLU
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.emb_dim)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.emb_dim)
        self.mlp = SwiGLU(config.emb_dim, config.ffn_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        normed = self.attention_norm(x)
        if use_cache:
            attn_out, new_kv = self.attention(normed, past_kv=past_kv, use_cache=True)
            x = x + attn_out
            x = x + self.mlp(self.mlp_norm(x))
            return x, new_kv
        else:
            x = x + self.attention(normed, past_kv=past_kv, use_cache=False)
            return x + self.mlp(self.mlp_norm(x))
