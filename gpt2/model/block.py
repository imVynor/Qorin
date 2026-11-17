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

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))
