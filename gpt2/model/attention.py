import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.query = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.key = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.value = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.output = nn.Linear(config.emb_dim, config.emb_dim)
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(
            self.head_dim, config.context_length, config.rope_theta
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        query = self.query(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.rope(query, key)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.output(attended)
