from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Multi-Head and Grouped-Query Attention with Rotary Position Embeddings and KV Cache.

    Supports:
    - Multi-Head Attention (MHA): num_kv_heads == n_heads
    - Grouped-Query Attention (GQA): n_heads % num_kv_heads == 0, num_kv_heads < n_heads
    - Multi-Query Attention (MQA): num_kv_heads == 1
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.num_kv_heads = config.kv_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.num_queries_per_kv = self.n_heads // self.num_kv_heads

        self.query = nn.Linear(config.emb_dim, config.emb_dim, bias=config.qkv_bias)
        self.key = nn.Linear(
            config.emb_dim, self.num_kv_heads * self.head_dim, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.emb_dim, self.num_kv_heads * self.head_dim, bias=config.qkv_bias
        )
        self.output = nn.Linear(
            config.emb_dim, config.emb_dim, bias=getattr(config, "out_proj_bias", True)
        )
        self.dropout = config.dropout
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.context_length,
            config.rope_theta,
            rope_scale=getattr(config, "rope_scale", 1.0),
            rope_type=getattr(config, "rope_type", "linear"),
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        batch_size, sequence_length, _ = x.shape

        # Compute Q, K, V projections
        query = self.query(x).view(
            batch_size, sequence_length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).view(
            batch_size, sequence_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        offset = 0
        if past_kv is not None:
            offset = past_kv[0].size(-2)

        # Apply RoPE to query and key with appropriate positional offset
        query, key = self.rope(query, key, offset=offset)

        # Update KV cache
        if past_kv is not None:
            key = torch.cat([past_kv[0], key], dim=-2)
            value = torch.cat([past_kv[1], value], dim=-2)

        new_kv = (key, value) if use_cache else None

        # Repeat KV heads for GQA/MQA if queries_per_kv > 1
        if self.num_queries_per_kv > 1:
            key_for_sdpa = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value_for_sdpa = torch.repeat_interleave(value, self.num_queries_per_kv, dim=1)
        else:
            key_for_sdpa = key
            value_for_sdpa = value

        # When incrementally decoding token-by-token (sequence_length == 1 with past),
        # the single query attends to all preceding keys (no future tokens to mask), so is_causal=False.
        # For full sequence training/prefill (sequence_length > 1), causal masking is required.
        is_causal = (sequence_length > 1)

        attended = F.scaled_dot_product_attention(
            query,
            key_for_sdpa,
            value_for_sdpa,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        out = self.output(attended)

        if use_cache:
            return out, new_kv
        return out
