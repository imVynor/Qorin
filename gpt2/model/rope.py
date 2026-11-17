import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inverse_frequency)
        self.register_buffer(
            "cos", torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
        )
        self.register_buffer(
            "sin", torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
        )

    def forward(self, query, key):
        sequence_length = query.size(-2)
        cos = self.cos[:, :, :sequence_length].to(dtype=query.dtype)
        sin = self.sin[:, :, :sequence_length].to(dtype=query.dtype)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)

    @staticmethod
    def _rotate(x, cos, sin):
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        return x * cos + rotated * sin
