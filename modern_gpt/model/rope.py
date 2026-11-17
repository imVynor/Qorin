import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10000.0, rope_scale=1.0, rope_type="linear"):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope_scale = float(rope_scale)
        self.rope_type = rope_type

        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        # Position interpolation: p' = p / rope_scale
        positions = torch.arange(max_seq_len).float() / self.rope_scale
        angles = torch.outer(positions, inverse_frequency)
        # Register buffer (persistent by default for backward compatibility with existing tests/checkpoints)
        self.register_buffer(
            "cos", torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
        )
        self.register_buffer(
            "sin", torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Custom loader that safely adapts or skips persistent cos/sin buffers if shapes differ."""
        for name in ("cos", "sin"):
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                buf = getattr(self, name)
                if val.shape != buf.shape:
                    # Shape mismatch (e.g. 512 in checkpoint vs 4096 in model):
                    # We copy the buffer reference to state_dict to satisfy strict checking
                    state_dict[key] = buf
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, query, key, offset=0):
        sequence_length = query.size(-2)
        req_len = offset + sequence_length
        if req_len > self.cos.size(-2):
            inverse_frequency = 1.0 / (
                self.theta ** (torch.arange(0, self.head_dim, 2, device=query.device).float() / self.head_dim)
            )
            # Maintain exact position interpolation for dynamic expansion
            positions = torch.arange(req_len, device=query.device).float() / self.rope_scale
            angles = torch.outer(positions, inverse_frequency)
            cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
            sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
        else:
            cos = self.cos[:, :, offset : offset + sequence_length]
            sin = self.sin[:, :, offset : offset + sequence_length]

        cos = cos.to(dtype=query.dtype, device=query.device)
        sin = sin.to(dtype=query.dtype, device=query.device)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)

    @staticmethod
    def _rotate(x, cos, sin):
        rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        return x * cos + rotated * sin
