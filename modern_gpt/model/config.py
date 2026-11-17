from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 512
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    ffn_hidden_dim: int = 1024
    dropout: float = 0.1
    qkv_bias: bool = False
    rope_theta: float = 10000.0
    num_kv_heads: Optional[int] = None
    gradient_checkpointing: bool = False
    rope_scale: float = 1.0
    rope_type: str = "linear"
    out_proj_bias: bool = True

    def __post_init__(self):
        if self.emb_dim % self.n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        if (self.emb_dim // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.rope_scale <= 0.0:
            raise ValueError("rope_scale must be positive")
        if self.rope_type not in ("linear", "default"):
            raise ValueError(f"Unsupported rope_type: {self.rope_type}. Must be 'linear' or 'default'")
        
        kv = self.num_kv_heads if self.num_kv_heads is not None else self.n_heads
        if kv <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.n_heads % kv != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by num_kv_heads ({kv})")

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.n_heads

    @classmethod
    def get_16m_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT-S 16M configuration (16,768,128 parameters with vocab_size=16000)."""
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=384,
            n_heads=6,
            n_layers=6,
            ffn_hidden_dim=1024,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=6,
        )

    @classmethod
    def get_50m_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1, num_kv_heads: Optional[int] = None) -> "ModelConfig":
        """ModernGPT 50M deep configuration (49,105,408 parameters for MHA with vocab_size=16000).
        12 layers, 8 heads, head_dim 64, emb_dim 512, SwiGLU ffn_hidden_dim 1536 (3.0x).
        Supports MHA (8 KV heads), GQA-4 (4 KV heads), GQA-2 (2 KV heads), and MQA (1 KV head).
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=512,
            n_heads=8,
            n_layers=12,
            ffn_hidden_dim=1536,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=num_kv_heads,
        )

    @classmethod
    def get_50m_gqa4_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT 50M with GQA-4 (4 KV heads, 45,959,680 parameters)."""
        return cls.get_50m_config(vocab_size=vocab_size, context_length=context_length, dropout=dropout, num_kv_heads=4)

    @classmethod
    def get_50m_gqa2_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT 50M with GQA-2 (2 KV heads, 44,386,816 parameters)."""
        return cls.get_50m_config(vocab_size=vocab_size, context_length=context_length, dropout=dropout, num_kv_heads=2)

    @classmethod
    def get_50m_mqa_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT 50M with MQA (1 KV head, 43,600,384 parameters)."""
        return cls.get_50m_config(vocab_size=vocab_size, context_length=context_length, dropout=dropout, num_kv_heads=1)

    @classmethod
    def get_50m_deep_thin_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT-S v2 Deep-and-Thin configuration (48,936,384 parameters with vocab_size=16000).
        16 layers, 7 heads, head_dim 64, emb_dim 448, SwiGLU ffn_hidden_dim 1344 (3.0x), 7 KV heads.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=448,
            n_heads=7,
            n_layers=16,
            ffn_hidden_dim=1344,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=7,
        )

    @classmethod
    def get_50m_wide_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1) -> "ModelConfig":
        """ModernGPT 50M wide configuration (49,905,280 parameters with vocab_size=16000).
        8 layers, 10 heads, head_dim 64, emb_dim 640, SwiGLU ffn_hidden_dim 1728 (2.7x).
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=640,
            n_heads=10,
            n_layers=8,
            ffn_hidden_dim=1728,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=10,
        )

    @classmethod
    def get_100m_gqa4_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1, gradient_checkpointing: bool = False) -> "ModelConfig":
        """ModernGPT-S 100M balanced configuration (100,401,408 parameters with vocab_size=16000).
        14 layers, 12 heads, head_dim 64, emb_dim 768, SwiGLU ffn_hidden_dim 2048 (2.67x), 4 KV heads.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=768,
            n_heads=12,
            n_layers=14,
            ffn_hidden_dim=2048,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=4,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    def get_100m_gqa4_4k_config(
        cls,
        vocab_size: int = 16000,
        context_length: int = 4096,
        rope_scale: float = 8.0,
        rope_type: str = "linear",
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> "ModelConfig":
        """ModernGPT-S 100M 4K context configuration with Linear RoPE Interpolation.
        14 layers, 12 heads, head_dim 64, emb_dim 768, SwiGLU ffn_hidden_dim 2048 (2.67x), 4 KV heads.
        Target context 4096 tokens, rope_scale 8.0.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=768,
            n_heads=12,
            n_layers=14,
            ffn_hidden_dim=2048,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=4,
            gradient_checkpointing=gradient_checkpointing,
            rope_scale=rope_scale,
            rope_type=rope_type,
        )

    @classmethod
    def get_100m_deep_gqa4_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1, gradient_checkpointing: bool = False) -> "ModelConfig":
        """ModernGPT-S 100M deep candidate configuration (112,988,928 parameters with vocab_size=16000).
        16 layers, 12 heads, head_dim 64, emb_dim 768, SwiGLU ffn_hidden_dim 2048 (2.67x), 4 KV heads.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=768,
            n_heads=12,
            n_layers=16,
            ffn_hidden_dim=2048,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=4,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    def get_100m_wide_gqa4_config(cls, vocab_size: int = 16000, context_length: int = 512, dropout: float = 0.1, gradient_checkpointing: bool = False) -> "ModelConfig":
        """ModernGPT-S 100M wide candidate configuration (94,891,776 parameters with vocab_size=16000).
        12 layers, 12 heads, head_dim 64, emb_dim 768, SwiGLU ffn_hidden_dim 2304 (3.0x), 4 KV heads.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=768,
            n_heads=12,
            n_layers=12,
            ffn_hidden_dim=2304,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=4,
            gradient_checkpointing=gradient_checkpointing,
        )


    @classmethod
    def get_500m_config(
        cls,
        vocab_size: int = 32768,
        context_length: int = 2048,
        dropout: float = 0.0,
        gradient_checkpointing: bool = True,
        rope_scale: float = 1.0,
        rope_type: str = "default",
    ) -> "ModelConfig":
        """ModernGPT-S 500M production configuration (514,389,504 parameters).
        20 layers, 24 query heads, 4 KV heads (6:1 GQA), head_dim 64, emb_dim 1536,
        SwiGLU ffn_hidden_dim 3840 (2.5x), linear biases disabled, tied embeddings.
        """
        return cls(
            vocab_size=vocab_size,
            context_length=context_length,
            emb_dim=1536,
            n_heads=24,
            n_layers=20,
            ffn_hidden_dim=3840,
            dropout=dropout,
            qkv_bias=False,
            rope_theta=10000.0,
            num_kv_heads=4,
            gradient_checkpointing=gradient_checkpointing,
            rope_scale=rope_scale,
            rope_type=rope_type,
            out_proj_bias=False,
        )


MODERN_GPT_16M = ModelConfig.get_16m_config()
MODERN_GPT_50M = ModelConfig.get_50m_config()
MODERN_GPT_50M_GQA4 = ModelConfig.get_50m_gqa4_config()
MODERN_GPT_50M_GQA2 = ModelConfig.get_50m_gqa2_config()
MODERN_GPT_50M_MQA = ModelConfig.get_50m_mqa_config()
MODERN_GPT_50M_DEEP_THIN = ModelConfig.get_50m_deep_thin_config()
MODERN_GPT_50M_WIDE = ModelConfig.get_50m_wide_config()
MODERN_GPT_100M_GQA4 = ModelConfig.get_100m_gqa4_config()
MODERN_GPT_100M_GQA4_4K = ModelConfig.get_100m_gqa4_4k_config()
MODERN_GPT_100M_DEEP_GQA4 = ModelConfig.get_100m_deep_gqa4_config()
MODERN_GPT_100M_WIDE_GQA4 = ModelConfig.get_100m_wide_gqa4_config()

# Module-level alias & convenience functions
ModernGPTConfig = ModelConfig
get_100m_gqa4_config = ModelConfig.get_100m_gqa4_config
get_100m_gqa4_4k_config = ModelConfig.get_100m_gqa4_4k_config
get_100m_deep_gqa4_config = ModelConfig.get_100m_deep_gqa4_config
get_100m_wide_gqa4_config = ModelConfig.get_100m_wide_gqa4_config


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str = "../gpt2/shakespeare.txt"
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 2
    stride: int = 512
    train_ratio: float = 0.9
    num_epochs: int = 10
    max_tokens: int = 0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    min_learning_rate: float = 4e-5
    warmup_steps: int = 100
    decay_steps: int = 10000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 5
    seed: int = 123
    compile_model: bool = False
    resume: str = ""
