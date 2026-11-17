from dataclasses import dataclass


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

    def __post_init__(self):
        if self.emb_dim % self.n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        if (self.emb_dim // self.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str = "shakespeare.txt"
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
