from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .block import TransformerBlock
from .norm import RMSNorm


class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Tag residual projection layers for scaled initialization
        for block in self.blocks:
            block.attention.output._residual_proj = True
            block.mlp.down_proj._residual_proj = True

        # Initialize weights with explicit policy
        self.apply(self._init_weights)

        # Ensure embedding and lm_head remain strictly tied after init
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module):
        """Explicit initialization policy:
        - nn.Linear: N(0, 0.02)
          - Residual projection layers (attention.output, mlp.down_proj):
            scaled by 1 / sqrt(2 * n_layers) (GPT-2 / nanoGPT / Llama standard)
          - Biases (if present): 0.0
        - nn.Embedding: N(0, 0.02)
        - RMSNorm: 1.0
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_residual_proj", False):
                std = 0.02 / (2 * self.config.n_layers) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        x = self.dropout(self.token_embedding(input_ids))

        new_kv_cache = [] if use_cache else None
        use_grad_ckpt = getattr(self.config, "gradient_checkpointing", False) and self.training and not use_cache
        for i, block in enumerate(self.blocks):
            past_kv = kv_cache[i] if kv_cache is not None else None
            if use_cache:
                x, new_kv = block(x, past_kv=past_kv, use_cache=True)
                new_kv_cache.append(new_kv)
            elif use_grad_ckpt:
                x = torch.utils.checkpoint.checkpoint(block, x, past_kv, False, use_reentrant=False)
            else:
                x = block(x, past_kv=past_kv, use_cache=False)

        logits = self.lm_head(self.final_norm(x))
        if targets is None:
            if use_cache:
                return logits, new_kv_cache
            return logits

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def num_parameters(self, only_trainable: bool = False) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.parameter_count
