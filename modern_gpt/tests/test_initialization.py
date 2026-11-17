"""Tests for ModernGPT weight initialization policy, numerical stability, and checkpoint compatibility."""

import math
from pathlib import Path
import pytest
import torch

from model.config import ModelConfig, MODERN_GPT_16M, MODERN_GPT_50M
from model.gpt import ModernGPT


def test_initialization_distributions():
    """Verify weight distributions adhere to initialization policy."""
    torch.manual_seed(42)
    model = ModernGPT(MODERN_GPT_50M)

    # 1. Embedding initialization: N(0, 0.02)
    emb = model.token_embedding.weight
    assert abs(emb.mean().item()) < 0.005, f"Embedding mean too far from 0: {emb.mean().item()}"
    assert abs(emb.std().item() - 0.02) < 0.002, f"Embedding std not ~0.02: {emb.std().item()}"

    # 2. LM head weight is strictly tied
    assert model.lm_head.weight is model.token_embedding.weight

    # 3. RMSNorm weights initialized to 1.0
    assert torch.all(model.final_norm.weight == 1.0)
    for block in model.blocks:
        assert torch.all(block.attention_norm.weight == 1.0)
        assert torch.all(block.mlp_norm.weight == 1.0)

    # 4. Standard linear projections: N(0, 0.02)
    for block in model.blocks:
        for linear in [
            block.attention.query,
            block.attention.key,
            block.attention.value,
            block.mlp.gate_proj,
            block.mlp.up_proj,
        ]:
            assert abs(linear.weight.mean().item()) < 0.005
            assert abs(linear.weight.std().item() - 0.02) < 0.003
            if linear.bias is not None:
                assert torch.all(linear.bias == 0.0)

    # 5. Residual projection layers: N(0, 0.02 / sqrt(2 * n_layers))
    expected_residual_std = 0.02 / math.sqrt(2 * MODERN_GPT_50M.n_layers)
    for block in model.blocks:
        for res_linear in [block.attention.output, block.mlp.down_proj]:
            assert abs(res_linear.weight.mean().item()) < 0.003
            assert abs(res_linear.weight.std().item() - expected_residual_std) < 0.001
            if res_linear.bias is not None:
                assert torch.all(res_linear.bias == 0.0)


def test_fresh_model_loss_and_logits():
    """Verify initial cross-entropy matches theoretical ln(vocab_size) within 5%."""
    torch.manual_seed(1337)
    model = ModernGPT(MODERN_GPT_50M)
    model.eval()

    batch_size = 4
    seq_len = 128
    inputs = torch.randint(0, MODERN_GPT_50M.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, MODERN_GPT_50M.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, loss = model(inputs, targets)

    theoretical_loss = math.log(MODERN_GPT_50M.vocab_size)  # ln(16000) ~ 9.6803
    loss_val = loss.item()

    # Must be close to ~9.68, definitely NOT ~430 or ~17
    assert 9.0 < loss_val < 10.5, f"Initial loss {loss_val} deviates significantly from theoretical {theoretical_loss}"
    assert abs(loss_val - theoretical_loss) / theoretical_loss < 0.05

    # Check logits distribution
    assert abs(logits.mean().item()) < 0.1
    assert 0.2 < logits.std().item() < 1.0, f"Logits std {logits.std().item()} out of expected range [0.2, 1.0]"
    assert logits.min().item() > -10.0
    assert logits.max().item() < 10.0


def test_gradient_finiteness_and_norm():
    """Verify backward pass computes finite gradients with healthy norm."""
    torch.manual_seed(42)
    model = ModernGPT(MODERN_GPT_50M)
    model.train()

    inputs = torch.randint(0, MODERN_GPT_50M.vocab_size, (2, 64))
    targets = torch.randint(0, MODERN_GPT_50M.vocab_size, (2, 64))

    logits, loss = model(inputs, targets)
    loss.backward()

    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    assert 1.0 < total_norm < 100.0, f"Gradient norm {total_norm} unexpected"


def test_checkpoint_compatibility_with_16m():
    """Verify that existing 16M checkpoint loads cleanly and tied weights remain intact."""
    ckpt_path = Path("checkpoints/exp_100m/best_val.pt")
    if not ckpt_path.exists():
        pytest.skip(f"Historical checkpoint {ckpt_path} not found")

    model = ModernGPT(MODERN_GPT_16M)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    assert model.lm_head.weight is model.token_embedding.weight
    assert model.parameter_count == 16_768_128

