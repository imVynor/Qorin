"""Test suite for ModernGPT 50M architecture, causal masking, shapes, gradients, and checkpoint compatibility."""

import tempfile
from pathlib import Path
import pytest
import torch

from model.config import ModelConfig, MODERN_GPT_16M, MODERN_GPT_50M, MODERN_GPT_50M_WIDE, MODERN_GPT_50M_DEEP_THIN, MODERN_GPT_50M_GQA4
from model.gpt import ModernGPT


def test_exact_parameter_counts():
    """Verify exact parameter counts for 16M and 50M architectural presets."""
    m_16m = ModernGPT(MODERN_GPT_16M)
    assert m_16m.parameter_count == 16_768_128, f"Expected 16,768,128, got {m_16m.parameter_count}"

    m_50m = ModernGPT(MODERN_GPT_50M)
    assert m_50m.parameter_count == 49_105_408, f"Expected 49,105,408, got {m_50m.parameter_count}"

    m_50m_wide = ModernGPT(MODERN_GPT_50M_WIDE)
    assert m_50m_wide.parameter_count == 49_905_280, f"Expected 49,905,280, got {m_50m_wide.parameter_count}"

    m_50m_gqa4 = ModernGPT(MODERN_GPT_50M_GQA4)
    assert m_50m_gqa4.parameter_count == 45_959_680, f"Expected 45,959,680, got {m_50m_gqa4.parameter_count}"

    m_50m_deep_thin = ModernGPT(MODERN_GPT_50M_DEEP_THIN)
    assert m_50m_deep_thin.parameter_count == 48_936_384, f"Expected 48,936,384, got {m_50m_deep_thin.parameter_count}"

    # Verify tied embedding weights
    for m in [m_16m, m_50m, m_50m_wide, m_50m_gqa4, m_50m_deep_thin]:
        assert m.lm_head.weight is m.token_embedding.weight


def test_tensor_shapes():
    """Verify output shapes with and without targets for 50M configuration."""
    # Use smaller seq_len / batch for fast unit test
    cfg = ModelConfig(
        vocab_size=1000,
        context_length=64,
        emb_dim=512,
        n_heads=8,
        n_layers=2,  # 2 layers for fast shape check
        ffn_hidden_dim=1536,
        dropout=0.0,
    )
    model = ModernGPT(cfg)
    model.eval()

    batch_size = 3
    seq_len = 16
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Inference forward pass
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)

    # Training forward pass with targets
    targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    logits, loss = model(x, targets)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)
    assert loss.dim() == 0  # scalar loss
    assert torch.isfinite(loss)


def test_causal_masking():
    """Verify causal attention: changing future tokens does not affect earlier token representations."""
    cfg = ModelConfig(
        vocab_size=500,
        context_length=32,
        emb_dim=256,
        n_heads=4,
        n_layers=2,
        ffn_hidden_dim=512,
        dropout=0.0,
    )
    model = ModernGPT(cfg)
    model.eval()

    # Create sequence of length 8
    seq1 = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    # Modify only the last two tokens (positions 6 and 7)
    seq2 = torch.tensor([[10, 20, 30, 40, 50, 60, 99, 100]])

    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)

    # Positions 0 through 5 must be completely identical
    assert torch.allclose(out1[:, :6, :], out2[:, :6, :], atol=1e-6), (
        "Causal masking violation: earlier positions changed when future tokens were modified!"
    )
    # Position 6 and 7 should differ
    assert not torch.allclose(out1[:, 6:, :], out2[:, 6:, :], atol=1e-4)


def test_forward_backward_pass():
    """Verify backward pass computes finite gradients across all parameters."""
    cfg = ModelConfig(
        vocab_size=200,
        context_length=32,
        emb_dim=128,
        n_heads=4,
        n_layers=2,
        ffn_hidden_dim=256,
        dropout=0.0,
    )
    model = ModernGPT(cfg)
    model.train()

    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))

    _, loss = model(inputs, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"


def test_checkpoint_save_load():
    """Verify state_dict roundtrip preserves exact weights and outputs."""
    cfg = ModelConfig(
        vocab_size=500,
        context_length=32,
        emb_dim=256,
        n_heads=4,
        n_layers=2,
        ffn_hidden_dim=512,
        dropout=0.0,
    )
    model_orig = ModernGPT(cfg)
    model_orig.eval()

    inputs = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits_orig = model_orig(inputs)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save({"model": model_orig.state_dict(), "config": cfg}, tmp_path)

        state = torch.load(tmp_path, map_location="cpu", weights_only=False)
        model_loaded = ModernGPT(state["config"])
        model_loaded.load_state_dict(state["model"])
        model_loaded.eval()

        with torch.no_grad():
            logits_loaded = model_loaded(inputs)

        assert torch.equal(logits_orig, logits_loaded), "Loaded model outputs do not match original"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_compatibility_with_existing_16m_checkpoint():
    """Verify that existing 16M pretraining checkpoint loads cleanly into ModernGPT."""
    ckpt_path = Path("checkpoints/exp_100m/best_val.pt")
    if not ckpt_path.exists():
        pytest.skip(f"{ckpt_path} not found")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(
        vocab_size=state["vocab_size"],
        context_length=state["model_config"].get("context_length", 512),
        emb_dim=state["model_config"].get("emb_dim", 384),
        n_heads=state["model_config"].get("n_heads", 6),
        n_layers=state["model_config"].get("n_layers", 6),
        ffn_hidden_dim=state["model_config"].get("ffn_hidden_dim", 1024),
    )
    model = ModernGPT(cfg)
    model.load_state_dict(state["model"])
    model.eval()

    inputs = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (1, 32, cfg.vocab_size)
    assert torch.isfinite(logits).all()

