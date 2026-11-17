import math
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import hashlib

from model.config import ModelConfig, get_100m_gqa4_config, get_100m_gqa4_4k_config
from model.gpt import ModernGPT
from model.rope import RotaryEmbedding
from generate import generate


# ---------------------------------------------------------
# A. Configuration Tests
# ---------------------------------------------------------
def test_default_config_preservation():
    cfg = ModelConfig()
    assert cfg.context_length == 512
    assert cfg.rope_scale == 1.0
    assert cfg.rope_type == "linear"

    cfg_100m = get_100m_gqa4_config()
    assert cfg_100m.context_length == 512
    assert cfg_100m.rope_scale == 1.0
    assert cfg_100m.kv_heads == 4
    assert cfg_100m.emb_dim == 768


def test_4k_config_instantiation():
    cfg_4k = get_100m_gqa4_4k_config()
    assert cfg_4k.context_length == 4096
    assert cfg_4k.rope_scale == 8.0
    assert cfg_4k.rope_type == "linear"
    assert cfg_4k.emb_dim == 768
    assert cfg_4k.kv_heads == 4


def test_invalid_configurations_rejected():
    with pytest.raises(ValueError, match="rope_scale must be positive"):
        ModelConfig(rope_scale=0.0)
    with pytest.raises(ValueError, match="rope_scale must be positive"):
        ModelConfig(rope_scale=-2.0)
    with pytest.raises(ValueError, match="context_length must be positive"):
        ModelConfig(context_length=0)
    with pytest.raises(ValueError, match="Unsupported rope_type"):
        ModelConfig(rope_type="yarn_unsupported")


# ---------------------------------------------------------
# B. RoPE Tests
# ---------------------------------------------------------
def test_rope_linear_scaling_formula():
    head_dim = 64
    rope_orig = RotaryEmbedding(head_dim, 512, theta=10000.0, rope_scale=1.0)
    rope_4k = RotaryEmbedding(head_dim, 4096, theta=10000.0, rope_scale=8.0)

    # In linear interpolation, position p in 4k corresponds to position p / 8
    # For p = 80 in 4K, scaled position is 10.0, which matches p = 10 in rope_orig!
    cos_orig_pos10 = rope_orig.cos[:, :, 10, :]
    cos_4k_pos80 = rope_4k.cos[:, :, 80, :]
    assert torch.allclose(cos_orig_pos10, cos_4k_pos80, atol=1e-6)

    sin_orig_pos10 = rope_orig.sin[:, :, 10, :]
    sin_4k_pos80 = rope_4k.sin[:, :, 80, :]
    assert torch.allclose(sin_orig_pos10, sin_4k_pos80, atol=1e-6)


def test_rope_boundary_and_finiteness():
    head_dim = 64
    rope_4k = RotaryEmbedding(head_dim, 4096, theta=10000.0, rope_scale=8.0)
    assert not torch.isnan(rope_4k.cos).any()
    assert not torch.isinf(rope_4k.cos).any()
    assert not torch.isnan(rope_4k.sin).any()
    assert not torch.isinf(rope_4k.sin).any()

    # Test position 4095
    cos_last = rope_4k.cos[:, :, 4095, :]
    sin_last = rope_4k.sin[:, :, 4095, :]
    assert not torch.isnan(cos_last).any()
    assert not torch.isnan(sin_last).any()


def test_interleaved_rotation_preserved():
    # Verify interleaved rotation formula: [-x1, x0, -x3, x2]
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    cos = torch.ones_like(x)
    sin = torch.ones_like(x)
    # rotated = [-2, 1, -4, 3]
    # out = x * cos + rotated * sin = [1 - 2, 2 + 1, 3 - 4, 4 + 3] = [-1, 3, -1, 7]
    out = RotaryEmbedding._rotate(x, cos, sin)
    expected = torch.tensor([[[[-1.0, 3.0, -1.0, 7.0]]]])
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------
# C. Checkpoint Loading Tests
# ---------------------------------------------------------
def test_checkpoint_strict_load_and_parameter_preservation():
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints/modern_gpts_100m/best_val.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    # Verify SHA-256
    expected_hash = "bded509c44eabc2932dc1e293967bb0524acc31182fe9df0295b9ad2a4671e31"
    computed_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert computed_hash == expected_hash

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_4k = get_100m_gqa4_4k_config()
    model = ModernGPT(cfg_4k)

    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0
    assert model.parameter_count == 100401408
    assert model.blocks[0].attention.rope.cos.shape == (1, 1, 4096, 64)


# ---------------------------------------------------------
# D. Forward Pass Tests Across Sequence Lengths
# ---------------------------------------------------------
@pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
def test_forward_pass_across_context_lengths(seq_len):
    # Test on CPU with a mini model to ensure fast unit test execution
    cfg = ModelConfig(
        vocab_size=1000,
        context_length=4096,
        emb_dim=128,
        n_heads=4,
        num_kv_heads=2,
        n_layers=2,
        ffn_hidden_dim=256,
        rope_scale=8.0,
    )
    model = ModernGPT(cfg)
    model.eval()

    tokens = torch.randint(0, 1000, (1, seq_len))
    targets = torch.randint(0, 1000, (1, seq_len))

    with torch.no_grad():
        logits, loss = model(tokens, targets=targets)

    assert logits.shape == (1, seq_len, 1000)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


# ---------------------------------------------------------
# E. KV-Cache Equivalence Tests
# ---------------------------------------------------------
def test_kv_cache_equivalence():
    cfg = ModelConfig(
        vocab_size=500,
        context_length=1024,
        emb_dim=128,
        n_heads=4,
        num_kv_heads=2,
        n_layers=2,
        ffn_hidden_dim=256,
        rope_scale=2.0,
    )
    model = ModernGPT(cfg)
    model.eval()

    torch.manual_seed(42)
    # Prompt of 600 tokens (beyond 512)
    prompt = torch.randint(0, 500, (1, 600))

    # 1. Full sequence forward
    with torch.no_grad():
        full_logits = model(prompt)

    # 2. Prefill (first 550 tokens) then step-by-step cached generation for 50 tokens
    prefill_len = 550
    with torch.no_grad():
        prefill_logits, kv_cache = model(prompt[:, :prefill_len], use_cache=True)
        cached_logits_list = [prefill_logits]

        for step in range(prefill_len, 600):
            next_input = prompt[:, step : step + 1]
            step_logits, kv_cache = model(next_input, kv_cache=kv_cache, use_cache=True)
            cached_logits_list.append(step_logits)

        all_cached_logits = torch.cat(cached_logits_list, dim=1)

    # Compare logits
    max_diff = (full_logits - all_cached_logits).abs().max().item()
    assert max_diff < 1e-4, f"KV cache divergence too high: {max_diff}"


# ---------------------------------------------------------
# F. Regression Tests on Original 512 Context
# ---------------------------------------------------------
def test_original_512_regression():
    cfg_orig = get_100m_gqa4_config()
    assert cfg_orig.context_length == 512
    assert cfg_orig.rope_scale == 1.0

    torch.manual_seed(1234)
    model_orig = ModernGPT(cfg_orig)
    model_orig.eval()

    inputs = torch.randint(0, 16000, (1, 512))
    with torch.no_grad():
        logits_orig = model_orig(inputs)

    assert logits_orig.shape == (1, 512, 16000)
    assert not torch.isnan(logits_orig).any()

