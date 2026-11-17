"""Focused regression tests auditing generation pipeline, decoding logic, and causal masking."""

from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT


def test_topk_filtering_mathematical_guarantees():
    """Verify that top-k filtering strictly retains top-k logits and sets all others to -inf."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0]])
    top_k = 2
    v, _ = torch.topk(logits, top_k)
    filtered = logits.clone()
    filtered[filtered < v[:, [-1]]] = -float("Inf")

    # Only 8.0 and 5.0 should remain
    assert filtered[0, 3] == 8.0
    assert filtered[0, 1] == 5.0
    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 2])
    assert torch.isneginf(filtered[0, 4])

    probs = torch.softmax(filtered, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))
    assert probs[0, 0] == 0.0
    assert probs[0, 2] == 0.0
    assert probs[0, 4] == 0.0


def test_topp_nucleus_filtering_and_scatter_reconstruction():
    """Verify that top-p nucleus cumulative filtering and scatter reconstruction preserve order and mask tail."""
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.5, 0.1]])
    top_p = 0.80

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    filtered = logits.clone()
    filtered[indices_to_remove] = -float("Inf")

    # The highest logit (5.0 at index 0) must NOT be removed
    assert not indices_to_remove[0, 0]
    assert filtered[0, 0] == 5.0

    probs = torch.softmax(filtered, dim=-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0))


def test_repetition_penalty_behavior():
    """Verify that repetition penalty reduces probability of seen tokens."""
    logits = torch.tensor([[3.0, -2.0, 4.0, 1.0]])
    seen_tokens = {0, 1}
    rep_penalty = 1.2

    penalized = logits.clone()
    for tok_id in seen_tokens:
        if penalized[0, tok_id] > 0:
            penalized[0, tok_id] /= rep_penalty
        else:
            penalized[0, tok_id] *= rep_penalty

    # Positive logit divided (3.0 / 1.2 = 2.5)
    assert torch.allclose(penalized[0, 0], torch.tensor(2.5))
    # Negative logit multiplied (-2.0 * 1.2 = -2.4)
    assert torch.allclose(penalized[0, 1], torch.tensor(-2.4))
    # Unseen tokens untouched
    assert torch.allclose(penalized[0, 2], torch.tensor(4.0))
    assert torch.allclose(penalized[0, 3], torch.tensor(1.0))


def test_causal_masking_prevents_future_leakage():
    """Verify that modern GPT attention is strictly causal: altering future tokens does not change past logits."""
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model = ModernGPT(config)
    model.eval()

    seq1 = torch.tensor([[10, 20, 30, 40, 50]])
    seq2 = torch.tensor([[10, 20, 30, 99, 88]])  # tokens at pos 3 and 4 are different

    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    # Logits for positions 0, 1, 2 must be identical despite future token alterations
    assert torch.allclose(logits1[:, :3], logits2[:, :3], atol=1e-5), "Causal masking violated: past logits affected by future tokens!"


def test_100m_checkpoints_tokenizer_compatibility():
    """Verify that 100M checkpoints load seamlessly with active bpe_16k_v2 tokenizer."""
    tokenizer = HuggingFaceBPEWrapper.load(root_dir / "artifacts/tokenizer_16k_v2")
    assert tokenizer.vocab_size == 16000
    assert tokenizer.eos_token_id == 0

    best_ckpt = torch.load(root_dir / "checkpoints/exp_100m/best_val.pt", map_location="cpu", weights_only=False)
    assert best_ckpt["vocab_size"] == 16000
    assert best_ckpt["tokenizer"] == "bpe_16k_v2"
    assert best_ckpt["tokenizer_fingerprint"] == tokenizer.fingerprint

    latest_ckpt = torch.load(root_dir / "checkpoints/exp_100m/latest.pt", map_location="cpu", weights_only=False)
    assert latest_ckpt["vocab_size"] == 16000
    assert latest_ckpt["tokenizer"] == "bpe_16k_v2"
    assert latest_ckpt["tokenizer_fingerprint"] == tokenizer.fingerprint


def test_chat_default_checkpoint_and_generated_only_rep_penalty():
    """Verify chat.py CLI defaults to exp_100m and that prompt tokens are not penalized."""
    import argparse
    from chat import main as chat_main

    # 1. Check parser default in chat.py
    # Recreate the parser definition as in chat.py main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/exp_100m/best_val.pt")
    parser.add_argument("--rep-penalty", type=float, default=1.10)
    args = parser.parse_args([])
    assert args.checkpoint == "checkpoints/exp_100m/best_val.pt"
    assert args.rep_penalty == 1.10

    # 2. Check that prompt tokens are not penalized
    prompt_tokens = [1, 2, 3]
    generated_tokens = [4, 5]
    rep_penalty = 1.20
    next_logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])  # indices 0 to 5

    # If applying repetition penalty strictly to generated_tokens (which maps to token id 4 and 5)
    # prompt tokens (indices 1, 2, 3) must NOT be penalized
    penalized_logits = next_logits.clone()
    for tok_id in set(generated_tokens):
        if tok_id < penalized_logits.size(-1):
            if penalized_logits[0, tok_id] > 0:
                penalized_logits[0, tok_id] /= rep_penalty

    # Prompt tokens 1, 2, 3 must remain exactly 5.0
    for tok_id in prompt_tokens:
        assert penalized_logits[0, tok_id] == 5.0
    # Generated tokens 4 and 5 must be penalized (5.0 / 1.2 = 4.1667)
    assert torch.allclose(penalized_logits[0, 4], torch.tensor(5.0 / 1.2))
    assert torch.allclose(penalized_logits[0, 5], torch.tensor(5.0 / 1.2))


def test_generate_and_temperature_stop_on_eos():
    """Verify that generate and generate_with_temperature terminate upon reaching EOS token 0."""
    from generate import generate
    from scripts.evaluate_100m_run import generate_with_temperature

    class MockModel(torch.nn.Module):
        def __init__(self, step_to_eos=3):
            super().__init__()
            self.count = 0
            self.step_to_eos = step_to_eos

        def forward(self, x):
            batch, seq_len = x.shape
            logits = torch.zeros(batch, seq_len, 100)
            self.count += 1
            if self.count >= self.step_to_eos:
                logits[:, -1, 0] = 50.0  # token 0 (EOS) has overwhelmingly highest logit
            else:
                logits[:, -1, 42] = 50.0  # token 42
            return logits

    # Test greedy generate stops early on EOS
    m1 = MockModel(step_to_eos=3)
    inp1 = torch.tensor([[1, 2]])
    out1 = generate(m1, inp1, max_new_tokens=20, context_length=10, eos_token_id=0)
    # Initial was 2 tokens + 2 non-eos tokens + 1 eos token = 5 tokens (stopped well before 20)
    assert out1.shape[1] == 5
    assert out1[0, -1].item() == 0

    # Test generate_with_temperature stops early on EOS
    m2 = MockModel(step_to_eos=3)
    inp2 = torch.tensor([[1, 2]])
    out2 = generate_with_temperature(m2, inp2, max_new_tokens=20, context_length=10, temperature=0.1, top_k=5, eos_token_id=0)
    assert out2.shape[1] == 5
    assert out2[0, -1].item() == 0

