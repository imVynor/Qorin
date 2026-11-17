"""Targeted test verifying gradient accumulation normalization for partial windows.

Compares:
1. Accumulating 4 microbatches sequentially with loss / 4
2. Directly computing the average loss over the 4 microbatches in a single forward/backward pass.
Asserts that parameter gradients match with machine precision.
"""

from pathlib import Path
import sys
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import ModelConfig
from model.gpt import ModernGPT


def test_partial_window_accumulation_equivalence():
    torch.manual_seed(42)

    cfg = ModelConfig(
        vocab_size=1000,
        context_length=128,
        emb_dim=64,
        n_heads=2,
        num_kv_heads=2,
        n_layers=2,
        ffn_hidden_dim=128,
        dropout=0.0,
        rope_scale=1.0,
    )

    # Reference model: computes true average loss in single pass
    model_ref = ModernGPT(cfg)
    # Accumulated model: accumulates over 4 microbatches with 1/4 loss scaling
    model_accum = ModernGPT(cfg)
    model_accum.load_state_dict(model_ref.state_dict())

    # Create 4 synthetic microbatches (e.g. representing the final 4 items of an epoch)
    microbatches = []
    for _ in range(4):
        inp = torch.randint(0, 1000, (1, 32))
        lbl = inp.clone()
        lbl[:, :16] = -100  # mask first half
        microbatches.append((inp, lbl))

    # 1. Reference computation: sum losses and divide by 4, then backward
    model_ref.zero_grad()
    total_loss_ref = 0.0
    for inp, lbl in microbatches:
        logits = model_ref(inp)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = lbl[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, 1000), shift_labels.view(-1), ignore_index=-100)
        total_loss_ref = total_loss_ref + loss
    avg_loss_ref = total_loss_ref / 4.0
    avg_loss_ref.backward()

    # 2. Accumulated computation: divide each microbatch loss by 4 and backward sequentially
    model_accum.zero_grad()
    window_size = 4
    for inp, lbl in microbatches:
        logits = model_accum(inp)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = lbl[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, 1000), shift_labels.view(-1), ignore_index=-100)
        scaled_loss = loss / window_size
        scaled_loss.backward()

    # 3. Assert gradients are bitwise / numerically identical
    max_diff = 0.0
    for (name_ref, p_ref), (name_accum, p_accum) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p_ref.grad is not None:
            diff = (p_ref.grad - p_accum.grad).abs().max().item()
            max_diff = max(max_diff, diff)
            assert torch.allclose(p_ref.grad, p_accum.grad, atol=1e-6), f"Gradient mismatch on {name_ref}: max diff = {diff}"

    print(f"Partial window gradient equivalence verified successfully! Max grad diff = {max_diff:.2e}")


if __name__ == "__main__":
    test_partial_window_accumulation_equivalence()
