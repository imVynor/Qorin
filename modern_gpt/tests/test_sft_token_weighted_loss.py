"""Rigorously tests token-weighted SFT loss calculation, gradient accumulation equivalence,
zero-token handling, variable padding, and validation partition invariance for ModernGPT-S.
"""

import copy
import math
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import ModelConfig
from model.gpt import ModernGPT
from model.loss import compute_token_weighted_cross_entropy


def get_deterministic_model(seed: int = 42) -> ModernGPT:
    """Instantiate a small deterministic ModernGPT model for exact mathematical tests."""
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=1000,
        context_length=256,
        emb_dim=64,
        n_heads=4,
        num_kv_heads=2,
        n_layers=2,
        ffn_hidden_dim=128,
        dropout=0.0,
        rope_scale=1.0,
    )
    model = ModernGPT(cfg)
    model.eval()  # ensure dropout=0.0 deterministic forward
    return model


def test_a_equal_supervised_token_counts():
    """Test A: When microbatches have equal supervised tokens, token-weighted loss

    is mathematically identical to the unweighted mean of individual microbatch means.
    """
    model_ref = get_deterministic_model(seed=101)
    model_accum = copy.deepcopy(model_ref)

    # Two microbatches with exactly 30 supervised tokens each
    N = 30
    batches = []
    for _ in range(2):
        inp = torch.randint(1, 1000, (1, 60))
        lbl = inp.clone()
        lbl[:, : (60 - N)] = -100  # 30 masked, 30 supervised
        batches.append((inp, lbl))

    # 1. Unweighted mean of microbatch means (old implementation behavior on equal counts)
    model_ref.zero_grad()
    loss_sum_old = 0.0
    for inp, lbl in batches:
        logits = model_ref(inp)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = lbl[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, 1000), shift_labels.view(-1), ignore_index=-100)
        (loss / 2.0).backward()
        loss_sum_old += (loss / 2.0).item()

    # 2. Token-weighted accumulation
    model_accum.zero_grad()
    N_total = sum((lbl[..., 1:] != -100).sum().item() for _, lbl in batches)
    assert N_total == 2 * N

    loss_sum_new = 0.0
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, _, l_sum = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_total)
        loss.backward()
        loss_sum_new += loss.item()

    assert math.isclose(loss_sum_old, loss_sum_new, rel_tol=1e-6, abs_tol=1e-7)

    # Parameter gradients must match exactly
    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p1.grad is not None:
            max_diff = (p1.grad - p2.grad).abs().max().item()
            assert max_diff < 1e-6, f"Mismatch on {n1}: max_diff={max_diff:.2e}"


def test_b_unequal_supervised_token_counts():
    """Test B: Unequal counts (20 tokens vs 1,600 tokens).

    Primary regression test proving token-weighted accumulation matches the true
    single-pass global mean.
    """
    model_ref = get_deterministic_model(seed=202)
    model_accum = copy.deepcopy(model_ref)

    # Microbatch A: exactly 20 supervised tokens in shift_labels
    inp_a = torch.randint(1, 1000, (1, 51))
    lbl_a = inp_a.clone()
    lbl_a[:, :31] = -100  # shift_labels has len 50, first 30 are -100, exactly 20 valid

    # Microbatch B: exactly 150 supervised tokens in shift_labels
    inp_b = torch.randint(1, 1000, (1, 201))
    lbl_b = inp_b.clone()
    lbl_b[:, :51] = -100  # shift_labels has len 200, first 50 are -100, exactly 150 valid

    batches = [(inp_a, lbl_a), (inp_b, lbl_b)]
    n_a = (lbl_a[..., 1:] != -100).sum().item()
    n_b = (lbl_b[..., 1:] != -100).sum().item()
    assert n_a == 20
    assert n_b == 150
    N_total = n_a + n_b

    # 1. Reference single global mean over concatenated valid tokens
    model_ref.zero_grad()
    all_logits = []
    all_targets = []
    for inp, lbl in batches:
        logits = model_ref(inp)
        all_logits.append(logits[..., :-1, :].reshape(-1, 1000))
        all_targets.append(lbl[..., 1:].reshape(-1))

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    ref_loss = F.cross_entropy(cat_logits, cat_targets, ignore_index=-100, reduction="mean")
    ref_loss.backward()

    # 2. Sequential token-weighted accumulation
    model_accum.zero_grad()
    accum_loss = 0.0
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, valid_toks, _ = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_total)
        loss.backward()
        accum_loss += loss.item()

    assert math.isclose(ref_loss.item(), accum_loss, rel_tol=1e-5, abs_tol=1e-6)

    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p1.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
            assert diff < 1e-6, f"Gradient divergence on {n1}: diff={diff:.2e}"

    assert max_grad_diff < 1e-6, f"Max gradient diff {max_grad_diff:.2e} exceeded tolerance"


def test_c_extreme_imbalance():
    """Test C: Extreme imbalance (1 supervised token vs 200 supervised tokens)."""
    model_ref = get_deterministic_model(seed=303)
    model_accum = copy.deepcopy(model_ref)

    # Exactly 1 valid shift target
    inp_a = torch.randint(1, 1000, (1, 31))
    lbl_a = inp_a.clone()
    lbl_a[:, :30] = -100  # shift_labels has len 30, first 29 are -100, exactly 1 valid

    # Exactly 200 valid shift targets
    inp_b = torch.randint(1, 1000, (1, 221))
    lbl_b = inp_b.clone()
    lbl_b[:, :21] = -100  # shift_labels has len 220, first 20 are -100, exactly 200 valid

    batches = [(inp_a, lbl_a), (inp_b, lbl_b)]
    N_total = sum((lbl[..., 1:] != -100).sum().item() for _, lbl in batches)
    assert N_total == 201

    # Reference
    model_ref.zero_grad()
    all_logits = []
    all_targets = []
    for inp, lbl in batches:
        logits = model_ref(inp)
        all_logits.append(logits[..., :-1, :].reshape(-1, 1000))
        all_targets.append(lbl[..., 1:].reshape(-1))
    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    ref_loss = F.cross_entropy(cat_logits, cat_targets, ignore_index=-100, reduction="mean")
    ref_loss.backward()

    # Accumulation
    model_accum.zero_grad()
    accum_loss = 0.0
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, _, _ = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_total)
        loss.backward()
        accum_loss += loss.item()

    assert math.isclose(ref_loss.item(), accum_loss, rel_tol=1e-5, abs_tol=1e-6)

    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p1.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            assert diff < 1e-6, f"Extreme imbalance divergence on {n1}: diff={diff:.2e}"


def test_d_variable_padding_and_ignored_labels():
    """Test D: Microbatches with differing lengths, trailing padding, and internal masking.

    Verifies padding tokens contribute zero loss and zero gradient.
    """
    model = get_deterministic_model(seed=404)

    # Sequence with padding vs unpadded version
    unpadded_inp = torch.tensor([[10, 20, 30, 40, 50]])
    unpadded_lbl = torch.tensor([[-100, -100, 30, 40, 50]])

    padded_inp = torch.tensor([[10, 20, 30, 40, 50, 0, 0, 0]])
    padded_lbl = torch.tensor([[-100, -100, 30, 40, 50, -100, -100, -100]])

    model.zero_grad()
    out_unpadded = model(unpadded_inp)
    loss_unpadded, n_unpadded, sum_unpadded = compute_token_weighted_cross_entropy(out_unpadded, unpadded_lbl)
    loss_unpadded.backward()
    grads_unpadded = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad()
    out_padded = model(padded_inp)
    loss_padded, n_padded, sum_padded = compute_token_weighted_cross_entropy(out_padded, padded_lbl)
    loss_padded.backward()
    grads_padded = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert n_unpadded == n_padded == 3
    assert math.isclose(sum_unpadded, sum_padded, rel_tol=1e-6)
    assert math.isclose(loss_unpadded.item(), loss_padded.item(), rel_tol=1e-6)

    # Gradients on shared active positions must match closely
    # (Note: RoPE position indices 0..4 match)
    for n in grads_unpadded:
        diff = (grads_unpadded[n] - grads_padded[n]).abs().max().item()
        assert diff < 1e-4, f"Padding leakage on {n}: diff={diff:.2e}"


def test_e_partial_accumulation_window():
    """Test E: Incomplete accumulation window (e.g.

    3 microbatches when grad_accum_steps=8).
    Normalization must use the actual tokens in the 3 microbatches.
    """
    model_ref = get_deterministic_model(seed=505)
    model_accum = copy.deepcopy(model_ref)

    # 3 microbatches with exactly 15, 45, and 70 supervised tokens
    batches = []
    for num_sup in [15, 45, 70]:
        inp = torch.randint(1, 1000, (1, 101))
        lbl = inp.clone()
        lbl[:, : (101 - num_sup)] = -100  # shift_labels has len 100, exactly num_sup valid
        batches.append((inp, lbl))

    N_actual = sum((lbl[..., 1:] != -100).sum().item() for _, lbl in batches)
    assert N_actual == 15 + 45 + 70

    # Reference
    model_ref.zero_grad()
    all_logits = []
    all_targets = []
    for inp, lbl in batches:
        logits = model_ref(inp)
        all_logits.append(logits[..., :-1, :].reshape(-1, 1000))
        all_targets.append(lbl[..., 1:].reshape(-1))
    ref_loss = F.cross_entropy(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), ignore_index=-100, reduction="mean")
    ref_loss.backward()

    # Partial window accumulation
    model_accum.zero_grad()
    accum_loss = 0.0
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, _, _ = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_actual)
        loss.backward()
        accum_loss += loss.item()

    assert math.isclose(ref_loss.item(), accum_loss, rel_tol=1e-5, abs_tol=1e-6)

    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p1.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            assert diff < 1e-6, f"Partial window mismatch on {n1}: diff={diff:.2e}"


def test_f_zero_supervised_tokens_safety():
    """Test F: Microbatch containing only ignored labels (-100).

    Must return 0.0 loss without NaN, and backward must not corrupt gradients.
    """
    model = get_deterministic_model(seed=606)

    inp = torch.randint(1, 1000, (1, 50))
    lbl_all_masked = torch.full((1, 50), -100, dtype=torch.long)

    model.zero_grad()
    logits = model(inp)

    # 1. Standalone zero-token microbatch
    loss, valid_tokens, loss_sum = compute_token_weighted_cross_entropy(logits, lbl_all_masked, normalizer=100)
    assert valid_tokens == 0
    assert loss_sum == 0.0
    assert torch.isfinite(loss)
    assert loss.item() == 0.0

    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0.0), "Zero-token batch produced non-zero gradient"

    # 2. Empty window (normalizer=0)
    model.zero_grad()
    loss_empty_win, v_toks, _ = compute_token_weighted_cross_entropy(logits, lbl_all_masked, normalizer=0)
    assert v_toks == 0
    assert torch.isfinite(loss_empty_win)
    assert loss_empty_win.item() == 0.0


def test_g_gradient_equivalence_full_model():
    """Test G: Exhaustive gradient equivalence comparison across every parameter

    in ModernGPT between single global batch and accumulated microbatches.
    """
    model_ref = get_deterministic_model(seed=707)
    model_accum = copy.deepcopy(model_ref)

    # Construct 4 diverse microbatches
    batches = [
        (torch.randint(1, 1000, (1, 40)), torch.randint(1, 1000, (1, 40))),
        (torch.randint(1, 1000, (1, 80)), torch.randint(1, 1000, (1, 80))),
        (torch.randint(1, 1000, (1, 120)), torch.randint(1, 1000, (1, 120))),
        (torch.randint(1, 1000, (1, 60)), torch.randint(1, 1000, (1, 60))),
    ]
    # Apply variable masking to each
    batches[0][1][:, :30] = -100   # 9 supervised
    batches[1][1][:, :20] = -100   # 59 supervised
    batches[2][1][:, :10] = -100   # 109 supervised
    batches[3][1][:, :] = -100     # 0 supervised (edge case within window)

    N_total = sum((lbl[..., 1:] != -100).sum().item() for _, lbl in batches)

    # Reference pass
    model_ref.zero_grad()
    all_logits = []
    all_targets = []
    for inp, lbl in batches:
        logits = model_ref(inp)
        all_logits.append(logits[..., :-1, :].reshape(-1, 1000))
        all_targets.append(lbl[..., 1:].reshape(-1))
    ref_loss = F.cross_entropy(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), ignore_index=-100, reduction="mean")
    ref_loss.backward()

    # Accumulated pass
    model_accum.zero_grad()
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, _, _ = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_total)
        loss.backward()

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for (n, p_ref), (_, p_accum) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        if p_ref.grad is not None:
            abs_diff = (p_ref.grad - p_accum.grad).abs().max().item()
            rel_diff = ((p_ref.grad - p_accum.grad).abs() / (p_ref.grad.abs() + 1e-8)).max().item()
            max_abs_diff = max(max_abs_diff, abs_diff)
            max_rel_diff = max(max_rel_diff, rel_diff)

    print(f"\n[Test G] Max Abs Grad Diff: {max_abs_diff:.2e} | Max Rel Grad Diff: {max_rel_diff:.2e}")
    assert max_abs_diff < 1e-5
    assert max_rel_diff < 1e-2


def test_h_optimizer_update_equivalence():
    """Test H: Verify resulting weights after one complete AdamW optimizer step

    match within floating-point precision between global-batch and accumulation.
    """
    model_ref = get_deterministic_model(seed=808)
    model_accum = copy.deepcopy(model_ref)

    opt_ref = torch.optim.AdamW(model_ref.parameters(), lr=1e-3, weight_decay=0.1)
    opt_accum = torch.optim.AdamW(model_accum.parameters(), lr=1e-3, weight_decay=0.1)

    batches = [
        (torch.randint(1, 1000, (1, 50)), torch.randint(1, 1000, (1, 50))),
        (torch.randint(1, 1000, (1, 100)), torch.randint(1, 1000, (1, 100))),
    ]
    batches[0][1][:, :35] = -100  # 14 supervised
    batches[1][1][:, :20] = -100  # 79 supervised
    N_total = sum((lbl[..., 1:] != -100).sum().item() for _, lbl in batches)

    # 1. Reference step
    opt_ref.zero_grad()
    all_logits = []
    all_targets = []
    for inp, lbl in batches:
        logits = model_ref(inp)
        all_logits.append(logits[..., :-1, :].reshape(-1, 1000))
        all_targets.append(lbl[..., 1:].reshape(-1))
    ref_loss = F.cross_entropy(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0), ignore_index=-100, reduction="mean")
    ref_loss.backward()
    opt_ref.step()

    # 2. Accumulated step
    opt_accum.zero_grad()
    for inp, lbl in batches:
        logits = model_accum(inp)
        loss, _, _ = compute_token_weighted_cross_entropy(logits, lbl, normalizer=N_total)
        loss.backward()
    opt_accum.step()

    # Verify updated parameter equivalence
    max_weight_diff = 0.0
    for (n, p_ref), (_, p_accum) in zip(model_ref.named_parameters(), model_accum.named_parameters()):
        diff = (p_ref - p_accum).abs().max().item()
        max_weight_diff = max(max_weight_diff, diff)
        assert diff < 1e-6, f"Weight divergence on {n}: diff={diff:.2e}"

    print(f"\n[Test H] Optimizer Step Weight Equivalence: max_diff = {max_weight_diff:.2e}")
    assert max_weight_diff < 1e-6


def test_j_validation_loss_partition_invariance():
    """Test J: Verify validation loss calculation is completely invariant

    to how examples are partitioned into batches (batch_size=1 vs batch_size=2 vs batch_size=4).
    """
    model = get_deterministic_model(seed=909)

    # Construct 4 validation examples of varying lengths and supervised counts
    samples = [
        (torch.randint(1, 1000, (1, 30)), torch.randint(1, 1000, (1, 30))),
        (torch.randint(1, 1000, (1, 70)), torch.randint(1, 1000, (1, 70))),
        (torch.randint(1, 1000, (1, 50)), torch.randint(1, 1000, (1, 50))),
        (torch.randint(1, 1000, (1, 90)), torch.randint(1, 1000, (1, 90))),
    ]
    samples[0][1][:, :15] = -100
    samples[1][1][:, :50] = -100
    samples[2][1][:, :10] = -100
    samples[3][1][:, :30] = -100

    def compute_val_loss(batch_list):
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for inp, lbl in batch_list:
                logits = model(inp)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = lbl[..., 1:].contiguous()

                for b in range(inp.size(0)):
                    item_logits = shift_logits[b]
                    item_labels = shift_labels[b]
                    mask = item_labels != -100
                    count = mask.sum().item()
                    if count > 0:
                        loss_sum = F.cross_entropy(item_logits[mask], item_labels[mask], reduction="sum").item()
                        total_loss += loss_sum
                        total_tokens += count
        return total_loss / max(1, total_tokens)

    # Partition 1: 4 separate microbatches of size 1
    val_loss_p1 = compute_val_loss(samples)

    # Partition 2: 2 batches of size 2 (with collation/padding)
    def collate_two(s1, s2):
        max_l = max(s1[0].size(1), s2[0].size(1))
        inps = torch.full((2, max_l), 0, dtype=torch.long)
        lbls = torch.full((2, max_l), -100, dtype=torch.long)
        inps[0, : s1[0].size(1)] = s1[0]
        inps[1, : s2[0].size(1)] = s2[0]
        lbls[0, : s1[1].size(1)] = s1[1]
        lbls[1, : s2[1].size(1)] = s2[1]
        return inps, lbls

    batch_2a = collate_two(samples[0], samples[1])
    batch_2b = collate_two(samples[2], samples[3])
    val_loss_p2 = compute_val_loss([batch_2a, batch_2b])

    # Partition 3: All 4 in a single batch
    max_all = max(s[0].size(1) for s in samples)
    inps_all = torch.full((4, max_all), 0, dtype=torch.long)
    lbls_all = torch.full((4, max_all), -100, dtype=torch.long)
    for idx, (inp, lbl) in enumerate(samples):
        inps_all[idx, : inp.size(1)] = inp
        lbls_all[idx, : lbl.size(1)] = lbl
    val_loss_p3 = compute_val_loss([(inps_all, lbls_all)])

    print(f"\n[Test J] Partition Invariance: P1={val_loss_p1:.8f}, P2={val_loss_p2:.8f}, P3={val_loss_p3:.8f}")
    assert math.isclose(val_loss_p1, val_loss_p2, rel_tol=1e-6, abs_tol=1e-7)
    assert math.isclose(val_loss_p1, val_loss_p3, rel_tol=1e-6, abs_tol=1e-7)
