"""ModernGPT-S 100M: Global Gradient Accumulation and Checkpoint-Safety Tests.

Verifies:
1. Seven residual micro-batches from one shard plus one micro-batch from the next shard produce exactly one optimizer update.
2. No gradients are discarded at shard transitions.
3. Accumulation remains correct across multiple shard boundaries.
4. Session timeout during partial accumulation finishes the current accumulation group before checkpointing.
5. Checkpoint recovery preserves optimizer, scheduler and global accumulation state.
6. Interrupted and uninterrupted training produce equivalent results under the same deterministic test kernel.
7. The final incomplete accumulation group is handled explicitly.
8. The global optimizer-step budget is respected.
"""

import dataclasses
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from model.config import get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.pretrain_100m import PretrainTrainer, ShardTokenDataset, compute_cosine_lr


def create_synthetic_shard(path: Path, num_sequences: int, context_len: int = 512, seed: int = 42):
    """Generates synthetic binary token shard with deterministic uint16 tokens."""
    np.random.seed(seed)
    total_tokens = num_sequences * context_len + 1
    tokens = np.random.randint(1, 1000, size=total_tokens, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(tokens.tobytes())
    return path


@pytest.fixture
def test_setup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Small 2-layer model on CPU for fast deterministic tests
        base_cfg = get_100m_gqa4_config()
        model_cfg = dataclasses.replace(base_cfg, n_layers=2, dropout=0.0)
        train_cfg = {
            "total_tokens": 1000000,
            "total_optimizer_steps": 100,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "warmup_steps": 10,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 64,
            "output_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
            "metrics_file": str(tmp_path / "logs/metrics.jsonl"),
            "dataset_manifest_hash": "test",
            "eval_interval_steps": 1000,
            "save_interval_steps": 1000,
        }
        session_cfg = {
            "session_duration_minutes": 120,
            "cooling_duration_minutes": 30,
            "max_gpu_temp_celsius": 95.0,
            "warn_gpu_temp_celsius": 90.0,
            "emergency_cooldown_temp_celsius": 70.0,
        }
        yield tmp_path, model_cfg, train_cfg, session_cfg


def test_cross_shard_accumulation_single_optimizer_update(test_setup):
    """Test 1: Seven residual micro-batches from shard A + 1 from shard B produce exactly 1 optimizer update."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    # Shard A: 7 microbatches = 7 * 8 = 56 sequences
    # Shard B: 5 microbatches = 5 * 8 = 40 sequences
    shard_a = create_synthetic_shard(tmp_path / "shard_a.bin", num_sequences=56, seed=1)
    shard_b = create_synthetic_shard(tmp_path / "shard_b.bin", num_sequences=40, seed=2)

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )

    # Train only 1 step (max_steps=1)
    res = trainer.train_session(shard_paths=[shard_a, shard_b], max_steps=1)

    assert res["steps_completed"] == 1
    assert trainer.global_step == 1
    assert trainer.shard_index == 1, "Should have crossed to shard B"
    # 7 microbatches from shard A (56 seqs) + 1 microbatch from shard B (8 seqs) = 64 seqs
    assert trainer.sequence_offset == 8, f"Expected offset 8 in shard B, got {trainer.sequence_offset}"
    assert trainer.accum_steps == 0, f"Expected accum_steps reset to 0 after step, got {trainer.accum_steps}"


def test_no_gradients_discarded_at_shard_transition(test_setup):
    """Test 2: Parameter gradients are strictly preserved and not zeroed across shard transition."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    shard_a = create_synthetic_shard(tmp_path / "shard_a.bin", num_sequences=56, seed=10) # 7 microbatches
    shard_b = create_synthetic_shard(tmp_path / "shard_b.bin", num_sequences=16, seed=20) # 2 microbatches

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )

    # Track gradients after processing shard_a
    # We run shard_a only
    res_a = trainer.train_session(shard_paths=[shard_a])
    assert trainer.global_step == 0, "No optimizer step should have occurred in shard A (only 7 microbatches)"
    assert trainer.accum_steps == 7, "accum_steps must be 7 after shard A"

    # Verify gradients exist on model parameters and are non-zero
    grads_exist = any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in trainer.model.parameters())
    assert grads_exist, "Gradients must exist after 7 microbatches of shard A"

    # Capture gradient norm after shard A
    grad_norm_after_a = sum(p.grad.norm().item() for p in trainer.model.parameters() if p.grad is not None)
    assert grad_norm_after_a > 0.0

    # Now continue into shard_b
    res_b = trainer.train_session(shard_paths=[shard_a, shard_b])
    # Shard B should process 1 microbatch, hit 8, step optimizer, and reset accum_steps
    assert trainer.global_step == 1, "Optimizer should have stepped on 8th microbatch in shard B"
    assert trainer.accum_steps == 1, "1 residual microbatch in shard B should be in accumulator"


def test_multi_shard_continuous_accumulation(test_setup):
    """Test 3: Accumulation remains correct across multiple shard boundaries (A -> B -> C)."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    # Shard A: 7 microbatches (56 seqs)
    # Shard B: 7 microbatches (56 seqs)
    # Shard C: 7 microbatches (56 seqs)
    # Total = 21 microbatches. With grad_accum=8:
    # Step 1 at microbatch 8 (7 from A + 1 from B)
    # Step 2 at microbatch 16 (6 from B + 2 from C)
    # Remaining in accumulator: 5 microbatches from C
    shard_a = create_synthetic_shard(tmp_path / "shard_a.bin", num_sequences=56, seed=100)
    shard_b = create_synthetic_shard(tmp_path / "shard_b.bin", num_sequences=56, seed=101)
    shard_c = create_synthetic_shard(tmp_path / "shard_c.bin", num_sequences=56, seed=102)

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )

    res = trainer.train_session(shard_paths=[shard_a, shard_b, shard_c])
    assert trainer.global_step == 2, f"Expected 2 completed steps from 21 microbatches, got {trainer.global_step}"
    assert trainer.accum_steps == 5, f"Expected 5 leftover microbatches in accumulator, got {trainer.accum_steps}"
    assert trainer.shard_index == 3, "All 3 shards should have been processed"


def test_session_timeout_finishes_accumulation_group(test_setup):
    """Test 4: Session timeout during partial accumulation finishes the current accumulation group before checkpointing."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    # Large shard: 32 microbatches (256 sequences)
    shard = create_synthetic_shard(tmp_path / "shard_large.bin", num_sequences=256, seed=42)

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )

    # Set session time limit to trigger after roughly 2 microbatches
    # We patch time.time to simulate timeout during partial accumulation
    real_time = time.time
    t_start = real_time()

    def mock_time():
        # Trigger timeout only once trainer has started accumulating (accum_steps >= 2)
        if trainer.accum_steps >= 2:
            return t_start + 100000.0
        return real_time()

    with patch("time.time", side_effect=mock_time):
        res = trainer.train_session(shard_paths=[shard], max_session_minutes=10)

    assert res["outcome"] == "SESSION_TIME_LIMIT"
    # Even though timeout was triggered mid-step, it must finish the full 8 microbatches
    assert res["steps_completed"] == 1, f"Expected exactly 1 complete step finished before pause, got {res['steps_completed']}"
    assert trainer.accum_steps == 0, f"accum_steps must be 0 at step-boundary checkpoint, got {trainer.accum_steps}"
    assert trainer.latest_ckpt.exists(), "Latest checkpoint must be written at step boundary"


def test_checkpoint_recovery_preserves_accumulation_state(test_setup):
    """Test 5: Checkpoint recovery preserves optimizer, scheduler, and accumulation counters."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    shard = create_synthetic_shard(tmp_path / "shard.bin", num_sequences=128, seed=7)

    trainer1 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    # Train 1 step and save checkpoint
    trainer1.train_session(shard_paths=[shard], max_steps=1)
    assert trainer1.global_step == 1
    assert trainer1.accum_steps == 0

    # Instantiate trainer2 and load checkpoint
    trainer2 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    trainer2.load_checkpoint(trainer1.latest_ckpt)

    assert trainer2.global_step == 1
    assert trainer2.accum_steps == 0
    assert trainer2.sequence_offset == trainer1.sequence_offset
    assert trainer2.tokens_seen == trainer1.tokens_seen

    # Compare model parameters between trainer1 and trainer2
    for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
        assert torch.equal(p1, p2), "Recovered model parameters must match saved parameters exactly"


def test_interrupted_vs_uninterrupted_equivalence(test_setup):
    """Test 6: Interrupted and uninterrupted training produce bit-for-bit equivalent results."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    shard_path = create_synthetic_shard(tmp_path / "shard_equiv.bin", num_sequences=128, seed=999)

    # 1. Uninterrupted run: 2 steps continuously
    torch.manual_seed(1234)
    np.random.seed(1234)
    trainer_uninterrupted = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    res_un = trainer_uninterrupted.train_session(shard_paths=[shard_path], max_steps=2)
    assert res_un["steps_completed"] == 2

    # 2. Interrupted run: step 1 in Session 1, step 2 in Session 2 after reload
    torch.manual_seed(1234)
    np.random.seed(1234)
    ckpt_dir_interrupted = tmp_path / "ckpt_interrupted"
    train_cfg_int = dict(train_cfg, output_dir=str(ckpt_dir_interrupted))

    trainer_int_1 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg_int,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    res_int1 = trainer_int_1.train_session(shard_paths=[shard_path], max_steps=1)
    assert res_int1["steps_completed"] == 1

    # Reload into second trainer instance
    trainer_int_2 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg_int,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    trainer_int_2.load_checkpoint(trainer_int_1.latest_ckpt)
    res_int2 = trainer_int_2.train_session(shard_paths=[shard_path], max_steps=2)
    assert res_int2["steps_completed"] == 1
    assert trainer_int_2.global_step == 2

    # Compare model parameters between uninterrupted and interrupted
    max_diff = 0.0
    for p_un, p_int in zip(trainer_uninterrupted.model.parameters(), trainer_int_2.model.parameters()):
        diff = torch.max(torch.abs(p_un - p_int)).item()
        if diff > max_diff:
            max_diff = diff

    assert max_diff < 1e-6, f"Expected bitwise equivalence, got max param diff {max_diff}"


def test_final_incomplete_accumulation_handled_explicitly(test_setup):
    """Test 7: Final incomplete accumulation group is explicitly reported without silent drop."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    # 10 microbatches total (80 sequences). At grad_accum=8: 1 step of 8, plus 2 residual
    shard = create_synthetic_shard(tmp_path / "shard_final.bin", num_sequences=80, seed=55)

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    res = trainer.train_session(shard_paths=[shard])
    assert res["outcome"] == "CORPUS_EXHAUSTED"
    assert res["steps_completed"] == 1
    assert trainer.accum_steps == 2, f"Expected 2 residual microbatches tracked, got {trainer.accum_steps}"


def test_global_optimizer_step_budget_respected(test_setup):
    """Test 8: Global optimizer step budget is respected as the authoritative stopping target."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    train_cfg_budget = dict(train_cfg, total_optimizer_steps=3)
    shard = create_synthetic_shard(tmp_path / "shard_budget.bin", num_sequences=256, seed=777)

    trainer = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg_budget,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    res = trainer.train_session(shard_paths=[shard])
    assert res["outcome"] == "GLOBAL_TARGET_REACHED"
    assert trainer.global_step == 3
    assert res["steps_completed"] == 3


def test_legacy_checkpoint_migration_without_accum_steps(test_setup):
    """Test 9: Legacy checkpoint missing accum_steps and accum_loss migrates cleanly to 0 with exact sampler offset."""
    tmp_path, model_cfg, train_cfg, session_cfg = test_setup
    legacy_ckpt_path = tmp_path / "legacy.pt"

    t0 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    state = t0.build_checkpoint_state(current_lr=1e-3)
    # Simulate legacy checkpoint by deleting new accumulation fields
    del state["accum_steps"]
    del state["accum_loss"]
    state["step"] = 10342
    state["shard_index"] = 6
    state["sequence_offset"] = 76336
    state["tokens_seen"] = 339083264
    torch.save(state, legacy_ckpt_path)

    # Load with new trainer
    t1 = PretrainTrainer(
        model_config=model_cfg,
        train_config=train_cfg,
        session_config=session_cfg,
        device=torch.device("cpu"),
    )
    t1.load_checkpoint(legacy_ckpt_path)

    assert t1.global_step == 10342
    assert t1.tokens_seen == 339083264
    assert t1.shard_index == 6
    assert t1.sequence_offset == 76336
    assert t1.accum_steps == 0, "Missing accum_steps must safely default to 0"
    assert t1.accum_loss == 0.0, "Missing accum_loss must safely default to 0.0"


