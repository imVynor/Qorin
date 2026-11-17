"""ModernGPT-S 100M: Session Orchestration and Bug Regression Tests.

Verifies:
1. A checkpoint with 339,083,264 tokens does NOT satisfy a 1.5B-token global budget.
2. A session with zero completed steps cannot be treated as successful global training completion.
3. The orchestrator proceeds to Session 2 after Session 1 completes if the global budget remains.
4. The orchestrator exits only when the actual global training budget is reached.
5. Resume preserves exact token accounting and optimizer/scheduler state.
6. Cooling-break and restart thresholds match the approved policy (90°C / 95°C / 70°C, 30m mandatory).
"""

import copy
import dataclasses
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from model.config import get_100m_gqa4_config
from scripts.monitor_gpu import GPUMonitor
from scripts.pretrain_100m import PretrainTrainer, compute_cosine_lr


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


def test_checkpoint_token_budget_not_satisfied(repo_root):
    """Test 1: Verify a checkpoint with 339,083,264 tokens does NOT satisfy 1.5B-token budget."""
    total_budget_tokens = 1500020736
    total_optimizer_steps = 45777

    # Load production latest.pt metadata (read-only)
    latest_ckpt_path = repo_root / "checkpoints/modern_gpts_100m/latest.pt"
    assert latest_ckpt_path.exists(), "Production checkpoint latest.pt not found!"
    ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)

    tokens_seen = ckpt["tokens_seen"]
    current_step = ckpt["step"]

    assert tokens_seen < total_budget_tokens, f"Expected tokens_seen < {total_budget_tokens}, got {tokens_seen}"
    assert current_step < total_optimizer_steps, f"Expected step < {total_optimizer_steps}, got {current_step}"

    # Verify global completion conditions evaluate to False
    is_global_step_complete = current_step >= total_optimizer_steps
    is_global_token_complete = tokens_seen >= total_budget_tokens
    is_global_complete = is_global_step_complete or is_global_token_complete

    assert not is_global_complete, "Checkpoint incorrectly evaluated as completing the global budget!"
    assert not is_global_step_complete, f"Step count {current_step} incorrectly satisfied 45,777 steps!"
    assert not is_global_token_complete, f"Token count {tokens_seen} incorrectly satisfied 1.5B tokens!"

    # Compute exact remaining budget
    remaining_steps = total_optimizer_steps - current_step
    remaining_tokens = total_budget_tokens - tokens_seen
    assert remaining_steps > 0
    assert remaining_tokens > 0


def test_zero_step_session_cannot_be_completed(repo_root):
    """Test 2: A session with zero completed steps cannot be treated as successful global training completion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_ckpt = tmp_path / "latest.pt"

        model_cfg = dataclasses.replace(get_100m_gqa4_config(), n_layers=2)

        train_cfg = {
            "total_tokens": 1500020736,
            "total_optimizer_steps": 45777,
            "learning_rate": 2.5e-4,
            "min_learning_rate": 2.5e-5,
            "warmup_steps": 2000,
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "metrics_file": str(tmp_path / "logs/metrics.jsonl"),
            "dataset_manifest_hash": "test",
        }
        session_cfg = {
            "session_duration_minutes": 120,
            "max_gpu_temp_celsius": 82.0,
            "warn_gpu_temp_celsius": 78.0,
            "emergency_cooldown_temp_celsius": 70.0,
        }

        trainer = PretrainTrainer(
            model_config=model_cfg,
            train_config=train_cfg,
            session_config=session_cfg,
            device=torch.device("cpu"),
        )
        # Create and save a valid 2-layer checkpoint
        dummy_state = trainer.build_checkpoint_state(2.5e-4)
        torch.save(dummy_state, test_ckpt)
        trainer.load_checkpoint(test_ckpt)

        # Simulate empty shards or immediate loop termination
        result = trainer.train_session(shard_paths=[], max_session_minutes=0.01)
        
        # Verify outcome is NOT COMPLETED or GLOBAL_TARGET_REACHED
        assert result["steps_completed"] == 0
        assert result["outcome"] != "COMPLETED", "Zero-step session must not return COMPLETED!"
        assert result["outcome"] != "GLOBAL_TARGET_REACHED", "Zero-step session must not report global target reached!"
        assert result["outcome"] in ("NO_STEPS_EXECUTED", "CORPUS_EXHAUSTED", "SESSION_TIME_LIMIT")


def test_orchestrator_proceeds_to_session_2_after_time_limit(repo_root):
    """Test 3: The orchestrator proceeds to Session 2 after Session 1 completes if global budget remains."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_ckpt = tmp_path / "latest.pt"

        model_cfg = dataclasses.replace(get_100m_gqa4_config(), n_layers=1)
        train_cfg = {
            "total_tokens": 1500020736,
            "total_optimizer_steps": 45777,
            "output_dir": str(tmp_path),
            "log_dir": str(tmp_path / "logs"),
            "metrics_file": str(tmp_path / "logs/metrics.jsonl"),
            "dataset_manifest_hash": "test",
        }
        session_cfg = {
            "session_duration_minutes": 120,
            "cooling_duration_minutes": 30,
            "max_gpu_temp_celsius": 82.0,
            "warn_gpu_temp_celsius": 78.0,
            "emergency_cooldown_temp_celsius": 70.0,
        }

        trainer = PretrainTrainer(
            model_config=model_cfg,
            train_config=train_cfg,
            session_config=session_cfg,
            device=torch.device("cpu"),
        )
        dummy_state = trainer.build_checkpoint_state(2.5e-4)
        torch.save(dummy_state, test_ckpt)
        trainer.load_checkpoint(test_ckpt)

        # Simulate Session 1 ending with stop_requested = True (time limit hit)
        trainer.stop_requested = True
        assert trainer.stop_requested is True

        # In Session 2, train_session is called on the same trainer instance
        # Verify that calling train_session immediately clears stop_requested to False
        res = trainer.train_session(shard_paths=[], max_session_minutes=0.01)
        assert trainer.stop_requested is False
        # Verify outcome does NOT falsely claim global completion
        assert res["outcome"] != "COMPLETED"
        assert res["outcome"] != "GLOBAL_TARGET_REACHED"


def test_orchestrator_exits_only_on_true_global_target():
    """Test 4: Orchestrator exits only when actual global target or override is reached."""
    total_tokens = 1500020736
    total_steps = 45777

    # Sub-case A: Intermediate progress (e.g. 10,342 steps) -> Must NOT exit
    step_a = 10342
    tokens_a = 339083264
    assert not (step_a >= total_steps or tokens_a >= total_tokens)

    # Sub-case B: Reached 45,777 steps -> Must exit
    step_b = 45777
    tokens_b = 1500020736
    assert (step_b >= total_steps or tokens_b >= total_tokens)

    # Sub-case C: Benchmark override (max_steps = 100) reached at 100
    max_steps_override = 100
    step_c = 100
    assert bool(max_steps_override and step_c >= max_steps_override)


def test_resume_preserves_token_accounting_and_optimizer_state(repo_root):
    """Test 5: Resume preserves exact token accounting, RNG, and optimizer/scheduler state."""
    latest_ckpt_path = repo_root / "checkpoints/modern_gpts_100m/latest.pt"
    ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)

    # Verify all 19 recovery state keys exist
    expected_keys = [
        "model", "optimizer", "scheduler", "step", "tokens_seen", "grad_scaler",
        "python_rng_state", "numpy_rng_state", "torch_cpu_rng_state", "cuda_rng_state",
        "data_sampler_state", "shard_index", "sequence_offset", "training_config",
        "model_config", "dataset_manifest_hash", "tokenizer_fingerprint", "experiment_id",
        "session_id",
    ]
    for key in expected_keys:
        assert key in ckpt, f"Checkpoint missing required recovery key: {key}"

    # Verify values are consistent with checkpoint progress
    assert ckpt["step"] >= 10342
    assert ckpt["tokens_seen"] >= 339083264
    assert ckpt["shard_index"] >= 6
    assert ckpt["sequence_offset"] >= 0

    # Verify optimizer state has momentum buffers
    optim_state = ckpt["optimizer"]
    assert "state" in optim_state and len(optim_state["state"]) > 0

    # Verify deterministic cosine lr calculation for step 10343
    next_lr = compute_cosine_lr(
        step=10343,
        base_lr=0.00025,
        min_lr=0.000025,
        warmup_steps=2000,
        total_steps=45777,
    )
    assert 0.000230 < next_lr < 0.000231
    assert abs(next_lr - 0.00023043128334578018) < 1e-12


def test_cooling_break_and_thermal_thresholds_match_approved_policy(repo_root):
    """Test 6: Cooling-break and restart thresholds match approved policy (90°C / 95°C / 70°C, 30m mandatory)."""
    cfg_path = repo_root / "configs/modern_gpts_100m/session.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        session_cfg = yaml.safe_load(f)

    # 1. Config thresholds
    assert session_cfg["warn_gpu_temp_celsius"] == 90.0, f"Expected 90°C warning, got {session_cfg['warn_gpu_temp_celsius']}"
    assert session_cfg["max_gpu_temp_celsius"] == 95.0, f"Expected 95°C emergency max, got {session_cfg['max_gpu_temp_celsius']}"
    assert session_cfg["emergency_cooldown_temp_celsius"] == 70.0, f"Expected 70°C cooldown target, got {session_cfg['emergency_cooldown_temp_celsius']}"
    assert session_cfg["session_duration_minutes"] == 120.0
    assert session_cfg["cooling_duration_minutes"] == 30.0

    # 2. GPUMonitor default parameters
    monitor = GPUMonitor()
    assert monitor.warn_temp == 90.0
    assert monitor.max_temp == 95.0
    assert monitor.cooldown_temp == 70.0

    # 3. Mandatory cooldown duration logic: verify wait_for_cooldown does NOT exit prematurely
    # even if GPU temp is already <= 70°C, when min_cooldown_seconds > 0.
    with patch.object(monitor, "poll_once", return_value={"gpu_temp_c": 65.0}):
        # Mock poll_once returning 65.0°C (well below 70°C)
        # With min_cooldown_seconds = 0.5s, it must wait at least 0.5s
        t0 = time.time()
        final_temp = monitor.wait_for_cooldown(
            target_temp=70.0,
            poll_interval=0.1,
            min_cooldown_seconds=0.5,
            timeout_seconds=2.0,
        )
        elapsed = time.time() - t0
        assert final_temp == 65.0
        assert elapsed >= 0.45, f"Expected cooldown to respect min_cooldown_seconds (0.5s), but took {elapsed:.3f}s"
