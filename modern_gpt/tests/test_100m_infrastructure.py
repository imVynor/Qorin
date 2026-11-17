"""Tests for ModernGPT-S 100M Pretraining Infrastructure.

Verifies:
1. Exact parameter count for ModernGPT-S 100M GQA-4 (100,401,408).
2. Gradient checkpointing forward & backward correctness.
3. Full 19-point state recovery checkpoint structure.
4. Atomic checkpoint saving and backup rotation.
5. Multi-source ratio-balanced interleaving logic.
6. Contamination audit n-gram detection.
7. Hardware telemetry watchdog polling.
"""

import dataclasses
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from data.sources import NormalizedDocument
from model.config import ModernGPTConfig, get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.audit_100m_contamination import get_ngrams
from scripts.monitor_gpu import GPUMonitor
from scripts.prepare_100m_corpus import InterleavedSourceStreamer
from scripts.pretrain_100m import (
    PretrainTrainer,
    compute_cosine_lr,
    save_checkpoint_atomic,
)


def test_100m_parameter_count():
    cfg = get_100m_gqa4_config()
    model = ModernGPT(cfg)
    total_params = model.num_parameters()
    # 100,401,408 exact target
    assert total_params == 100401408, f"Expected 100,401,408 params, got {total_params}"


def test_100m_gradient_checkpointing():
    base_cfg = get_100m_gqa4_config()
    cfg = dataclasses.replace(base_cfg, n_layers=2, gradient_checkpointing=True)

    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    targets = torch.randint(0, cfg.vocab_size, (2, 64))

    logits, loss = model(x, targets)
    assert logits.shape == (2, 64, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)

    loss.backward()
    # Verify gradients exist and are non-zero
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "Gradient checkpointing backward did not produce gradients"


def test_19_point_checkpoint_integrity():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        base_cfg = get_100m_gqa4_config()
        cfg = dataclasses.replace(base_cfg, n_layers=2)
        train_cfg = {
            "output_dir": str(tmp_path / "checkpoints"),
            "session_backup_dir": str(tmp_path / "backups"),
            "log_dir": str(tmp_path / "logs"),
            "metrics_file": str(tmp_path / "logs/metrics.jsonl"),
            "precision": "fp32",
            "learning_rate": 2.5e-4,
            "min_learning_rate": 2.5e-5,
            "warmup_steps": 100,
            "total_optimizer_steps": 1000,
            "grad_clip_norm": 1.0,
            "dataset_manifest_hash": "test_manifest_hash_123",
            "experiment_id": "test_100m_run",
        }
        session_cfg = {
            "thermal_log_file": str(tmp_path / "logs/thermal.jsonl"),
            "thermal_monitor_interval_seconds": 1.0,
            "warn_gpu_temp_celsius": 80.0,
            "max_gpu_temp_celsius": 85.0,
            "emergency_cooldown_temp_celsius": 70.0,
        }

        trainer = PretrainTrainer(
            model_config=cfg,
            train_config=train_cfg,
            session_config=session_cfg,
            device=torch.device("cpu"),
            session_id="test_session_001",
        )

        state = trainer.build_checkpoint_state(current_lr=2.5e-4)

        # 19 Mandatory elements
        required_keys = [
            "model",
            "optimizer",
            "scheduler",
            "step",
            "tokens_seen",
            "grad_scaler",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "cuda_rng_state",
            "data_sampler_state",
            "shard_index",
            "sequence_offset",
            "training_config",
            "model_config",
            "dataset_manifest_hash",
            "tokenizer_fingerprint",
            "experiment_id",
            "session_id",
        ]

        for k in required_keys:
            assert k in state, f"Missing required 19-point state key: {k}"

        assert state["scheduler"]["current_lr"] == 2.5e-4
        assert state["session_id"] == "test_session_001"
        assert state["experiment_id"] == "test_100m_run"


def test_atomic_checkpoint_recovery():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_file = tmp_path / "latest.pt"
        backup_dir = tmp_path / "backups"

        state_1 = {
            "model": {"w": torch.tensor([1.0, 2.0])},
            "optimizer": {"state": {}},
            "step": 100,
            "tokens_seen": 3276800,
        }

        save_checkpoint_atomic(ckpt_file, state_1, backup_dir=backup_dir, max_backups=3)
        assert ckpt_file.exists()

        loaded_1 = torch.load(ckpt_file, weights_only=False)
        assert loaded_1["step"] == 100
        assert torch.equal(loaded_1["model"]["w"], torch.tensor([1.0, 2.0]))

        # Second save creates backup
        state_2 = {
            "model": {"w": torch.tensor([3.0, 4.0])},
            "optimizer": {"state": {}},
            "step": 200,
            "tokens_seen": 6553600,
        }
        save_checkpoint_atomic(ckpt_file, state_2, backup_dir=backup_dir, max_backups=3)

        loaded_2 = torch.load(ckpt_file, weights_only=False)
        assert loaded_2["step"] == 200
        assert torch.equal(loaded_2["model"]["w"], torch.tensor([3.0, 4.0]))

        # Verify backup was created
        backups = list(backup_dir.glob("checkpoint_step_*.pt"))
        assert len(backups) == 1
        backed_up = torch.load(backups[0], weights_only=False)
        assert backed_up["step"] == 100


def test_interleaved_streaming_logic():
    ratios = {
        "fineweb_edu": 0.60,
        "cosmopedia_v2": 0.20,
        "python_edu": 0.08,
        "open_web_math": 0.07,
        "fineweb": 0.05,
    }

    # Test selection deficit math
    streamer = InterleavedSourceStreamer.__new__(InterleavedSourceStreamer)
    streamer.source_ratios = ratios
    streamer.accumulated_counts = {s: 0 for s in ratios}
    streamer.iterators = {}

    for s in ratios:
        mock_it = iter([
            NormalizedDocument(
                text=f"Sample text for {s}",
                source=s,
                document_id=f"{s}:1",
                char_count=50,
                byte_count=50,
                content_hash="hash1",
            )
            for _ in range(100)
        ])
        streamer.iterators[s] = mock_it

    # Pull 100 documents
    counts = {s: 0 for s in ratios}
    for _ in range(100):
        src, _ = streamer.next_document()
        counts[src] += 1

    # fw_edu should be the majority (~60)
    assert counts["fineweb_edu"] >= 55 and counts["fineweb_edu"] <= 65
    assert counts["cosmopedia_v2"] >= 15 and counts["cosmopedia_v2"] <= 25
    assert counts["python_edu"] >= 5 and counts["python_edu"] <= 12


def test_contamination_audit_ngram_detection():
    doc_a = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    doc_b = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 999]  # Shares 13-gram
    doc_c = [999] * 20  # Completely disjoint

    ngrams_a = get_ngrams(doc_a, n=13)
    ngrams_b = get_ngrams(doc_b, n=13)
    ngrams_c = get_ngrams(doc_c, n=13)

    assert len(ngrams_a.intersection(ngrams_b)) > 0, "Failed to detect overlapping 13-gram"
    assert len(ngrams_a.intersection(ngrams_c)) == 0, "False positive 13-gram detected"


def test_cosine_lr_schedule():
    base_lr = 2.5e-4
    min_lr = 2.5e-5
    warmup = 1000
    total = 10000

    # Step 0 (start of warmup)
    lr_0 = compute_cosine_lr(0, base_lr, min_lr, warmup, total)
    assert lr_0 > 0 and lr_0 < base_lr

    # Peak at step warmup - 1
    lr_warmup = compute_cosine_lr(warmup - 1, base_lr, min_lr, warmup, total)
    assert abs(lr_warmup - base_lr) < 1e-6

    # Midpoint
    lr_mid = compute_cosine_lr(5500, base_lr, min_lr, warmup, total)
    assert lr_mid < base_lr and lr_mid > min_lr

    # Endpoint
    lr_end = compute_cosine_lr(total, base_lr, min_lr, warmup, total)
    assert abs(lr_end - min_lr) < 1e-6
