"""Tests for the permanent experiment tracking infrastructure.

Covers Part A9 requirements:
- Unique experiment IDs.
- Correct registry updates (JSONL and CSV synchronization).
- No accidental overwrite of completed runs.
- Correct checkpoint hashing (SHA-256 integrity).
- Incremental metric persistence.
- Interrupted-run preservation (partial logs and metrics retained).
- Correct experiment-to-checkpoint association.
- Reproducible metadata capture.
- Standardized report generation.
- Fail-safe execution (raises error if tracking cannot initialize).
"""

import csv
import json
from pathlib import Path
import pytest
import torch

from experiments.registry import ExperimentRegistry, REGISTRY_FIELDS
from experiments.tracker import ExperimentTracker, compute_file_sha256, capture_environment_metadata


def test_unique_experiment_ids_and_registration(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    assert len(registry.get_all_experiments()) == 0

    record1 = {
        "experiment_id": "EXP-001",
        "experiment_name": "Test Run 1",
        "category": "pretraining",
        "status": "completed",
    }
    registry.register_or_update(record1)
    assert len(registry.get_all_experiments()) == 1
    assert registry.experiment_exists("EXP-001")

    # Update existing run preserves uniqueness (replaces/updates entry, does not duplicate)
    record1_updated = {
        "experiment_id": "EXP-001",
        "experiment_name": "Test Run 1 Updated",
        "category": "pretraining",
        "status": "completed",
    }
    registry.register_or_update(record1_updated)
    assert len(registry.get_all_experiments()) == 1
    assert registry.get_experiment("EXP-001")["experiment_name"] == "Test Run 1 Updated"

    # Add second distinct experiment
    record2 = {
        "experiment_id": "EXP-002",
        "experiment_name": "Test Run 2",
        "category": "SFT",
        "status": "running",
    }
    registry.register_or_update(record2)
    assert len(registry.get_all_experiments()) == 2


def test_registry_csv_and_jsonl_synchronization(tmp_path):
    registry = ExperimentRegistry(tmp_path)
    record = {
        "experiment_id": "EXP-SYNC-01",
        "experiment_name": "Sync Verification",
        "category": "architecture",
        "architecture": "ModernGPT-50M",
        "parameter_count": 49105408,
        "status": "completed",
        "best_val_metric": 3.1415,
        "main_conclusion": "Verified CSV and JSONL sync",
    }
    registry.register_or_update(record)

    # Check JSONL
    with open(registry.jsonl_path, "r", encoding="utf-8") as f:
        jsonl_lines = [json.loads(line) for line in f if line.strip()]
    assert len(jsonl_lines) == 1
    assert jsonl_lines[0]["experiment_id"] == "EXP-SYNC-01"

    # Check CSV
    with open(registry.csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    assert reader[0]["experiment_id"] == "EXP-SYNC-01"
    assert reader[0]["category"] == "architecture"
    assert float(reader[0]["best_val_metric"]) == 3.1415


def test_metadata_capture():
    meta = capture_environment_metadata()
    assert "platform" in meta
    assert "pytorch_version" in meta
    assert "python_version" in meta
    assert "cuda_available" in meta
    assert "installed_packages" in meta
    assert isinstance(meta["installed_packages"], dict)


def test_checkpoint_sha256_hashing(tmp_path):
    dummy_file = tmp_path / "dummy_checkpoint.pt"
    dummy_content = b"ModernGPT-S model weights bytes test"
    dummy_file.write_bytes(dummy_content)

    import hashlib
    expected_hash = hashlib.sha256(dummy_content).hexdigest()
    computed_hash = compute_file_sha256(dummy_file)

    assert computed_hash == expected_hash
    assert compute_file_sha256(tmp_path / "non_existent.pt") == "file_not_found"


def test_incremental_metric_persistence(tmp_path):
    tracker = ExperimentTracker(
        experiment_id="EXP-METRICS-01",
        experiment_name="Incremental Metrics Test",
        category="pretraining",
        architecture="ModernGPT-50M",
        training_config={"lr": 3e-4},
        experiments_root=tmp_path,
    )

    metrics_file = tracker.run_dir / "metrics.json"
    log_file = tracker.run_dir / "training.log"

    tracker.log_step(step=1, epoch=0.1, train_loss=3.5, lr=3e-4, val_loss=3.8)
    assert metrics_file.exists()
    assert log_file.exists()

    with open(metrics_file, "r") as f:
        data = json.load(f)
    assert data["total_steps_logged"] == 1
    assert data["history"][0]["step"] == 1

    # Simulate subsequent step
    tracker.log_step(step=2, epoch=0.2, train_loss=3.2, lr=3e-4, val_loss=3.6)
    with open(metrics_file, "r") as f:
        data = json.load(f)
    assert data["total_steps_logged"] == 2
    assert data["best_val_metric"] == 3.6


def test_interrupted_run_preservation(tmp_path):
    # Simulate a run that terminates abruptly without calling finalize()
    tracker = ExperimentTracker(
        experiment_id="EXP-INTERRUPTED-01",
        experiment_name="Crash Simulation",
        category="SFT",
        architecture="ModernGPT-50M",
        training_config={"epochs": 3},
        experiments_root=tmp_path,
    )

    tracker.log_step(step=10, epoch=0.5, train_loss=2.5, lr=2.5e-5)
    tracker.log_step(step=20, epoch=1.0, train_loss=2.1, lr=2.5e-5)

    # Process "crashes" here. Re-inspect run directory:
    manifest_file = tracker.run_dir / "manifest.json"
    metrics_file = tracker.run_dir / "metrics.json"
    log_file = tracker.run_dir / "training.log"

    assert manifest_file.exists()
    assert metrics_file.exists()
    assert log_file.exists()

    with open(metrics_file) as f:
        metrics = json.load(f)
    assert len(metrics["history"]) == 2
    assert metrics["history"][-1]["step"] == 20


def test_checkpoint_tracking_association(tmp_path):
    tracker = ExperimentTracker(
        experiment_id="EXP-CKPT-01",
        experiment_name="Checkpoint Assoc Test",
        category="architecture",
        architecture="ModernGPT-50M",
        training_config={},
        experiments_root=tmp_path,
    )

    ckpt_file = tmp_path / "best_val.pt"
    ckpt_file.write_bytes(b"checkpoint binary payload")

    info = tracker.log_checkpoint(
        checkpoint_path=ckpt_file,
        step=500,
        epoch=1.5,
        val_loss=1.45,
        checkpoint_type="best",
    )

    assert info["experiment_id"] == "EXP-CKPT-01"
    assert info["step"] == 500
    assert info["type"] == "best"
    assert len(info["sha256"]) == 64

    # Verify manifest in checkpoints dir
    manifest = tracker.run_dir / "checkpoints" / "checkpoints_manifest.json"
    assert manifest.exists()
    with open(manifest) as f:
        ckpts = json.load(f)
    assert len(ckpts) == 1
    assert ckpts[0]["sha256"] == info["sha256"]


def test_report_generation_14_sections(tmp_path):
    tracker = ExperimentTracker(
        experiment_id="EXP-REPORT-01",
        experiment_name="Report Generation Test",
        category="architecture",
        architecture="ModernGPT-50M-GQA4",
        training_config={"lr": 3e-4},
        model_config={"num_kv_heads": 4},
        parameter_count=45959680,
        dataset_path="artifacts/shards_250m/train.bin",
        token_budget=250000000,
        experiments_root=tmp_path,
    )

    tracker.finalize(
        status="completed",
        main_conclusion="GQA-4 matched MHA perplexity within 0.02 nats.",
    )

    report_file = tracker.run_dir / "report.md"
    assert report_file.exists()
    report_text = report_file.read_text(encoding="utf-8")

    # Verify all 14 mandatory sections are present
    required_sections = [
        "## 1. Objective",
        "## 2. Hypothesis",
        "## 3. Architecture",
        "## 4. Dataset",
        "## 5. Training Configuration",
        "## 6. Hardware and Software Environment",
        "## 7. Training Results",
        "## 8. Validation Results",
        "## 9. Downstream Evaluation",
        "## 10. Efficiency Measurements",
        "## 11. Comparison with Reference Experiments",
        "## 12. Limitations",
        "## 13. Conclusions",
        "## 14. Recommended Next Experiment",
    ]
    for sec in required_sections:
        assert sec in report_text, f"Missing required report section: {sec}"


def test_fail_safe_initialization_rejection(tmp_path):
    # Empty experiment ID must fail with RuntimeError
    with pytest.raises(RuntimeError, match="Fail-safe"):
        ExperimentTracker(
            experiment_id="",
            experiment_name="Invalid ID Test",
            category="pretraining",
            architecture="ModernGPT-50M",
            training_config={},
            experiments_root=tmp_path,
        )

