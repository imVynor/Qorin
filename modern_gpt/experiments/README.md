# ModernGPT-S Permanent Experiment Tracking Infrastructure

This directory contains the central experiment registry, metadata capture system, incremental metric logger, and benchmark tracking repository for all ModernGPT-S research.

---

## 1. Directory Structure

```text
experiments/
├── README.md                      # Architecture & usage documentation (this file)
├── experiment_registry.jsonl      # Complete, append-only JSONL experiment registry
├── experiment_registry.csv        # Tabular CSV summary of all experiments
├── registry.py                    # Registry manager module (enforces uniqueness & atomic writes)
├── tracker.py                     # ExperimentTracker class (metadata, metrics, checkpoints, reports)
├── register_historical.py         # Utility script registering baseline historical runs
└── runs/
    └── <experiment_id>/           # Dedicated isolated run directory per experiment
        ├── manifest.json          # Complete experiment manifest & status
        ├── config.yaml            # Exact model and training hyperparameter config
        ├── environment.json       # Captured hardware, software, Git, and package metadata
        ├── training.log           # Streaming plaintext training logs
        ├── metrics.json           # Incremental step-by-step metric history
        ├── evaluation/            # Downstream benchmark evaluation tracking
        │   ├── evaluation_config.json
        │   ├── results.json
        │   ├── failure_analysis.json
        │   └── raw_generations/   # Full raw model outputs for reproducibility
        ├── profiling/             # Hardware efficiency & latency measurements
        │   ├── throughput.json
        │   ├── hardware_metrics.json
        │   └── memory_usage.json
        ├── checkpoints/           # Checkpoints manifest & SHA-256 hashes
        └── report.md              # 14-section standardized experiment report
```

---

## 2. Mandatory Experiment Metadata Capture

Every experiment initialized through `ExperimentTracker` automatically captures:
- **System & Hardware:** Platform, OS release, CPU, GPU name, total VRAM, CUDA capability, CUDA version, PyTorch version.
- **Git State:** Current commit hash, working tree clean status.
- **Packages:** Installed versions of PyTorch, Transformers, Datasets, NumPy, PyYAML.
- **Datasets:** File paths, record counts, and SHA-256 cryptographic hashes.
- **Tokenizer:** Tokenizer name and exact fingerprint (`be6c3d60c5baf217`).
- **Reproducibility:** Random seeds, batch sizes, accumulation steps, precision mode, context window.
- **Launch Command:** Complete verbatim CLI command string.

---

## 3. Python API Usage

### Initializing an Experiment Run:
```python
from experiments.tracker import ExperimentTracker

tracker = ExperimentTracker(
    experiment_id="EXP-20261103-GQA4-PRETRAIN",
    experiment_name="50M GQA-4 Pretraining Pilot",
    category="architecture",
    architecture="ModernGPT-50M-GQA4",
    training_config={"lr": 3e-4, "batch_size": 16, "grad_accum": 4},
    model_config=asdict(config),
    parameter_count=model.parameter_count,
    dataset_path="artifacts/shards_250m/train.bin",
    tokenizer_name="bpe_16k_v2",
    tokenizer_fingerprint="be6c3d60c5baf217",
    token_budget=25000000,
    parent_experiment_id="EXP-PRETRAIN-50M-250M",
)
```

### Logging Training Steps Incrementally:
```python
tracker.log_step(
    step=step,
    epoch=epoch,
    train_loss=loss.item(),
    lr=current_lr,
    val_loss=val_loss,
    grad_norm=float(grad_norm),
    tokens_processed=tokens_seen,
    tokens_per_sec=tokens_per_second,
    vram_allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
)
```

### Logging a Checkpoint with SHA-256 Hashing:
```python
tracker.log_checkpoint(
    checkpoint_path=ckpt_file,
    step=step,
    epoch=epoch,
    val_loss=val_loss,
    checkpoint_type="best" if is_best else "periodic",
)
```

### Finalizing the Run & Auto-Generating Report:
```python
tracker.finalize(
    status="completed",
    main_conclusion="GQA-4 matched MHA loss within 0.02 nats while reducing KV cache by 50%.",
    custom_report_sections={
        "objective": "Evaluate KV cache memory savings and pretraining convergence of GQA-4 vs MHA.",
        "hypothesis": "GQA-4 retains >=98% of MHA language modeling capacity while doubling inference throughput.",
    }
)
```

---

## 4. Fail-Safe Invariant

If `ExperimentTracker` fails to initialize or the run directory cannot be established, training is immediately aborted with a `RuntimeError`. Untracked training runs are strictly prohibited in the ModernGPT-S research framework.

