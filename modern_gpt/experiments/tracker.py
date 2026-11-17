import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import yaml

import torch

from .registry import ExperimentRegistry


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file efficiently in chunks."""
    filepath = Path(filepath)
    if not filepath.exists() or not filepath.is_file():
        return "file_not_found"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def capture_environment_metadata() -> Dict[str, Any]:
    """Capture comprehensive hardware, software, and Git metadata."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

    # GPU Metadata
    if torch.cuda.is_available():
        meta["gpu_count"] = torch.cuda.device_count()
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        meta["gpu_vram_total_mb"] = round(props.total_memory / (1024 * 1024), 2)
        meta["gpu_capability"] = f"{props.major}.{props.minor}"
    else:
        meta["gpu_count"] = 0
        meta["gpu_name"] = None
        meta["gpu_vram_total_mb"] = 0.0

    # Git Metadata
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_commit"] = git_commit
        meta["git_clean"] = len(git_status) == 0
    except Exception:
        meta["git_commit"] = "git_unavailable"
        meta["git_clean"] = False

    # Package Versions
    pkg_versions = {}
    for pkg in ["numpy", "torch", "transformers", "datasets", "pytest", "yaml"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkg_versions[pkg] = "not_installed"
    meta["installed_packages"] = pkg_versions

    return meta


class ExperimentTracker:
    """Manages tracking, logging, checkpointing, and reporting for a single experiment run."""

    def __init__(
        self,
        experiment_id: str,
        experiment_name: str,
        category: str,
        architecture: str,
        training_config: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        parameter_count: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_version: Optional[str] = "v1",
        tokenizer_name: Optional[str] = None,
        tokenizer_fingerprint: Optional[str] = None,
        token_budget: Optional[int] = None,
        parent_experiment_id: Optional[str] = "None",
        experiments_root: Optional[Path] = None,
        launch_command: Optional[str] = None,
    ):
        if not experiment_id or not experiment_id.strip():
            raise RuntimeError("Fail-safe trigger: Experiment initialization failed: experiment_id cannot be empty")

        if experiments_root is None:
            experiments_root = Path(__file__).parent
        self.experiments_root = Path(experiments_root)
        self.registry = ExperimentRegistry(self.experiments_root)

        self.experiment_id = experiment_id.strip()
        self.experiment_name = experiment_name
        self.category = category
        self.architecture = architecture
        self.training_config = training_config or {}
        self.model_config = model_config or {}
        self.parameter_count = parameter_count
        self.dataset_path = dataset_path
        self.dataset_version = dataset_version
        self.tokenizer_name = tokenizer_name
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.token_budget = token_budget
        self.parent_experiment_id = parent_experiment_id
        self.launch_command = launch_command or " ".join(sys.argv)
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        # Establish run directory structure
        self.run_dir = self.experiments_root / "runs" / self.experiment_id
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.eval_dir = self.run_dir / "evaluation"
            self.eval_dir.mkdir(exist_ok=True)
            (self.eval_dir / "raw_generations").mkdir(exist_ok=True)
            self.profiling_dir = self.run_dir / "profiling"
            self.profiling_dir.mkdir(exist_ok=True)
            self.checkpoints_dir = self.run_dir / "checkpoints"
            self.checkpoints_dir.mkdir(exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Fail-safe trigger: Failed to initialize run directory at {self.run_dir}: {e}")

        # Metrics state
        self.metrics_history: List[Dict[str, Any]] = []
        self.checkpoints_logged: List[Dict[str, Any]] = []
        self.evaluations_logged: List[Dict[str, Any]] = []
        self.best_val_metric: Optional[float] = None
        self.final_val_metric: Optional[float] = None

        # Capture and save environment & config
        self.environment = capture_environment_metadata()
        with open(self.run_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(self.environment, f, indent=2)

        config_payload = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "category": self.category,
            "architecture": self.architecture,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "launch_command": self.launch_command,
        }
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config_payload, f, default_flow_style=False)

        # Dataset hashes
        self.dataset_sha256 = "unavailable"
        if self.dataset_path and Path(self.dataset_path).exists():
            self.dataset_sha256 = compute_file_sha256(Path(self.dataset_path))

        # Initial manifest
        self._update_manifest(status="running")

        # Register in central registry
        self._sync_registry(status="running", main_conclusion="Experiment in progress")

    def _update_manifest(self, status: str = "running", main_conclusion: str = "In progress"):
        manifest = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "category": self.category,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "dataset": self.dataset_path,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "training_token_budget": self.token_budget,
            "start_time": self.start_time,
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) if status != "running" else "running",
            "status": status,
            "launch_command": self.launch_command,
            "parent_experiment_id": self.parent_experiment_id,
            "best_val_metric": self.best_val_metric,
            "final_val_metric": self.final_val_metric,
            "checkpoints_count": len(self.checkpoints_logged),
            "evaluations_count": len(self.evaluations_logged),
            "main_conclusion": main_conclusion,
        }
        with open(self.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _sync_registry(self, status: str, main_conclusion: str):
        record = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "category": self.category,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "dataset": self.dataset_path or "unavailable",
            "dataset_version": self.dataset_version or "v1",
            "tokenizer_fingerprint": self.tokenizer_fingerprint or "unavailable",
            "training_token_budget": self.token_budget or 0,
            "training_config": self.training_config,
            "start_time": self.start_time,
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) if status != "running" else "running",
            "status": status,
            "checkpoint_locations": [c["path"] for c in self.checkpoints_logged],
            "best_val_metric": self.best_val_metric,
            "final_val_metric": self.final_val_metric,
            "evaluation_report_location": str(self.run_dir / "evaluation" / "results.json"),
            "experiment_report_location": str(self.run_dir / "report.md"),
            "parent_experiment_id": self.parent_experiment_id,
            "main_conclusion": main_conclusion,
        }
        self.registry.register_or_update(record)

    def log_step(
        self,
        step: int,
        epoch: float,
        train_loss: float,
        lr: float,
        val_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        tokens_processed: Optional[int] = None,
        tokens_per_sec: Optional[float] = None,
        vram_allocated_mb: Optional[float] = None,
    ):
        """Incrementally log a training step to metrics.json and training.log."""
        entry = {
            "step": step,
            "epoch": round(epoch, 4),
            "train_loss": round(train_loss, 4),
            "lr": lr,
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "grad_norm": round(grad_norm, 4) if grad_norm is not None else None,
            "tokens_processed": tokens_processed,
            "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
            "vram_allocated_mb": round(vram_allocated_mb, 2) if vram_allocated_mb is not None else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        self.metrics_history.append(entry)

        if val_loss is not None:
            self.final_val_metric = val_loss
            if self.best_val_metric is None or val_loss < self.best_val_metric:
                self.best_val_metric = val_loss

        # Incremental write to metrics.json so partial runs are never lost
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "experiment_id": self.experiment_id,
                "total_steps_logged": len(self.metrics_history),
                "best_val_metric": self.best_val_metric,
                "final_val_metric": self.final_val_metric,
                "history": self.metrics_history
            }, f, indent=2)

        # Append to streaming training.log
        log_line = (
            f"[{entry['timestamp']}] Step {step:5d} | Epoch {epoch:.2f} | "
            f"TrainLoss {train_loss:.4f} | "
            f"ValLoss {val_loss if val_loss is not None else 'N/A'} | "
            f"LR {lr:.2e} | "
            f"GradNorm {grad_norm if grad_norm is not None else 'N/A'}\n"
        )
        with open(self.run_dir / "training.log", "a", encoding="utf-8") as f:
            f.write(log_line)

    def log_checkpoint(
        self,
        checkpoint_path: Path,
        step: int,
        epoch: float,
        val_loss: Optional[float] = None,
        checkpoint_type: str = "periodic",
    ) -> Dict[str, Any]:
        """Track and cryptographically hash a checkpoint."""
        ckpt_path = Path(checkpoint_path)
        sha256 = compute_file_sha256(ckpt_path) if ckpt_path.exists() else "unavailable"
        file_size = ckpt_path.stat().st_size if ckpt_path.exists() else 0

        info = {
            "path": str(ckpt_path),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "type": checkpoint_type,
            "file_size_bytes": file_size,
            "sha256": sha256,
            "experiment_id": self.experiment_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "logged_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        self.checkpoints_logged.append(info)

        with open(self.run_dir / "checkpoints" / "checkpoints_manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.checkpoints_logged, f, indent=2)

        return info

    def log_evaluation(
        self,
        eval_name: str,
        checkpoint_path: Path,
        benchmark_name: str,
        metrics: Dict[str, Any],
        raw_generations: Optional[List[Dict[str, Any]]] = None,
        failure_analysis: Optional[Dict[str, Any]] = None,
    ):
        """Track downstream benchmark evaluations and preserve raw model generations."""
        ckpt_path = Path(checkpoint_path)
        ckpt_hash = compute_file_sha256(ckpt_path) if ckpt_path.exists() else "unavailable"

        eval_record = {
            "eval_name": eval_name,
            "benchmark_name": benchmark_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_hash,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "metrics": metrics,
        }
        self.evaluations_logged.append(eval_record)

        # Write results.json
        with open(self.eval_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(self.evaluations_logged, f, indent=2)

        # Write raw generations if provided
        if raw_generations:
            gen_file = self.eval_dir / "raw_generations" / f"{eval_name}_raw_generations.json"
            with open(gen_file, "w", encoding="utf-8") as f:
                json.dump(raw_generations, f, indent=2)

        # Write failure analysis if provided
        if failure_analysis:
            with open(self.eval_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
                json.dump(failure_analysis, f, indent=2)

    def log_profiling(
        self,
        throughput_tokens_sec: float,
        prefill_latency_ms: Optional[float] = None,
        decode_latency_tok_sec: Optional[float] = None,
        peak_vram_mb: Optional[float] = None,
        kv_cache_memory_mb: Optional[float] = None,
    ):
        """Save profiling and efficiency measurements."""
        profile_data = {
            "throughput_tokens_sec": throughput_tokens_sec,
            "prefill_latency_ms": prefill_latency_ms,
            "decode_latency_tok_sec": decode_latency_tok_sec,
            "peak_vram_mb": peak_vram_mb,
            "kv_cache_memory_mb": kv_cache_memory_mb,
            "logged_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        }
        with open(self.profiling_dir / "throughput.json", "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

    def finalize(
        self,
        status: str = "completed",
        main_conclusion: str = "Run completed successfully",
        custom_report_sections: Optional[Dict[str, str]] = None,
    ):
        """Finalize the experiment run, update registry, and generate report.md."""
        self._update_manifest(status=status, main_conclusion=main_conclusion)
        self._sync_registry(status=status, main_conclusion=main_conclusion)
        self.generate_report(main_conclusion=main_conclusion, custom_sections=custom_report_sections)

    def generate_report(
        self,
        main_conclusion: str = "Completed successfully",
        custom_sections: Optional[Dict[str, str]] = None,
    ):
        """Automatically produce a comprehensive 14-section Markdown report."""
        custom = custom_sections or {}

        report_md = f"""# Experiment Report: {self.experiment_id} ({self.experiment_name})

**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}  
**Status:** {self.get_manifest().get('status', 'completed')}  
**Category:** {self.category}  
**Architecture:** {self.architecture} ({self.parameter_count:,} parameters)  

---

## 1. Objective
{custom.get("objective", f"Execute {self.category} run for {self.architecture} under controlled configuration.")}

## 2. Hypothesis
{custom.get("hypothesis", "Theoretical expectation based on scaling laws or architectural inductive bias.")}

## 3. Architecture
- **Model Architecture:** `{self.architecture}`
- **Parameters:** {self.parameter_count:,}
- **Configuration Details:**
```yaml
{yaml.dump(self.model_config, default_flow_style=False)}
```

## 4. Dataset
- **Dataset Path:** `{self.dataset_path}`
- **Dataset Version:** `{self.dataset_version}`
- **SHA-256 Hash:** `{self.dataset_sha256}`
- **Tokenizer Name / Fingerprint:** `{self.tokenizer_name}` (`{self.tokenizer_fingerprint}`)
- **Token Budget:** {self.token_budget:,} tokens

## 5. Training Configuration
- **Launch Command:** `{self.launch_command}`
- **Optimizer & Scheduler:**
```yaml
{yaml.dump(self.training_config, default_flow_style=False)}
```

## 6. Hardware and Software Environment
- **Platform:** {self.environment.get('platform')}
- **PyTorch / CUDA:** PyTorch {self.environment.get('pytorch_version')} | CUDA {self.environment.get('cuda_version')}
- **GPU:** {self.environment.get('gpu_name')} ({self.environment.get('gpu_vram_total_mb')} MB VRAM)
- **Git Commit:** `{self.environment.get('git_commit')}` (Clean: {self.environment.get('git_clean')})

## 7. Training Results
- **Total Steps Logged:** {len(self.metrics_history)}
- **Initial Loss:** {self.metrics_history[0]['train_loss'] if self.metrics_history else 'N/A'}
- **Final Train Loss:** {self.metrics_history[-1]['train_loss'] if self.metrics_history else 'N/A'}

## 8. Validation Results
- **Best Validation Metric:** {self.best_val_metric}
- **Final Validation Metric:** {self.final_val_metric}

## 9. Downstream Evaluation
{custom.get("downstream_evaluation", "Evaluated on standard held-out capability suites. See evaluation/results.json.")}

## 10. Efficiency Measurements
{custom.get("efficiency_measurements", "Throughput, VRAM, and latency measurements recorded in profiling/.")}

## 11. Comparison with Reference Experiments
- **Parent / Reference Experiment:** `{self.parent_experiment_id}`
{custom.get("comparison", "Comparative benchmark against baseline checkpoints.")}

## 12. Limitations
{custom.get("limitations", "Statistical limits, context limits, or compute budget boundaries.")}

## 13. Conclusions
{main_conclusion}

## 14. Recommended Next Experiment
{custom.get("next_experiment", "Proceed to comparative architecture benchmarking or next scaling tier.")}
"""
        with open(self.run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

    def get_manifest(self) -> Dict[str, Any]:
        manifest_file = self.run_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
