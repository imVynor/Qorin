"""ModernGPT-S 100M: Pretraining Engine with 19-Point State Recovery & Thermal Guardrails.

Features:
- Full 19-point state recovery preserving exact RNG, shard position, and optimizer states.
- Atomic two-stage checkpointing with verification and rolling backups.
- Integrated thermal watchdog with graceful pausing and cooling breaks.
- Session-based execution (e.g. 2-hour max sessions with auto-resumption).
- Mixed precision (bf16/fp16/fp32) and gradient accumulation.
- Real-time logging of loss, throughput (tok/s), VRAM, and GPU temperatures.
"""

import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import ModernGPTConfig, get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.monitor_gpu import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.Pretrain100M")


def _serialize_numpy_rng(s):
    if s is None:
        return None
    try:
        return (s[0], s[1].tolist() if hasattr(s[1], "tolist") else s[1], s[2], s[3], s[4])
    except Exception:
        return None


def _deserialize_numpy_rng(s):
    if s is None:
        return None
    try:
        return (s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4])
    except Exception:
        return None


class ShardTokenDataset(Dataset):
    """Memory-mapped dataset reading fixed-length sequences from uint16 token shard."""

    def __init__(self, shard_path: Path, context_length: int = 512, start_offset: int = 0):
        self.shard_path = Path(shard_path)
        self.context_length = context_length
        self.tokens = np.memmap(self.shard_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.tokens)
        self.num_sequences = max(0, (self.total_tokens - 1) // self.context_length)
        self.start_offset = start_offset

    def __len__(self) -> int:
        return max(0, self.num_sequences - self.start_offset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.start_offset + idx
        start = actual_idx * self.context_length
        end = start + self.context_length
        chunk = self.tokens[start : end + 1]
        inputs = torch.from_numpy(chunk[:-1].astype(np.int64))
        targets = torch.from_numpy(chunk[1:].astype(np.int64))
        return inputs, targets


def compute_cosine_lr(
    step: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint_atomic(
    checkpoint_path: Path,
    state_dict: Dict[str, Any],
    backup_dir: Optional[Path] = None,
    max_backups: int = 5,
) -> None:
    """Atomically writes 19-point checkpoint state to disk with validation and rolling backups."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".pt.tmp")

    # 1. Write state to temporary file
    torch.save(state_dict, tmp_path)

    # 2. Verify readability
    try:
        check = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert "model" in check, "Checkpoint missing 'model' key"
        assert "optimizer" in check, "Checkpoint missing 'optimizer' key"
        assert "step" in check, "Checkpoint missing 'step' key"
        del check
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Atomic checkpoint verification failed: {e}")

    # 3. Maintain rolling backups of existing checkpoint
    if checkpoint_path.exists() and backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        step = state_dict.get("step", 0)
        ts = int(time.time())
        backup_file = backup_dir / f"checkpoint_step_{step}_{ts}.pt"
        shutil.copy2(checkpoint_path, backup_file)

        # Prune old backups
        backups = sorted(backup_dir.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            try:
                oldest.unlink()
            except Exception:
                pass

    # 4. Atomic file rename
    os.replace(tmp_path, checkpoint_path)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    precision_dtype: torch.dtype,
    max_batches: int = 50,
) -> Tuple[float, float]:
    """Computes validation cross-entropy loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for b_idx, (inputs, targets) in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(20.0, avg_loss))
    model.train()
    return avg_loss, perplexity


class PretrainTrainer:
    def __init__(
        self,
        model_config: ModernGPTConfig,
        train_config: Dict[str, Any],
        session_config: Dict[str, Any],
        device: torch.device,
        session_id: Optional[str] = None,
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.session_config = session_config
        self.device = device
        self.session_id = session_id or f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self.output_dir = REPO_ROOT / train_config.get("output_dir", "checkpoints/modern_gpts_100m")
        self.backup_dir = REPO_ROOT / train_config.get("session_backup_dir", "checkpoints/modern_gpts_100m/session_backups")
        self.latest_ckpt = self.output_dir / "latest.pt"
        self.best_ckpt = self.output_dir / "best_val.pt"

        self.log_dir = REPO_ROOT / train_config.get("log_dir", "logs/pretrain_100m")
        self.metrics_file = REPO_ROOT / train_config.get("metrics_file", "logs/pretrain_100m/training_metrics.jsonl")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Hardware & Precision
        precision_str = train_config.get("precision", "bf16").lower()
        if precision_str == "bf16" and torch.cuda.is_bf16_supported():
            self.precision_dtype = torch.bfloat16
            self.scaler = None
        elif precision_str == "fp16":
            self.precision_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.precision_dtype = torch.float32
            self.scaler = None

        # Build Model
        logger.info(f"Initializing ModernGPT 100M model on {device}...")
        self.model = ModernGPT(model_config).to(device)
        logger.info(f"Model parameters: {self.model.num_parameters():,}")

        # Optimizer
        self.base_lr = float(train_config.get("learning_rate", 2.5e-4))
        self.min_lr = float(train_config.get("min_learning_rate", 2.5e-5))
        self.weight_decay = float(train_config.get("weight_decay", 0.1))
        self.warmup_steps = int(train_config.get("warmup_steps", 2000))
        self.total_steps = int(train_config.get("total_optimizer_steps", 45776))
        self.grad_clip = float(train_config.get("grad_clip_norm", 1.0))

        # Weight decay grouping
        decay_params = [p for p in self.model.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.model.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.base_lr,
            betas=(train_config.get("beta1", 0.9), train_config.get("beta2", 0.95)),
            eps=float(train_config.get("eps", 1e-8)),
        )

        # Training state trackers
        self.global_step = 0
        self.tokens_seen = 0
        self.shard_index = 0
        self.sequence_offset = 0
        self.accum_steps = 0
        self.accum_loss = 0.0
        self.best_val_loss = float("inf")
        self.stop_requested = False

        # Thermal Monitor
        thermal_log = REPO_ROOT / session_config.get("thermal_log_file", "logs/sessions/thermal_telemetry.jsonl")
        self.monitor = GPUMonitor(
            log_file=thermal_log,
            poll_interval_seconds=session_config.get("thermal_monitor_interval_seconds", 10.0),
            warn_temp_celsius=session_config.get("warn_gpu_temp_celsius", 90.0),
            max_temp_celsius=session_config.get("max_gpu_temp_celsius", 95.0),
            emergency_cooldown_temp=session_config.get("emergency_cooldown_temp_celsius", 70.0),
        )

        # Signal handlers for graceful termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.warning(f"Termination signal {signum} received. Requesting graceful pause after current step...")
        self.stop_requested = True

    def build_checkpoint_state(self, current_lr: float) -> Dict[str, Any]:
        """Constructs complete 19-point state recovery dictionary."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {
                "step": self.global_step,
                "current_lr": current_lr,
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "base_lr": self.base_lr,
                "min_lr": self.min_lr,
                "schedule": "cosine",
            },
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "grad_scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _serialize_numpy_rng(np.random.get_state()),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data_sampler_state": {
                "shard_index": self.shard_index,
                "sequence_offset": self.sequence_offset,
            },
            "shard_index": self.shard_index,
            "sequence_offset": self.sequence_offset,
            "accum_steps": self.accum_steps,
            "accum_loss": self.accum_loss,
            "training_config": self.train_config,
            "model_config": asdict(self.model_config),
            "dataset_manifest_hash": self.train_config.get("dataset_manifest_hash", "none"),
            "tokenizer_fingerprint": self.model_config.tokenizer_fingerprint if hasattr(self.model_config, "tokenizer_fingerprint") else "be6c3d60c5baf217",
            "experiment_id": self.train_config.get("experiment_id", "pretrain_100m_1.5b"),
            "session_id": self.session_id,
        }

    def load_checkpoint(self, path: Path) -> None:
        """Restores all 19 state recovery points."""
        logger.info(f"Loading checkpoint from {path}...")
        state = torch.load(path, map_location=self.device, weights_only=False)

        # 1. Model & Optimizer
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

        # 2. Counters
        self.global_step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        self.shard_index = state.get("shard_index", 0)
        self.sequence_offset = state.get("sequence_offset", 0)
        self.accum_steps = state.get("accum_steps", 0)
        self.accum_loss = state.get("accum_loss", 0.0)

        # 3. Scaler
        if self.scaler is not None and state.get("grad_scaler") is not None:
            self.scaler.load_state_dict(state["grad_scaler"])

        # 4. RNG states
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
        if "numpy_rng_state" in state and state["numpy_rng_state"] is not None:
            deserialized = _deserialize_numpy_rng(state["numpy_rng_state"])
            if deserialized is not None:
                np.random.set_state(deserialized)
        if "torch_cpu_rng_state" in state and state["torch_cpu_rng_state"] is not None:
            cpu_state = state["torch_cpu_rng_state"]
            if hasattr(cpu_state, "cpu"):
                cpu_state = cpu_state.cpu()
            torch.set_rng_state(cpu_state)
        if "cuda_rng_state" in state and state["cuda_rng_state"] is not None and torch.cuda.is_available():
            cuda_states = [s.cpu() if hasattr(s, "cpu") else s for s in state["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_states)

        logger.info(
            f"✓ Resumed from step {self.global_step:,} ({self.tokens_seen:,} tokens, "
            f"shard {self.shard_index}, seq_offset {self.sequence_offset})"
        )

    def train_session(
        self,
        shard_paths: List[Path],
        val_path: Optional[Path] = None,
        max_session_minutes: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs pretraining for a bounded session duration or step target."""
        # Reset stop signal and update session id for new session run
        self.stop_requested = False
        self.session_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Immediate exit if global training target is already satisfied
        total_tokens_budget = int(self.train_config.get("total_tokens", 1500020736))
        if self.global_step >= self.total_steps or self.tokens_seen >= total_tokens_budget:
            logger.info(
                f"🎯 Global training target already satisfied: Step {self.global_step:,}/{self.total_steps:,} | "
                f"Tokens {self.tokens_seen:,}/{total_tokens_budget:,}."
            )
            return {
                "session_id": self.session_id,
                "outcome": "GLOBAL_TARGET_REACHED",
                "duration_minutes": 0.0,
                "steps_completed": 0,
                "tokens_processed": 0,
                "final_global_step": self.global_step,
                "total_tokens_seen": self.tokens_seen,
                "checkpoint_path": str(self.latest_ckpt),
            }

        self.monitor.start()
        session_minutes = max_session_minutes or float(self.session_config.get("session_duration_minutes", 120.0))
        max_seconds = session_minutes * 60.0
        session_start_time = time.time()

        micro_batch = int(self.train_config.get("micro_batch_size", 8))
        grad_accum = int(self.train_config.get("gradient_accumulation_steps", 8))
        eval_interval = int(self.train_config.get("eval_interval_steps", 500))
        save_interval = int(self.train_config.get("save_interval_steps", 500))
        eval_batches = int(self.train_config.get("eval_batches", 50))
        context_len = self.model_config.context_length

        # Validation DataLoader if val_path provided
        val_loader = None
        if val_path and Path(val_path).exists():
            val_dataset = ShardTokenDataset(val_path, context_length=context_len)
            val_loader = DataLoader(val_dataset, batch_size=micro_batch, shuffle=False)
            logger.info(f"Validation dataset loaded from {val_path} ({len(val_dataset):,} sequences)")

        logger.info(f"=== Starting Session: {self.session_id} ===")
        logger.info(f"Target duration: {session_minutes} min | Micro-batch: {micro_batch} | Grad accum: {grad_accum}")
        logger.info(f"Effective batch size: {micro_batch * grad_accum} ({micro_batch * grad_accum * context_len:,} tokens/step)")

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        tokens_this_session = 0
        steps_this_session = 0
        session_outcome = "IN_PROGRESS"
        session_time_limit_hit = False

        if self.accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)
        step_start_time = time.time()

        while self.shard_index < len(shard_paths) and not self.stop_requested:
            current_shard = shard_paths[self.shard_index]
            logger.info(f"Loading shard {self.shard_index}/{len(shard_paths)}: {current_shard.name} (offset: {self.sequence_offset})")
            
            shard_dataset = ShardTokenDataset(
                current_shard,
                context_length=context_len,
                start_offset=self.sequence_offset,
            )
            shard_loader = DataLoader(
                shard_dataset,
                batch_size=micro_batch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for inputs, targets in shard_loader:
                if self.stop_requested:
                    session_outcome = "INTERRUPTED_SIGNAL"
                    break

                # Check time boundary: complete active accumulation group before pausing
                elapsed_session = time.time() - session_start_time
                if elapsed_session >= max_seconds and not session_time_limit_hit:
                    session_time_limit_hit = True
                    if self.accum_steps > 0:
                        logger.info(
                            f"Session time limit ({session_minutes:.1f}m) reached during gradient accumulation "
                            f"({self.accum_steps}/{grad_accum} micro-batches). "
                            f"Completing current accumulation group before pausing..."
                        )
                    else:
                        logger.info(f"Session time limit ({session_minutes:.1f}m) reached at step boundary. Initiating clean pause.")
                        session_outcome = "SESSION_TIME_LIMIT"
                        self.stop_requested = True
                        break

                # Thermal guardrail check
                is_hot, current_temp = self.monitor.is_overheating()
                if is_hot:
                    logger.warning(f"⚠️ GPU Temp {current_temp}°C >= {self.monitor.max_temp}°C! Cooling down.")
                    self.monitor.wait_for_cooldown()

                # Micro-batch forward and backward
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=self.precision_dtype):
                    logits = self.model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss = loss / grad_accum

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # accum_loss sums the per-microbatch loss scaled by 1/grad_accum,
                # yielding the exact per-token cross entropy mean across the full optimizer step.
                self.accum_loss += loss.item()
                self.accum_steps += 1
                self.sequence_offset += micro_batch
                self.tokens_seen += inputs.numel()
                tokens_this_session += inputs.numel()

                # Optimizer step upon full accumulation
                if self.accum_steps == grad_accum:
                    lr = compute_cosine_lr(
                        self.global_step,
                        self.base_lr,
                        self.min_lr,
                        self.warmup_steps,
                        self.total_steps,
                    )
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    steps_this_session += 1

                    step_time = time.time() - step_start_time
                    step_tokens = micro_batch * grad_accum * context_len
                    tok_per_sec = step_tokens / max(1e-5, step_time)

                    # Log metrics
                    gpu_m = self.monitor.get_latest()
                    mem_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
                    mem_res_mb = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0

                    if self.global_step % 10 == 0 or self.global_step == 1:
                        logger.info(
                            f"Step {self.global_step:,}/{self.total_steps:,} | "
                            f"Loss: {self.accum_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tokens: {self.tokens_seen:,} | "
                            f"{tok_per_sec:,.0f} tok/s | "
                            f"VRAM: {mem_alloc_mb:.0f}/{mem_res_mb:.0f} MB | "
                            f"GPU: {gpu_m.get('gpu_temp_c', 0)}°C"
                        )

                    log_entry = {
                        "step": self.global_step,
                        "loss": round(self.accum_loss, 4),
                        "lr": lr,
                        "tokens_seen": self.tokens_seen,
                        "tok_per_sec": round(tok_per_sec, 1),
                        "step_time_sec": round(step_time, 3),
                        "mem_alloc_mb": round(mem_alloc_mb, 1),
                        "mem_res_mb": round(mem_res_mb, 1),
                        "gpu_temp_c": gpu_m.get("gpu_temp_c"),
                        "gpu_power_w": gpu_m.get("power_draw_w"),
                        "shard_index": self.shard_index,
                        "sequence_offset": self.sequence_offset,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(self.metrics_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    # Periodic Evaluation
                    if val_loader is not None and (self.global_step % eval_interval == 0):
                        v_loss, v_ppl = evaluate(
                            self.model, val_loader, self.device, self.precision_dtype, max_batches=eval_batches
                        )
                        logger.info(f"📊 Validation at Step {self.global_step}: Loss {v_loss:.4f} | Perplexity {v_ppl:.2f}")
                        if v_loss < self.best_val_loss:
                            self.best_val_loss = v_loss
                            logger.info(f"⭐ New Best Val Loss {v_loss:.4f}! Saving {self.best_ckpt.name}...")
                            best_state = self.build_checkpoint_state(lr)
                            save_checkpoint_atomic(self.best_ckpt, best_state)

                    # Periodic Save
                    if self.global_step % save_interval == 0:
                        logger.info(f"💾 Saving periodic checkpoint at step {self.global_step}...")
                        save_state = self.build_checkpoint_state(lr)
                        save_checkpoint_atomic(self.latest_ckpt, save_state, self.backup_dir)

                    self.accum_loss = 0.0
                    self.accum_steps = 0
                    step_start_time = time.time()

                    # Check if session time limit was requested during this accumulation group
                    if session_time_limit_hit:
                        logger.info(f"Accumulation group completed at step {self.global_step}. Pausing session cleanly at step boundary.")
                        current_lr = compute_cosine_lr(self.global_step, self.base_lr, self.min_lr, self.warmup_steps, self.total_steps)
                        save_state = self.build_checkpoint_state(current_lr)
                        save_checkpoint_atomic(self.latest_ckpt, save_state, self.backup_dir)
                        session_outcome = "SESSION_TIME_LIMIT"
                        self.stop_requested = True
                        break

                    # Check global target completion
                    if self.global_step >= self.total_steps:
                        logger.info(f"🎯 Global training target reached ({self.global_step:,}/{self.total_steps:,} steps).")
                        session_outcome = "GLOBAL_TARGET_REACHED"
                        self.stop_requested = True
                        break

                    # Check max steps limit (for benchmarks/tests)
                    if max_steps and self.global_step >= max_steps:
                        logger.info(f"Step limit {max_steps} reached.")
                        session_outcome = "MAX_STEPS_REACHED"
                        self.stop_requested = True
                        break

            # End of shard
            if not self.stop_requested:
                self.shard_index += 1
                self.sequence_offset = 0

        # Resolve final session outcome if still IN_PROGRESS
        if session_outcome == "IN_PROGRESS":
            if self.global_step >= self.total_steps:
                session_outcome = "GLOBAL_TARGET_REACHED"
            elif self.shard_index >= len(shard_paths):
                session_outcome = "CORPUS_EXHAUSTED"
                if self.accum_steps > 0:
                    logger.warning(
                        f"Corpus exhausted with {self.accum_steps} residual micro-batches "
                        f"({self.accum_steps * micro_batch * context_len:,} tokens) remaining unstepped."
                    )
            elif steps_this_session == 0:
                session_outcome = "NO_STEPS_EXECUTED"
                logger.error("Session terminated without executing any steps!")
            else:
                session_outcome = "SESSION_FINISHED"

        # End of session: save final checkpoint
        current_lr = compute_cosine_lr(self.global_step, self.base_lr, self.min_lr, self.warmup_steps, self.total_steps)
        final_state = self.build_checkpoint_state(current_lr)
        save_checkpoint_atomic(self.latest_ckpt, final_state, self.backup_dir)
        self.monitor.stop()

        elapsed_total = time.time() - session_start_time
        logger.info(f"=== Session {self.session_id} Finished ===")
        logger.info(f"Outcome: {session_outcome}")
        logger.info(f"Duration: {elapsed_total / 60.0:.2f} min | Steps completed: {steps_this_session:,}")
        logger.info(f"Tokens processed: {tokens_this_session:,} (Total lifetime: {self.tokens_seen:,})")
        logger.info(f"Checkpoint saved to {self.latest_ckpt}")

        return {
            "session_id": self.session_id,
            "outcome": session_outcome,
            "duration_minutes": elapsed_total / 60.0,
            "steps_completed": steps_this_session,
            "tokens_processed": tokens_this_session,
            "final_global_step": self.global_step,
            "total_tokens_seen": self.tokens_seen,
            "checkpoint_path": str(self.latest_ckpt),
        }
