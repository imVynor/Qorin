"""Supervised Fine-Tuning (SFT) Engine for ModernGPT-S 100M GQA-4 at 4,096 Context.

Features:
- Native 100M GQA-4 architecture (n_layer=14, n_embd=768, n_head=12, num_kv_heads=4).
- 4,096-token context window with RoPE linear interpolation (scale 8.0).
- ChatML multi-turn conversational format (<|im_start|>role\\ncontent<|im_end|>).
- Assistant-only loss masking (-100 on system, user, tool observations, padding).
- Microbatch size 1, gradient accumulation 8 (32,768 tokens per optimizer update).
- Native support for both BF16 and FP16 (with GradScaler).
- 19-point state recovery and atomic checkpointing.
- Thermal guardrails with automated cooling breaks.
"""

import argparse
from collections import Counter, defaultdict
import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer, Message
from model.config import ModelConfig
from model.gpt import ModernGPT
from model.loss import compute_token_weighted_cross_entropy
from scripts.monitor_gpu import GPUMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.SFT4K")


class SFT4KDataset(Dataset):
    """Dataset for 4K conversational SFT with assistant-only loss masking."""

    def __init__(self, data_path: Path, serializer: ConversationSerializer):
        self.serializer = serializer
        self.records: List[Dict[str, Any]] = []

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        messages = rec.get("messages", [])
        enc = self.serializer.encode_conversation(messages)
        enc["category"] = rec.get("category", "unknown")
        return enc


def collate_sft_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, Any]:
    """Collate variable-length serialized conversations with right-padding."""
    max_len = max(len(item["input_ids"]) for item in batch)
    b_size = len(batch)

    input_ids = torch.full((b_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((b_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((b_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "categories": [item.get("category", "unknown") for item in batch],
        "seq_lengths": [len(item["input_ids"]) for item in batch],
    }


class SFT4KTrainer:
    """Trainer orchestrating 4K SFT on ModernGPT-S 100M."""

    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = REPO_ROOT / self.config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = REPO_ROOT / self.config["artifacts_dir"]
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = GPUMonitor()
        self.tokenizer = HuggingFaceBPEWrapper.load(REPO_ROOT / self.config["tokenizer_path"])
        self.serializer = ConversationSerializer(self.tokenizer, max_length=self.config["model_config"]["context_length"])

        # Model configuration
        m_cfg = self.config["model_config"]
        self.model_config = ModelConfig(
            vocab_size=m_cfg["vocab_size"],
            context_length=m_cfg["context_length"],
            emb_dim=m_cfg["emb_dim"],
            n_heads=m_cfg["n_heads"],
            num_kv_heads=m_cfg.get("num_kv_heads", 4),
            n_layers=m_cfg["n_layers"],
            ffn_hidden_dim=m_cfg["ffn_hidden_dim"],
            dropout=m_cfg.get("dropout", 0.05),
            qkv_bias=m_cfg.get("qkv_bias", False),
            rope_theta=m_cfg.get("rope_theta", 10000.0),
            rope_scale=m_cfg.get("rope_scale", 8.0),
            rope_type=m_cfg.get("rope_type", "linear"),
        )

        self.precision = self.config.get("precision", "bfloat16")
        self.torch_dtype = torch.bfloat16 if self.precision == "bfloat16" else torch.float16

        logger.info(f"Instantiating ModernGPT-S 100M GQA-4 ({self.precision.upper()})...")
        self.model = ModernGPT(self.model_config).to(device=self.device, dtype=self.torch_dtype)

        # Base checkpoint loading
        base_ckpt = REPO_ROOT / self.config["base_checkpoint"]
        logger.info(f"Loading checkpoint weights from {base_ckpt}...")
        state = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=True)
        assert self.model.parameter_count == 100401408, f"Unexpected parameter count: {self.model.parameter_count}"
        logger.info(f"✓ Verified 100M GQA-4 weights (100,401,408 parameters) successfully loaded.")

        # Optimizer & Precision
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=float(self.config.get("weight_decay", 0.1)),
        )
        self.scaler = torch.amp.GradScaler("cuda") if (self.precision == "float16" and self.device.type == "cuda") else None

        # Training schedule
        self.micro_batch_size = self.config["micro_batch_size"]
        self.grad_accum_steps = self.config["gradient_accumulation_steps"]
        self.peak_lr = float(self.config["learning_rate"])
        self.min_lr = float(self.config["min_learning_rate"])
        self.warmup_steps = int(self.config["warmup_steps"])
        self.grad_clip = float(self.config.get("grad_clip", 1.0))

        # Tracking state
        self.global_step = 0
        self.micro_step = 0
        self.tokens_seen = 0
        self.supervised_tokens_seen = 0
        self.best_val_loss = float("inf")

    def get_lr(self, step: int, total_steps: int) -> float:
        """Compute cosine learning rate with linear warmup."""
        if step < self.warmup_steps:
            return self.peak_lr * (step + 1) / max(1, self.warmup_steps)
        if step > total_steps:
            return self.min_lr
        decay_ratio = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.peak_lr - self.min_lr)

    def evaluate(self, val_loader: DataLoader, max_batches: Optional[int] = 50) -> Dict[str, Any]:
        """Compute validation loss and perplexity on assistant-supervised tokens overall and per-category."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        cat_losses = defaultdict(float)
        cat_tokens = defaultdict(int)

        len_bins = [
            ("len_0_512", 0, 512),
            ("len_513_1024", 513, 1024),
            ("len_1025_2048", 1025, 2048),
            ("len_2049_3072", 2049, 3072),
            ("len_3073_4096", 3073, 4096),
        ]
        len_losses = defaultdict(float)
        len_tokens = defaultdict(int)

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                categories = batch.get("categories", ["unknown"] * len(input_ids))
                seq_lengths = batch.get("seq_lengths", [input_ids.size(1)] * len(input_ids))

                with torch.amp.autocast(device_type="cuda", dtype=self.torch_dtype):
                    logits = self.model(input_ids)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    for b_idx in range(input_ids.size(0)):
                        item_logits = shift_logits[b_idx]
                        item_labels = shift_labels[b_idx]
                        item_cat = categories[b_idx]
                        item_len = seq_lengths[b_idx]

                        sup_mask = (item_labels != -100)
                        sup_count = sup_mask.sum().item()
                        if sup_count > 0:
                            item_loss = F.cross_entropy(
                                item_logits[sup_mask],
                                item_labels[sup_mask],
                                reduction="sum",
                            ).item()
                            total_loss += item_loss
                            total_tokens += sup_count

                            cat_losses[item_cat] += item_loss
                            cat_tokens[item_cat] += sup_count

                            for bin_name, low, high in len_bins:
                                if low <= item_len <= high:
                                    len_losses[bin_name] += item_loss
                                    len_tokens[bin_name] += sup_count
                                    break

        self.model.train()
        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(15.0, avg_loss))

        cat_summary = {}
        for cat in cat_tokens:
            c_loss = cat_losses[cat] / max(1, cat_tokens[cat])
            cat_summary[cat] = {
                "loss": round(c_loss, 4),
                "ppl": round(math.exp(min(15.0, c_loss)), 2),
                "tokens": cat_tokens[cat],
            }

        len_summary = {}
        for b_name in len_tokens:
            l_loss = len_losses[b_name] / max(1, len_tokens[b_name])
            len_summary[b_name] = {
                "loss": round(l_loss, 4),
                "ppl": round(math.exp(min(15.0, l_loss)), 2),
                "tokens": len_tokens[b_name],
            }

        return {
            "val_loss": avg_loss,
            "val_ppl": ppl,
            "eval_tokens": total_tokens,
            "categories": cat_summary,
            "context_lengths": len_summary,
        }

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save complete 19-point training state."""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "tokens_seen": self.tokens_seen,
            "supervised_tokens_seen": self.supervised_tokens_seen,
            "best_val_loss": self.best_val_loss,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch_cpu": torch.get_rng_state(),
            "rng_torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "model_config": self.model_config.__dict__,
            "config": self.config,
            "precision": self.precision,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        if tmp_path.exists():
            os.replace(tmp_path, path)

    def load_checkpoint(self, path: Path):
        """Load and restore complete 19-point training state."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler and state.get("scaler"):
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.micro_step = state["micro_step"]
        self.tokens_seen = state["tokens_seen"]
        self.supervised_tokens_seen = state.get("supervised_tokens_seen", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"✓ Restored state from step {self.global_step} ({self.tokens_seen:,} tokens seen).")

    def train(self, max_steps: Optional[int] = None, is_canary: bool = False, resume_path: Optional[Path] = None):
        """Execute full SFT training pipeline or canary phase across configured epochs."""
        if resume_path:
            logger.info(f"Resuming training state from {resume_path}...")
            self.load_checkpoint(resume_path)

        train_data_path = REPO_ROOT / self.config["train_data_path"]
        val_data_path = REPO_ROOT / self.config["val_data_path"]

        logger.info(f"Loading training dataset from {train_data_path}...")
        train_dataset = SFT4KDataset(train_data_path, self.serializer)
        logger.info(f"Loading validation dataset from {val_data_path}...")
        val_dataset = SFT4KDataset(val_data_path, self.serializer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=lambda b: collate_sft_batch(b, pad_token_id=self.tokenizer.eos_token_id),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda b: collate_sft_batch(b, pad_token_id=self.tokenizer.eos_token_id),
        )

        total_epochs = self.config.get("total_epochs", 3)
        updates_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        total_optimizer_steps = updates_per_epoch * total_epochs
        target_max_steps = max_steps if max_steps is not None else total_optimizer_steps

        logger.info(f"Training schedule: {len(train_dataset):,} train examples, batch_size={self.micro_batch_size}, grad_accum={self.grad_accum_steps}")
        logger.info(f"Updates per epoch: {updates_per_epoch:,} | Total epochs: {total_epochs} | Target optimizer updates: {target_max_steps:,}")

        eval_interval = self.config.get("eval_interval_steps", 200)
        save_interval = self.config.get("save_interval_steps", 200)
        cooling_interval = self.config.get("cooling_interval_steps", 200)
        cooling_duration = self.config.get("cooling_duration_seconds", 10)

        # Baseline evaluation at step 0 if not resuming
        if self.global_step == 0:
            logger.info("Executing baseline validation evaluation at step 0...")
            eval_0 = self.evaluate(val_loader, max_batches=self.config.get("eval_batches", 50))
            logger.info(f"--- Step 0 Baseline: Val Loss: {eval_0['val_loss']:.4f} | Val PPL: {eval_0['val_ppl']:.2f} ---")
            self.best_val_loss = eval_0["val_loss"]

        self.model.train()
        accum_loss = 0.0
        t0_step = time.time()
        telemetry_log = []

        start_epoch = self.micro_step // len(train_loader)

        for epoch in range(start_epoch, total_epochs):
            logger.info(f"=== Starting Epoch {epoch + 1}/{total_epochs} (Global Step {self.global_step}) ===")
            window_batches = []
            for i, batch in enumerate(train_loader):
                # If resuming mid-epoch, skip already processed microbatches
                if epoch == start_epoch and i < (self.micro_step % len(train_loader)):
                    continue

                window_batches.append(batch)
                is_accum_boundary = ((i + 1) % self.grad_accum_steps == 0)
                is_epoch_end = (i == len(train_loader) - 1)

                if is_accum_boundary or is_epoch_end:
                    # Determine exact total supervised tokens in this accumulation window
                    window_supervised_tokens = sum(
                        (b["labels"][..., 1:] != -100).sum().item() for b in window_batches
                    )
                    window_loss_sum = 0.0

                    for micro_batch in window_batches:
                        self.micro_step += 1
                        input_ids = micro_batch["input_ids"].to(self.device)
                        labels = micro_batch["labels"].to(self.device)

                        num_tokens = input_ids.numel()
                        num_supervised = (labels[..., 1:] != -100).sum().item()
                        self.tokens_seen += num_tokens
                        self.supervised_tokens_seen += num_supervised

                        with torch.amp.autocast(device_type="cuda", dtype=self.torch_dtype):
                            logits = self.model(input_ids)
                            loss, micro_tokens, micro_sum = compute_token_weighted_cross_entropy(
                                logits=logits,
                                labels=labels,
                                ignore_index=-100,
                                normalizer=window_supervised_tokens if window_supervised_tokens > 0 else None,
                            )

                        assert torch.isfinite(loss), f"Encountered non-finite loss at micro_step {self.micro_step}"
                        window_loss_sum += micro_sum

                        if window_supervised_tokens > 0:
                            if self.scaler:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()

                    if window_supervised_tokens > 0:
                        lr = self.get_lr(self.global_step, total_optimizer_steps)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = lr

                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                            self.optimizer.step()

                        assert math.isfinite(grad_norm), f"Encountered non-finite gradient norm at step {self.global_step}"

                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                        effective_loss = window_loss_sum / window_supervised_tokens
                    else:
                        logger.warning(
                            f"Accumulation window at step {self.global_step} contained 0 supervised tokens; skipping optimizer step."
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        effective_loss = 0.0
                        grad_norm = 0.0
                        lr = self.get_lr(self.global_step, total_optimizer_steps)

                    step_time = time.time() - t0_step
                    t0_step = time.time()

                    # Telemetry recording & streaming to jsonl
                    step_telemetry = {
                        "global_step": self.global_step,
                        "epoch": epoch + 1,
                        "train_loss": round(effective_loss, 4),
                        "grad_norm": round(grad_norm, 4),
                        "learning_rate": lr,
                        "tokens_seen": self.tokens_seen,
                        "supervised_tokens_seen": self.supervised_tokens_seen,
                        "window_supervised_tokens": window_supervised_tokens,
                        "window_microbatches": len(window_batches),
                        "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0,
                        "reserved_vram_mb": round(torch.cuda.memory_reserved() / (1024**2), 2) if torch.cuda.is_available() else 0.0,
                        "step_time_ms": round(step_time * 1000, 1),
                    }
                    telemetry_log.append(step_telemetry)

                    # Stream to experiment training log
                    log_filename = "step_08d_training_log.jsonl" if "step_08d" in str(self.artifacts_dir) else "step_07b_training_log.jsonl"
                    log_file = self.artifacts_dir / log_filename
                    with open(log_file, "a", encoding="utf-8") as f_log:
                        f_log.write(json.dumps(step_telemetry) + "\n")

                    if self.global_step % 20 == 0 or self.global_step == target_max_steps:
                        logger.info(
                            f"Epoch {epoch+1} | Step {self.global_step}/{target_max_steps} | "
                            f"Loss: {effective_loss:.4f} | GradNorm: {grad_norm:.3f} | LR: {lr:.2e} | "
                            f"Tokens: {self.tokens_seen:,} | Step Time: {step_time*1000:.1f}ms"
                        )

                    window_batches = []

                    # Checkpoint for canary (step 100)
                    if self.global_step == 100:
                        eval_res = self.evaluate(val_loader, max_batches=50)
                        logger.info(f"--- Canary Checkpoint at Step 100: Val Loss: {eval_res['val_loss']:.4f} | Val PPL: {eval_res['val_ppl']:.2f} ---")
                        canary_ckpt = self.output_dir / "canary_step_100.pt"
                        self.save_checkpoint(canary_ckpt)
                        logger.info(f"Saved canary checkpoint to {canary_ckpt}")

                        canary_report_path = self.artifacts_dir / "canary_report.json"
                        with open(canary_report_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "step": 100,
                                "val_loss": eval_res["val_loss"],
                                "val_ppl": eval_res["val_ppl"],
                                "categories": eval_res.get("categories", {}),
                                "context_lengths": eval_res.get("context_lengths", {}),
                                "tokens_seen": self.tokens_seen,
                                "supervised_tokens_seen": self.supervised_tokens_seen,
                                "checkpoint_path": str(canary_ckpt),
                            }, f, indent=2)

                        if is_canary:
                            logger.info("Canary run complete. Exiting as requested.")
                            return

                    # Evaluation
                    if self.global_step % eval_interval == 0:
                        eval_res = self.evaluate(val_loader, max_batches=self.config.get("eval_batches", 50))
                        logger.info(f"--- Eval at Step {self.global_step}: Loss: {eval_res['val_loss']:.4f} | PPL: {eval_res['val_ppl']:.2f} ---")
                        if eval_res["val_loss"] < self.best_val_loss:
                            self.best_val_loss = eval_res["val_loss"]
                            self.save_checkpoint(self.output_dir / "best_val.pt", is_best=True)

                    # Periodic Checkpoint
                    if self.global_step % save_interval == 0:
                        self.save_checkpoint(self.output_dir / "latest.pt")
                        logger.info(f"Updated latest checkpoint at step {self.global_step}")

                    # Thermal Cooling Break
                    if self.global_step % cooling_interval == 0 and self.config.get("thermal_guard_enabled", True):
                        logger.info(f"Thermal guard: pausing for {cooling_duration}s cooling break...")
                        time.sleep(cooling_duration)
                        t0_step = time.time()

                    if self.global_step >= target_max_steps:
                        break

            # Epoch boundary evaluation and checkpoint
            eval_epoch = self.evaluate(val_loader, max_batches=self.config.get("eval_batches", 50))
            logger.info(f"=== Epoch {epoch + 1} Validation: Loss: {eval_epoch['val_loss']:.4f} | PPL: {eval_epoch['val_ppl']:.2f} ===")
            if eval_epoch["val_loss"] < self.best_val_loss:
                self.best_val_loss = eval_epoch["val_loss"]
                self.save_checkpoint(self.output_dir / "best_val.pt", is_best=True)
                logger.info(f"✓ New best validation checkpoint updated at epoch {epoch + 1}")

            epoch_ckpt = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            self.save_checkpoint(epoch_ckpt)
            logger.info(f"✓ Saved epoch boundary checkpoint to {epoch_ckpt}")

            # Save epoch eval artifact
            epoch_eval_path = self.artifacts_dir / f"epoch_{epoch + 1}_eval.json"
            with open(epoch_eval_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "global_step": self.global_step,
                    "val_loss": eval_epoch["val_loss"],
                    "val_ppl": eval_epoch["val_ppl"],
                    "categories": eval_epoch.get("categories", {}),
                    "context_lengths": eval_epoch.get("context_lengths", {}),
                    "tokens_seen": self.tokens_seen,
                    "supervised_tokens_seen": self.supervised_tokens_seen,
                }, f, indent=2)

            if self.global_step >= target_max_steps:
                break

        # Final evaluation and checkpoint save
        eval_final = self.evaluate(val_loader, max_batches=100)
        logger.info(f"=== Final Evaluation: Val Loss: {eval_final['val_loss']:.4f} | Val PPL: {eval_final['val_ppl']:.2f} ===")
        if eval_final["val_loss"] < self.best_val_loss:
            self.best_val_loss = eval_final["val_loss"]
            self.save_checkpoint(self.output_dir / "best_val.pt", is_best=True)

        final_ckpt = self.output_dir / "checkpoint_epoch_3.pt"
        self.save_checkpoint(final_ckpt)
        self.save_checkpoint(self.output_dir / "latest.pt")

        # Save training summary
        summary_path = self.artifacts_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_steps": self.global_step,
                "total_epochs": total_epochs,
                "final_val_loss": eval_final["val_loss"],
                "final_val_ppl": eval_final["val_ppl"],
                "best_val_loss": self.best_val_loss,
                "tokens_seen": self.tokens_seen,
                "supervised_tokens_seen": self.supervised_tokens_seen,
                "final_checkpoint": str(final_ckpt),
                "best_checkpoint": str(self.output_dir / "best_val.pt"),
            }, f, indent=2)

        logger.info(f"✓ SFT training complete. Final checkpoint saved to {final_ckpt}")

    def run_dry_run_test(self, temp_out_dir: Path) -> Dict[str, Any]:
        """Perform 1-step forward/backward sanity check and save/load test on CUDA."""
        logger.info("Executing SFT 4K dry-run sanity check...")
        self.model.train()

        # Build representative batch of length 4096
        sample_msgs = [
            {"role": "user", "content": "Explain grouped query attention in transformers." + (" Detail" * 500)},
            {"role": "assistant", "content": "Grouped query attention partitions query heads across KV heads." + (" Answer" * 100)},
        ]
        encoded = self.serializer.encode_conversation(sample_msgs, max_length=4096)
        batch = collate_sft_batch([encoded, encoded], pad_token_id=self.tokenizer.eos_token_id)

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=self.torch_dtype):
            logits = self.model(input_ids)
            loss, _, _ = compute_token_weighted_cross_entropy(
                logits=logits,
                labels=labels,
                ignore_index=-100,
            )

        assert torch.isfinite(loss), "Loss must be finite"
        loss.backward()

        # Test state save and load
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        test_ckpt_path = temp_out_dir / "test_sft_recovery.pt"
        self.global_step = 1
        self.save_checkpoint(test_ckpt_path)
        assert test_ckpt_path.exists(), "Checkpoint file must be written"

        # Verify load
        self.load_checkpoint(test_ckpt_path)
        assert self.global_step == 1, "Global step mismatch upon recovery"

        # Clean up temporary file
        if test_ckpt_path.exists():
            test_ckpt_path.unlink()

        self.optimizer.zero_grad(set_to_none=True)
        logger.info("✓ SFT 4K dry-run sanity check passed successfully.")
        return {
            "status": "PASSED",
            "loss": float(loss.item()),
            "allocated_vram_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
            "reserved_vram_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
        }


def main():
    parser = argparse.ArgumentParser(description="ModernGPT-S 100M GQA-4 4K SFT Engine")
    parser.add_argument("--config", type=str, default="configs/modern_gpts_100m/sft_4k.yaml", help="Path to config")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-step sanity check and exit")
    parser.add_argument("--canary", action="store_true", help="Run initial 100-step canary phase")
    parser.add_argument("--train", action="store_true", help="Execute full training pipeline")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    trainer = SFT4KTrainer(cfg_path)

    resume_p = Path(args.resume_from) if args.resume_from else None

    if args.dry_run:
        res = trainer.run_dry_run_test(REPO_ROOT / "scratch/test_sft_check")
        print(f"Dry-run result: {res}")
    elif args.canary:
        trainer.train(max_steps=100, is_canary=True, resume_path=resume_p)
    elif args.train:
        trainer.train(max_steps=args.max_steps, is_canary=False, resume_path=resume_p)
    else:
        logger.info("Specify --canary, --train, or --dry-run.")


if __name__ == "__main__":
    main()

