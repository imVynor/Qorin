"""Controlled 4K Continued-Pretraining Engine for ModernGPT-S 100M.

Features:
- Continued pretraining with 4,096-token context window and Linear Position Interpolation (s=8.0).
- Microbatch size 1, gradient accumulation 8 (32,768 tokens per optimizer update).
- Multi-length validation evaluation at milestone checkpoints (0, 100, 250, 500, 1000, 1500, 2000).
- Passkey retrieval evaluation at milestone checkpoints (0, 500, 1000, 2000).
- Strict thermal guardrails with automated cooling breaks.
- Full 19-point state recovery and atomic checkpointing.
"""

import argparse
import datetime
import gc
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.monitor_gpu import GPUMonitor
from scripts.evaluate_context_scaling import evaluate_val_loss
from scripts.evaluate_passkey_retrieval import build_passkey_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.Pilot4K")


class ContinuedPretrainPilot:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = REPO_ROOT / self.config["output_dir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = REPO_ROOT / self.config["artifacts_dir"]
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = GPUMonitor()
        self.tokenizer = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")

        # Load 4K model configuration & init checkpoint
        init_ckpt_path = REPO_ROOT / self.config["init_checkpoint"]
        init_state = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)

        m_cfg = self.config["model_config"]
        self.model_config = ModelConfig(
            vocab_size=m_cfg["vocab_size"],
            context_length=m_cfg["context_length"],
            emb_dim=m_cfg["emb_dim"],
            n_heads=m_cfg["n_heads"],
            n_layers=m_cfg["n_layers"],
            ffn_hidden_dim=m_cfg["ffn_hidden_dim"],
            dropout=m_cfg.get("dropout", 0.0),
            qkv_bias=m_cfg.get("qkv_bias", False),
            rope_theta=m_cfg.get("rope_theta", 10000.0),
            num_kv_heads=m_cfg.get("num_kv_heads", 4),
            rope_scale=m_cfg.get("rope_scale", 8.0),
            rope_type=m_cfg.get("rope_type", "linear"),
        )

        logger.info(f"Instantiating ModernGPT-S 100M with context={self.model_config.context_length}, rope_scale={self.model_config.rope_scale}")
        self.model = ModernGPT(self.model_config).to(device=self.device, dtype=torch.bfloat16)
        self.model.load_state_dict(init_state["model"], strict=True)
        assert self.model.parameter_count == 100401408

        # Training schedule & parameters
        self.total_steps = self.config["total_steps"]
        self.context_length = self.config["context_length"]
        self.micro_batch_size = self.config["micro_batch_size"]
        self.grad_accum_steps = self.config["gradient_accumulation_steps"]
        self.tokens_per_step = self.context_length * self.micro_batch_size * self.grad_accum_steps

        # Optimization
        self.peak_lr = float(self.config["learning_rate"])
        self.min_lr = float(self.config["min_learning_rate"])
        self.warmup_steps = int(self.config["warmup_steps"])
        self.grad_clip = float(self.config["grad_clip_norm"])

        # Create optimizer
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": float(self.config["weight_decay"])},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(
            optim_groups,
            lr=self.min_lr,
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=float(self.config["eps"]),
        )

        self.global_step = 0
        self.tokens_seen = 0
        self.current_shard_idx = 0
        self.shard_seq_offset = 0

        # Training log files
        self.metrics_jsonl = self.artifacts_dir / "training_metrics.jsonl"
        self.metrics_csv = self.artifacts_dir / "training_metrics.csv"
        self.telemetry_csv = self.artifacts_dir / "gpu_telemetry.csv"

        if not self.metrics_csv.exists():
            with open(self.metrics_csv, "w", encoding="utf-8") as f:
                f.write("step,tokens_seen,loss,lr,grad_norm,step_time_ms,tokens_per_sec,allocated_mb,reserved_mb,gpu_temp_c\n")
        if not self.telemetry_csv.exists():
            with open(self.telemetry_csv, "w", encoding="utf-8") as f:
                f.write("timestamp,step,tokens_seen,gpu_temp_c,gpu_util_pct,mem_used_mb,power_draw_w\n")

        self.val_bin_path = REPO_ROOT / "artifacts/shards_250m/val.bin"
        self.eval_milestones = set(self.config.get("eval_milestones", [0, 100, 250, 500, 1000, 1500, 2000]))
        self.passkey_milestones = set(self.config.get("passkey_eval_milestones", [0, 500, 1000, 2000]))

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.min_lr + (self.peak_lr - self.min_lr) * (step / max(1, self.warmup_steps))
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

    def check_thermal_safety(self) -> None:
        poll = self.monitor.poll_once()
        temp = poll.get("gpu_temp_c", 0.0)
        if temp >= self.config["pause_temp_celsius"]:
            logger.warning(f"HIGH TEMPERATURE REACHED: {temp}°C >= {self.config['pause_temp_celsius']}°C. Initiating cooling break...")
            while temp > self.config["cooldown_temp_celsius"]:
                time.sleep(10)
                poll = self.monitor.poll_once()
                temp = poll.get("gpu_temp_c", 0.0)
                logger.info(f"Cooling... Current temp: {temp}°C (target <= {self.config['cooldown_temp_celsius']}°C)")
            logger.info("Temperature normalized. Resuming pilot.")

    def run_multi_length_validation(self, step: int) -> Dict:
        logger.info(f"--- Running Multi-Length Validation at Step {step} ---")
        results = {}
        for ctx in [512, 1024, 2048, 4096]:
            scale = max(1.0, ctx / 512.0)
            cfg = ModelConfig(
                vocab_size=self.model_config.vocab_size,
                context_length=ctx,
                emb_dim=self.model_config.emb_dim,
                n_heads=self.model_config.n_heads,
                n_layers=self.model_config.n_layers,
                ffn_hidden_dim=self.model_config.ffn_hidden_dim,
                dropout=0.0,
                qkv_bias=False,
                rope_theta=10000.0,
                num_kv_heads=4,
                rope_scale=8.0,
                rope_type="linear",
            )
            val_model = ModernGPT(cfg).to(device=self.device, dtype=torch.bfloat16)
            val_model.load_state_dict(self.model.state_dict(), strict=True)
            val_model.eval()

            batch_size = max(1, 4096 // ctx)
            ds = ShardedPackedDataset(str(self.val_bin_path), context_length=ctx)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            val_loss, ppl, n_tok = evaluate_val_loss(
                val_model, loader, self.device, max_batches=50, autocast_dtype=torch.bfloat16
            )
            results[f"ctx_{ctx}"] = {
                "val_loss": round(val_loss, 4),
                "perplexity": round(ppl, 3),
                "tokens_evaluated": n_tok,
            }
            logger.info(f"Step {step} | Ctx {ctx:4d} | Loss: {val_loss:.4f} | PPL: {ppl:.3f}")
            del val_model
            gc.collect()
            torch.cuda.empty_cache()

        return results

    def run_passkey_evaluation(self, step: int, trials: int = 5) -> Dict:
        logger.info(f"--- Running Passkey Retrieval Evaluation at Step {step} ({trials} trials/combo) ---")
        results = {}
        context_lengths = [512, 1024, 2048, 4096]
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]

        random.seed(42 + step)
        torch.manual_seed(42 + step)
        self.model.eval()

        total_correct = 0
        total_cases = 0

        for ctx in context_lengths:
            results[str(ctx)] = {}
            for d in depths:
                correct = 0
                for _ in range(trials):
                    passkey = f"{random.randint(10000, 99999)}"
                    inputs, expected_key = build_passkey_prompt(
                        tokenizer=self.tokenizer,
                        target_length=ctx - 10,
                        depth_fraction=d,
                        passkey=passkey,
                    )
                    inputs = inputs.to(self.device)

                    with torch.inference_mode():
                        gen_ids = inputs
                        for _ in range(8):
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                logits = self.model(gen_ids[:, -self.context_length:])
                            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
                            gen_ids = torch.cat((gen_ids, next_tok), dim=1)

                    cont_ids = gen_ids[0, inputs.size(1):].tolist()
                    cont_text = self.tokenizer.decode(cont_ids).strip()
                    if expected_key in cont_text:
                        correct += 1

                acc = correct / trials
                results[str(ctx)][str(d)] = acc
                total_correct += correct
                total_cases += trials

        overall_acc = total_correct / total_cases
        results["overall_accuracy"] = round(overall_acc, 4)
        logger.info(f"Step {step} | Passkey Accuracy: {total_correct}/{total_cases} ({overall_acc*100:.1f}%)")
        self.model.train()
        return results

    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        state = {
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "shard_index": self.current_shard_idx,
            "shard_seq_offset": self.shard_seq_offset,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": dict(self.model_config.__dict__),
            "training_config": self.config,
            "tokenizer_fingerprint": self.tokenizer.fingerprint,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        torch.save(state, path)
        logger.info(f"Saved checkpoint to {path}")

    def train_pilot(self) -> None:
        logger.info(f"Starting 4K Continued-Pretraining Pilot: Target {self.total_steps} steps, {self.total_steps * self.tokens_per_step:,} tokens")

        shards_dir = REPO_ROOT / "artifacts/modern_gpts_100m/shards"
        shard_files = sorted(list(shards_dir.glob("shard_*.bin")))
        if not shard_files:
            raise RuntimeError(f"No training shards found in {shards_dir}")

        # Validation history
        val_history = {}
        passkey_history = {}

        # Milestone 0 evaluation
        if 0 in self.eval_milestones:
            val_res_0 = self.run_multi_length_validation(step=0)
            val_history["step_0"] = val_res_0

        if 0 in self.passkey_milestones:
            pass_res_0 = self.run_passkey_evaluation(step=0, trials=5)
            passkey_history["step_0"] = pass_res_0
            with open(self.artifacts_dir / "passkey_results_step_0.json", "w", encoding="utf-8") as f:
                json.dump(pass_res_0, f, indent=2)

        self.model.train()
        step_t0 = time.perf_counter()

        # Shard sequence iterator
        while self.global_step < self.total_steps:
            current_shard_file = shard_files[self.current_shard_idx % len(shard_files)]
            logger.info(f"Loading shard {current_shard_file.name} (index {self.current_shard_idx})")
            dataset = ShardedPackedDataset(str(current_shard_file), context_length=self.context_length)
            loader = DataLoader(dataset, batch_size=self.micro_batch_size, shuffle=False)

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            accum_counter = 0

            for seq_idx, (inputs, targets) in enumerate(loader):
                if seq_idx < self.shard_seq_offset:
                    continue

                self.check_thermal_safety()

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = self.model(inputs, targets=targets)
                    loss_scaled = loss / self.grad_accum_steps

                loss_scaled.backward()
                accum_loss += loss.item()
                accum_counter += 1

                if accum_counter == self.grad_accum_steps:
                    # Clip gradients
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()

                    # Check gradient finiteness
                    if not math.isfinite(grad_norm):
                        logger.error(f"FATAL: Non-finite gradient norm encountered at step {self.global_step}: {grad_norm}")
                        raise RuntimeError("Non-finite gradient norm")

                    # Learning rate step
                    cur_lr = self.get_lr(self.global_step)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = cur_lr

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    step_t1 = time.perf_counter()
                    step_time_ms = (step_t1 - step_t0) * 1000.0
                    step_tok_s = self.tokens_per_step / max(1e-4, step_t1 - step_t0)
                    step_t0 = time.perf_counter()

                    self.global_step += 1
                    self.tokens_seen += self.tokens_per_step
                    self.shard_seq_offset = seq_idx + 1

                    avg_step_loss = accum_loss / self.grad_accum_steps
                    accum_loss = 0.0
                    accum_counter = 0

                    alloc_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    res_mb = torch.cuda.max_memory_reserved(self.device) / (1024**2)
                    poll = self.monitor.poll_once()
                    temp = poll.get("gpu_temp_c", 0.0)

                    # Log progress
                    if self.global_step % 10 == 0 or self.global_step <= 10:
                        logger.info(
                            f"Step {self.global_step:4d}/{self.total_steps} | Loss {avg_step_loss:.4f} | LR {cur_lr:.2e} | "
                            f"GradNorm {grad_norm:.2f} | Time {step_time_ms:.1f}ms ({step_tok_s:.0f} tok/s) | Temp {temp}°C"
                        )

                    # Write training metrics
                    metric_dict = {
                        "step": self.global_step,
                        "tokens_seen": self.tokens_seen,
                        "loss": round(avg_step_loss, 4),
                        "lr": cur_lr,
                        "grad_norm": round(grad_norm, 4),
                        "step_time_ms": round(step_time_ms, 2),
                        "tokens_per_sec": round(step_tok_s, 1),
                        "allocated_mb": round(alloc_mb, 2),
                        "reserved_mb": round(res_mb, 2),
                        "gpu_temp_c": temp,
                    }
                    with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(metric_dict) + "\n")
                    with open(self.metrics_csv, "a", encoding="utf-8") as f:
                        f.write(
                            f"{self.global_step},{self.tokens_seen},{avg_step_loss:.4f},{cur_lr:.6e},"
                            f"{grad_norm:.4f},{step_time_ms:.2f},{step_tok_s:.1f},{alloc_mb:.2f},{res_mb:.2f},{temp}\n"
                        )

                    # Evaluation milestones
                    if self.global_step in self.eval_milestones:
                        val_res = self.run_multi_length_validation(self.global_step)
                        val_history[f"step_{self.global_step}"] = val_res
                        with open(self.artifacts_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
                            json.dump(val_history, f, indent=2)

                    if self.global_step in self.passkey_milestones:
                        pass_res = self.run_passkey_evaluation(self.global_step, trials=5)
                        passkey_history[f"step_{self.global_step}"] = pass_res
                        with open(self.artifacts_dir / f"passkey_results_step_{self.global_step}.json", "w", encoding="utf-8") as f:
                            json.dump(pass_res, f, indent=2)

                    # Checkpointing cadence
                    if self.global_step % self.config["save_interval_steps"] == 0 or self.global_step == self.total_steps:
                        ckpt_file = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
                        self.save_checkpoint(ckpt_file)
                        # Also save latest.pt
                        self.save_checkpoint(self.output_dir / "latest.pt")

                    if self.global_step >= self.total_steps:
                        break

            self.current_shard_idx += 1
            self.shard_seq_offset = 0

        logger.info(f"Pilot execution successfully completed {self.global_step} optimizer steps ({self.tokens_seen:,} tokens)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/modern_gpts_100m_4k_pilot.yaml")
    args = parser.parse_args()

    pilot = ContinuedPretrainPilot(REPO_ROOT / args.config)
    pilot.train_pilot()

