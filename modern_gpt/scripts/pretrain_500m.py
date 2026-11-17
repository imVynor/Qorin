import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))



import os

import gc

import time

import math

import json

import signal

import subprocess

from datetime import datetime

from typing import Dict, Optional, Tuple, Any



import numpy as np

import torch

import torch.nn as nn



from modern_gpt.model.config import ModelConfig

from modern_gpt.model.gpt import ModernGPT

from modern_gpt.data_500m.domain_sampler import InterleavedDomainSampler



# Target pretraining hyperparameters

VOCAB_SIZE = 32768

CONTEXT_LENGTH = 2048

EXPECTED_TOKENIZER_FP = "3559c7435e1d9330"

MICROBATCH_SIZE = 1

GRAD_ACCUM_STEPS = 16 # 32,768 tokens per optimizer update

TOKENS_PER_UPDATE = MICROBATCH_SIZE * GRAD_ACCUM_STEPS * CONTEXT_LENGTH



MAX_LR = 3.0e-4

MIN_LR = 3.0e-5

WARMUP_UPDATES = 2000

WEIGHT_DECAY = 0.1

GRAD_CLIP_NORM = 1.0



# 30-minute training / 5-minute cooldown parameters

ACTIVE_TRAIN_DURATION_SEC = 30 * 60   # 30 minutes

COOLDOWN_DURATION_SEC = 5 * 60        # 5 minutes

MAX_SAFE_TEMP_C = 92.0                 # Pre-emptive thermal guard (well below 95°C)

COOLDOWN_TARGET_TEMP_C = 65.0          # Resumption threshold



class PretrainingOrchestrator:

    def __init__(

        self,

        target_tokens: int = 5_000_000_000,

        checkpoint_dir: str = "modern_gpt/checkpoints/500m_pretrain",

        log_dir: str = "modern_gpt/reports/phase1c_pretraining",

        val_eval_interval_tokens: int = 50_000_000, # Eval every 50M tokens

        save_interval_tokens: int = 50_000_000,     # Checkpoint every 50M tokens

        pilot_mode: bool = False,

        pilot_updates: int = 10

    ):

        self.target_tokens = target_tokens

        self.total_target_updates = target_tokens // TOKENS_PER_UPDATE

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)



        self.val_eval_interval_tokens = val_eval_interval_tokens

        self.save_interval_tokens = save_interval_tokens

        self.pilot_mode = pilot_mode

        self.pilot_updates = pilot_updates



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type != "cuda":

            raise RuntimeError("CUDA is required for ModernGPT-S 500M pretraining.")



        # Instantiate Model

        self.cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)

        self.model = ModernGPT(self.cfg).to(device=self.device, dtype=torch.bfloat16)



        # Optimizer with weight decay separation

        decay_params = [p for n, p in self.model.named_parameters() if p.requires_grad and p.dim() >= 2]

        nodecay_params = [p for n, p in self.model.named_parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [

            {"params": decay_params, "weight_decay": WEIGHT_DECAY},

            {"params": nodecay_params, "weight_decay": 0.0}

        ]

        self.optimizer = torch.optim.AdamW(optim_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8, fused=True)



        # Datasets & Samplers

        data_base = Path("modern_gpt/data_500m/final_5b")

        self.train_sampler = InterleavedDomainSampler(data_base, sequence_length=CONTEXT_LENGTH, split="train")

        self.val_sampler = InterleavedDomainSampler(data_base, sequence_length=CONTEXT_LENGTH, split="val")



        # Tracking state

        self.global_update = 0

        self.total_tokens_processed = 0

        self.best_val_loss = float("inf")

        self.cumulative_cycle_train_sec = 0.0

        self.cycle_count = 1

        self.stop_requested = False



        # Register signal handlers for graceful shutdown

        signal.signal(signal.SIGINT, self._handle_signal)

        signal.signal(signal.SIGTERM, self._handle_signal)



        self.telemetry_file = self.log_dir / "training_telemetry.jsonl"

        self.cycle_log_file = self.log_dir / "gpu_cycle_log.jsonl"



    def _handle_signal(self, signum, frame):

        print(f"\n[ALERT] Received shutdown signal {signum}. Requesting graceful pause after current update...", flush=True)

        self.stop_requested = True



    def get_lr(self, update: int) -> float:

        """Cosine learning rate schedule with linear warmup."""

        if update < WARMUP_UPDATES:

            return MAX_LR * (update + 1) / WARMUP_UPDATES

        if update >= self.total_target_updates:

            return MIN_LR

        decay_ratio = (update - WARMUP_UPDATES) / (self.total_target_updates - WARMUP_UPDATES)

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return MIN_LR + coeff * (MAX_LR - MIN_LR)



    def query_gpu_stats(self) -> Dict[str, Any]:

        """Queries nvidia-smi for current temperature, power, and utilization."""

        try:

            out = subprocess.check_output(

                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"],

                encoding="utf-8"

            ).strip().split(",")

            return {

                "temp_c": float(out[0].strip()),

                "power_w": float(out[1].strip()) if out[1].strip() != "[N/A]" else None,

                "util_pct": float(out[2].strip())

            }

        except Exception:

            return {"temp_c": None, "power_w": None, "util_pct": None}



    def save_checkpoint(self, path: Path, is_milestone: bool = False):

        """Atomic checkpoint saving with full bit-exact reproducibility state."""

        tmp_path = path.with_suffix(".pt.tmp")

        state = {

            "model_state_dict": self.model.state_dict(),

            "optimizer_state_dict": self.optimizer.state_dict(),

            "global_update": self.global_update,

            "total_tokens_processed": self.total_tokens_processed,

            "best_val_loss": self.best_val_loss,

            "train_sampler_state": self.train_sampler.state_dict(),

            "val_sampler_state": self.val_sampler.state_dict(),

            "python_rng": np.random.get_state(),

            "torch_rng": torch.get_rng_state(),

            "cuda_rng": torch.cuda.get_rng_state(),

            "cycle_count": self.cycle_count,

            "cumulative_cycle_train_sec": self.cumulative_cycle_train_sec,

            "config": self.cfg.__dict__,

            "tokenizer_fingerprint": EXPECTED_TOKENIZER_FP,

            "timestamp": datetime.now().isoformat()

        }

        torch.save(state, tmp_path)

        tmp_path.replace(path)

        print(f"[CHECKPOINT] Saved atomic checkpoint to {path.name} ({self.total_tokens_processed:,} tokens)", flush=True)



    def load_checkpoint(self, path: Path):

        """Loads and restores bit-exact checkpoint state."""

        if not path.exists():

            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"[RESUME] Loading checkpoint from {path}...", flush=True)

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])

        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.global_update = ckpt["global_update"]

        self.total_tokens_processed = ckpt["total_tokens_processed"]

        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        self.train_sampler.load_state_dict(ckpt["train_sampler_state"])

        self.val_sampler.load_state_dict(ckpt["val_sampler_state"])

        np.random.set_state(ckpt["python_rng"])

        torch.set_rng_state(ckpt["torch_rng"].cpu())

        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())

        self.cycle_count = ckpt.get("cycle_count", 1)

        self.cumulative_cycle_train_sec = ckpt.get("cumulative_cycle_train_sec", 0.0)

        print(f"[RESUME] Successfully restored at update {self.global_update:,} ({self.total_tokens_processed:,} tokens).", flush=True)



    def run_validation(self, eval_sequences: int = 50) -> Dict[str, float]:

        """Runs validation across isolated validation shards without updating weights."""

        self.model.eval()

        val_losses = []

        domain_losses = {d: [] for d in self.val_sampler.domains}



        with torch.no_grad():

            for _ in range(eval_sequences):

                x, y, d = self.val_sampler.get_batch()

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    _, loss = self.model(x, targets=y)

                l_val = loss.item()

                val_losses.append(l_val)

                domain_losses[d].append(l_val)



        self.model.train()

        mean_val = float(np.mean(val_losses)) if val_losses else 0.0

        ppl = math.exp(min(mean_val, 20.0))

        per_domain_means = {d: round(float(np.mean(vals)), 4) if vals else None for d, vals in domain_losses.items()}



        print(f"\n[VALIDATION] Update {self.global_update:,} | Val Loss: {mean_val:.4f} | PPL: {ppl:.2f}", flush=True)

        print(f"  Per-Domain Val Losses: {per_domain_means}\n", flush=True)

        return {

            "mean_val_loss": round(mean_val, 4),

            "perplexity": round(ppl, 2),

            "per_domain_val_losses": per_domain_means

        }



    def execute_cooldown(self, pre_temp: float, reason: str = "30-minute scheduled cycle"):

        """Saves a checkpoint and pauses training for the configured 5-minute cooldown."""

        print(f"\n=================================================================", flush=True)

        print(f"[COOLDOWN] Entering mandatory GPU cooldown ({reason}).", flush=True)

        print(f"  Pre-cooldown Temp: {pre_temp}°C | Scheduled duration: {COOLDOWN_DURATION_SEC // 60} minutes", flush=True)

        print(f"=================================================================\n", flush=True)



        # Save checkpoint before idle

        self.save_checkpoint(self.checkpoint_dir / "ckpt_latest.pt")



        # Free CUDA memory during cooldown

        torch.cuda.empty_cache()

        gc.collect()



        t_cool_start = time.time()

        cooldown_target_sec = COOLDOWN_DURATION_SEC if not self.pilot_mode else 10 # 10s in pilot mode



        while time.time() - t_cool_start < cooldown_target_sec:

            rem = int(cooldown_target_sec - (time.time() - t_cool_start))

            if rem % 60 == 0 or rem <= 10:

                stats = self.query_gpu_stats()

                print(f"  Cooling down... {rem}s remaining | Current GPU Temp: {stats['temp_c']}°C", flush=True)

            time.sleep(5)



        post_stats = self.query_gpu_stats()

        post_temp = post_stats["temp_c"]

        print(f"\n[COOLDOWN COMPLETE] Post-cooldown GPU Temp: {post_temp}°C (Delta: {pre_temp - post_temp:.1f}°C drop).", flush=True)



        # Log cycle event

        cycle_record = {

            "cycle_number": self.cycle_count,

            "timestamp": datetime.now().isoformat(),

            "reason": reason,

            "active_training_duration_sec": round(self.cumulative_cycle_train_sec, 1),

            "pre_cooldown_temp_c": pre_temp,

            "post_cooldown_temp_c": post_temp,

            "cooldown_duration_sec": round(time.time() - t_cool_start, 1)

        }

        with open(self.cycle_log_file, "a", encoding="utf-8") as f:

            f.write(json.dumps(cycle_record) + "\n")



        self.cumulative_cycle_train_sec = 0.0

        self.cycle_count += 1



    def train(self):

        """Main training loop enforcing 30 minutes of training followed by a 5-minute cooldown."""

        print(f"\n=================================================================")

        print(f"MODERNGPT-S 500M PRETRAINING INITIALIZED")

        print(f"Target Tokens: {self.target_tokens:,} ({self.total_target_updates:,} updates)")

        print(f"Batch Size: {MICROBATCH_SIZE} x {GRAD_ACCUM_STEPS} accum x {CONTEXT_LENGTH} seq = {TOKENS_PER_UPDATE:,} tokens/update")

        print(f"Operating Cycle: 30 min active GPU training / 5 min mandatory cooldown")

        print(f"=================================================================\n", flush=True)



        max_updates = self.pilot_updates if self.pilot_mode else self.total_target_updates

        last_eval_tokens = self.total_tokens_processed

        last_save_tokens = self.total_tokens_processed



        cycle_start_time = time.time()



        while self.global_update < max_updates and not self.stop_requested:

            t0 = time.perf_counter()

            lr = self.get_lr(self.global_update)

            for param_group in self.optimizer.param_groups:

                param_group["lr"] = lr



            self.optimizer.zero_grad(set_to_none=True)

            accum_loss = 0.0

            domain_counts = {}



            # Gradient accumulation microbatches

            for _ in range(GRAD_ACCUM_STEPS):

                x, y, d = self.train_sampler.get_batch()

                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

                domain_counts[d] = domain_counts.get(d, 0) + 1



                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                    _, loss = self.model(x, targets=y)

                    loss = loss / GRAD_ACCUM_STEPS



                loss.backward()

                accum_loss += loss.item()



            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)

            self.optimizer.step()

            torch.cuda.synchronize()



            t1 = time.perf_counter()

            dt = t1 - t0

            tps = TOKENS_PER_UPDATE / dt



            self.global_update += 1

            self.total_tokens_processed += TOKENS_PER_UPDATE

            self.cumulative_cycle_train_sec += dt



            # Query GPU telemetry

            gpu_stats = self.query_gpu_stats()

            curr_temp = gpu_stats["temp_c"]



            # Console log

            if self.global_update % 1 == 0:

                print(

                    f"Update {self.global_update:05d}/{max_updates:05d} | "

                    f"Tokens: {self.total_tokens_processed / 1e6:6.2f}M | "

                    f"Loss: {accum_loss:.4f} | "

                    f"LR: {lr:.2e} | "

                    f"GradNorm: {grad_norm:.2f} | "

                    f"Throughput: {tps:,.1f} tok/s | "

                    f"Temp: {curr_temp}°C",

                    flush=True

                )



            # Telemetry record

            telem_record = {

                "update": self.global_update,

                "tokens": self.total_tokens_processed,

                "loss": round(accum_loss, 4),

                "lr": lr,

                "grad_norm": round(float(grad_norm), 4),

                "throughput_tok_per_sec": round(tps, 1),

                "step_time_sec": round(dt, 3),

                "gpu_temp_c": curr_temp,

                "gpu_power_w": gpu_stats["power_w"],

                "proportions": self.train_sampler.get_proportions(),

                "timestamp": datetime.now().isoformat()

            }

            with open(self.telemetry_file, "a", encoding="utf-8") as f:

                f.write(json.dumps(telem_record) + "\n")



            # Periodic Validation

            if self.total_tokens_processed - last_eval_tokens >= self.val_eval_interval_tokens:

                val_res = self.run_validation()

                last_eval_tokens = self.total_tokens_processed

                if val_res["mean_val_loss"] < self.best_val_loss:

                    self.best_val_loss = val_res["mean_val_loss"]

                    self.save_checkpoint(self.checkpoint_dir / "ckpt_best_val.pt")



            # Periodic Checkpointing (every 50M tokens)

            if self.total_tokens_processed - last_save_tokens >= self.save_interval_tokens:

                self.save_checkpoint(self.checkpoint_dir / "ckpt_latest.pt")

                # Milestone checkpoint every 500M tokens

                if self.total_tokens_processed % 500_000_000 == 0:

                    milestone_tokens_m = self.total_tokens_processed // 1_000_000

                    self.save_checkpoint(self.checkpoint_dir / f"ckpt_{milestone_tokens_m}m.pt", is_milestone=True)

                last_save_tokens = self.total_tokens_processed



            # THERMAL SAFETY OVERRIDE: Check if GPU temp exceeds safe threshold

            if curr_temp is not None and curr_temp >= MAX_SAFE_TEMP_C:

                print(f"\n[THERMAL WARNING] GPU temp {curr_temp}°C >= safety limit {MAX_SAFE_TEMP_C}°C!", flush=True)

                self.execute_cooldown(curr_temp, reason=f"Thermal override trigger ({curr_temp}°C >= {MAX_SAFE_TEMP_C}°C)")

                continue



            # Mandatory cycle: pause after 30 minutes of active training

            if self.cumulative_cycle_train_sec >= ACTIVE_TRAIN_DURATION_SEC:

                self.execute_cooldown(curr_temp if curr_temp else 0.0, reason="30-minute scheduled cycle completed")

                continue



        # Final save upon completion or graceful pause

        self.save_checkpoint(self.checkpoint_dir / "ckpt_latest.pt")

        print(f"\n[DONE] Training session finished at update {self.global_update:,} ({self.total_tokens_processed:,} tokens).", flush=True)



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--tokens", type=int, default=5_000_000_000)

    parser.add_argument("--pilot", action="store_true")

    parser.add_argument("--updates", type=int, default=10)

    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()



    orchestrator = PretrainingOrchestrator(

        target_tokens=args.tokens,

        pilot_mode=args.pilot,

        pilot_updates=args.updates

    )

    if args.resume:

        orchestrator.load_checkpoint(Path(args.resume))

    orchestrator.train()
