"""
QORIN Phase 2E: Controlled Training Runner for Branch B and Branch C
Strictly implements Stage 8 protocols, safeguards, and telemetry logging.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig

PARENT_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt"
EXPECTED_PARENT_SHA = "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOK_SHA = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

DATA_DIR = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_1_data"

BRANCH_CONFIGS = {
    "branch_b": {
        "name": "Branch B: Decision Calibration Only",
        "train_file": "corpus1_calibration_train.jsonl",
        "val_file": "corpus1_calibration_val.jsonl",
        "expected_train_sha": "e7fb51c24ca9028cd15e5d7898c17273cec231f19441b3cab83d53691846ee03",
        "expected_val_sha": "afcf897cb83a51a887df4c3bb1d26a23b9e3bde2e1d34d9d26db193cbf591cd0",
        "out_dir_name": "branch_b_calibration",
        "target_updates": 200,
        "epochs": 2,
    },
    "branch_c": {
        "name": "Branch C: Evidence-Grounded Synthesis Only",
        "train_file": "corpus2_synthesis_train.jsonl",
        "val_file": "corpus2_synthesis_val.jsonl",
        "expected_train_sha": "da289a90e3325d7b2b6aa7ec0bc3611ddba5cb31ccbbccc3461e8c905b2ea94f",
        "expected_val_sha": "0ce60831baea478262a8499b73378d9c644abde59303c39d3f048e7b895050f7",
        "out_dir_name": "branch_c_synthesis",
        "target_updates": 200,
        "epochs": 2,
    }
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_gpu_telemetry() -> Dict[str, Any]:
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, encoding="utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temp_c": float(parts[0]),
            "power_w": float(parts[1]),
            "utilization_pct": float(parts[2]),
            "memory_used_mib": float(parts[3]),
            "memory_total_mib": float(parts[4]),
            "timestamp": time.time()
        }
    except Exception as e:
        return {"error": str(e), "temp_c": -1, "memory_used_mib": -1, "timestamp": time.time()}

class ChatMLDataset(Dataset):
    def __init__(self, jsonl_path: Path, tok: Tokenizer):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                chatml = rec["serialized_chatml"]
                prefix_len = rec["prefix_token_length"]
                target_len = rec["target_token_length"]
                
                input_ids = tok.encode(chatml).ids
                # Assistant-target only loss masking:
                labels = [-100] * prefix_len + input_ids[prefix_len:]
                assert len(input_ids) == len(labels)
                
                self.samples.append({
                    "id": rec["id"],
                    "category": rec.get("category", "unknown"),
                    "input_ids": input_ids,
                    "labels": labels,
                    "prefix_len": prefix_len,
                    "target_len": target_len
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_chatml(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for b in batch:
        inp = b["input_ids"]
        lab = b["labels"]
        pad_len = max_len - len(inp)

        input_ids_batch.append(inp + [0] * pad_len)
        labels_batch.append(lab + [-100] * pad_len)
        attention_mask_batch.append([1] * len(inp) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "categories": [b["category"] for b in batch],
        "ids": [b["id"] for b in batch]
    }

def evaluate_validation(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    cat_loss = {}
    cat_tokens = {}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            categories = batch["categories"]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                for i in range(input_ids.size(0)):
                    s_logits = shift_logits[i]
                    s_labels = shift_labels[i]
                    mask = (s_labels != -100)
                    n_tok = mask.sum().item()
                    if n_tok == 0:
                        continue
                    sample_loss = loss_fn(s_logits[mask], s_labels[mask]).item()
                    cat = categories[i]

                    total_loss += sample_loss
                    total_target_tokens += n_tok
                    cat_loss[cat] = cat_loss.get(cat, 0.0) + sample_loss
                    cat_tokens[cat] = cat_tokens.get(cat, 0.0) + n_tok

    model.train()
    overall_loss = total_loss / max(1, total_target_tokens)
    per_cat = {cat: round(cat_loss[cat] / max(1, cat_tokens[cat]), 4) for cat in cat_loss}
    return {
        "val_loss": round(overall_loss, 4),
        "total_target_tokens": total_target_tokens,
        "per_category_loss": per_cat
    }

def get_lr(step: int, total_steps: int = 200, warmup_steps: int = 20, peak_lr: float = 1.5e-5, min_lr: float = 1.5e-6) -> float:
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def train_branch(branch_key: str):
    bcfg = BRANCH_CONFIGS[branch_key]
    stage_dir = Path(__file__).resolve().parent / bcfg["out_dir_name"]
    ckpt_dir = stage_dir / "checkpoints"
    log_dir = stage_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"QORIN EXPERIMENT E: {bcfg['name'].upper()}")
    print(f"Output Directory: {stage_dir}")
    print("=" * 80)

    # Preflight Verification
    print("\n[Step 1/6] Running Mandatory Pre-training Integrity Audits...")
    parent_sha = sha256_file(PARENT_CKPT_PATH)
    assert parent_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch! {parent_sha}"
    tok_sha = sha256_file(TOK_PATH)
    assert tok_sha == EXPECTED_TOK_SHA, f"Tokenizer SHA mismatch! {tok_sha}"
    
    train_path = DATA_DIR / bcfg["train_file"]
    val_path = DATA_DIR / bcfg["val_file"]
    assert sha256_file(train_path) == bcfg["expected_train_sha"], "Train dataset SHA mismatch!"
    assert sha256_file(val_path) == bcfg["expected_val_sha"], "Val dataset SHA mismatch!"
    
    print(f"  Parent Checkpoint Verified: {PARENT_CKPT_PATH.name} (SHA: {parent_sha[:16]}...) [IMMUTABLE]")
    print(f"  Tokenizer Verified:         {TOK_PATH.name} (SHA: {tok_sha[:16]}...)")
    print(f"  Train Dataset Verified:     {train_path.name}")
    print(f"  Val Dataset Verified:       {val_path.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute Device:             {device} ({torch.cuda.get_device_name(0)})")
    initial_telem = get_gpu_telemetry()
    print(f"  Initial GPU State:          Temp={initial_telem['temp_c']}C, VRAM={initial_telem['memory_used_mib']}MiB")

    # Load Tokenizer & Model
    print("\n[Step 2/6] Initializing Model from Frozen Parent Checkpoint...")
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg).to(device)

    # Load frozen weights
    parent_ckpt_data = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = parent_ckpt_data["model_state_dict"] if "model_state_dict" in parent_ckpt_data else parent_ckpt_data
    model.load_state_dict(state_dict, strict=True)
    model.train()
    print(f"  Loaded parent weights into ModernGPT 500M ({sum(p.numel() for p in model.parameters()):,} params).")

    # Load Datasets
    print("\n[Step 3/6] Preparing DataLoaders...")
    train_dataset = ChatMLDataset(train_path, tok)
    val_dataset = ChatMLDataset(val_path, tok)
    assert len(train_dataset) == 400, f"Expected 400 train samples, got {len(train_dataset)}"
    assert len(val_dataset) == 50, f"Expected 50 val samples, got {len(val_dataset)}"

    # Effective batch size = 4 sequences per step.
    # Microbatch size = 2, gradient accumulation = 2 (2 * 2 = 4).
    # 400 records / 4 sequences per step = 100 optimizer steps per epoch.
    # 2 epochs = exactly 200 optimizer steps!
    micro_batch_size = 2
    accum_steps = 2
    total_planned_updates = 200
    warmup_steps = 20
    peak_lr = 1.5e-5
    min_lr = 1.5e-6
    weight_decay = 0.05

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_chatml)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_chatml)
    print(f"  Train: {len(train_dataset)} records | Val: {len(val_dataset)} records")
    print(f"  Kinematics: MicroBatch={micro_batch_size}, Accumulation={accum_steps}, EffectiveBatch=4, Updates/Epoch=100, TotalUpdates=200")

    # Optimizer & Loss
    print("\n[Step 4/6] Initializing Optimizer & Safeguards...")
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=get_lr(0, total_planned_updates, warmup_steps, peak_lr, min_lr),
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Initial Validation
    print("  Running pre-training validation (Step 0)...")
    init_val = evaluate_validation(model, val_loader, device)
    best_val_loss = init_val["val_loss"]
    best_step = 0
    print(f"  Step 0 Val Loss: {best_val_loss:.4f} (Target tokens evaluated: {init_val['total_target_tokens']})")
    print(f"  Per-category loss: {init_val['per_category_loss']}")

    # Logging Setup
    train_log_file = log_dir / "training_log.jsonl"
    val_log_file = log_dir / "validation_log.jsonl"
    telemetry_log_file = log_dir / "telemetry_log.jsonl"

    with open(val_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step": 0,
            "val_loss": best_val_loss,
            "target_tokens": init_val["total_target_tokens"],
            "per_category_loss": init_val["per_category_loss"],
            "timestamp": time.time()
        }) + "\n")

    # Safeguard Tracking State
    eval_interval = 25
    patience_plateau = 3
    patience_stop = 5
    no_improvement_count = 0
    val_loss_history = [(0, best_val_loss)]
    thermal_pause_temp = 78.0
    thermal_abort_temp = 82.0
    max_session_gpu_seconds = 105 * 60

    best_ckpt_path = ckpt_dir / "ckpt_branch_best.pt"
    final_ckpt_path = ckpt_dir / "ckpt_branch_final.pt"

    # Save initial best checkpoint (Step 0 baseline)
    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "branch": branch_key,
        "parent_sha256": EXPECTED_PARENT_SHA
    }, best_ckpt_path)

    # Training Loop
    print("\n[Step 5/6] Commencing Controlled Training Loop (200 Updates, 2 Epochs)...")
    current_update = 0
    microbatch_count = 0
    running_loss = 0.0
    running_target_tokens = 0
    total_tokens_processed = 0
    total_target_tokens_processed = 0

    optimizer.zero_grad()
    t_start_train = time.perf_counter()
    active_gpu_seconds = 0.0
    early_stopped = False
    stop_reason = "Completed full 200 optimizer updates"

    for epoch in range(1, bcfg["epochs"] + 1):
        if early_stopped:
            break
        print(f"\n--- Starting Epoch {epoch}/{bcfg['epochs']} ---")
        
        for batch in train_loader:
            t0_mb = time.perf_counter()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss_scaled = loss / accum_steps

            loss_scaled.backward()
            
            # Token accounting
            n_target = (shift_labels != -100).sum().item()
            running_loss += loss.item() * n_target
            running_target_tokens += n_target
            total_tokens_processed += input_ids.numel()
            total_target_tokens_processed += n_target
            
            microbatch_count += 1
            t_mb = time.perf_counter() - t0_mb
            active_gpu_seconds += t_mb

            # Optimizer Step on Accumulation Window
            if microbatch_count % accum_steps == 0:
                current_update += 1
                
                # Check for NaN / Inf
                if math.isnan(loss.item()) or math.isinf(loss.item()):
                    early_stopped = True
                    stop_reason = f"Numerical Instability: Loss is {loss.item()} at step {current_update}"
                    print(f"\n[ALERT] {stop_reason} -> ABORTING")
                    break

                # Gradient clipping to max norm 1.0
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 100.0:
                    early_stopped = True
                    stop_reason = f"Gradient norm explosion: {grad_norm:.2f} at step {current_update}"
                    print(f"\n[ALERT] {stop_reason} -> ABORTING")
                    break

                # Optimizer step & LR update
                lr = get_lr(current_update, total_planned_updates, warmup_steps, peak_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

                step_loss = running_loss / max(1, running_target_tokens)
                running_loss = 0.0
                running_target_tokens = 0

                # GPU telemetry & thermal guard
                telem = get_gpu_telemetry()
                cur_temp = telem["temp_c"]

                if cur_temp >= thermal_abort_temp:
                    early_stopped = True
                    stop_reason = f"Thermal critical condition ({cur_temp}C >= {thermal_abort_temp}C)"
                    print(f"\n[ALERT] {stop_reason} -> ABORTING TO PREVENT HARDWARE DAMAGE")
                    break
                elif cur_temp >= thermal_pause_temp:
                    print(f"\n[THERMAL PAUSE] GPU temp {cur_temp}C >= {thermal_pause_temp}C. Cooling down 60s...")
                    time.sleep(60)

                # Duty cycle check
                if active_gpu_seconds >= max_session_gpu_seconds:
                    early_stopped = True
                    stop_reason = f"Reached maximum active GPU compute window ({active_gpu_seconds / 60:.1f} min)"
                    print(f"\n[SAFEGUARD] {stop_reason} -> STOPPING")
                    break

                # Log training step
                train_entry = {
                    "step": current_update,
                    "epoch": epoch,
                    "train_loss": round(step_loss, 4),
                    "grad_norm": round(grad_norm, 4),
                    "lr": lr,
                    "gpu_temp_c": cur_temp,
                    "vram_used_mib": telem["memory_used_mib"],
                    "total_target_tokens": total_target_tokens_processed,
                    "timestamp": time.time()
                }
                with open(train_log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(train_entry) + "\n")

                if current_update % 5 == 0 or current_update == 1:
                    print(f"Step {current_update:>3}/200 | Ep {epoch} | Loss: {step_loss:.4f} | Grad: {grad_norm:.2f} | LR: {lr:.2e} | Temp: {cur_temp:.0f}C")

                # Validation & Checkpointing at Cadence
                if current_update % eval_interval == 0:
                    val_res = evaluate_validation(model, val_loader, device)
                    v_loss = val_res["val_loss"]
                    val_loss_history.append((current_update, v_loss))
                    
                    val_entry = {
                        "step": current_update,
                        "epoch": epoch,
                        "val_loss": v_loss,
                        "target_tokens": val_res["total_target_tokens"],
                        "per_category_loss": val_res["per_category_loss"],
                        "timestamp": time.time()
                    }
                    with open(val_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(val_entry) + "\n")

                    print(f"\n>>> [EVALUATION STEP {current_update}] Val Loss: {v_loss:.4f} (Best: {best_val_loss:.4f})")
                    print(f"    Per-category: {val_res['per_category_loss']}")

                    # Milestone Checkpoint at Step 100
                    if current_update == 100:
                        ms_path = ckpt_dir / "ckpt_branch_step100.pt"
                        torch.save({
                            "step": 100,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": v_loss,
                            "branch": branch_key
                        }, ms_path)
                        print(f"    [CHECKPOINT SAVED] Milestone Step 100 -> {ms_path.name}")

                    # Check improvement
                    if v_loss < best_val_loss:
                        improvement = best_val_loss - v_loss
                        best_val_loss = v_loss
                        best_step = current_update
                        no_improvement_count = 0
                        # Save Best Checkpoint
                        torch.save({
                            "step": current_update,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": v_loss,
                            "branch": branch_key,
                            "per_category_loss": val_res["per_category_loss"],
                            "parent_sha256": EXPECTED_PARENT_SHA
                        }, best_ckpt_path)
                        print(f"    [NEW BEST CHECKPOINT] Improvement of {improvement:.4f} -> Saved to {best_ckpt_path.name}")
                    else:
                        no_improvement_count += 1
                        print(f"    [NO IMPROVEMENT] Count: {no_improvement_count}/{patience_stop} (Best remains Step {best_step} @ {best_val_loss:.4f})")

                        # Plateau check (3 consecutive)
                        if no_improvement_count >= patience_plateau:
                            print(f"    [SAFEGUARD] Validation plateau detected ({no_improvement_count} evals without improvement).")

                        # Early stopping tripwire (5 consecutive)
                        if no_improvement_count >= patience_stop:
                            early_stopped = True
                            stop_reason = f"Early stopping triggered: 5 consecutive evaluations without improvement (Best: Step {best_step} @ {best_val_loss:.4f})"
                            print(f"\n[EARLY STOP] {stop_reason}")
                            break

                        # Sustained degradation check (>10% above minimum)
                        if v_loss > best_val_loss * 1.10:
                            early_stopped = True
                            stop_reason = f"Sustained degradation: Val loss {v_loss:.4f} exceeded best {best_val_loss:.4f} by >10%"
                            print(f"\n[EARLY STOP] {stop_reason}")
                            break

                if current_update >= total_planned_updates:
                    break

    # Save Final Checkpoint
    print(f"\n[Step 6/6] Finalizing Training Artifacts...")
    torch.save({
        "step": current_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss_history[-1][1],
        "branch": branch_key,
        "parent_sha256": EXPECTED_PARENT_SHA
    }, final_ckpt_path)
    print(f"  Final Checkpoint Saved: {final_ckpt_path.name} (Step {current_update})")

    # Record summary metadata
    total_wall_time = time.perf_counter() - t_start_train
    summary_report = {
        "branch": branch_key,
        "branch_name": bcfg["name"],
        "stop_reason": stop_reason,
        "total_updates": current_update,
        "planned_updates": total_planned_updates,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_loss_history[-1][1],
        "total_tokens_processed": total_tokens_processed,
        "total_target_tokens_processed": total_target_tokens_processed,
        "wall_clock_seconds": round(total_wall_time, 2),
        "active_gpu_seconds": round(active_gpu_seconds, 2),
        "val_loss_history": val_loss_history,
        "best_checkpoint_sha256": sha256_file(best_ckpt_path),
        "final_checkpoint_sha256": sha256_file(final_ckpt_path)
    }

    summary_file = stage_dir / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE FOR {branch_key.upper()}")
    print(f"Stop Reason:             {stop_reason}")
    print(f"Total Updates:           {current_update}/{total_planned_updates}")
    print(f"Best Step:               Step {best_step} (Val Loss: {best_val_loss:.4f})")
    print(f"Target Tokens Processed: {total_target_tokens_processed:,}")
    print(f"Wall Clock Time:         {total_wall_time / 60:.2f} minutes")
    print(f"Best Checkpoint SHA-256: {summary_report['best_checkpoint_sha256']}")
    print("=" * 80)

    # Release GPU memory cleanly
    del model
    del optimizer
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["branch_b", "branch_c"], required=True, help="Branch to train")
    args = parser.parse_args()
    train_branch(args.branch)
