"""
QORIN Phase 2E: Branch D (Combined Multi-Task Training) Runner
Implements Phase 1 specifications:
- Matched-update training: Step 1 to Step 200
- 800 combined training records (400 Calibration + 400 Synthesis)
- Effective batch size = 4 (Micro-batch = 2, Gradient Accumulation = 2)
- Alternating 1:1 balanced batches (Microbatch 1: Calibration, Microbatch 2: Synthesis)
- Balanced loss weighting (alpha = 0.5)
- Fresh AdamW 8-bit optimizer, lr=1.5e-5, cosine schedule with 20 warmup steps
- Strict safeguards (plateau detection, 10% degradation tripwire, thermal guard, grad clipping)
- Saves Step 200 checkpoint (ckpt_branch_d_step200.pt) and best checkpoint (ckpt_branch_d_best.pt)
- Stops after Step 200 (Mandatory stopping point)
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
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
COMBINED_TRAIN_PATH = DATA_DIR / "combined_train.jsonl"
EXPECTED_TRAIN_SHA = "bac3916840b1135b1b5303d312f8214f378bf3950b131c9abd410a44ba4ecf2e"
COMBINED_VAL_PATH = DATA_DIR / "combined_val.jsonl"
EXPECTED_VAL_SHA = "22df08bf2d7dd298eb91a0f98b01998de80c6c68465ad2e3e10c02628d5813ca"

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

class ChatMLRecordDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tok: Tokenizer):
        self.samples = []
        for rec in records:
            chatml = rec["serialized_chatml"]
            prefix_len = rec["prefix_token_length"]
            target_len = rec["target_token_length"]
            
            input_ids = tok.encode(chatml).ids
            # Assistant-target only loss masking:
            labels = [-100] * prefix_len + input_ids[prefix_len:]
            assert len(input_ids) == len(labels)
            
            self.samples.append({
                "id": rec["id"],
                "corpus": rec.get("corpus", "unknown"),
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
        "corpus": [b["corpus"] for b in batch],
        "categories": [b["category"] for b in batch],
        "ids": [b["id"] for b in batch]
    }

def evaluate_validation(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0
    corpus_loss = {}
    corpus_tokens = {}
    cat_loss = {}
    cat_tokens = {}
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            corpus_list = batch["corpus"]
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
                    corp = corpus_list[i]
                    cat = categories[i]

                    total_loss += sample_loss
                    total_target_tokens += n_tok
                    corpus_loss[corp] = corpus_loss.get(corp, 0.0) + sample_loss
                    corpus_tokens[corp] = corpus_tokens.get(corp, 0.0) + n_tok
                    cat_loss[cat] = cat_loss.get(cat, 0.0) + sample_loss
                    cat_tokens[cat] = cat_tokens.get(cat, 0.0) + n_tok

    model.train()
    overall_loss = total_loss / max(1, total_target_tokens)
    per_corpus = {corp: round(corpus_loss[corp] / max(1, corpus_tokens[corp]), 4) for corp in corpus_loss}
    per_cat = {cat: round(cat_loss[cat] / max(1, cat_tokens[cat]), 4) for cat in cat_loss}
    return {
        "val_loss": round(overall_loss, 4),
        "total_target_tokens": total_target_tokens,
        "per_corpus_loss": per_corpus,
        "per_category_loss": per_cat
    }

def get_lr(step: int, total_steps: int = 200, warmup_steps: int = 20, peak_lr: float = 1.5e-5, min_lr: float = 1.5e-6) -> float:
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def train_branch_d():
    stage_dir = Path(__file__).resolve().parent / "branch_d_combined"
    ckpt_dir = stage_dir / "checkpoints"
    log_dir = stage_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("QORIN EXPERIMENT E: BRANCH D (COMBINED MULTI-TASK SFT)")
    print(f"Output Directory: {stage_dir}")
    print("=" * 80)

    # Preflight Verification
    print("\n[Step 1/6] Running Mandatory Pre-training Integrity Audits...")
    parent_sha = sha256_file(PARENT_CKPT_PATH)
    assert parent_sha == EXPECTED_PARENT_SHA, f"Parent SHA mismatch! {parent_sha}"
    tok_sha = sha256_file(TOK_PATH)
    assert tok_sha == EXPECTED_TOK_SHA, f"Tokenizer SHA mismatch! {tok_sha}"
    
    assert sha256_file(COMBINED_TRAIN_PATH) == EXPECTED_TRAIN_SHA, "Train dataset SHA mismatch!"
    assert sha256_file(COMBINED_VAL_PATH) == EXPECTED_VAL_SHA, "Val dataset SHA mismatch!"
    
    print(f"  Parent Checkpoint Verified: {PARENT_CKPT_PATH.name} (SHA: {parent_sha[:16]}...) [IMMUTABLE]")
    print(f"  Tokenizer Verified:         {TOK_PATH.name} (SHA: {tok_sha[:16]}...)")
    print(f"  Combined Train Verified:    {COMBINED_TRAIN_PATH.name} ({EXPECTED_TRAIN_SHA[:16]}...)")
    print(f"  Combined Val Verified:      {COMBINED_VAL_PATH.name} ({EXPECTED_VAL_SHA[:16]}...)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Compute Device:             {device} ({torch.cuda.get_device_name(0)})")
    initial_telem = get_gpu_telemetry()
    print(f"  Initial GPU State:          Temp={initial_telem['temp_c']}C, VRAM={initial_telem['memory_used_mib']}MiB")

    # Load Tokenizer & Model
    print("\n[Step 2/6] Initializing Model from Frozen Parent Checkpoint (Exp D Step 425)...")
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg).to(device)

    parent_ckpt_data = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    state_dict = parent_ckpt_data["model_state_dict"] if "model_state_dict" in parent_ckpt_data else parent_ckpt_data
    model.load_state_dict(state_dict, strict=True)
    model.train()
    print(f"  Loaded parent weights into ModernGPT 500M ({sum(p.numel() for p in model.parameters()):,} params).")

    # Load and Partition Combined Datasets into Calibration & Synthesis
    print("\n[Step 3/6] Setting Up Balanced Alternating 1:1 Batching (alpha = 0.5)...")
    with open(COMBINED_TRAIN_PATH, "r", encoding="utf-8") as f:
        all_train_records = [json.loads(line) for line in f if line.strip()]
    cal_records = [r for r in all_train_records if r.get("corpus") == "corpus1_calibration"]
    syn_records = [r for r in all_train_records if r.get("corpus") == "corpus2_synthesis"]
    assert len(cal_records) == 400 and len(syn_records) == 400

    # Shuffle both splits with fixed seed 42
    random.seed(42)
    random.shuffle(cal_records)
    random.shuffle(syn_records)

    with open(COMBINED_VAL_PATH, "r", encoding="utf-8") as f:
        all_val_records = [json.loads(line) for line in f if line.strip()]
    val_dataset = ChatMLRecordDataset(all_val_records, tok)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_chatml)

    cal_dataset = ChatMLRecordDataset(cal_records, tok)
    syn_dataset = ChatMLRecordDataset(syn_records, tok)
    cal_loader = DataLoader(cal_dataset, batch_size=2, shuffle=True, collate_fn=collate_chatml)
    syn_loader = DataLoader(syn_dataset, batch_size=2, shuffle=True, collate_fn=collate_chatml)

    print(f"  Train: 400 Calibration records + 400 Synthesis records = 800 total records")
    print(f"  Val:   50 Calibration records + 50 Synthesis records = 100 total records")
    print(f"  Kinematics:")
    print(f"    Microbatch: 2 sequences | Accumulation: 2 steps")
    print(f"    Sub-step 1: Microbatch of 2 Calibration sequences (alpha = 0.5)")
    print(f"    Sub-step 2: Microbatch of 2 Synthesis sequences (alpha = 0.5)")
    print(f"    Effective Batch: 4 sequences per optimizer update")
    print(f"    Planned Updates: Exactly 200 optimizer steps (1.0 epoch across 800 records)")

    # Optimizer & Hyperparameters
    print("\n[Step 4/6] Initializing Fresh Optimizer & Safeguards...")
    total_planned_updates = 200
    warmup_steps = 20
    peak_lr = 1.5e-5
    min_lr = 1.5e-6
    weight_decay = 0.05

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=get_lr(0, total_planned_updates, warmup_steps, peak_lr, min_lr),
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Initial Step 0 Validation
    print("  Running pre-training validation (Step 0 baseline)...")
    init_val = evaluate_validation(model, val_loader, device)
    best_val_loss = init_val["val_loss"]
    best_step = 0
    print(f"  Step 0 Val Loss: {best_val_loss:.4f} (Target tokens evaluated: {init_val['total_target_tokens']})")
    print(f"  Per-corpus: {init_val['per_corpus_loss']}")

    train_log_file = log_dir / "training_log.jsonl"
    val_log_file = log_dir / "validation_log.jsonl"
    with open(val_log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "step": 0,
            "val_loss": best_val_loss,
            "target_tokens": init_val["total_target_tokens"],
            "per_corpus_loss": init_val["per_corpus_loss"],
            "per_category_loss": init_val["per_category_loss"],
            "timestamp": time.time()
        }) + "\n")

    eval_interval = 25
    patience_plateau = 3
    patience_stop = 5
    no_improvement_count = 0
    val_loss_history = [(0, best_val_loss)]
    thermal_pause_temp = 78.0
    thermal_abort_temp = 82.0
    max_session_gpu_seconds = 105 * 60

    best_ckpt_path = ckpt_dir / "ckpt_branch_d_best.pt"
    step200_ckpt_path = ckpt_dir / "ckpt_branch_d_step200.pt"

    torch.save({
        "step": 0,
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "branch": "branch_d_combined",
        "parent_sha256": EXPECTED_PARENT_SHA
    }, best_ckpt_path)

    # Commencing Controlled Training Loop
    print("\n[Step 5/6] Commencing Combined Multi-Task Training Loop (Step 1 to Step 200)...")
    current_update = 0
    running_cal_loss = 0.0
    running_syn_loss = 0.0
    running_target_tokens = 0
    total_tokens_processed = 0
    total_target_tokens_processed = 0

    optimizer.zero_grad()
    t_start_train = time.perf_counter()
    active_gpu_seconds = 0.0
    early_stopped = False
    stop_reason = "Completed Step 200 evaluation gate (Mandatory Stop)"

    cal_iter = iter(cal_loader)
    syn_iter = iter(syn_loader)

    while current_update < total_planned_updates and not early_stopped:
        t0_step = time.perf_counter()
        
        # 1. Calibration Microbatch (alpha = 0.5)
        try:
            batch_cal = next(cal_iter)
        except StopIteration:
            cal_iter = iter(cal_loader)
            batch_cal = next(cal_iter)

        input_cal = batch_cal["input_ids"].to(device)
        labels_cal = batch_cal["labels"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_cal = model(input_cal)
            shift_logits_cal = logits_cal[..., :-1, :].contiguous()
            shift_labels_cal = labels_cal[..., 1:].contiguous()
            loss_cal = loss_fn(shift_logits_cal.view(-1, shift_logits_cal.size(-1)), shift_labels_cal.view(-1))
            loss_cal_scaled = 0.5 * loss_cal  # alpha = 0.5, accum = 2

        loss_cal_scaled.backward()
        n_cal_target = (shift_labels_cal != -100).sum().item()
        running_cal_loss += loss_cal.item() * n_cal_target
        running_target_tokens += n_cal_target
        total_tokens_processed += input_cal.numel()
        total_target_tokens_processed += n_cal_target

        # 2. Synthesis Microbatch (alpha = 0.5)
        try:
            batch_syn = next(syn_iter)
        except StopIteration:
            syn_iter = iter(syn_loader)
            batch_syn = next(syn_iter)

        input_syn = batch_syn["input_ids"].to(device)
        labels_syn = batch_syn["labels"].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_syn = model(input_syn)
            shift_logits_syn = logits_syn[..., :-1, :].contiguous()
            shift_labels_syn = labels_syn[..., 1:].contiguous()
            loss_syn = loss_fn(shift_logits_syn.view(-1, shift_logits_syn.size(-1)), shift_labels_syn.view(-1))
            loss_syn_scaled = 0.5 * loss_syn  # alpha = 0.5, accum = 2

        loss_syn_scaled.backward()
        n_syn_target = (shift_labels_syn != -100).sum().item()
        running_syn_loss += loss_syn.item() * n_syn_target
        running_target_tokens += n_syn_target
        total_tokens_processed += input_syn.numel()
        total_target_tokens_processed += n_syn_target

        current_update += 1
        t_step = time.perf_counter() - t0_step
        active_gpu_seconds += t_step

        # Numerical sanity check
        if math.isnan(loss_cal.item()) or math.isnan(loss_syn.item()):
            early_stopped = True
            stop_reason = f"Numerical Instability: NaN loss at step {current_update}"
            print(f"\n[ALERT] {stop_reason} -> ABORTING")
            break

        # Gradient clipping to max norm 1.0 (L2)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        if math.isnan(grad_norm) or math.isinf(grad_norm) or grad_norm > 100.0:
            early_stopped = True
            stop_reason = f"Gradient norm explosion ({grad_norm:.2f}) at step {current_update}"
            print(f"\n[ALERT] {stop_reason} -> ABORTING")
            break

        # Optimizer update & LR
        lr = get_lr(current_update, total_planned_updates, warmup_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
        optimizer.zero_grad()

        avg_step_loss = (running_cal_loss + running_syn_loss) / max(1, running_target_tokens)
        cal_step_loss = running_cal_loss / max(1, n_cal_target)
        syn_step_loss = running_syn_loss / max(1, n_syn_target)
        running_cal_loss = 0.0
        running_syn_loss = 0.0
        running_target_tokens = 0

        # GPU Telemetry & Thermal Guard
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

        # Duty cycle limit
        if active_gpu_seconds >= max_session_gpu_seconds:
            early_stopped = True
            stop_reason = f"Reached maximum active GPU compute window ({active_gpu_seconds / 60:.1f} min)"
            print(f"\n[SAFEGUARD] {stop_reason} -> STOPPING")
            break

        # Log training step
        train_entry = {
            "step": current_update,
            "train_loss": round(avg_step_loss, 4),
            "cal_loss": round(cal_step_loss, 4),
            "syn_loss": round(syn_step_loss, 4),
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
            print(f"Step {current_update:>3}/200 | Loss: {avg_step_loss:.4f} (Cal: {cal_step_loss:.4f}, Syn: {syn_step_loss:.4f}) | Grad: {grad_norm:.2f} | LR: {lr:.2e} | Temp: {cur_temp:.0f}C")

        # Evaluation & Checkpointing Cadence
        if current_update % eval_interval == 0:
            val_res = evaluate_validation(model, val_loader, device)
            v_loss = val_res["val_loss"]
            val_loss_history.append((current_update, v_loss))

            val_entry = {
                "step": current_update,
                "val_loss": v_loss,
                "target_tokens": val_res["total_target_tokens"],
                "per_corpus_loss": val_res["per_corpus_loss"],
                "per_category_loss": val_res["per_category_loss"],
                "timestamp": time.time()
            }
            with open(val_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_entry) + "\n")

            print(f"\n>>> [EVALUATION STEP {current_update}] Val Loss: {v_loss:.4f} (Best: {best_val_loss:.4f})")
            print(f"    Per-corpus: {val_res['per_corpus_loss']}")

            if v_loss < best_val_loss:
                improvement = best_val_loss - v_loss
                best_val_loss = v_loss
                best_step = current_update
                no_improvement_count = 0
                torch.save({
                    "step": current_update,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "branch": "branch_d_combined",
                    "per_corpus_loss": val_res["per_corpus_loss"],
                    "parent_sha256": EXPECTED_PARENT_SHA
                }, best_ckpt_path)
                print(f"    [NEW BEST CHECKPOINT] Improvement of {improvement:.4f} -> Saved to {best_ckpt_path.name}")
            else:
                no_improvement_count += 1
                print(f"    [NO IMPROVEMENT] Count: {no_improvement_count}/{patience_stop} (Best remains Step {best_step} @ {best_val_loss:.4f})")

                if no_improvement_count >= patience_plateau:
                    print(f"    [SAFEGUARD] Validation plateau detected ({no_improvement_count} evals without improvement).")

                if no_improvement_count >= patience_stop:
                    early_stopped = True
                    stop_reason = f"Early stopping triggered: 5 consecutive evaluations without improvement (Best: Step {best_step} @ {best_val_loss:.4f})"
                    print(f"\n[EARLY STOP] {stop_reason}")
                    break

                if v_loss > best_val_loss * 1.10:
                    early_stopped = True
                    stop_reason = f"Sustained degradation: Val loss {v_loss:.4f} exceeded best {best_val_loss:.4f} by >10%"
                    print(f"\n[EARLY STOP] {stop_reason}")
                    break

    # Save Step 200 Milestone Checkpoint
    print(f"\n[Step 6/6] Finalizing Step 200 Checkpoint Artifacts...")
    torch.save({
        "step": current_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss_history[-1][1],
        "branch": "branch_d_combined",
        "parent_sha256": EXPECTED_PARENT_SHA
    }, step200_ckpt_path)
    print(f"  Step 200 Checkpoint Saved: {step200_ckpt_path.name} (Step {current_update})")

    total_wall_time = time.perf_counter() - t_start_train
    summary_report = {
        "branch": "branch_d_combined",
        "stop_reason": stop_reason,
        "total_updates": current_update,
        "planned_updates": total_planned_updates,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "step200_val_loss": val_loss_history[-1][1],
        "total_tokens_processed": total_tokens_processed,
        "total_target_tokens_processed": total_target_tokens_processed,
        "wall_clock_seconds": round(total_wall_time, 2),
        "active_gpu_seconds": round(active_gpu_seconds, 2),
        "val_loss_history": val_loss_history,
        "best_checkpoint_sha256": sha256_file(best_ckpt_path),
        "step200_checkpoint_sha256": sha256_file(step200_ckpt_path)
    }

    summary_file = stage_dir / "training_summary_step200.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 200 TRAINING COMPLETE FOR BRANCH D")
    print(f"Stop Reason:             {stop_reason}")
    print(f"Total Updates:           {current_update}/{total_planned_updates}")
    print(f"Best Step:               Step {best_step} (Val Loss: {best_val_loss:.4f})")
    print(f"Step 200 Val Loss:       {val_loss_history[-1][1]:.4f}")
    print(f"Target Tokens Processed: {total_target_tokens_processed:,}")
    print(f"Wall Clock Time:         {total_wall_time / 60:.2f} minutes")
    print(f"Step 200 Checkpoint SHA: {summary_report['step200_checkpoint_sha256']}")
    print("MANDATORY STOP ENFORCED: Pausing before Step 201-400.")
    print("=" * 80)

    del model
    del optimizer
    del cal_loader
    del syn_loader
    del val_loader
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    train_branch_d()
