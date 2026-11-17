"""
QORIN Phase 2D: Experiment D — Stage 3 Pilot (50 Steps)
Parent Model: ckpt_experiment_c_final.pt (SHA-256: 6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b)
Tokenizer: tokenizer.json (SHA-256: d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59)
Target: 50 optimizer updates @ microbatch=1, accum=8 (400 records, ~150k tokens)
Optimizer: bitsandbytes 8-bit AdamW (fresh)
Supervision: Assistant-target-only loss masking
"""

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer
from modern_gpt.model import ModernGPT, ModelConfig

# Paths
PILOT_DIR = Path(__file__).resolve().parent
EXP_D_DIR = PILOT_DIR.parent
CKPT_DIR = PILOT_DIR / "checkpoints"
LOG_DIR = PILOT_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt"
EXPECTED_PARENT_SHA = "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOK_SHA = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

TRAIN_DATA_PATH = EXP_D_DIR / "stage_1_data/train.jsonl"
VAL_DATA_PATH = EXP_D_DIR / "stage_1_data/val.jsonl"
MANIFEST_PATH = EXP_D_DIR / "stage_1_data/dataset_manifest.json"

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
        return {"error": str(e), "timestamp": time.time()}

class ConversationJSONLDataset(Dataset):
    def __init__(self, path: Path, serializer: ConversationSerializer, max_samples: Optional[int] = None):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                encoded = serializer.encode_conversation(rec["messages"], max_length=2048, append_eos=True)
                self.samples.append({
                    "input_ids": encoded["input_ids"],
                    "labels": encoded["labels"],
                    "category": rec.get("category", "unknown"),
                    "id": rec.get("id", f"sample_{idx}")
                })
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_conversations(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for b in batch:
        inp = b["input_ids"].tolist()
        lab = b["labels"].tolist()
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

def evaluate_val_metrics(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
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
                
                # Per-sample loss calculation for category breakdown
                for i in range(input_ids.size(0)):
                    sample_logits = shift_logits[i]
                    sample_labels = shift_labels[i]
                    mask = (sample_labels != -100)
                    n_tok = mask.sum().item()
                    if n_tok == 0:
                        continue
                    s_loss = loss_fn(sample_logits[mask], sample_labels[mask]).item()
                    cat = categories[i]

                    total_loss += s_loss
                    total_tokens += n_tok
                    cat_loss[cat] = cat_loss.get(cat, 0.0) + s_loss
                    cat_tokens[cat] = cat_tokens.get(cat, 0.0) + n_tok

    model.train()
    overall_loss = total_loss / max(1, total_tokens)
    per_cat = {cat: cat_loss[cat] / max(1, cat_tokens[cat]) for cat in cat_loss}
    return {
        "val_loss": round(overall_loss, 4),
        "total_tokens": total_tokens,
        "per_category_loss": {k: round(v, 4) for k, v in per_cat.items()}
    }

def get_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    print("=" * 80)
    print("QORIN Phase 2D: Experiment D — Stage 3 Pilot (50 Steps)")
    print("=" * 80)

    # 1. Preflight Validations
    print("\n[1/6] Running Preflight Integrity Checks...")
    init_sha = sha256_file(PARENT_CKPT_PATH)
    print(f"Parent Checkpoint: {PARENT_CKPT_PATH}")
    print(f"Verified SHA-256:  {init_sha}")
    if init_sha != EXPECTED_PARENT_SHA:
        raise ValueError(f"CRITICAL: Parent checkpoint SHA mismatch! Expected {EXPECTED_PARENT_SHA}, got {init_sha}")

    tok_sha = sha256_file(TOK_PATH)
    print(f"Tokenizer:         {TOK_PATH}")
    print(f"Verified SHA-256:  {tok_sha}")
    if tok_sha != EXPECTED_TOK_SHA:
        raise ValueError(f"CRITICAL: Tokenizer SHA mismatch! Expected {EXPECTED_TOK_SHA}, got {tok_sha}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    train_sha = sha256_file(TRAIN_DATA_PATH)
    val_sha = sha256_file(VAL_DATA_PATH)
    expected_train_sha = manifest_data["dataset_files"]["train"]["sha256"]
    expected_val_sha = manifest_data["dataset_files"]["val"]["sha256"]
    print(f"Train Dataset SHA-256: {train_sha} (Matches manifest: {train_sha == expected_train_sha})")
    print(f"Val Dataset SHA-256:   {val_sha} (Matches manifest: {val_sha == expected_val_sha})")
    assert train_sha == expected_train_sha, "Train dataset SHA mismatch!"
    assert val_sha == expected_val_sha, "Val dataset SHA mismatch!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Tokenizer and Serializer
    print("\n[2/6] Loading Tokenizer and Conversation Serializer...")
    tok_raw = Tokenizer.from_file(str(TOK_PATH))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")
    serializer = ConversationSerializer(tokenizer, max_length=2048)

    # 3. Model Loading
    print("\n[3/6] Initializing Model from Experiment C Final Checkpoint...")
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg).to(device)
    ckpt_data = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    model.train()
    param_count = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters: {param_count:,} total, {unique_params:,} unique (tied embeddings)")
    assert unique_params == 514389504, f"Unexpected unique parameter count: {unique_params}"

    # 4. Datasets & Dataloaders
    print("\n[4/6] Loading Datasets...")
    # For pilot: exactly 400 train records (50 updates * 8 accum steps)
    accum_steps = 8
    pilot_updates = 50
    records_needed = pilot_updates * accum_steps
    train_dataset = ConversationJSONLDataset(TRAIN_DATA_PATH, serializer, max_samples=records_needed)
    val_dataset = ConversationJSONLDataset(VAL_DATA_PATH, serializer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_conversations)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_conversations)
    print(f"Loaded {len(train_dataset)} pilot train conversations, {len(val_dataset)} validation conversations.")

    # 5. Optimizer, Scheduler, Hyperparameters
    warmup_steps = 25
    peak_lr = 1.2e-5
    min_lr = 1.2e-6
    weight_decay = 0.01

    print(f"\nPilot Configuration:")
    print(f"  Microbatch Size:        1")
    print(f"  Gradient Accumulation:  {accum_steps}")
    print(f"  Total Optimizer Steps:  {pilot_updates}")
    print(f"  Warmup Steps:           {warmup_steps} (50.0% of pilot)")
    print(f"  Peak Learning Rate:     {peak_lr}")
    print(f"  Min Learning Rate:      {min_lr}")
    print(f"  Weight Decay:           {weight_decay}")
    print(f"  Checkpoints:            Step 50 (final pilot)")

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=min_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Pre-training initial validation loss (Step 0)
    print("\nMeasuring Initial Validation Metrics (Step 0)...")
    val_metrics_step0 = evaluate_val_metrics(model, val_loader, device)
    print(f"Step 0 Val Loss: {val_metrics_step0['val_loss']}")
    print(f"Step 0 Category Breakdown: {json.dumps(val_metrics_step0['per_category_loss'], indent=2)}")

    # Logging setup
    train_log_file = LOG_DIR / "pilot_training_log.jsonl"
    thermal_log_file = LOG_DIR / "pilot_thermal_log.json"
    with open(train_log_file, "w", encoding="utf-8") as f:
        pass
    thermal_records = []
    val_checkpoints = {"step_0": val_metrics_step0}

    # 6. Pilot Training Loop
    print("\n[5/6] Starting Stage 3 Pilot Loop (50 Updates)...")
    optimizer.zero_grad()
    t_start = time.perf_counter()
    running_loss = 0.0
    total_conv_tokens = 0
    total_asst_tokens = 0
    category_tokens = {}
    current_update = 0
    microbatch_count = 0

    for batch in train_loader:
        microbatch_count += 1
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        category = batch["categories"][0]

        conv_tok = input_ids.size(1)
        asst_tok = (labels != -100).sum().item()
        total_conv_tokens += conv_tok
        total_asst_tokens += asst_tok

        if category not in category_tokens:
            category_tokens[category] = {"conv_tokens": 0, "asst_tokens": 0, "records": 0}
        category_tokens[category]["conv_tokens"] += conv_tok
        category_tokens[category]["asst_tokens"] += asst_tok
        category_tokens[category]["records"] += 1

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            scaled_loss = loss / accum_steps

        scaled_loss.backward()
        running_loss += loss.item()

        if microbatch_count % accum_steps == 0:
            current_update += 1
            cur_lr = get_lr(current_update, pilot_updates, warmup_steps, peak_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_before = float(unclipped_norm.item())
            grad_norm_after = min(grad_norm_before, 1.0)

            optimizer.step()
            optimizer.zero_grad()

            avg_step_loss = running_loss / accum_steps
            running_loss = 0.0

            cur_time = time.perf_counter()
            elapsed_sec = cur_time - t_start
            tok_per_sec = total_conv_tokens / max(0.1, elapsed_sec)

            telem = get_gpu_telemetry()
            thermal_records.append(telem)

            # Thermal throttle check
            if telem.get("temp_c", 0) > 78.0:
                print(f"[THERMAL WARNING] GPU temp {telem['temp_c']}C exceeds 78C. Cooling 30s...")
                time.sleep(30)

            log_entry = {
                "step": current_update,
                "microbatch": microbatch_count,
                "loss": round(avg_step_loss, 4),
                "learning_rate": cur_lr,
                "grad_norm_before_clip": round(grad_norm_before, 4),
                "grad_norm_after_clip": round(grad_norm_after, 4),
                "cumulative_conv_tokens": total_conv_tokens,
                "cumulative_asst_tokens": total_asst_tokens,
                "throughput_tok_per_sec": round(tok_per_sec, 2),
                "gpu_temp_c": telem.get("temp_c"),
                "gpu_vram_mib": telem.get("memory_used_mib"),
                "elapsed_sec": round(elapsed_sec, 1)
            }
            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if current_update % 5 == 0 or current_update == pilot_updates:
                print(
                    f"Step {current_update:02d}/{pilot_updates:02d} | "
                    f"Loss: {avg_step_loss:.4f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"GradNorm: {grad_norm_before:.2f} | "
                    f"Tok/s: {tok_per_sec:.1f} | "
                    f"VRAM: {telem.get('memory_used_mib', 0):.0f}MiB | "
                    f"Temp: {telem.get('temp_c', 0)}C | "
                    f"Elapsed: {elapsed_sec:.1f}s"
                )

            # Midpoint Validation (Step 25)
            if current_update == 25:
                print("\n--> Evaluating Midpoint Validation Metrics (Step 25)...")
                val_m25 = evaluate_val_metrics(model, val_loader, device)
                val_checkpoints["step_25"] = val_m25
                print(f"Step 25 Val Loss: {val_m25['val_loss']}")

            if current_update >= pilot_updates:
                break

    # 7. Post-Pilot Final Validation & Checkpointing
    t_end = time.perf_counter()
    total_elapsed_sec = t_end - t_start
    print(f"\nPilot Loop Completed in {total_elapsed_sec:.1f}s ({total_elapsed_sec/60:.2f} mins).")

    print("\nMeasuring Final Validation Metrics (Step 50)...")
    val_m50 = evaluate_val_metrics(model, val_loader, device)
    val_checkpoints["step_50"] = val_m50
    print(f"Step 50 Val Loss: {val_m50['val_loss']}")
    print(f"Step 50 Category Breakdown: {json.dumps(val_m50['per_category_loss'], indent=2)}")

    # Save pilot checkpoint
    pilot_ckpt_path = CKPT_DIR / "ckpt_pilot_step50.pt"
    print(f"\n[6/6] Saving Pilot Checkpoint to {pilot_ckpt_path}...")
    torch.save({
        "step": pilot_updates,
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "val_loss": val_m50["val_loss"],
        "parent_checkpoint": str(PARENT_CKPT_PATH),
        "parent_sha256": EXPECTED_PARENT_SHA
    }, pilot_ckpt_path)

    pilot_sha = sha256_file(pilot_ckpt_path)
    print(f"Pilot Checkpoint SHA-256: {pilot_sha}")

    # Save summary and thermal logs
    with open(thermal_log_file, "w", encoding="utf-8") as f:
        json.dump(thermal_records, f, indent=2)

    summary = {
        "experiment": "modern_gpt_experiment_d_pilot",
        "parent_checkpoint": {
            "path": str(PARENT_CKPT_PATH),
            "sha256": EXPECTED_PARENT_SHA
        },
        "pilot_checkpoint": {
            "path": str(pilot_ckpt_path),
            "sha256": pilot_sha,
            "step": pilot_updates
        },
        "validation_trajectory": val_checkpoints,
        "token_metrics": {
            "total_conv_tokens_trained": total_conv_tokens,
            "total_asst_tokens_trained": total_asst_tokens,
            "category_token_tally": category_tokens
        },
        "training_dynamics": {
            "total_updates": pilot_updates,
            "accum_steps": accum_steps,
            "elapsed_seconds": round(total_elapsed_sec, 2),
            "elapsed_minutes": round(total_elapsed_sec / 60, 2),
            "avg_throughput_tok_per_sec": round(total_conv_tokens / max(0.1, total_elapsed_sec), 2),
            "val_loss_step_0": val_checkpoints["step_0"]["val_loss"],
            "val_loss_step_25": val_checkpoints["step_25"]["val_loss"],
            "val_loss_step_50": val_checkpoints["step_50"]["val_loss"]
        },
        "thermal_summary": {
            "max_gpu_temp_c": max((r.get("temp_c", 0) for r in thermal_records), default=0),
            "avg_power_w": round(sum(r.get("power_w", 0) for r in thermal_records) / max(1, len(thermal_records)), 2),
            "max_vram_used_mib": max((r.get("memory_used_mib", 0) for r in thermal_records), default=0)
        }
    }

    summary_file = LOG_DIR / "pilot_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_file}")
    print("\nStage 3 Pilot Execution Complete.")

if __name__ == "__main__":
    main()
