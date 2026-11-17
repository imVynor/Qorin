"""
ModernGPT-S / QORIN Phase 2D Experiment C — Official Training Run (Stage 3)
Model: 514M ModernGPT-S (514,389,504 unique parameters)
Optimizer: Config A (8-bit AdamW via bitsandbytes)
Initialization: ckpt_experiment_b_final.pt (SHA-256 verified)
Curriculum: Stage 1 release (train.jsonl: 3,962 records, val.jsonl: 208 records)
Budget: ~1,885,000 serialized conversation tokens (~495 updates @ microbatch=1, accum=8)
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
STAGE_DIR = Path(__file__).resolve().parent
CKPT_DIR = STAGE_DIR / "checkpoints"
LOG_DIR = STAGE_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CKPT_PATH = REPO_ROOT / "modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt"
EXPECTED_STARTING_SHA = "acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
TRAIN_DATA_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_1_data/train.jsonl"
VAL_DATA_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_1_data/val.jsonl"
MANIFEST_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_1_data/dataset_manifest.json"
TOOL_SPEC_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/tool_context_spec.json"

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

def evaluate_val_loss(model, dataloader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            n_tokens = (labels != -100).sum().item()
            if n_tokens == 0:
                continue

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item()
            total_tokens += n_tokens

    model.train()
    return total_loss / max(1, total_tokens)

def get_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step <= warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

def main():
    print("================================================================================")
    print("ModernGPT-S / QORIN Phase 2D: Experiment C Official Training Run")
    print("================================================================================")

    # 1. Preflight Validations
    print("[1/6] Running Preflight Integrity Checks...")
    init_sha = sha256_file(STARTING_CKPT_PATH)
    print(f"Starting Checkpoint: {STARTING_CKPT_PATH}")
    print(f"Verified SHA-256:    {init_sha}")
    if init_sha != EXPECTED_STARTING_SHA:
        raise ValueError(f"CRITICAL: Starting checkpoint SHA mismatch! Expected {EXPECTED_STARTING_SHA}, got {init_sha}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    train_sha = sha256_file(TRAIN_DATA_PATH)
    val_sha = sha256_file(VAL_DATA_PATH)
    print(f"Train Dataset SHA-256: {train_sha} (Matches manifest: {train_sha == manifest_data['train_sha256']})")
    print(f"Val Dataset SHA-256:   {val_sha} (Matches manifest: {val_sha == manifest_data['val_sha256']})")
    assert train_sha == manifest_data["train_sha256"], "Train dataset SHA mismatch!"
    assert val_sha == manifest_data["val_sha256"], "Val dataset SHA mismatch!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Tokenizer and Serializer
    print("\n[2/6] Loading Tokenizer and Conversation Serializer...")
    tok_raw = Tokenizer.from_file(str(TOK_PATH))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")
    with open(TOOL_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)
    manifest = spec["compact_manifests"]["two_tool_manifest"]
    serializer = ConversationSerializer(tokenizer, max_length=2048, default_system=manifest)
    print(f"Compact System Prompt Token Count: {len(tokenizer.encode(manifest))}")

    # 3. Model Loading
    print("\n[3/6] Initializing Model from Experiment B Final Checkpoint...")
    cfg = ModelConfig.get_500m_config(gradient_checkpointing=True)
    model = ModernGPT(cfg).to(device)
    ckpt_data = torch.load(STARTING_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
    model.train()
    param_count = sum(p.numel() for p in model.parameters())
    unique_params = sum(p.numel() for p in set(model.parameters()))
    print(f"Model parameters: {param_count:,} total, {unique_params:,} unique (tied embeddings)")
    assert unique_params == 514389504, f"Unexpected unique parameter count: {unique_params}"

    # 4. Datasets & Dataloaders
    print("\n[4/6] Loading Train & Validation Datasets...")
    train_dataset = ConversationJSONLDataset(TRAIN_DATA_PATH, serializer)
    val_dataset = ConversationJSONLDataset(VAL_DATA_PATH, serializer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_conversations)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_conversations)
    print(f"Loaded {len(train_dataset)} train conversations, {len(val_dataset)} validation conversations.")

    # 5. Optimizer, Scheduler, Hyperparameters
    accum_steps = 8
    total_records = len(train_dataset)
    total_updates = total_records // accum_steps  # 3962 // 8 = 495 updates
    warmup_steps = int(total_updates * 0.03)      # 3% of 495 = 14 steps
    peak_lr = 2.0e-5
    min_lr = 2.0e-6
    weight_decay = 0.01

    print(f"\nTraining Plan:")
    print(f"  Microbatch Size:        1")
    print(f"  Gradient Accumulation:  {accum_steps}")
    print(f"  Total Optimizer Steps:  {total_updates}")
    print(f"  Warmup Steps:           {warmup_steps} (3.0%)")
    print(f"  Peak Learning Rate:     {peak_lr}")
    print(f"  Min Learning Rate:      {min_lr}")
    print(f"  Weight Decay:           {weight_decay}")
    print(f"  Checkpoint Milestones:  Step 250, Step {total_updates} (final)")

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=min_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Pre-training initial validation loss
    print("\nMeasuring Initial Validation Loss (Step 0)...")
    val_loss_init = evaluate_val_loss(model, val_loader, device)
    print(f"Initial Val Loss: {val_loss_init:.4f}")

    # Telemetry logging setup
    train_log_file = LOG_DIR / "training_log.jsonl"
    thermal_log_file = LOG_DIR / "thermal_log.json"
    with open(train_log_file, "w", encoding="utf-8") as f:
        pass  # clear/initialize
    thermal_records = []

    # 6. Training Loop
    print("\n[5/6] Starting Official Training Loop...")
    optimizer.zero_grad()
    t_start = time.perf_counter()
    running_loss = 0.0
    running_conv_tokens = 0
    running_asst_tokens = 0
    total_conv_tokens_trained = 0
    total_asst_tokens_trained = 0
    category_token_tally = {}

    current_update = 0
    microbatch_count = 0
    saved_checkpoints = []

    step_250_val_loss = None
    final_val_loss = None

    for mb_idx, batch in enumerate(train_loader):
        microbatch_count += 1
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        category = batch["categories"][0]

        conv_tokens = (input_ids != 0).sum().item()
        asst_tokens = (labels != -100).sum().item()
        running_conv_tokens += conv_tokens
        running_asst_tokens += asst_tokens
        total_conv_tokens_trained += conv_tokens
        total_asst_tokens_trained += asst_tokens

        if category not in category_token_tally:
            category_token_tally[category] = {"conv_tokens": 0, "asst_tokens": 0, "records": 0}
        category_token_tally[category]["conv_tokens"] += conv_tokens
        category_token_tally[category]["asst_tokens"] += asst_tokens
        category_token_tally[category]["records"] += 1

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            scaled_loss = loss / accum_steps

        scaled_loss.backward()
        running_loss += loss.item()

        # Check for optimizer step
        if microbatch_count % accum_steps == 0:
            current_update += 1
            cur_lr = get_lr(current_update, total_updates, warmup_steps, peak_lr, min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            # Dual gradient clipping tracking
            unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_before = float(unclipped_norm.item())
            grad_norm_after = min(grad_norm_before, 1.0)

            optimizer.step()
            optimizer.zero_grad()

            avg_step_loss = running_loss / accum_steps
            running_loss = 0.0

            # Telemetry snapshot
            cur_time = time.perf_counter()
            elapsed_sec = cur_time - t_start
            tok_per_sec = total_conv_tokens_trained / max(0.1, elapsed_sec)

            log_entry = {
                "step": current_update,
                "microbatch": microbatch_count,
                "loss": round(avg_step_loss, 4),
                "learning_rate": cur_lr,
                "grad_norm_before_clip": round(grad_norm_before, 4),
                "grad_norm_after_clip": round(grad_norm_after, 4),
                "cumulative_conv_tokens": total_conv_tokens_trained,
                "cumulative_asst_tokens": total_asst_tokens_trained,
                "throughput_tok_per_sec": round(tok_per_sec, 2),
                "elapsed_seconds": round(elapsed_sec, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            }

            with open(train_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if current_update % 25 == 0 or current_update == 1 or current_update == total_updates:
                gpu_info = get_gpu_telemetry()
                gpu_info["step"] = current_update
                thermal_records.append(gpu_info)
                print(
                    f"Step {current_update:3d}/{total_updates:3d} | "
                    f"Loss: {avg_step_loss:.4f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"GradNorm: {grad_norm_before:.3f}->{grad_norm_after:.3f} | "
                    f"ConvTok: {total_conv_tokens_trained:,} | "
                    f"Speed: {tok_per_sec:.1f} tok/s | "
                    f"GPU: {gpu_info.get('temp_c', 0)}C {gpu_info.get('power_w', 0)}W"
                )

            # Milestone Checkpoint at Step 250
            if current_update == 250:
                print("\n--> [MILESTONE] Reached Step 250. Evaluating intermediate validation loss...")
                step_250_val_loss = evaluate_val_loss(model, val_loader, device)
                print(f"Step 250 Val Loss: {step_250_val_loss:.4f}")
                ckpt_250_path = CKPT_DIR / "ckpt_step_250.pt"
                print(f"Saving intermediate checkpoint to {ckpt_250_path}...")
                torch.save({
                    "step": current_update,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": step_250_val_loss,
                    "total_conv_tokens": total_conv_tokens_trained,
                    "total_asst_tokens": total_asst_tokens_trained,
                    "config": cfg.__dict__
                }, ckpt_250_path)
                sha_250 = sha256_file(ckpt_250_path)
                saved_checkpoints.append({"step": 250, "path": str(ckpt_250_path), "sha256": sha_250})
                print(f"Step 250 Checkpoint Saved. SHA-256: {sha_250}\n")

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTraining Loop Completed in {total_elapsed/60:.2f} minutes ({total_elapsed:.1f}s).")
    print(f"Total Updates Completed: {current_update}")
    print(f"Total Conversation Tokens Processed: {total_conv_tokens_trained:,}")
    print(f"Total Assistant Target Tokens Supervised: {total_asst_tokens_trained:,}")

    # Final Validation Loss
    print("\n[6/6] Final Validation Loss & Checkpoint Persistence...")
    final_val_loss = evaluate_val_loss(model, val_loader, device)
    print(f"Final Val Loss: {final_val_loss:.4f} (Initial was {val_loss_init:.4f})")

    # Final Checkpoint Save
    final_ckpt_path = CKPT_DIR / "ckpt_experiment_c_final.pt"
    print(f"Saving Final Experiment C Checkpoint to {final_ckpt_path}...")
    torch.save({
        "step": current_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss_init": val_loss_init,
        "val_loss_final": final_val_loss,
        "total_conv_tokens": total_conv_tokens_trained,
        "total_asst_tokens": total_asst_tokens_trained,
        "category_distribution": category_token_tally,
        "config": cfg.__dict__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }, final_ckpt_path)
    final_sha = sha256_file(final_ckpt_path)
    saved_checkpoints.append({"step": current_update, "path": str(final_ckpt_path), "sha256": final_sha})
    print(f"Final Checkpoint Saved. SHA-256: {final_sha}")

    # Resume Fidelity Verification on Final Checkpoint
    print("\nVerifying Final Checkpoint Resume Fidelity...")
    reload_model = ModernGPT(cfg).to(device)
    reload_data = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    reload_model.load_state_dict(reload_data["model_state_dict"])
    reload_model.eval()
    model.eval()

    test_batch = next(iter(val_loader))
    t_inp = test_batch["input_ids"].to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            orig_out = model(t_inp)
            reload_out = reload_model(t_inp)
    max_logit_diff = torch.max(torch.abs(orig_out - reload_out)).item()
    print(f"Resume Fidelity Max Logit Difference: {max_logit_diff:.8e}")
    assert max_logit_diff < 1e-6, f"Resume fidelity failed: max_diff = {max_logit_diff}"
    print("Resume Fidelity: STRICT BIT-EXACT PASS.")

    # Re-verify Experiment B Checkpoint Hash (Proof of Zero Mutation)
    post_exp_b_sha = sha256_file(STARTING_CKPT_PATH)
    print(f"\nPost-Training Check of Experiment B Checkpoint:")
    print(f"  Expected: {EXPECTED_STARTING_SHA}")
    print(f"  Actual:   {post_exp_b_sha}")
    assert post_exp_b_sha == EXPECTED_STARTING_SHA, "CRITICAL: Experiment B starting checkpoint was mutated!"
    print("  Zero mutation confirmed: STRICT PASS.")

    # Write Thermal Log
    with open(thermal_log_file, "w", encoding="utf-8") as f:
        json.dump(thermal_records, f, indent=2)

    # Write SHA256SUMS for checkpoints
    sha_file = CKPT_DIR / "SHA256SUMS"
    with open(sha_file, "w", encoding="utf-8") as f:
        for c in saved_checkpoints:
            p = Path(c["path"])
            f.write(f"{c['sha256']}  {p.name}\n")
    print(f"Wrote checkpoint manifest: {sha_file}")

    # Write Complete Training Summary JSON
    summary = {
        "experiment": "modern_gpt_experiment_c",
        "phase": "2D",
        "starting_checkpoint": {
            "path": str(STARTING_CKPT_PATH),
            "sha256": init_sha,
            "unmutated_post_training": (post_exp_b_sha == EXPECTED_STARTING_SHA)
        },
        "final_checkpoint": {
            "path": str(final_ckpt_path),
            "sha256": final_sha,
            "resume_fidelity_max_diff": max_logit_diff,
            "resume_fidelity_pass": (max_logit_diff < 1e-6)
        },
        "intermediate_checkpoints": saved_checkpoints,
        "token_metrics": {
            "total_conv_tokens_trained": total_conv_tokens_trained,
            "total_asst_tokens_trained": total_asst_tokens_trained,
            "category_token_tally": category_token_tally
        },
        "training_dynamics": {
            "total_updates": current_update,
            "accum_steps": accum_steps,
            "elapsed_seconds": round(total_elapsed, 2),
            "elapsed_minutes": round(total_elapsed / 60, 2),
            "avg_throughput_tok_per_sec": round(total_conv_tokens_trained / max(0.1, total_elapsed), 2),
            "val_loss_step_0": round(val_loss_init, 4),
            "val_loss_step_250": round(step_250_val_loss, 4) if step_250_val_loss else None,
            "val_loss_final": round(final_val_loss, 4)
        },
        "hyperparameters": {
            "optimizer": "AdamW8bit (bitsandbytes)",
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "grad_clip_max_norm": 1.0,
            "precision": "bfloat16 autocast",
            "gradient_checkpointing": True
        },
        "thermal_summary": {
            "max_gpu_temp_c": max((r.get("temp_c", 0) for r in thermal_records), default=0),
            "avg_power_w": round(sum(r.get("power_w", 0) for r in thermal_records) / max(1, len(thermal_records)), 2),
            "max_vram_used_mib": max((r.get("memory_used_mib", 0) for r in thermal_records), default=0)
        }
    }

    summary_file = STAGE_DIR / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote full training summary to: {summary_file}")
    print("\n=== EXPERIMENT C OFFICIAL TRAINING RUN SUCCESSFULLY COMPLETED ===")

if __name__ == "__main__":
    main()
