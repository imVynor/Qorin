"""Supervised Fine-Tuning (SFT) implementation for ModernGPT-S.

Features:
- Initializes from pre-trained 16M or 50M checkpoint.
- Standard Alpaca instruction prompt template formatting.
- Assistant-only loss masking: prompt tokens masked with -100; only response tokens and EOS contribute to loss.
- Strict EOS token 0 supervision at the end of each response.
- Train and validation isolation with support for custom split files.
- Configurable LR schedule (warmup + cosine decay).
- Intermediate checkpoints and best_val tracking based on validation loss.
- Support for bf16 precision, checkpoint resume, and dry-run verification.
"""

import argparse
import contextlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT


def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    """Format input into standard Alpaca instruction prompt."""
    inst = instruction.strip()
    inp = input_text.strip() if input_text else ""
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{inst}\n\n### Response:\n"
    )


class SFTDataset(Dataset):
    """Instruction dataset with assistant-only loss masking and EOS supervision."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: HuggingFaceBPEWrapper,
        context_length: int = 512,
        min_resp_tokens: int = 16,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.min_resp_tokens = min_resp_tokens
        self.examples = []

        jsonl_path = Path(jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.examples.append(rec)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        rec = self.examples[idx]
        prompt_text = format_instruction_prompt(rec["instruction"], rec.get("input", ""))
        response_text = rec["output"].strip()

        p_ids = self.tokenizer.encode(prompt_text)
        r_ids = self.tokenizer.encode(response_text)
        eos_id = self.tokenizer.eos_token_id  # 0
        r_ids.append(eos_id)

        # Truncation strategy preserving response tokens
        max_prompt_len = self.context_length - min(len(r_ids), self.min_resp_tokens)
        if len(p_ids) > max_prompt_len:
            p_ids = p_ids[:max_prompt_len]

        input_ids = p_ids + r_ids
        labels = [-100] * len(p_ids) + r_ids

        # Truncate to context length
        input_ids = input_ids[: self.context_length]
        labels = labels[: self.context_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "category": rec.get("category", "general"),
            "id": rec.get("id", str(idx)),
        }


def collate_sft_batch(batch: list, pad_token_id: int = 0) -> tuple:
    """Collate variable-length sequences with right-padding and -100 label padding."""
    max_len = max(len(b["input_ids"]) for b in batch)

    batch_input_ids = []
    batch_labels = []

    for b in batch:
        cur_len = len(b["input_ids"])
        pad_len = max_len - cur_len

        # Pad input_ids with pad_token_id (0) and labels with -100 (ignored in cross-entropy)
        padded_inputs = b["input_ids"] + [pad_token_id] * pad_len
        padded_labels = b["labels"] + [-100] * pad_len

        batch_input_ids.append(padded_inputs)
        batch_labels.append(padded_labels)

    inputs = torch.tensor(batch_input_ids, dtype=torch.long)
    targets = torch.tensor(batch_labels, dtype=torch.long)
    return inputs, targets


def compute_sft_loss(model: ModernGPT, inputs: torch.Tensor, labels: torch.Tensor, autocast_ctx=None) -> torch.Tensor:
    """Compute causal LM cross-entropy loss ignoring prompt tokens (-100)."""
    if autocast_ctx is not None:
        with autocast_ctx:
            logits = model(inputs)
    else:
        logits = model(inputs)
    # Next-token prediction: position t predicts token at t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def evaluate_sft_loss(model: ModernGPT, val_loader: DataLoader, device: torch.device, autocast_ctx=None) -> tuple:
    """Compute full validation loss and perplexity across SFT validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            shift_labels = targets[..., 1:].contiguous()
            active_tokens = (shift_labels != -100).sum().item()
            if active_tokens == 0:
                continue

            loss = compute_sft_loss(model, inputs, targets, autocast_ctx=autocast_ctx)
            total_loss += loss.item() * active_tokens
            total_tokens += active_tokens

    mean_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity


def run_sft_dry_run(
    model: ModernGPT,
    train_dataset: SFTDataset,
    val_dataset: SFTDataset,
    tokenizer: HuggingFaceBPEWrapper,
    args,
    device: torch.device,
    autocast_ctx=None,
):
    """Execute complete SFT sanity and memory dry run without training."""
    print("\n" + "=" * 70)
    print("MODERNGPT-S: SFT DRY-RUN VERIFICATION")
    print("=" * 70)

    # 1. Dataset stats
    print(f"✓ Train dataset size:       {len(train_dataset):,} examples")
    print(f"✓ Validation dataset size:  {len(val_dataset):,} examples")
    print(f"✓ Model parameters:         {model.parameter_count:,}")
    print(f"✓ Context length:           {model.config.context_length}")
    print(f"✓ Initial Learning Rate:    {args.lr}")
    print(f"✓ Batch Size:               {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"✓ Precision:                {args.precision}")

    # 2. Check loss masking on sample batch
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tokenizer.eos_token_id),
    )
    sample_inputs, sample_targets = next(iter(train_loader))

    sample_inputs = sample_inputs.to(device)
    sample_targets = sample_targets.to(device)

    print(f"✓ Sample batch inputs shape:  {sample_inputs.shape}")
    print(f"✓ Sample batch targets shape: {sample_targets.shape}")

    shift_targets = sample_targets[..., 1:].contiguous()
    active_tokens = (shift_targets != -100).sum().item()
    total_tokens = shift_targets.numel()
    pct_supervised = active_tokens / total_tokens * 100

    print(f"✓ Supervised response tokens: {active_tokens}/{total_tokens} ({pct_supervised:.1f}% active, remainder masked)")
    assert active_tokens > 0, "Error: zero supervised tokens in sample batch!"

    # 3. Forward pass & loss check
    with torch.no_grad():
        sample_loss = compute_sft_loss(model, sample_inputs, sample_targets, autocast_ctx=autocast_ctx)
    assert torch.isfinite(sample_loss), f"Error: sample SFT loss is non-finite: {sample_loss.item()}"
    sample_ppl = math.exp(min(sample_loss.item(), 20.0))
    print(f"✓ Initial base model SFT loss: {sample_loss.item():.4f} (Perplexity: {sample_ppl:.2f})")

    # 4. Check GPU memory if available
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f"✓ CUDA memory allocated:      {allocated_mb:.1f} MB (Reserved: {reserved_mb:.1f} MB)")
        print(f"✓ Device:                     {torch.cuda.get_device_name(device)}")
    else:
        print("✓ Running on CPU (no CUDA GPU detected).")

    # 5. Planned steps calculation
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"✓ Steps per epoch:            {steps_per_epoch:,} (Effective batch size: {effective_batch})")
    print(f"✓ Total planned steps:        {total_steps:,} for {args.epochs} epoch(s)")
    print("=" * 70)
    print("DRY-RUN STATUS: ALL CHECKS PASSED. SFT RUN IS READY TO LAUNCH.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for ModernGPT-S")
    parser.add_argument("--base-checkpoint", default="checkpoints/pretrain_50m_250m/latest.pt", help="Base pre-trained checkpoint")
    parser.add_argument("--tokenizer-dir", default="artifacts/tokenizer_16k_v2", help="Tokenizer directory")
    parser.add_argument("--data-dir", default="data/midscale_500", help="Instruction data directory")
    parser.add_argument("--train-file", default=None, help="Explicit path to train jsonl")
    parser.add_argument("--val-file", default=None, help="Explicit path to validation jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft_50m_midscale_500", help="Output directory for SFT checkpoints")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16", help="Training precision")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-6, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=30, help="Warmup steps")
    parser.add_argument("--epochs", type=float, default=6.0, help="Number of training epochs")
    parser.add_argument("--eval-interval", type=int, default=25, help="Evaluation interval in optimizer steps")
    parser.add_argument("--save-interval", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient clipping norm")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest.pt in output-dir if present")
    parser.add_argument("--dry-run", action="store_true", help="Perform sanity verification without training")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autocast context
    if args.precision == "bf16" and torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    # Load tokenizer
    tokenizer = HuggingFaceBPEWrapper.load(args.tokenizer_dir)

    # Load base checkpoint
    base_ckpt_path = Path(args.base_checkpoint)
    if not base_ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found at: {base_ckpt_path}")

    print(f"Loading base checkpoint: {base_ckpt_path} on {device}...")
    ckpt_state = torch.load(base_ckpt_path, map_location=device, weights_only=False)

    cfg_dict = ckpt_state.get("model_config", {})
    arch = "50m" if cfg_dict.get("emb_dim", 384) == 512 or ckpt_state.get("arch") == "50m" else "16m"
    config = ModelConfig(
        vocab_size=ckpt_state.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 512),
        emb_dim=cfg_dict.get("emb_dim", 512 if arch == "50m" else 384),
        n_heads=cfg_dict.get("n_heads", 8 if arch == "50m" else 6),
        n_layers=cfg_dict.get("n_layers", 12 if arch == "50m" else 6),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 1536 if arch == "50m" else 1024),
        dropout=cfg_dict.get("dropout", 0.1),
        qkv_bias=cfg_dict.get("qkv_bias", False),
        rope_theta=cfg_dict.get("rope_theta", 10000.0),
    )

    model = ModernGPT(config).to(device)
    model.load_state_dict(ckpt_state["model"])
    print(f"✓ Loaded {arch.upper()} model ({model.parameter_count:,} parameters).")

    # Datasets
    data_dir = Path(args.data_dir)
    train_file = Path(args.train_file) if args.train_file else data_dir / "train.jsonl"
    val_file = Path(args.val_file) if args.val_file else data_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Train dataset file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset file not found: {val_file}")

    train_dataset = SFTDataset(train_file, tokenizer, context_length=config.context_length)
    val_dataset = SFTDataset(val_file, tokenizer, context_length=config.context_length)
    print(f"✓ Loaded training set ({len(train_dataset):,} records from {train_file})")
    print(f"✓ Loaded validation set ({len(val_dataset):,} records from {val_file})")

    # Dry-run execution
    if args.dry_run:
        run_sft_dry_run(model, train_dataset, val_dataset, tokenizer, args, device, autocast_ctx=autocast_ctx)
        return

    # SFT Training execution
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory {output_dir} exists and is not empty. Use --overwrite or --resume to proceed.")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tokenizer.eos_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=tokenizer.eos_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = int(steps_per_epoch * args.epochs)

    # Save effective config
    effective_config = {
        "base_checkpoint": str(base_ckpt_path),
        "tokenizer_dir": str(args.tokenizer_dir),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": effective_batch,
        "precision": args.precision,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "total_steps": total_steps,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "model_config": asdict(config),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=2)

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    print("\n" + "=" * 70)
    print("STARTING MODERNGPT-S SUPERVISED FINE-TUNING")
    print("=" * 70)
    print(f"Total training examples:  {len(train_dataset):,}")
    print(f"Total validation examples:{len(val_dataset):,}")
    print(f"Batch size:               {args.batch_size} (Grad Accum: {args.grad_accum}, Effective: {effective_batch})")
    print(f"Precision:                {args.precision}")
    print(f"Epochs:                   {args.epochs} ({total_steps:,} total optimizer steps)")
    print(f"Learning rate:            {args.lr:.2e} -> {args.min_lr:.2e} (Warmup: {args.warmup_steps} steps)")
    print(f"Output directory:         {output_dir}")
    print("=" * 70)

    # Initial validation before fine-tuning
    print("Evaluating initial base model on SFT validation set...")
    init_val_loss, init_val_ppl = evaluate_sft_loss(model, val_loader, device, autocast_ctx=autocast_ctx)
    print(f"Initial Base Val Loss: {init_val_loss:.4f} | Perplexity: {init_val_ppl:.2f}\n")

    best_val_loss = float("inf")
    step = 0
    tokens_seen = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    train_history = []
    val_history = [
        {"step": 0, "val_loss": round(init_val_loss, 4), "perplexity": round(init_val_ppl, 2), "is_best": False}
    ]

    # Resume handling
    if args.resume and (output_dir / "latest.pt").exists():
        print(f"Resuming from {output_dir / 'latest.pt'}...")
        res_state = torch.load(output_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(res_state["model"])
        if "optimizer" in res_state:
            optimizer.load_state_dict(res_state["optimizer"])
        step = res_state.get("step", 0)
        tokens_seen = res_state.get("tokens_seen", 0)
        best_val_loss = res_state.get("manifest_meta", {}).get("best_val_loss", float("inf"))
        print(f"Resumed at step {step}/{total_steps}, best_val_loss {best_val_loss:.4f}")

    for epoch in range(math.ceil(args.epochs)):
        model.train()
        num_batches = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if step >= total_steps:
                break

            # Exact gradient accumulation window calculation
            accum_start = (batch_idx // args.grad_accum) * args.grad_accum
            accum_end = min(accum_start + args.grad_accum, num_batches)
            window_size = accum_end - accum_start
            is_step_batch = (batch_idx + 1 == accum_end) or (step == total_steps - 1)

            inputs = inputs.to(device)
            targets = targets.to(device)

            loss = compute_sft_loss(model, inputs, targets, autocast_ctx=autocast_ctx)
            (loss / window_size).backward()

            shift_targets = targets[..., 1:].contiguous()
            active_tokens = (shift_targets != -100).sum().item()
            if active_tokens == 0:
                raise RuntimeError(f"Aborting training: batch {batch_idx} in epoch {epoch} has 0 supervised response tokens!")
            tokens_seen += active_tokens

            if is_step_batch:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                current_lr = get_lr(step)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                train_history.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 4),
                    "lr": current_lr,
                    "grad_norm": round(float(grad_norm), 4),
                })

                if step > 0 and step % args.eval_interval == 0:
                    val_loss, val_ppl = evaluate_sft_loss(model, val_loader, device, autocast_ctx=autocast_ctx)
                    is_best = val_loss < best_val_loss
                    val_history.append({
                        "step": step,
                        "val_loss": round(val_loss, 4),
                        "perplexity": round(val_ppl, 2),
                        "is_best": is_best,
                    })
                    if is_best:
                        best_val_loss = val_loss
                        best_ckpt_path = output_dir / "best_val.pt"
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "tokens_seen": tokens_seen,
                            "vocab_size": config.vocab_size,
                            "tokenizer": tokenizer.name,
                            "tokenizer_fingerprint": tokenizer.fingerprint,
                            "model_config": asdict(config),
                            "manifest_meta": {
                                "val_loss": val_loss,
                                "perplexity": val_ppl,
                                "best_step": step,
                                "tokens_seen": tokens_seen,
                                "base_checkpoint": str(base_ckpt_path),
                            },
                        }, best_ckpt_path)

                    print(
                        f"Step {step:4d}/{total_steps} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} (Ppl: {val_ppl:.2f}) | "
                        f"LR: {current_lr:.2e} | "
                        f"GradNorm: {float(grad_norm):.2f} | "
                        f"{'★ BEST' if is_best else ''}"
                    )
                    model.train()
                elif step % 10 == 0:
                    print(
                        f"Step {step:4d}/{total_steps} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"GradNorm: {float(grad_norm):.2f}"
                    )

                if step > 0 and step % args.save_interval == 0:
                    step_ckpt_path = output_dir / f"step_{step}.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "vocab_size": config.vocab_size,
                        "model_config": asdict(config),
                    }, step_ckpt_path)

                step += 1

        if step >= total_steps:
            break

    # Final validation pass
    final_val_loss, final_val_ppl = evaluate_sft_loss(model, val_loader, device, autocast_ctx=autocast_ctx)
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_ckpt_path = output_dir / "best_val.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "tokens_seen": tokens_seen,
            "vocab_size": config.vocab_size,
            "tokenizer": tokenizer.name,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "model_config": asdict(config),
            "manifest_meta": {
                "val_loss": final_val_loss,
                "perplexity": final_val_ppl,
                "best_step": step,
                "tokens_seen": tokens_seen,
                "base_checkpoint": str(base_ckpt_path),
            },
        }, best_ckpt_path)
        print(f"★ Updated best_val.pt at terminal step {step} with val loss {final_val_loss:.4f}")

    latest_ckpt_path = output_dir / "latest.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "vocab_size": config.vocab_size,
        "tokenizer": tokenizer.name,
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "model_config": asdict(config),
        "manifest_meta": {
            "final_val_loss": final_val_loss,
            "final_val_ppl": final_val_ppl,
            "best_val_loss": best_val_loss,
            "total_steps": step,
            "base_checkpoint": str(base_ckpt_path),
        },
    }, latest_ckpt_path)

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "base_checkpoint": str(base_ckpt_path),
            "total_steps": step,
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "final_val_ppl": final_val_ppl,
            "train_history": train_history,
            "val_history": val_history,
        }, f, indent=2)
    print(f"  Training Log:    {log_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"SFT TRAINING COMPLETE in {elapsed:.2f}s")
    print(f"  Final Val Loss:  {final_val_loss:.4f} (Perplexity: {final_val_ppl:.2f})")
    print(f"  Best Val Loss:   {best_val_loss:.4f}")
    print(f"  Checkpoints:     {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
