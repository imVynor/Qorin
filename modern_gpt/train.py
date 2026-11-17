import argparse
import json
import math
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.modern_gpt_s import MODEL, TRAINING
from data.dataset import PackedTokenDataset
from data.prepare import read_split
from data.sharding import ShardedPackedDataset
from data.tokenizer import get_tokenizer
from evaluate import validation_loss
from model.config import ModelConfig
from model.gpt import ModernGPT


def compute_learning_rate(
    step: int,
    schedule: str = "cosine",
    warmup_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    base_lr: Optional[float] = None,
    min_lr: Optional[float] = None,
) -> float:
    """Compute learning rate supporting cosine, linear, and constant schedules."""
    if schedule not in ("cosine", "linear", "constant"):
        raise ValueError(f"Unknown schedule: {schedule}. Choose from 'cosine', 'linear', 'constant'.")

    warmup = warmup_steps if warmup_steps is not None else TRAINING.warmup_steps
    decay = decay_steps if decay_steps is not None else TRAINING.decay_steps
    lr0 = base_lr if base_lr is not None else TRAINING.learning_rate
    lr_min = min_lr if min_lr is not None else TRAINING.min_learning_rate

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if lr0 <= 0:
        raise ValueError(f"base_lr must be positive, got {lr0}")
    if lr_min < 0:
        raise ValueError(f"min_lr must be non-negative, got {lr_min}")
    if lr_min > lr0:
        raise ValueError(f"min_lr ({lr_min}) cannot exceed base_lr ({lr0})")
    if warmup < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup}")
    if decay < 0:
        raise ValueError(f"decay_steps must be non-negative, got {decay}")

    # Linear warmup phase
    if warmup > 0 and step < warmup:
        return lr0 * (step + 1) / warmup

    if schedule == "constant":
        return lr0

    # Decay phase progress [0.0, 1.0]
    if decay <= warmup:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (step - warmup) / (decay - warmup)))

    if schedule == "cosine":
        return lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress))
    elif schedule == "linear":
        return lr0 - progress * (lr0 - lr_min)
    return lr_min


def learning_rate(step, warmup_steps=None, decay_steps=None, lr=None, min_lr=None):
    """Backward-compatible wrapper for cosine learning rate."""
    return compute_learning_rate(
        step,
        schedule="cosine",
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        base_lr=lr,
        min_lr=min_lr,
    )


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


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    tokens_seen,
    tokenizer_name="gpt2",
    tokenizer_fingerprint=None,
    manifest_meta=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "vocab_size": model.config.vocab_size,
        "tokenizer": tokenizer_name,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "model_config": asdict(model.config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _serialize_numpy_rng(np.random.get_state()),
        "manifest_meta": manifest_meta,
    }
    torch.save(state, path)


def normalize_tokenizer_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).replace("\\", "/")
    if "16k_v2" in s:
        return "bpe_16k_v2"
    if "16k" in s:
        return "bpe_16k"
    if "gpt2" in s:
        return "gpt2"
    return name


def load_checkpoint(path, model, optimizer, device, expected_tokenizer="gpt2", expected_fingerprint=None):
    state = torch.load(path, map_location=device, weights_only=False)

    # 1. Strict vocab_size check
    if "vocab_size" in state and state["vocab_size"] != model.config.vocab_size:
        raise ValueError(
            f"Checkpoint vocabulary size ({state['vocab_size']}) does not match "
            f"model vocabulary size ({model.config.vocab_size})."
        )

    # 2. Strict tokenizer identity check
    if "tokenizer" in state and expected_tokenizer is not None:
        if normalize_tokenizer_name(state["tokenizer"]) != normalize_tokenizer_name(expected_tokenizer):
            raise ValueError(
                f"Checkpoint tokenizer '{state['tokenizer']}' does not match "
                f"expected tokenizer '{expected_tokenizer}'."
            )

    # 3. Strict tokenizer fingerprint check
    if "tokenizer_fingerprint" in state and expected_fingerprint is not None:
        ckpt_fp = state.get("tokenizer_fingerprint")
        if ckpt_fp and ckpt_fp != expected_fingerprint:
            raise ValueError(
                f"Checkpoint tokenizer fingerprint '{ckpt_fp}' does not match "
                f"active tokenizer fingerprint '{expected_fingerprint}'. Checkpoints from different tokenizer revisions are incompatible."
            )

    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    # 4. Restore RNG states if available
    if "torch_rng_state" in state and state["torch_rng_state"] is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "cuda_rng_state" in state and state["cuda_rng_state"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng_state"]])
        except Exception:
            pass
    if "python_rng_state" in state and state["python_rng_state"] is not None:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state and state["numpy_rng_state"] is not None:
        rng = _deserialize_numpy_rng(state["numpy_rng_state"])
        if rng is not None:
            np.random.set_state(rng)

    return state["step"], state["tokens_seen"]


def dry_run_check(
    config: ModelConfig,
    model: ModernGPT,
    train_data,
    val_data,
    tokenizer,
    tok_fingerprint: Optional[str],
    args,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute pretraining verification without performing training or checkpoint writes."""
    print("\n" + "=" * 70)
    print("MODERNGPT-S: PRETRAINING DRY-RUN INTEGRITY VERIFICATION")
    print("=" * 70)

    # 1. Shard / Dataset integrity
    if hasattr(train_data, "tokens"):
        min_tok_id = int(np.min(train_data.tokens))
        max_tok_id = int(np.max(train_data.tokens))
        eos_count_train = int(np.sum(train_data.tokens == 0))
        eos_count_val = int(np.sum(val_data.tokens == 0))
        assert min_tok_id >= 0 and max_tok_id < config.vocab_size, (
            f"Token IDs out of range: [{min_tok_id}, {max_tok_id}] vs vocab {config.vocab_size}"
        )
        assert eos_count_train > 0, "No EOS tokens found in train shard!"
    else:
        eos_count_train = "N/A"
        eos_count_val = "N/A"

    unique_train_tokens = len(train_data) * config.context_length
    unique_val_tokens = len(val_data) * config.context_length

    # 2. Target shift verification
    sample_inp, sample_tgt = train_data[0]
    assert sample_inp.shape == (config.context_length,), f"Unexpected input shape: {sample_inp.shape}"
    assert sample_tgt.shape == (config.context_length,), f"Unexpected target shape: {sample_tgt.shape}"
    assert torch.equal(sample_inp[1:], sample_tgt[:-1]), "Target shift mismatch: targets[i] != inputs[i+1]"

    # 3. Model & Architecture verification
    param_count = model.parameter_count
    arch = getattr(args, "arch", "16m")
    if arch == "50m":
        expected_params = 49105408
    elif arch == "50m_gqa4":
        expected_params = 45959680
    elif arch == "50m_gqa2":
        expected_params = 44386816
    elif arch == "50m_mqa":
        expected_params = 43600384
    elif arch == "50m_deep_thin":
        expected_params = 48936384
    elif arch == "50m_wide":
        expected_params = 49905280
    else:
        expected_params = 16768128 if config.vocab_size == 16000 else 29922816
    assert param_count == expected_params, f"Parameter count mismatch: {param_count} vs expected {expected_params}"
    assert model.lm_head.weight is model.token_embedding.weight, "LM head weight is not tied to token embedding!"

    # 4. Tokenizer verification
    assert tokenizer.vocab_size == config.vocab_size, (
        f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match model vocab ({config.vocab_size})"
    )
    if "16k" in args.tokenizer:
        assert getattr(tokenizer, "eos_token_id", None) == 0, "EOS token ID must be 0"
        assert getattr(tokenizer, "pad_token_id", None) is None, "pad_token_id must be None"
        assert getattr(tokenizer, "unk_token_id", None) is None, "unk_token_id must be None"

    # 5. Checkpoint compatibility check (if resume specified)
    if args.resume:
        temp_model = ModernGPT(config).to(device)
        load_checkpoint(
            args.resume,
            temp_model,
            optimizer=None,
            device=device,
            expected_tokenizer=args.tokenizer,
            expected_fingerprint=tok_fingerprint,
        )
        print(f"✓ Checkpoint compatibility verified for resume: '{args.resume}'")

    # 6. Budget Accounting & Validation
    grad_accum = getattr(args, "grad_accum", TRAINING.gradient_accumulation_steps)
    tokens_per_step = args.batch_size * config.context_length * grad_accum
    budget = args.max_tokens if args.max_tokens > 0 else unique_train_tokens
    epoch_multiplier = budget / max(1, unique_train_tokens)
    planned_steps = budget // max(1, tokens_per_step)

    print(f"✓ Active Tokenizer:         {args.tokenizer} (Vocab: {tokenizer.vocab_size}, Fingerprint: {tok_fingerprint})")
    print(f"✓ Exact Model Parameters:   {param_count:,} (Architecture: {config.n_layers}L, {config.n_heads}H, {config.emb_dim}D, Tied Head)")
    print(f"✓ Dataset Sequence Format:  Context length = {config.context_length}, Target shift verified")
    print(f"✓ Shard Unique Tokens:      Train = {unique_train_tokens:,} ({len(train_data):,} seqs), Val = {unique_val_tokens:,} ({len(val_data):,} seqs)")
    print(f"✓ Shard EOS Boundaries:     Train = {eos_count_train}, Val = {eos_count_val}")
    print(f"✓ Target Budget:            {budget:,} processed tokens")
    print(f"✓ Epoch Multiplier:         {epoch_multiplier:.2f}x passes over training data")
    print(f"✓ Planned Optimizer Steps:  {planned_steps:,} steps ({tokens_per_step:,} tokens/step)")

    if args.strict_budget and budget > unique_train_tokens:
        raise ValueError(
            f"Strict budget violation: requested {budget:,} processed tokens exceeds unique dataset tokens ({unique_train_tokens:,}). "
            f"Multi-epoch training is blocked under --strict-budget."
        )

    if args.max_epochs is not None and epoch_multiplier > args.max_epochs:
        raise ValueError(
            f"Epoch cap exceeded: requested budget requires {epoch_multiplier:.2f} epochs, "
            f"which exceeds --max-epochs={args.max_epochs:.2f}. "
            f"To prevent rote memorization and overfitting, expand your dataset or adjust the token budget."
        )

    if epoch_multiplier > 2.0:
        print(f"⚠️  WARNING: Target budget will cycle through the training shard {epoch_multiplier:.2f} times.")
        print(f"    In LLM pretraining, training beyond 1-2 epochs on fixed data causes rote memorization.")
        print(f"    For optimal generalization, expand unique tokens to match the budget.")
    else:
        print("✓ Epoch multiplier is within optimal single/low-epoch regime (<= 2.0x).")

    print("=" * 70)
    print("DRY-RUN STATUS: VERIFIED & READY")
    print("No training steps were executed and no checkpoints were modified.")
    print("=" * 70)

    return {
        "status": "VERIFIED",
        "tokenizer": args.tokenizer,
        "tokenizer_fingerprint": tok_fingerprint,
        "vocab_size": config.vocab_size,
        "model_parameters": param_count,
        "unique_train_tokens": unique_train_tokens,
        "unique_val_tokens": unique_val_tokens,
        "budget_processed_tokens": budget,
        "epoch_multiplier": round(epoch_multiplier, 3),
        "planned_steps": planned_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", default=TRAINING.resume)
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name ('gpt2', 'bpe_16k', 'bpe_16k_v2') or path")
    parser.add_argument(
        "--arch",
        default="16m",
        choices=["16m", "50m", "50m_wide", "50m_gqa4", "50m_gqa2", "50m_mqa", "50m_deep_thin"],
        help="Model architecture preset",
    )
    parser.add_argument("--shards-dir", default=None, help="Path to directory containing train.bin and val.bin")
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--grad-accum", type=int, default=TRAINING.gradient_accumulation_steps, help="Gradient accumulation steps")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"], help="Precision mode")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=TRAINING.max_tokens)
    parser.add_argument("--eval-interval", type=int, default=TRAINING.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=20, help="Number of validation batches per periodic evaluation")
    parser.add_argument("--save-interval", type=int, default=500, help="Interval for saving periodic checkpoints")
    parser.add_argument("--checkpoint-dir", default=TRAINING.checkpoint_dir)
    parser.add_argument("--experiment-id", type=str, default=None, help="Experiment ID for permanent tracking")
    parser.add_argument("--parent-experiment-id", type=str, default=None, help="Parent experiment ID")
    parser.add_argument("--dry-run", action="store_true", help="Validate corpus, shards, tokenizer, and model without training")
    parser.add_argument("--max-epochs", type=float, default=None, help="Maximum allowed epoch multiplier (budget / unique tokens)")
    parser.add_argument("--strict-budget", action="store_true", help="Disallow multi-epoch cycling (requires budget <= unique tokens)")
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant", "linear"], help="Learning rate schedule type")
    parser.add_argument("--lr", type=float, default=TRAINING.learning_rate, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=TRAINING.min_learning_rate, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=TRAINING.warmup_steps, help="Number of linear warmup steps")
    parser.add_argument("--decay-steps", type=int, default=None, help="Steps over which to decay learning rate (defaults to total planned steps)")
    parser.add_argument("--weight-decay", type=float, default=TRAINING.weight_decay, help="Weight decay")
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="AdamW beta2")
    args = parser.parse_args()

    torch.manual_seed(TRAINING.seed)
    random.seed(TRAINING.seed)
    np.random.seed(TRAINING.seed)

    # Resolve tokenizer
    tok_name = args.tokenizer
    tok_path = None
    if Path(tok_name).is_dir() or Path(tok_name).is_file():
        tok_path = Path(tok_name)
        if "16k_v2" in str(tok_path):
            tok_name = "bpe_16k_v2"
        elif "16k" in str(tok_path):
            tok_name = "bpe_16k"
    tokenizer = get_tokenizer(tok_name, path=tok_path)
    tok_fingerprint = getattr(tokenizer, "fingerprint", None)

    # Select configuration
    vocab_size = 16000 if ("16k" in args.tokenizer or "16k" in tok_name) else 50257
    if args.arch == "50m":
        config = ModelConfig.get_50m_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    elif args.arch == "50m_gqa4":
        config = ModelConfig.get_50m_gqa4_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    elif args.arch == "50m_gqa2":
        config = ModelConfig.get_50m_gqa2_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    elif args.arch == "50m_mqa":
        config = ModelConfig.get_50m_mqa_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    elif args.arch == "50m_deep_thin":
        config = ModelConfig.get_50m_deep_thin_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    elif args.arch == "50m_wide":
        config = ModelConfig.get_50m_wide_config(vocab_size=vocab_size, context_length=MODEL.context_length)
    else:
        config = ModelConfig.get_16m_config(vocab_size=vocab_size, context_length=MODEL.context_length)

    # Data loading: Sharded vs raw text
    manifest_meta = None
    if args.shards_dir:
        shards_path = Path(args.shards_dir)
        train_bin = shards_path / "train.bin"
        val_bin = shards_path / "val.bin"
        train_manifest = shards_path / "train.json"
        if train_manifest.exists():
            with train_manifest.open("r", encoding="utf-8") as f:
                manifest_meta = json.load(f)

        train_data = ShardedPackedDataset(train_bin, context_length=config.context_length)
        val_data = ShardedPackedDataset(val_bin, context_length=config.context_length)
    else:
        train_text, val_text = read_split(TRAINING.data_path, TRAINING.train_ratio)
        train_data = PackedTokenDataset(
            train_text, config.context_length, TRAINING.stride, tokenizer
        )
        val_data = PackedTokenDataset(
            val_text, config.context_length, TRAINING.stride, tokenizer
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernGPT(config).to(device)

    # Dry-run execution
    if args.dry_run:
        dry_run_check(config, model, train_data, val_data, tokenizer, tok_fingerprint, args, device)
        return

    output_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_data, args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False)

    if TRAINING.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    step = 0
    tokens_seen = 0
    if args.resume:
        step, tokens_seen = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            expected_tokenizer=tok_name,
            expected_fingerprint=tok_fingerprint,
        )

    started = time.perf_counter()
    max_steps = 2 if args.smoke_test else args.max_steps
    accumulation_steps = args.grad_accum
    optimizer.zero_grad(set_to_none=True)

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if args.precision == "fp16" else None)
    use_autocast = (device.type == "cuda" and autocast_dtype is not None)
    training_history = []

    max_tokens_budget = args.max_tokens
    tokens_per_step = args.batch_size * config.context_length * accumulation_steps
    unique_train_tokens = len(train_data) * config.context_length
    unique_val_tokens = len(val_data) * config.context_length
    epoch_multiplier = max_tokens_budget / max(1, unique_train_tokens) if max_tokens_budget > 0 else 1.0

    # Auto-configure decay steps matching planned tokens if not explicitly specified
    if args.decay_steps is not None:
        decay_steps = args.decay_steps
    elif max_tokens_budget > 0 and tokens_per_step > 0:
        decay_steps = math.ceil(max_tokens_budget / tokens_per_step)
    else:
        decay_steps = TRAINING.decay_steps

    tracker = None
    if args.experiment_id:
        try:
            from experiments.tracker import ExperimentTracker
            tracker = ExperimentTracker(
                experiment_id=args.experiment_id,
                experiment_name=args.experiment_id,
                category="pretrain",
                architecture=args.arch,
                training_config={
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "learning_rate": args.lr,
                    "min_learning_rate": args.min_lr,
                    "warmup_steps": args.warmup_steps,
                    "decay_steps": decay_steps,
                    "lr_schedule": args.lr_schedule,
                    "precision": args.precision,
                    "max_tokens": max_tokens_budget,
                    "eval_interval": args.eval_interval,
                    "save_interval": args.save_interval,
                    "weight_decay": args.weight_decay,
                    "adam_beta1": args.adam_beta1,
                    "adam_beta2": args.adam_beta2,
                },
                model_config=asdict(config),
                parameter_count=model.parameter_count,
                dataset_path=str(args.shards_dir) if args.shards_dir else TRAINING.data_path,
                dataset_version="v2_250m" if (args.shards_dir and "250m" in str(args.shards_dir)) else "v1",
                tokenizer_name=tok_name,
                tokenizer_fingerprint=tok_fingerprint,
                token_budget=max_tokens_budget,
                parent_experiment_id=args.parent_experiment_id or "None",
                launch_command=" ".join(sys.argv),
            )
            print(f"✓ Experiment tracking active: registered run '{args.experiment_id}' in experiments/registry")
        except Exception as e:
            print(f"⚠️ Warning: Experiment tracker failed to initialize: {e}")
            tracker = None

    print("=" * 70)
    print("MODERNGPT-S PRETRAINING INITIALIZATION")
    print("=" * 70)
    print(f"Tokenizer:                {tok_name} (Fingerprint: {tok_fingerprint})")
    print(f"Architecture:             {args.arch} ({model.parameter_count:,} parameters)")
    print(f"Batch Size:               {args.batch_size} (Tokens/step: {tokens_per_step:,}, Grad Accum: {accumulation_steps})")
    print(f"Context Length:           {config.context_length}")
    print(f"LR Schedule:              {args.lr_schedule}")
    print(f"Peak LR / Min LR:         {args.lr:.2e} / {args.min_lr:.2e} (Warmup: {args.warmup_steps}, Decay: {decay_steps:,} steps)")
    print(f"Eval Interval / Batches:  Every {args.eval_interval} steps ({args.eval_batches} batches)")
    print(f"Save Interval:            Every {args.save_interval} steps")
    print("-" * 70)
    print("Dataset Accounting:")
    print(f"  Unique Training Tokens:   {unique_train_tokens:,} ({len(train_data):,} packed sequences)")
    print(f"  Unique Validation Tokens: {unique_val_tokens:,} ({len(val_data):,} packed sequences)")
    print("-" * 70)
    print("Budget Accounting:")
    print(f"  Target Processed Tokens:  {max_tokens_budget:,}")
    print(f"  Planned Epochs:           {epoch_multiplier:.2f}x passes over shard")
    print("=" * 70)

    # Budget validation
    if args.strict_budget and max_tokens_budget > unique_train_tokens:
        raise ValueError(
            f"Strict budget violation: requested {max_tokens_budget:,} processed tokens "
            f"exceeds unique training tokens ({unique_train_tokens:,}). "
            f"Multi-epoch training is blocked under --strict-budget."
        )

    if args.max_epochs is not None and epoch_multiplier > args.max_epochs:
        raise ValueError(
            f"Epoch cap exceeded: requested budget requires {epoch_multiplier:.2f} epochs, "
            f"which exceeds --max-epochs={args.max_epochs:.2f}. "
            f"To prevent rote memorization and overfitting, expand your dataset or adjust the token budget."
        )

    if epoch_multiplier > 2.0:
        print(f"⚠️  WARNING: Target budget will cycle {epoch_multiplier:.2f} times across {unique_train_tokens:,} unique tokens.")
        print("    Risk of rote memorization beyond ~1-2 epochs.")

    tokens_per_epoch = len(train_loader) * args.batch_size * config.context_length
    if max_tokens_budget > 0 and tokens_per_epoch > 0:
        needed_epochs = math.ceil((max_tokens_budget - tokens_seen) / tokens_per_epoch)
        total_epochs = max(TRAINING.num_epochs, needed_epochs)
    else:
        total_epochs = TRAINING.num_epochs

    best_val_loss = float("inf")
    best_step = 0
    best_ckpt_path = output_dir / "best_val.pt"
    ckpt_path = output_dir / "latest.pt"

    if args.resume and best_ckpt_path.exists():
        try:
            best_state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            if "manifest_meta" in best_state and "val_loss" in best_state["manifest_meta"]:
                best_val_loss = float(best_state["manifest_meta"]["val_loss"])
                best_step = best_state.get("step", best_state["manifest_meta"].get("best_step", 0))
                print(f"✓ Resumed best validation tracking: best_val_loss = {best_val_loss:.4f} at step {best_step}")
        except Exception:
            pass

    micro_step = 0
    try:
        for epoch in range(total_epochs):
            for inputs, targets in train_loader:
                if max_steps is not None and step >= max_steps:
                    break

                if max_tokens_budget > 0:
                    remaining_tokens = max_tokens_budget - tokens_seen
                    if remaining_tokens <= 0:
                        break
                    remaining_seqs = remaining_tokens // config.context_length
                    if remaining_seqs == 0:
                        break
                    if remaining_seqs < inputs.size(0):
                        inputs = inputs[:remaining_seqs]
                        targets = targets[:remaining_seqs]

                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        logits, loss = model(inputs.to(device), targets.to(device))
                else:
                    logits, loss = model(inputs.to(device), targets.to(device))

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                (loss / accumulation_steps).backward()
                tokens_seen += inputs.numel()
                micro_step += 1

                should_step = (
                    micro_step % accumulation_steps == 0
                    or (max_tokens_budget > 0 and (tokens_seen >= max_tokens_budget or (max_tokens_budget - tokens_seen) < config.context_length))
                )
                if should_step:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), TRAINING.max_grad_norm
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(f"Non-finite gradient at step {step}")

                    curr_lr = compute_learning_rate(
                        step=step,
                        schedule=args.lr_schedule,
                        warmup_steps=args.warmup_steps,
                        decay_steps=decay_steps,
                        base_lr=args.lr,
                        min_lr=args.min_lr,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = curr_lr

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    current_epoch_val = tokens_seen / max(1, unique_train_tokens)
                    tokens_per_sec = round(tokens_seen / max(1e-9, time.perf_counter() - started), 1)

                    if step % args.eval_interval == 0:
                        val_loss, perplexity = validation_loss(
                            model, val_loader, device, args.eval_batches, autocast_dtype=autocast_dtype
                        )
                        is_best = val_loss < best_val_loss
                        if is_best:
                            best_val_loss = val_loss
                            best_step = step
                            save_checkpoint(
                                best_ckpt_path,
                                model,
                                optimizer,
                                step,
                                tokens_seen,
                                tokenizer_name=tok_name,
                                tokenizer_fingerprint=tok_fingerprint,
                                manifest_meta={
                                    "val_loss": val_loss,
                                    "perplexity": perplexity,
                                    "best_step": step,
                                    "tokens_seen": tokens_seen,
                                    "eval_batches": args.eval_batches,
                                    "lr_schedule": args.lr_schedule,
                                },
                            )
                            if tracker is not None:
                                tracker.log_checkpoint(
                                    checkpoint_path=best_ckpt_path,
                                    step=step,
                                    epoch=current_epoch_val,
                                    val_loss=val_loss,
                                    checkpoint_type="best_val",
                                )

                        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                        log_entry = {
                            "step": step, "loss": round(loss.item(), 4), "val_loss": round(val_loss, 4),
                            "perplexity": round(perplexity, 2), "tokens_seen": tokens_seen,
                            "lr": curr_lr,
                            "grad_norm": round(float(grad_norm), 4),
                            "parameters": model.parameter_count,
                            "tokens_per_second": tokens_per_sec,
                            "is_best": is_best,
                            "best_val_loss": round(best_val_loss, 4),
                        }
                        training_history.append(log_entry)
                        print(json.dumps(log_entry))

                        if tracker is not None:
                            tracker.log_step(
                                step=step,
                                epoch=current_epoch_val,
                                train_loss=loss.item(),
                                lr=curr_lr,
                                val_loss=val_loss,
                                grad_norm=float(grad_norm),
                                tokens_processed=tokens_seen,
                                tokens_per_sec=tokens_per_sec,
                                vram_allocated_mb=peak_vram,
                            )

                    # Periodic checkpointing
                    if args.save_interval and step > 0 and step % args.save_interval == 0:
                        periodic_ckpt = output_dir / f"step_{step}.pt"
                        save_checkpoint(
                            periodic_ckpt,
                            model,
                            optimizer,
                            step,
                            tokens_seen,
                            tokenizer_name=tok_name,
                            tokenizer_fingerprint=tok_fingerprint,
                            manifest_meta={"step": step, "tokens_seen": tokens_seen},
                        )
                        save_checkpoint(
                            ckpt_path,
                            model,
                            optimizer,
                            step,
                            tokens_seen,
                            tokenizer_name=tok_name,
                            tokenizer_fingerprint=tok_fingerprint,
                            manifest_meta={"step": step, "tokens_seen": tokens_seen},
                        )
                        if tracker is not None:
                            tracker.log_checkpoint(
                                checkpoint_path=periodic_ckpt,
                                step=step,
                                epoch=current_epoch_val,
                                val_loss=None,
                                checkpoint_type="periodic",
                            )

                    step += 1

                if max_tokens_budget > 0 and (tokens_seen >= max_tokens_budget or (max_tokens_budget - tokens_seen) < config.context_length):
                    break
            if max_steps is not None and step >= max_steps:
                break
            if max_tokens_budget > 0 and (tokens_seen >= max_tokens_budget or (max_tokens_budget - tokens_seen) < config.context_length):
                break

    except Exception as e:
        if tracker is not None:
            tracker.fail(failure_reason=f"Training encountered exception: {e}")
        raise e

    # Full validation audit across all validation sequences
    print("=" * 70)
    print("POST-PRETRAINING FULL VALIDATION AUDIT")
    print("=" * 70)
    audit_batches = 2 if args.smoke_test else None
    print(f"Auditing latest model (step {step}) across full validation set ({len(val_data):,} seqs)...")
    full_latest_loss, full_latest_ppl = validation_loss(model, val_loader, device, max_batches=audit_batches, autocast_dtype=autocast_dtype)
    print(f"✓ Latest Checkpoint Full Validation: Loss = {full_latest_loss:.4f}, Perplexity = {full_latest_ppl:.2f}")

    latest_meta = {
        "final_step": step,
        "tokens_seen": tokens_seen,
        "full_val_loss": full_latest_loss,
        "full_val_perplexity": full_latest_ppl,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "lr_schedule": args.lr_schedule,
    }

    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        step,
        tokens_seen,
        tokenizer_name=tok_name,
        tokenizer_fingerprint=tok_fingerprint,
        manifest_meta=latest_meta,
    )
    print(f"Saved final checkpoint with full validation audit to {ckpt_path}.")

    full_best_loss, full_best_ppl = None, None
    if best_ckpt_path.exists():
        best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"])
        print(f"Auditing best model (step {best_step}) across full validation set ({len(val_data):,} seqs)...")
        full_best_loss, full_best_ppl = validation_loss(model, val_loader, device, max_batches=audit_batches, autocast_dtype=autocast_dtype)
        print(f"✓ Best Checkpoint Full Validation:   Loss = {full_best_loss:.4f}, Perplexity = {full_best_ppl:.2f}")
        if best_state.get("manifest_meta") is None:
            best_state["manifest_meta"] = {}
        best_state["manifest_meta"]["full_val_loss"] = full_best_loss
        best_state["manifest_meta"]["full_val_perplexity"] = full_best_ppl
        torch.save(best_state, best_ckpt_path)
        print(f"Updated {best_ckpt_path} with full validation audit metrics.")

    log_file = output_dir / "training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(config),
            "architecture": args.arch,
            "precision": args.precision,
            "tokens_processed": tokens_seen,
            "total_steps": step,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "final_val_loss": full_latest_loss,
            "final_val_perplexity": full_latest_ppl,
            "training_history": training_history,
        }, f, indent=2)
    print(f"Saved complete training log to {log_file}.")

    if tracker is not None:
        tracker.log_checkpoint(
            checkpoint_path=ckpt_path,
            step=step,
            epoch=tokens_seen / max(1, unique_train_tokens),
            val_loss=full_latest_loss,
            checkpoint_type="latest",
        )
        if best_ckpt_path.exists() and full_best_loss is not None:
            tracker.log_checkpoint(
                checkpoint_path=best_ckpt_path,
                step=best_step,
                epoch=tokens_seen / max(1, unique_train_tokens),
                val_loss=full_best_loss,
                checkpoint_type="best_val",
            )
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        final_tok_sec = round(tokens_seen / max(1e-9, time.perf_counter() - started), 1)
        tracker.log_profiling(
            throughput_tokens_sec=final_tok_sec,
            prefill_latency_ms=0.0,
            decode_latency_tok_sec=0.0,
            peak_vram_mb=round(peak_vram, 2),
            kv_cache_memory_mb=0.0,
        )
        tracker.finalize(
            status="completed",
            main_conclusion=(
                f"Pretrained {args.arch} ({model.parameter_count:,} params) on {tokens_seen:,} tokens "
                f"in {round((time.perf_counter() - started) / 3600, 2)}h. "
                f"Best val loss: {best_val_loss:.4f}, Full val loss: {full_latest_loss:.4f} (ppl {full_latest_ppl:.2f})."
            ),
        )


if __name__ == "__main__":
    main()
