"""Tests for pretraining controls: dry-run mode, token budget accounting, and best_val.pt checkpointing."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from model.config import ModelConfig
from model.gpt import ModernGPT
from train import compute_learning_rate, dry_run_check, learning_rate, load_checkpoint, save_checkpoint


@pytest.fixture
def tokenizer_v2():
    tok_dir = Path("artifacts/tokenizer_16k_v2")
    if not tok_dir.exists():
        pytest.skip(f"Tokenizer v2 not found at {tok_dir}")
    return HuggingFaceBPEWrapper.load(tok_dir, name="bpe_16k_v2")


def test_dry_run_verification(tokenizer_v2):
    """Test that dry_run_check runs without errors and validates all pretraining components."""
    shards_dir = Path("artifacts/shards/bpe_16k_v2")
    if not (shards_dir / "train.bin").exists():
        pytest.skip("Shards not found")

    config = ModelConfig(vocab_size=16000, context_length=512)
    device = torch.device("cpu")
    model = ModernGPT(config).to(device)

    train_data = ShardedPackedDataset(shards_dir / "train.bin", context_length=512)
    val_data = ShardedPackedDataset(shards_dir / "val.bin", context_length=512)

    args = SimpleNamespace(
        tokenizer="bpe_16k_v2",
        shards_dir=str(shards_dir),
        batch_size=2,
        max_tokens=2000000,
        resume=None,
        strict_budget=False,
        max_epochs=None,
    )

    result = dry_run_check(
        config=config,
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer_v2,
        tok_fingerprint=tokenizer_v2.fingerprint,
        args=args,
        device=device,
    )

    assert result["status"] == "VERIFIED"
    assert result["model_parameters"] == 16768128
    assert result["vocab_size"] == 16000
    assert result["unique_train_tokens"] > 0
    assert result["unique_val_tokens"] > 0
    assert result["epoch_multiplier"] > 1.0


def test_token_budget_strict_enforcement(tokenizer_v2):
    """Verify that --strict-budget and --max-epochs block excessive dataset recycling."""
    shards_dir = Path("artifacts/shards/bpe_16k_v2")
    if not (shards_dir / "train.bin").exists():
        pytest.skip("Shards not found")

    config = ModelConfig(vocab_size=16000, context_length=512)
    device = torch.device("cpu")
    model = ModernGPT(config).to(device)

    train_data = ShardedPackedDataset(shards_dir / "train.bin", context_length=512)
    val_data = ShardedPackedDataset(shards_dir / "val.bin", context_length=512)

    # 1. Test --strict-budget violation
    args_strict = SimpleNamespace(
        tokenizer="bpe_16k_v2",
        shards_dir=str(shards_dir),
        batch_size=2,
        max_tokens=2000000,  # 2M budget vs ~70K unique tokens
        resume=None,
        strict_budget=True,
        max_epochs=None,
    )
    with pytest.raises(ValueError, match="Strict budget violation"):
        dry_run_check(
            config, model, train_data, val_data, tokenizer_v2, tokenizer_v2.fingerprint, args_strict, device
        )

    # 2. Test --max-epochs violation
    args_max_epoch = SimpleNamespace(
        tokenizer="bpe_16k_v2",
        shards_dir=str(shards_dir),
        batch_size=2,
        max_tokens=2000000,
        resume=None,
        strict_budget=False,
        max_epochs=2.0,  # 2 epochs allowed vs ~28 epochs requested
    )
    with pytest.raises(ValueError, match="Epoch cap exceeded"):
        dry_run_check(
            config, model, train_data, val_data, tokenizer_v2, tokenizer_v2.fingerprint, args_max_epoch, device
        )


def test_best_val_checkpoint_saving(tmp_path, tokenizer_v2):
    """Verify that best_val.pt is saved when validation loss improves while preserving latest.pt."""
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Simulate step 10: initial val loss 15.0 -> saves best_val.pt
    val_loss_1 = 15.0
    save_checkpoint(
        ckpt_dir / "best_val.pt",
        model,
        optimizer,
        step=10,
        tokens_seen=10240,
        tokenizer_name="bpe_16k_v2",
        tokenizer_fingerprint=tokenizer_v2.fingerprint,
        manifest_meta={"val_loss": val_loss_1, "best_step": 10},
    )

    # Simulate step 20: improved val loss 12.0 -> updates best_val.pt
    val_loss_2 = 12.0
    save_checkpoint(
        ckpt_dir / "best_val.pt",
        model,
        optimizer,
        step=20,
        tokens_seen=20480,
        tokenizer_name="bpe_16k_v2",
        tokenizer_fingerprint=tokenizer_v2.fingerprint,
        manifest_meta={"val_loss": val_loss_2, "best_step": 20},
    )

    # Save latest.pt at step 30 (where val loss was 18.0)
    save_checkpoint(
        ckpt_dir / "latest.pt",
        model,
        optimizer,
        step=30,
        tokens_seen=30720,
        tokenizer_name="bpe_16k_v2",
        tokenizer_fingerprint=tokenizer_v2.fingerprint,
        manifest_meta={"val_loss": 18.0, "step": 30},
    )

    assert (ckpt_dir / "best_val.pt").exists()
    assert (ckpt_dir / "latest.pt").exists()

    # Verify best_val.pt has step 20 and val_loss 12.0
    best_state = torch.load(ckpt_dir / "best_val.pt", map_location="cpu", weights_only=False)
    assert best_state["step"] == 20
    assert best_state["manifest_meta"]["val_loss"] == 12.0

    # Verify latest.pt has step 30 and val_loss 18.0
    latest_state = torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert latest_state["step"] == 30
    assert latest_state["manifest_meta"]["val_loss"] == 18.0


def test_5m_shards_and_budget_feasibility(tokenizer_v2):
    """Verify that the 5M shards satisfy token budget and feasibility under --max-epochs 1.2."""
    shards_dir = Path("artifacts/shards/bpe_16k_v2_5m")
    if not (shards_dir / "train.bin").exists():
        pytest.skip("5M shards not found at artifacts/shards/bpe_16k_v2_5m")

    # Load shard manifests
    with (shards_dir / "train.json").open("r", encoding="utf-8") as f:
        train_m = json.load(f)
    with (shards_dir / "val.json").open("r", encoding="utf-8") as f:
        val_m = json.load(f)

    assert train_m["total_tokens"] >= 5_000_000
    assert train_m["tokenizer_name"] == "bpe_16k_v2"
    assert train_m["eos_token_id"] == 0

    # Load corpus manifest
    manifest_path = Path("artifacts/corpus_5m/corpus_manifest.json")
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        c_manifest = json.load(f)

    train_tokens = c_manifest["train_stats"]["measured_tokens"]
    val_tokens = c_manifest["val_stats"]["measured_tokens"]
    assert train_tokens >= 5_000_000
    assert val_tokens > 0

    # Verify source proportions in train split (~80:20)
    tok_props = c_manifest["train_stats"]["source_proportions_by_tokens"]
    assert 0.78 <= tok_props["fineweb-edu"] <= 0.82
    assert 0.18 <= tok_props["cosmopedia-v2"] <= 0.22

    # Verify dry-run check with 5M budget and max_epochs=1.2
    config = ModelConfig(vocab_size=16000, context_length=512)
    device = torch.device("cpu")
    model = ModernGPT(config).to(device)

    train_data = ShardedPackedDataset(shards_dir / "train.bin", context_length=512)
    val_data = ShardedPackedDataset(shards_dir / "val.bin", context_length=512)

    args = SimpleNamespace(
        tokenizer="bpe_16k_v2",
        shards_dir=str(shards_dir),
        batch_size=2,
        max_tokens=5_000_000,
        resume=None,
        strict_budget=False,
        max_epochs=1.2,
    )

    result = dry_run_check(
        config=config,
        model=model,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer_v2,
        tok_fingerprint=tokenizer_v2.fingerprint,
        args=args,
        device=device,
    )

    assert result["status"] == "VERIFIED"
    assert result["epoch_multiplier"] <= 1.0
    assert result["unique_train_tokens"] >= 5_000_000


def test_configurable_lr_scheduler_behavior():
    """Verify cosine, constant, and linear learning rate schedules across warmup and decay."""
    base_lr = 4e-4
    min_lr = 4e-5
    warmup_steps = 100
    decay_steps = 1000

    # 1. Warmup behavior (common across schedules)
    lr_0 = compute_learning_rate(0, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    lr_50 = compute_learning_rate(49, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    lr_100 = compute_learning_rate(99, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    assert lr_0 == pytest.approx(base_lr * 1 / 100)
    assert lr_50 == pytest.approx(base_lr * 50 / 100)
    assert lr_100 == pytest.approx(base_lr)

    # 2. Cosine schedule decay
    lr_mid = compute_learning_rate(550, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    lr_end = compute_learning_rate(1000, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    lr_past = compute_learning_rate(1500, "cosine", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    assert min_lr < lr_mid < base_lr
    assert lr_end == pytest.approx(min_lr)
    assert lr_past == pytest.approx(min_lr)  # Clamps to min_lr past decay

    # 3. Constant schedule
    assert compute_learning_rate(500, "constant", warmup_steps=warmup_steps, base_lr=base_lr) == pytest.approx(base_lr)
    assert compute_learning_rate(2000, "constant", warmup_steps=warmup_steps, base_lr=base_lr) == pytest.approx(base_lr)

    # 4. Linear schedule
    lr_lin_mid = compute_learning_rate(550, "linear", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr)
    expected_lin_mid = base_lr - 0.5 * (base_lr - min_lr)
    assert lr_lin_mid == pytest.approx(expected_lin_mid)
    assert compute_learning_rate(1000, "linear", warmup_steps=warmup_steps, decay_steps=decay_steps, base_lr=base_lr, min_lr=min_lr) == pytest.approx(min_lr)

    # 5. Backward-compatibility wrapper
    assert learning_rate(99, warmup_steps=100, lr=base_lr) == pytest.approx(base_lr)

    # 6. Invalid schedule check
    with pytest.raises(ValueError, match="Unknown schedule"):
        compute_learning_rate(50, schedule="polynomial")


def test_checkpoint_resumption_and_full_validation_meta(tmp_path, tokenizer_v2):
    """Verify saving, loading, and resuming checkpoints with full-validation audit metadata."""
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_file = tmp_path / "latest.pt"
    manifest_meta = {
        "final_step": 500,
        "tokens_seen": 512000,
        "full_val_loss": 5.4321,
        "full_val_perplexity": 228.63,
        "best_step": 450,
        "best_val_loss": 5.3120,
        "lr_schedule": "cosine",
    }

    # Save checkpoint with full validation audit meta
    save_checkpoint(
        ckpt_file,
        model,
        optimizer,
        step=500,
        tokens_seen=512000,
        tokenizer_name="bpe_16k_v2",
        tokenizer_fingerprint=tokenizer_v2.fingerprint,
        manifest_meta=manifest_meta,
    )

    # Resume into fresh model and optimizer
    model_resumed = ModernGPT(config)
    optimizer_resumed = torch.optim.AdamW(model_resumed.parameters(), lr=1e-3)

    device = torch.device("cpu")
    step, tokens_seen = load_checkpoint(
        ckpt_file,
        model_resumed,
        optimizer_resumed,
        device=device,
        expected_tokenizer="bpe_16k_v2",
        expected_fingerprint=tokenizer_v2.fingerprint,
    )

    assert step == 500
    assert tokens_seen == 512000

    # Verify weights match exactly
    for p1, p2 in zip(model.parameters(), model_resumed.parameters()):
        assert torch.equal(p1, p2)

    # Verify full-validation metadata persisted
    saved_state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    assert saved_state["manifest_meta"]["full_val_loss"] == 5.4321
    assert saved_state["manifest_meta"]["full_val_perplexity"] == 228.63


def test_tokenizer_compatibility_and_rejection(tmp_path, tokenizer_v2):
    """Verify that checkpoint loading rejects incompatible tokenizers or vocab sizes."""
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_file = tmp_path / "test_compat.pt"
    save_checkpoint(
        ckpt_file,
        model,
        optimizer,
        step=10,
        tokens_seen=10240,
        tokenizer_name="bpe_16k_v2",
        tokenizer_fingerprint=tokenizer_v2.fingerprint,
    )

    device = torch.device("cpu")

    # 1. Tokenizer name mismatch
    with pytest.raises(ValueError, match="Checkpoint tokenizer.*does not match"):
        load_checkpoint(ckpt_file, model, optimizer, device, expected_tokenizer="gpt2")

    # 2. Tokenizer fingerprint mismatch
    with pytest.raises(ValueError, match="Checkpoint tokenizer fingerprint.*does not match"):
        load_checkpoint(
            ckpt_file,
            model,
            optimizer,
            device,
            expected_tokenizer="bpe_16k_v2",
            expected_fingerprint="invalid_fingerprint_12345",
        )

    # 3. Model vocab size mismatch
    config_gpt2 = ModelConfig(vocab_size=50257, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model_gpt2 = ModernGPT(config_gpt2)
    optimizer_gpt2 = torch.optim.AdamW(model_gpt2.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="Checkpoint vocabulary size.*does not match"):
        load_checkpoint(ckpt_file, model_gpt2, optimizer_gpt2, device, expected_tokenizer="bpe_16k_v2")


def test_token_accounting_and_budget_scaling(tokenizer_v2):
    """Verify token budget accounting and epoch cap calculations for 25M scale."""
    config = ModelConfig(vocab_size=16000, context_length=512)
    device = torch.device("cpu")
    model = ModernGPT(config).to(device)

    shards_dir = Path("artifacts/shards/bpe_16k_v2_5m")
    if not (shards_dir / "train.bin").exists():
        pytest.skip("5M shards not found")

    train_data = ShardedPackedDataset(shards_dir / "train.bin", context_length=512)
    val_data = ShardedPackedDataset(shards_dir / "val.bin", context_length=512)

    unique_tokens = len(train_data) * 512
    assert unique_tokens == 5_070_336

    # 1. Test that 25M budget across 5M shard requires ~4.93 epochs and is rejected under max-epochs=1.2
    args_25m_on_5m = SimpleNamespace(
        tokenizer="bpe_16k_v2",
        shards_dir=str(shards_dir),
        batch_size=2,
        max_tokens=25_000_000,
        resume=None,
        strict_budget=False,
        max_epochs=1.2,
    )
    with pytest.raises(ValueError, match="Epoch cap exceeded"):
        dry_run_check(
            config, model, train_data, val_data, tokenizer_v2, tokenizer_v2.fingerprint, args_25m_on_5m, device
        )

    # 2. Test that planned step calculation matches tokens_per_step
    batch_size = 4
    context_length = 512
    tokens_per_step = batch_size * context_length  # 2,048
    budget = 25_000_000
    planned_steps = math.ceil(budget / tokens_per_step)
    assert planned_steps == 12208


def test_validation_loss_preserves_training_state_and_safe_overflow():
    """Verify validation_loss preserves model.training flag and handles high loss overflow safely."""
    from evaluate import validation_loss
    from torch.utils.data import TensorDataset, DataLoader

    config = ModelConfig(vocab_size=16000, context_length=16, emb_dim=32, n_heads=2, n_layers=1)
    model = ModernGPT(config)
    model.train()
    assert model.training is True

    # Dummy loader with 2 batches of different sizes (e.g. 3 and 1)
    x = torch.randint(0, 16000, (4, 16))
    y = torch.randint(0, 16000, (4, 16))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=3)

    loss, ppl = validation_loss(model, loader, device=torch.device("cpu"))
    assert model.training is True, "validation_loss did not restore model.training mode!"
    assert math.isfinite(loss)
    assert math.isfinite(ppl) or math.isinf(ppl)


def test_overwrite_safety_protection(tmp_path):
    """Verify that existing corpus and shard directories cannot be overwritten without overwrite=True."""
    from data.pipeline import cmd_prepare_corpus, cmd_create_shards

    # 1. Simulate existing corpus manifest
    corpus_dir = tmp_path / "corpus_test"
    corpus_dir.mkdir()
    (corpus_dir / "corpus_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Corpus manifest already exists"):
        cmd_prepare_corpus(output_dir=corpus_dir, total_docs=10, overwrite=False)

    # 2. Simulate existing shard
    shard_dir = tmp_path / "shard_test"
    shard_dir.mkdir()
    (shard_dir / "train.bin").write_bytes(b"dummy_data")

    with pytest.raises(FileExistsError, match="Target shards already exist"):
        cmd_create_shards(corpus_dir=corpus_dir, tokenizer_dir=Path("artifacts/tokenizer_16k_v2"), output_dir=shard_dir, overwrite=False)


def test_lr_schedule_edge_cases_and_bounds():
    """Verify compute_learning_rate rejects invalid bounds and parameters."""
    with pytest.raises(ValueError, match="step must be non-negative"):
        compute_learning_rate(step=-1, schedule="cosine")

    with pytest.raises(ValueError, match="base_lr must be positive"):
        compute_learning_rate(step=10, schedule="cosine", base_lr=0.0)

    with pytest.raises(ValueError, match="min_lr must be non-negative"):
        compute_learning_rate(step=10, schedule="cosine", min_lr=-1e-4)

    with pytest.raises(ValueError, match="min_lr.*cannot exceed base_lr"):
        compute_learning_rate(step=10, schedule="cosine", base_lr=1e-4, min_lr=2e-4)

    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        compute_learning_rate(step=10, schedule="cosine", warmup_steps=-5)

    with pytest.raises(ValueError, match="decay_steps must be non-negative"):
        compute_learning_rate(step=10, schedule="cosine", decay_steps=-10)
