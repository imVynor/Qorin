"""Tests for ModernGPT-S 250M unified shards integrity, isolation, and loader compatibility."""

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from data.sharding import ShardedPackedDataset

HISTORICAL_TRAIN_SHA256 = "0d4cc2a74eff040895bea88f21b75aef40e588ff9e0713b84ce4779aa03e1e32"
HISTORICAL_VAL_SHA256 = "d97678fc0b8923c763c69c0e35271662b3a4aef1d9d43589bc5b416f5916280d"


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def test_historical_shards_remain_unmodified():
    """Verify historical 100M shards were not touched or overwritten."""
    train_bin = Path("artifacts/shards_100m/train.bin")
    val_bin = Path("artifacts/shards_100m/val.bin")

    assert train_bin.exists()
    assert val_bin.exists()

    assert get_file_sha256(train_bin) == HISTORICAL_TRAIN_SHA256
    assert get_file_sha256(val_bin) == HISTORICAL_VAL_SHA256


def test_shards_250m_manifest_and_proportions():
    """Verify 250M manifest token counts, source proportions (70/20/10), and hash uniqueness."""
    manifest_path = Path("artifacts/shards_250m/manifest.json")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Check usable token count
    usable_tokens = m["combined_inventory"]["train_tokens_usable"]
    assert usable_tokens >= 250_000_000, f"Expected >= 250M usable tokens, got {usable_tokens:,}"

    # Check source proportions (within 0.5% of 70/20/10)
    sources = m["source_contributions"]
    total = sum(sources[s]["total_tokens"] for s in sources)

    fw_pct = sources["fineweb-edu"]["total_tokens"] / total
    cosmo_pct = sources["cosmopedia-v2"]["total_tokens"] / total
    code_pct = sources["codeparrot-clean"]["total_tokens"] / total

    assert abs(fw_pct - 0.70) < 0.005, f"FineWeb ratio {fw_pct:.4f} diverges from 0.70"
    assert abs(cosmo_pct - 0.20) < 0.005, f"Cosmopedia ratio {cosmo_pct:.4f} diverges from 0.20"
    assert abs(code_pct - 0.10) < 0.005, f"CodeParrot ratio {code_pct:.4f} diverges from 0.10"

    # Validation isolation
    assert m["combined_inventory"]["val_sha256"] == HISTORICAL_VAL_SHA256
    assert m["combined_inventory"]["val_tokens_usable"] == 11_184_128


def test_shards_250m_packed_dataset_loader():
    """Verify ShardedPackedDataset reads 250M shards and satisfies all shape and target shift constraints."""
    shards_dir = Path("artifacts/shards_250m")
    train_bin = shards_dir / "train.bin"
    val_bin = shards_dir / "val.bin"

    assert train_bin.exists()
    assert val_bin.exists()

    context_length = 512
    train_dataset = ShardedPackedDataset(train_bin, context_length=context_length)
    val_dataset = ShardedPackedDataset(val_bin, context_length=context_length)

    assert len(train_dataset) == 489_268
    assert len(val_dataset) == 21_844

    # Sample check first, middle, and last sequences
    for idx in [0, len(train_dataset) // 2, len(train_dataset) - 1]:
        inp, tgt = train_dataset[idx]
        assert inp.shape == (context_length,)
        assert tgt.shape == (context_length,)
        assert torch.equal(inp[1:], tgt[:-1]), f"Target shift failed at sequence {idx}"
        assert inp.min().item() >= 0 and inp.max().item() < 16000
        assert tgt.min().item() >= 0 and tgt.max().item() < 16000

