"""Unit and regression tests for SFT v2 dataset pipeline, serialization, and isolation."""

import json
from pathlib import Path
import sys
import pytest
import torch

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from train_sft import SFTDataset, collate_sft_batch, format_instruction_prompt
from scripts.build_sft_v2_dataset import normalize_text, hash_text


@pytest.fixture
def tokenizer():
    return HuggingFaceBPEWrapper.load(root_dir / "artifacts/tokenizer_16k_v2")


def test_tokenizer_fingerprint_integrity(tokenizer):
    """Verify tokenizer matches exact required fingerprint."""
    assert tokenizer.name == "bpe_16k_v2"
    assert tokenizer.fingerprint == "be6c3d60c5baf217"
    assert tokenizer.eos_token_id == 0


def test_sft_v2_dataset_loading_and_masking(tokenizer):
    """Verify that SFTDataset properly loads sft_v2_public/train.jsonl with correct masking."""
    train_file = root_dir / "data/sft_v2_public/train.jsonl"
    assert train_file.exists(), "train.jsonl does not exist!"

    dataset = SFTDataset(train_file, tokenizer, context_length=512)
    assert len(dataset) == 8597

    # Test first 10 items for loss masking structure
    for idx in range(10):
        item = dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]

        assert len(input_ids) == len(labels)
        assert len(input_ids) <= 512

        # Terminal token must be EOS 0
        assert input_ids[-1] == 0
        assert labels[-1] == 0

        # Masking: prompt tokens must be -100
        first_supervised_idx = None
        for i, lab in enumerate(labels):
            if lab != -100:
                first_supervised_idx = i
                break

        assert first_supervised_idx is not None, f"Item {idx} has zero supervised tokens!"
        assert first_supervised_idx > 0, f"Item {idx} did not mask prompt tokens!"
        # All prompt tokens before response must be -100
        assert all(lab == -100 for lab in labels[:first_supervised_idx])
        # Supervised tokens must not be -100
        assert all(lab != -100 for lab in labels[first_supervised_idx:])


def test_split_isolation_and_zero_leakage():
    """Verify strict zero leakage between SFT v2 training data and protected evaluation splits."""
    train_file = root_dir / "data/sft_v2_public/train.jsonl"
    val_file = root_dir / "data/midscale_500/val.jsonl"
    sel_val_file = root_dir / "data/midscale_500/checkpoint_selection_val.jsonl"

    protected_prompts = set()
    protected_responses = set()

    for p in [val_file, sel_val_file]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    protected_prompts.add(normalize_text(rec["instruction"]))
                    protected_responses.add(normalize_text(rec["output"]))

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                p_norm = normalize_text(rec["instruction"])
                r_norm = normalize_text(rec["output"])

                assert p_norm not in protected_prompts, f"LEAKAGE DETECTED: prompt '{p_norm}' matches protected evaluation set!"
                if len(r_norm) > 20:
                    assert r_norm not in protected_responses, f"LEAKAGE DETECTED: response matches protected evaluation set!"


def test_collate_and_right_padding(tokenizer):
    """Verify batch collation with padding and -100 target ignore index."""
    train_file = root_dir / "data/sft_v2_public/train.jsonl"
    dataset = SFTDataset(train_file, tokenizer, context_length=512)

    batch_samples = [dataset[0], dataset[1], dataset[2], dataset[3]]
    inputs, targets = collate_sft_batch(batch_samples, pad_token_id=0)

    max_len = max(len(s["input_ids"]) for s in batch_samples)
    assert inputs.shape == (4, max_len)
    assert targets.shape == (4, max_len)

    # Check padding positions
    for i, s in enumerate(batch_samples):
        cur_len = len(s["input_ids"])
        if cur_len < max_len:
            assert (inputs[i, cur_len:] == 0).all()
            assert (targets[i, cur_len:] == -100).all()


def test_category_distribution_and_schema_types():
    """Verify all 5 required categories exist with valid schemas."""
    manifest_file = root_dir / "data/sft_v2_public/dataset_audit.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    cats = audit["categories"]
    assert "python" in cats
    assert "instruction" in cats
    assert "qa" in cats
    assert "json" in cats
    assert "math_reasoning" in cats

    assert audit["total_examples"] >= 5000
    assert audit["max_sequence_length"] <= 512
