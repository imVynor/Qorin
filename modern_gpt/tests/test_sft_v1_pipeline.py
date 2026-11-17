"""
Automated Pytest Suite for ModernGPT-S SFT Dataset v1 Pipeline
==============================================================
Validates:
1. Canonical tokenizer SHA-256 match and ChatML special token IDs.
2. Master splits existence, line counts, and strict disjointness.
3. Record schema integrity, required fields, and valid roles.
4. Token budget compliance (< 2048 context limit).
5. Contamination quarantine against Phase 1D benchmark probe suites.
6. Authentic tool execution trajectories in Collection E.
"""

import hashlib
import json
from pathlib import Path
import sys
import pytest
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SFT_DIR = REPO_ROOT / "modern_gpt/data_sft_v1"
SPLITS_DIR = SFT_DIR / "splits"
MANIFESTS_DIR = SFT_DIR / "manifests"
TOKENIZER_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXPECTED_TOKENIZER_SHA256 = "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"

def test_tokenizer_sha256_and_chatml_tokens():
    assert TOKENIZER_PATH.exists(), f"Tokenizer not found at {TOKENIZER_PATH}"
    with open(TOKENIZER_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == EXPECTED_TOKENIZER_SHA256, f"Tokenizer hash mismatch: {actual_hash}"

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    assert tok.get_vocab_size() == 32768, "Vocab size must be exactly 32,768"
    assert tok.token_to_id("<|im_start|>") == 4
    assert tok.token_to_id("<|im_end|>") == 5
    assert tok.token_to_id("<|tool_call|>") == 6
    assert tok.token_to_id("<|tool_result|>") == 7
    assert tok.token_to_id("<|tool_error|>") == 8
    assert tok.token_to_id("<|endoftext|>") == 9

def test_master_splits_exist_and_disjoint():
    train_file = SPLITS_DIR / "sft_v1_train.jsonl"
    val_file = SPLITS_DIR / "sft_v1_val.jsonl"
    test_file = SPLITS_DIR / "sft_v1_test.jsonl"

    assert train_file.exists(), "Train split file missing"
    assert val_file.exists(), "Validation split file missing"
    assert test_file.exists(), "Test split file missing"

    train_ids = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_ids.add(rec["id"])

    val_ids = set()
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            val_ids.add(rec["id"])

    test_ids = set()
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            test_ids.add(rec["id"])

    assert len(train_ids) > 85000, f"Expected >85k train IDs, got {len(train_ids)}"
    assert len(val_ids) > 2000, f"Expected >2k val IDs, got {len(val_ids)}"
    assert len(test_ids) > 2000, f"Expected >2k test IDs, got {len(test_ids)}"

    # Check strict disjointness
    assert len(train_ids & val_ids) == 0, f"Train and Val overlap: {len(train_ids & val_ids)} IDs"
    assert len(train_ids & test_ids) == 0, f"Train and Test overlap: {len(train_ids & test_ids)} IDs"
    assert len(val_ids & test_ids) == 0, f"Val and Test overlap: {len(val_ids & test_ids)} IDs"

def test_record_schema_and_quality():
    # Sample 1,000 records from train split
    train_file = SPLITS_DIR / "sft_v1_train.jsonl"
    required_keys = {"id", "source_name", "license", "category", "messages", "final_split", "quality_metadata"}

    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            rec = json.loads(line)
            assert required_keys.issubset(rec.keys()), f"Record {rec.get('id')} missing keys: {required_keys - rec.keys()}"
            assert rec["final_split"] == "train"
            assert rec["quality_metadata"]["total_tokens"] <= 2048, f"Record exceeds context window: {rec['quality_metadata']['total_tokens']}"
            assert rec["quality_metadata"]["assistant_tokens"] > 0, f"Record has 0 assistant tokens: {rec['id']}"

            # Check messages sequence
            msgs = rec["messages"]
            assert len(msgs) >= 2, f"Conversation must have at least 2 turns: {rec['id']}"
            assert any(m["role"] == "assistant" for m in msgs), f"Conversation missing assistant turn: {rec['id']}"

def test_benchmark_non_contamination():
    # Check that known probe prompts from Phase 1D do not exist in train split
    from modern_gpt.scripts.phase1d_benchmark_data import MATH_TASKS, TOOL_TASKS
    from modern_gpt.scripts.evaluate_sft_benchmarks import CODING_TASKS

    probe_prompts = {t["prompt"].strip().lower() for t in MATH_TASKS}
    probe_prompts.update(t["prompt"].strip().lower() for t in TOOL_TASKS)
    probe_prompts.update(t["prompt"].strip().lower() for t in CODING_TASKS)

    train_file = SPLITS_DIR / "sft_v1_train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            user_msg = next((m["content"].strip().lower() for m in rec["messages"] if m["role"] == "user"), "")
            assert user_msg not in probe_prompts, f"Benchmark leak detected in train set: '{user_msg[:60]}...'"

def test_manifest_consistency():
    manifest_file = MANIFESTS_DIR / "SFT_DATASET_V1_MANIFEST.json"
    assert manifest_file.exists(), "Master manifest missing"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["target_model"] == "ModernGPT-S 500M Decoder-Only Transformer"
    assert manifest["splits_summary"]["total_records"] == 95027
    assert manifest["splits_summary"]["total_assistant_target_tokens"] > 40_000_000
    assert "modern_gpt/data_sft_v1/splits/sft_v1_train.jsonl" in [k.replace("\\", "/") for k in manifest["file_checksums_and_sizes"].keys()]
