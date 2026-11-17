"""Regression and integrity test suite for ModernGPT-S 500-example midscale dataset."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from train_sft import format_instruction_prompt, SFTDataset

DATA_DIR = Path("data/midscale_500")

def test_manifest_and_counts():
    manifest_path = DATA_DIR / "manifest.json"
    assert manifest_path.exists(), "Manifest file not found"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_records"] == 500
    assert manifest["splits"]["train"]["record_count"] == 400
    assert manifest["splits"]["validation"]["record_count"] == 100

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["splits"]["train"]["sha256"]
    assert hashlib.sha256(val_path.read_bytes()).hexdigest() == manifest["splits"]["validation"]["sha256"]


def test_category_distributions():
    train_counts = {"python": 0, "json": 0, "qa": 0, "explanation": 0}
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_counts[rec["category"]] += 1

    assert train_counts == {"python": 160, "json": 160, "qa": 40, "explanation": 40}

    val_counts = {"python": 0, "json": 0, "qa": 0, "explanation": 0}
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            val_counts[rec["category"]] += 1

    assert val_counts == {"python": 40, "json": 40, "qa": 10, "explanation": 10}


def test_zero_family_leakage():
    train_families = set()
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            train_families.add(json.loads(line)["task_family"])

    val_families = set()
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_families.add(json.loads(line)["task_family"])

    leakage = train_families.intersection(val_families)
    assert len(leakage) == 0, f"Task family leakage found: {leakage}"


def test_python_and_json_validity():
    with open(DATA_DIR / "unit_tests.json", "r", encoding="utf-8") as f:
        specs = json.load(f)

    for path in [DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cat = rec["category"]
                out = rec["output"].strip()
                if cat == "python":
                    clean = out
                    if clean.startswith("```python"):
                        clean = clean[len("```python"):].strip()
                    if clean.endswith("```"):
                        clean = clean[:-3].strip()
                    ast.parse(clean)
                elif cat == "json":
                    data = json.loads(out)
                    assert isinstance(data, dict)
                    req_keys = specs["json_schemas"][rec["id"]]["required_keys"]
                    for k in req_keys:
                        assert k in data, f"Key {k} missing in {rec['id']}"


def test_tokenization_and_context_length():
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
    for path in [DATA_DIR / "train.jsonl", DATA_DIR / "val.jsonl"]:
        ds = SFTDataset(path, tok, context_length=512)
        assert len(ds) > 0
        for i in range(len(ds)):
            item = ds[i]
            input_ids = item["input_ids"]
            labels = item["labels"]
            assert len(input_ids) <= 512
            assert len(labels) <= 512
            assert input_ids[-1] == tok.eos_token_id
            active_tokens = sum(1 for x in labels if x != -100)
            assert active_tokens >= 8, f"Too few active tokens ({active_tokens}) in item {i}"
