"""Regression and Unit Tests for Forensic Contamination Analysis & Safeguards."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from scripts.forensic_100m_contamination import (
    BOILERPLATE_PATTERNS,
    NATURAL_LANGUAGE_PATTERNS,
    ForensicAuditor,
    classify_match,
    compute_file_sha256,
    get_ngrams,
)
from scripts.pretrain_100m_session import (
    APPROVED_HASH_REL_PATH,
    MANIFEST_REL_PATH,
    verify_manifest_and_approval,
)


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tokenizer(repo_root):
    tok_dir = repo_root / "artifacts/tokenizer_16k_v2"
    tok = HuggingFaceBPEWrapper.load(tok_dir)
    assert tok.fingerprint == "be6c3d60c5baf217"
    return tok


class TestContaminationForensics:
    def test_ngram_extraction(self):
        tokens = list(range(20))
        ngrams = get_ngrams(tokens, n=13)
        assert len(ngrams) == 20 - 13 + 1
        assert (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) in ngrams
        assert (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19) in ngrams

    def test_short_token_sequence_yields_empty_ngrams(self):
        assert get_ngrams([1, 2, 3], n=13) == set()

    def test_boundary_artifact_classification(self):
        # Token 0 is <|endoftext|>
        tokens = (10, 20, 0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)
        cls_type, reason = classify_match("some text <|endoftext|> more", tokens, is_boundary=True)
        assert cls_type == "BOUNDARY_ARTIFACT"

        # Explicit 0 in tokens even if is_boundary is False
        cls_type, reason = classify_match("some text", tokens, is_boundary=False)
        assert cls_type == "BOUNDARY_ARTIFACT"

    def test_boilerplate_classification(self):
        # Even/odd boilerplate
        cls_type, reason = classify_match("def is_even(n):\n    return n % 2 == 0", (1,) * 13, False)
        assert cls_type == "COMMON_BOILERPLATE"

        cls_type, reason = classify_match("def is_odd(n):\n    return n % 2 != 0", (1,) * 13, False)
        assert cls_type == "COMMON_BOILERPLATE"

        # Loop boilerplate
        cls_type, reason = classify_match("for i in range(len(arr)):", (1,) * 13, False)
        assert cls_type == "COMMON_BOILERPLATE"

        # Short fragments (< 30 characters)
        cls_type, reason = classify_match("return False\n else:", (1,) * 13, False)
        assert cls_type == "COMMON_BOILERPLATE"

    def test_natural_language_boilerplate(self):
        cls_type, reason = classify_match("Write a python function that takes a list of integers", (1,) * 13, False)
        assert cls_type == "COMMON_NATURAL_LANGUAGE"

        cls_type, reason = classify_match("vowels = ['a', 'e', 'i', 'o', 'u']", (1,) * 13, False)
        assert cls_type == "COMMON_NATURAL_LANGUAGE"

    def test_reference_field_indexing(self, repo_root, tokenizer):
        """Ensures instruction, input, and output fields are all extracted, preventing empty-hash bug."""
        sample_jsonl = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        rec = {
            "id": "test_rec_001",
            "instruction": "Explain quantum computing in comprehensive and meticulous detail using very basic elementary terms for beginner students.",
            "input": "No additional context.",
            "output": "Quantum computing uses qubits that exist in superpositions of zero and one.",
        }
        sample_jsonl.write(json.dumps(rec) + "\n")
        sample_jsonl.close()

        try:
            auditor = ForensicAuditor(repo_root)
            auditor.suite_definitions = {
                "test_suite": {
                    "category": "Test Suite",
                    "path": Path(sample_jsonl.name),
                    "fields": ["instruction", "input", "output"],
                    "description": "Test indexing",
                }
            }
            auditor.index_all_reference_suites()

            # Verify n-grams extracted from both instruction and output
            inst_tokens = tokenizer.encode(rec["instruction"])
            inst_ng = tuple(inst_tokens[:13])
            out_tokens = tokenizer.encode(rec["output"])
            out_ng = tuple(out_tokens[:13])

            assert inst_ng in auditor.ngram_to_refs
            assert out_ng in auditor.ngram_to_refs

            # Verify ref fields point to correct source
            inst_ref = [r for r in auditor.ngram_to_refs[inst_ng] if r["field"] == "instruction"]
            assert len(inst_ref) == 1
            assert inst_ref[0]["record_id"] == "test_rec_001"

            out_ref = [r for r in auditor.ngram_to_refs[out_ng] if r["field"] == "output"]
            assert len(out_ref) == 1
            assert out_ref[0]["record_id"] == "test_rec_001"

        finally:
            os.unlink(sample_jsonl.name)

    def test_reference_deduplication(self, repo_root, tokenizer):
        """Ensures identical n-grams within a dataset do not multiply occurrence counts."""
        sample_jsonl = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        # Write 5 records with identical boilerplate output
        for i in range(5):
            rec = {
                "id": f"dup_rec_{i}",
                "instruction": f"Problem statement number {i}",
                "output": "def is_even(n):\n    return n % 2 == 0",
            }
            sample_jsonl.write(json.dumps(rec) + "\n")
        sample_jsonl.close()

        try:
            auditor = ForensicAuditor(repo_root)
            auditor.suite_definitions = {
                "dup_test": {
                    "category": "Deduplication Test",
                    "path": Path(sample_jsonl.name),
                    "fields": ["instruction", "output"],
                    "description": "Testing deduplication",
                }
            }
            auditor.index_all_reference_suites()

            target_tokens = tokenizer.encode("def is_even(n):\n    return n % 2 == 0")
            ng = tuple(target_tokens[:13])

            # In the reference index, the n-gram should only be recorded once per suite
            matching_refs = [r for r in auditor.ngram_to_refs[ng] if r["suite_key"] == "dup_test"]
            assert len(matching_refs) == 1, f"Expected 1 unique ref for suite, got {len(matching_refs)}"
        finally:
            os.unlink(sample_jsonl.name)

    def test_pretrain_guardrail_blocks_without_approval_file(self, repo_root, tmp_path):
        """Verifies verify_manifest_and_approval refuses to launch if approved_dataset_manifest.sha256 is missing."""
        # Mock manifest file but NO approval file
        manifest_dir = tmp_path / "artifacts/modern_gpts_100m"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / "dataset_manifest.json"
        manifest_file.write_text('{"dummy": "manifest"}', encoding="utf-8")
        shard_dir = manifest_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="No certified contamination audit approval found"):
            verify_manifest_and_approval(repo_root=tmp_path, shard_dir=shard_dir, expected_shards=0)

    def test_pretrain_guardrail_blocks_on_hash_mismatch(self, repo_root, tmp_path):
        """Verifies verify_manifest_and_approval refuses to launch if approved hash differs from actual manifest hash."""
        manifest_dir = tmp_path / "artifacts/modern_gpts_100m"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / "dataset_manifest.json"
        manifest_file.write_text('{"actual": "manifest_v2"}', encoding="utf-8")
        shard_dir = manifest_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        reports_dir = tmp_path / "artifacts/reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        approved_file = reports_dir / "approved_dataset_manifest.sha256"
        approved_file.write_text("0000000000000000000000000000000000000000000000000000000000000000\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Corpus manifest mismatch"):
            verify_manifest_and_approval(repo_root=tmp_path, shard_dir=shard_dir, expected_shards=0)

    def test_pretrain_guardrail_passes_on_matching_approved_hash(self, repo_root, tmp_path):
        """Verifies verify_manifest_and_approval succeeds when manifest matches approved hash."""
        manifest_dir = tmp_path / "artifacts/modern_gpts_100m"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = manifest_dir / "dataset_manifest.json"
        content = '{"valid": "manifest_content"}'
        manifest_file.write_text(content, encoding="utf-8")
        shard_dir = manifest_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)

        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest().lower()

        reports_dir = tmp_path / "artifacts/reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        approved_file = reports_dir / "approved_dataset_manifest.sha256"
        approved_file.write_text(f"{expected_hash}  artifacts/modern_gpts_100m/dataset_manifest.json\n", encoding="utf-8")

        # Must not raise
        result_hash = verify_manifest_and_approval(repo_root=tmp_path, shard_dir=shard_dir, expected_shards=0)
        assert result_hash == expected_hash

    def test_live_repository_artifacts_certified(self, repo_root):
        """Verifies that the live repository's 30 shards, manifest, and approval file are strictly valid."""
        shard_dir = repo_root / "artifacts/modern_gpts_100m/shards"
        assert shard_dir.exists()
        shards = list(shard_dir.glob("shard_*.bin"))
        assert len(shards) == 30

        certified_hash = verify_manifest_and_approval(repo_root=repo_root, shard_dir=shard_dir, expected_shards=30)
        assert certified_hash == "6c6eeda62f7d0c68403b481e82c678138b95c17131ec194cefe6c7b34b2b33cb"
