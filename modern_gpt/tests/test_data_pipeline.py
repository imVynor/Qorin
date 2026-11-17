"""Unit tests for ModernGPT-S dataset and tokenizer pipeline.

All tests use synthetic fixtures and run entirely offline without network access.
Covers:
- Dataset record normalization
- Empty-document filtering
- Deterministic splitting
- Duplicate prevention across splits
- Source proportion calculations
- Tokenizer save/load (v1 legacy and v2 no-UNK)
- Vocabulary size and dynamic special-token IDs
- Encode/decode round trips for unseen English, Unicode, Emoji, Math, Code
- Correct document-boundary handling
- Tokenized shard integrity
- Training/validation isolation
- Checkpoint-tokenizer compatibility and fingerprint verification
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from train import compute_learning_rate

from data.bpe_tokenizer import (
    DEFAULT_SPECIAL_TOKENS,
    HuggingFaceBPEWrapper,
    TiktokenGPT2Wrapper,
    train_bpe_tokenizer,
)
from data.sharding import ShardedPackedDataset, create_tokenized_shard
from data.sources import (
    FINEWEB_EDU_CONFIG,
    COSMOPEDIA_V2_CONFIG,
    NormalizedDocument,
    compute_content_hash,
    normalize_record,
)
from data.split import (
    compute_split_stats,
    save_split_corpus,
    split_documents_leakage_safe,
)
from model.config import ModelConfig
from model.gpt import ModernGPT
from train import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# 1. Dataset record normalization & empty-document filtering
# ---------------------------------------------------------------------------

def test_record_normalization():
    raw_record = {
        "text": "This is a valid educational passage discussing machine learning algorithms in detail.",
        "id": "12345",
        "score": 4.5,
    }
    doc = normalize_record(raw_record, FINEWEB_EDU_CONFIG, min_chars=20)
    assert doc is not None
    assert doc.source == "fineweb-edu"
    assert doc.document_id == "fineweb-edu:12345"
    assert doc.char_count == len(raw_record["text"])
    assert doc.byte_count == len(raw_record["text"].encode("utf-8"))
    assert doc.content_hash == compute_content_hash(raw_record["text"])
    assert doc.metadata["score"] == 4.5


def test_empty_and_short_document_filtering():
    raw_empty = {"text": "   \n\t   ", "id": "e1"}
    raw_short = {"text": "Too short", "id": "s1"}
    raw_none = {"text": None, "id": "n1"}

    assert normalize_record(raw_empty, FINEWEB_EDU_CONFIG, min_chars=10) is None
    assert normalize_record(raw_short, FINEWEB_EDU_CONFIG, min_chars=20) is None
    assert normalize_record(raw_none, FINEWEB_EDU_CONFIG, min_chars=10) is None


# ---------------------------------------------------------------------------
# 2. Deterministic splitting, duplicate prevention, and source proportions
# ---------------------------------------------------------------------------

def test_deterministic_splitting_and_isolation():
    docs = []
    for i in range(16):
        text = f"FineWeb-Edu synthetic document number {i} with sufficient length for testing."
        docs.append(
            NormalizedDocument(
                text=text,
                source="fineweb-edu",
                document_id=f"fineweb-edu:{i}",
                char_count=len(text),
                byte_count=len(text.encode("utf-8")),
                content_hash=compute_content_hash(text),
            )
        )
    for i in range(4):
        text = f"Cosmopedia-v2 synthetic document number {i} covering science topics."
        docs.append(
            NormalizedDocument(
                text=text,
                source="cosmopedia-v2",
                document_id=f"cosmopedia-v2:{i}",
                char_count=len(text),
                byte_count=len(text.encode("utf-8")),
                content_hash=compute_content_hash(text),
            )
        )

    train1, val1, _ = split_documents_leakage_safe(docs, val_ratio=0.2, seed=42)
    train2, val2, _ = split_documents_leakage_safe(docs, val_ratio=0.2, seed=42)

    assert [d.document_id for d in train1] == [d.document_id for d in train2]
    assert [d.document_id for d in val1] == [d.document_id for d in val2]

    train_ids = {d.document_id for d in train1}
    val_ids = {d.document_id for d in val1}
    assert len(train_ids.intersection(val_ids)) == 0

    train_hashes = {d.content_hash for d in train1}
    val_hashes = {d.content_hash for d in val1}
    assert len(train_hashes.intersection(val_hashes)) == 0


def test_duplicate_prevention_across_splits():
    duplicate_text = "Exact identical duplicate text appearing in multiple records."
    h = compute_content_hash(duplicate_text)
    doc1 = NormalizedDocument(
        text=duplicate_text, source="fineweb-edu", document_id="id1",
        char_count=len(duplicate_text), byte_count=len(duplicate_text.encode("utf-8")),
        content_hash=h
    )
    doc2 = NormalizedDocument(
        text=duplicate_text, source="fineweb-edu", document_id="id2",
        char_count=len(duplicate_text), byte_count=len(duplicate_text.encode("utf-8")),
        content_hash=h
    )
    other_doc = NormalizedDocument(
        text="A completely different distinct document.", source="cosmopedia-v2", document_id="id3",
        char_count=40, byte_count=40, content_hash=compute_content_hash("A completely different distinct document.")
    )

    train, val, _ = split_documents_leakage_safe([doc1, doc2, other_doc], val_ratio=0.5, seed=123)
    train_hashes = {d.content_hash for d in train}
    val_hashes = {d.content_hash for d in val}
    assert len(train_hashes.intersection(val_hashes)) == 0


def test_source_proportion_calculation():
    docs = [
        NormalizedDocument(
            text="FW" * 100, source="fineweb-edu", document_id=f"fw_{i}",
            char_count=200, byte_count=200, content_hash=f"h_fw_{i}"
        )
        for i in range(8)
    ] + [
        NormalizedDocument(
            text="COSMO" * 50, source="cosmopedia-v2", document_id=f"cosmo_{i}",
            char_count=250, byte_count=250, content_hash=f"h_cosmo_{i}"
        )
        for i in range(2)
    ]
    stats = compute_split_stats(docs)
    assert stats.documents_by_source["fineweb-edu"] == 8
    assert stats.documents_by_source["cosmopedia-v2"] == 2
    assert stats.source_proportions_by_docs["fineweb-edu"] == 0.8
    assert stats.source_proportions_by_docs["cosmopedia-v2"] == 0.2


# ---------------------------------------------------------------------------
# 3. Tokenizer training, save/load, vocab size, special tokens, round trip
# ---------------------------------------------------------------------------

def test_tokenizer_training_v2_no_unk(tmp_path):
    training_data = [
        "Language models predict next tokens given preceding context.",
        "Differential calculus computes derivatives of functions: f'(x) = 2x.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Deep learning leverages backpropagation and stochastic gradient descent.",
    ] * 20

    tok, metrics = train_bpe_tokenizer(
        iter(training_data),
        vocab_size=350,
        special_tokens=DEFAULT_SPECIAL_TOKENS,  # ["<|endoftext|>"]
        name="bpe_test_v2",
    )

    # In v2 without UNK and without PAD:
    assert tok.eos_token_id == 0
    assert tok.unk_token_id is None
    assert tok.bos_token_id is None
    assert tok.pad_token_id is None  # no PAD token; sequence packing is dense
    assert tok.vocab_size <= 350
    assert tok.fingerprint is not None

    # Save and reload
    save_dir = tmp_path / "test_tok_v2"
    tok.save(save_dir)
    assert (save_dir / "tokenizer.json").exists()
    assert (save_dir / "tokenizer_config.json").exists()

    reloaded = HuggingFaceBPEWrapper.load(save_dir, name="bpe_test_v2")
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.eos_token_id == tok.eos_token_id
    assert reloaded.unk_token_id is None
    assert reloaded.fingerprint == tok.fingerprint


def test_v2_unseen_roundtrip_categories(tmp_path):
    # Train small byte-level BPE
    training_data = ["Basic English training sentences for tokenizer build."] * 10
    tok, _ = train_bpe_tokenizer(iter(training_data), vocab_size=300, special_tokens=["<|endoftext|>"])

    probes = [
        "Unseen English: Quantum entanglement demonstrates non-local correlations.",
        "Unicode & Accents: L'élève étudie à l'Université de Genève; très bien!",
        "Emoji: ModernGPT-S 🚀 ✨ 🤖 🦄 💻",
        r"Math: \int_0^1 x^2 dx = \frac{1}{3}, \sum_{i=1}^n i = \frac{n(n+1)}{2}",
        "Code: class SwiGLU(nn.Module):\n    def forward(self, x): return self.w(x)",
    ]

    for p in probes:
        enc = tok.encode(p)
        dec = tok.decode(enc)
        assert dec == p, f"Roundtrip failed for {p}"


def test_gpt2_wrapper_interface():
    gpt2_tok = TiktokenGPT2Wrapper()
    assert gpt2_tok.name == "gpt2"
    assert gpt2_tok.vocab_size == 50257
    assert gpt2_tok.eos_token_id == 50256
    assert gpt2_tok.fingerprint == "tiktoken_gpt2_50257"
    encoded = gpt2_tok.encode("Hello world")
    assert gpt2_tok.decode(encoded) == "Hello world"


# ---------------------------------------------------------------------------
# 4. Document boundary preservation and shard integrity
# ---------------------------------------------------------------------------

def test_sharding_and_boundary_preservation(tmp_path):
    tok = TiktokenGPT2Wrapper()
    doc1 = NormalizedDocument(
        text="First complete document.", source="fineweb-edu", document_id="d1",
        char_count=24, byte_count=24, content_hash="h1"
    )
    doc2 = NormalizedDocument(
        text="Second distinct document.", source="cosmopedia-v2", document_id="d2",
        char_count=25, byte_count=25, content_hash="h2"
    )

    shard_path = tmp_path / "shard.bin"
    context_length = 8
    create_tokenized_shard(
        [doc1, doc2], tok, shard_path, context_length=context_length, split_name="test"
    )

    assert shard_path.exists()
    assert shard_path.with_suffix(".json").exists()

    arr = np.memmap(shard_path, dtype=np.uint16, mode="r")
    eos_indices = np.where(arr == tok.eos_token_id)[0]
    assert len(eos_indices) == 2

    ds = ShardedPackedDataset(shard_path, context_length=context_length)
    if len(ds) > 0:
        inputs, targets = ds[0]
        assert inputs.shape == (context_length,)
        assert targets.shape == (context_length,)
        assert torch.equal(inputs[1:], targets[:-1])


# ---------------------------------------------------------------------------
# 5. Checkpoint-tokenizer compatibility & fingerprint checks
# ---------------------------------------------------------------------------

def test_checkpoint_tokenizer_compatibility_and_fingerprints(tmp_path):
    config = ModelConfig(vocab_size=16000, context_length=16, emb_dim=32, n_heads=4, n_layers=2, ffn_hidden_dim=64)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "ckpt_v1.pt"
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        step=5,
        tokens_seen=500,
        tokenizer_name="bpe_16k",
        tokenizer_fingerprint="fp_legacy_v1",
    )

    device = torch.device("cpu")

    # 1. Loading with different tokenizer name should fail
    with pytest.raises(ValueError, match="does not match expected tokenizer 'bpe_16k_v2'"):
        load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            device,
            expected_tokenizer="bpe_16k_v2",
            expected_fingerprint="fp_v2",
        )

    # 2. Loading with different fingerprint should fail
    with pytest.raises(ValueError, match="does not match active tokenizer fingerprint"):
        load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            device,
            expected_tokenizer="bpe_16k",
            expected_fingerprint="fp_mismatch",
        )

    # 3. Loading with matching tokenizer and fingerprint should succeed
    step, tokens_seen = load_checkpoint(
        ckpt_path,
        model,
        optimizer,
        device,
        expected_tokenizer="bpe_16k",
        expected_fingerprint="fp_legacy_v1",
    )
    assert step == 5
    assert tokens_seen == 500


# ---------------------------------------------------------------------------
# 6. Three-source pipeline & CodeParrot license and quality filters
# ---------------------------------------------------------------------------

def test_codeparrot_permissive_license_filtering():
    """Verify that only permissive licenses are accepted for CodeParrot."""
    from data.sources import CODEPARROT_CLEAN_CONFIG, normalize_record

    base_rec = {
        "content": "def calculate_area(width: float, height: float) -> float:\n    return width * height\n",
        "hash": "abc1234567890def",
        "autogenerated": False,
        "line_max": 80,
        "alpha_frac": 0.85,
    }

    # Permissive licenses must be accepted
    for lic in ("MIT", "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "unlicense", "cc0-1.0"):
        rec = dict(base_rec, license=lic)
        doc = normalize_record(rec, CODEPARROT_CLEAN_CONFIG)
        assert doc is not None, f"Expected license '{lic}' to be accepted!"
        assert doc.source == "codeparrot-clean"

    # Copyleft and viral licenses must be rejected
    for lic in ("gpl-3.0", "gpl-2.0", "agpl-3.0", "lgpl-3.0", "unknown", "", None):
        rec = dict(base_rec, license=lic)
        doc = normalize_record(rec, CODEPARROT_CLEAN_CONFIG)
        assert doc is None, f"Expected non-permissive license '{lic}' to be rejected!"


def test_codeparrot_quality_and_noise_filtering():
    """Verify that autogenerated code, extreme line lengths, and noise are rejected."""
    from data.sources import CODEPARROT_CLEAN_CONFIG, normalize_record

    valid_content = "def fibonacci(n: int) -> int:\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n"

    # 1. Autogenerated code rejection
    rec_autogen = {
        "content": valid_content,
        "license": "mit",
        "hash": "h_autogen",
        "autogenerated": True,
        "line_max": 80,
        "alpha_frac": 0.85,
    }
    assert normalize_record(rec_autogen, CODEPARROT_CLEAN_CONFIG) is None

    # 2. Extreme line length rejection (> 1000 chars)
    rec_long_line = {
        "content": valid_content,
        "license": "mit",
        "hash": "h_long",
        "autogenerated": False,
        "line_max": 1200,
        "alpha_frac": 0.85,
    }
    assert normalize_record(rec_long_line, CODEPARROT_CLEAN_CONFIG) is None

    # 3. Non-alphanumeric noise rejection (alpha_frac < 0.25)
    rec_noise = {
        "content": valid_content,
        "license": "mit",
        "hash": "h_noise",
        "autogenerated": False,
        "line_max": 80,
        "alpha_frac": 0.15,
    }
    assert normalize_record(rec_noise, CODEPARROT_CLEAN_CONFIG) is None


def test_three_source_proportions_and_deterministic_isolation():
    """Verify that 3 sources maintain deterministic isolation and 70/20/10 ratio."""
    from data.split import split_documents_leakage_safe, compute_split_stats

    # Construct 100 dummy documents: 70 FineWeb, 20 Cosmo, 10 Code
    docs = []
    for i in range(70):
        docs.append(NormalizedDocument(
            text=f"Educational web document content number {i} for FineWeb test.",
            source="fineweb-edu",
            document_id=f"fw_{i:03d}",
            char_count=60,
            byte_count=60,
            content_hash=f"hash_fw_{i:03d}",
        ))
    for i in range(20):
        docs.append(NormalizedDocument(
            text=f"Synthetic textbook lesson content number {i} for Cosmo test.",
            source="cosmopedia-v2",
            document_id=f"cos_{i:03d}",
            char_count=60,
            byte_count=60,
            content_hash=f"hash_cos_{i:03d}",
        ))
    for i in range(10):
        docs.append(NormalizedDocument(
            text=f"def python_code_sample_{i}():\n    return 'Clean code sample'\n",
            source="codeparrot-clean",
            document_id=f"code_{i:03d}",
            char_count=60,
            byte_count=60,
            content_hash=f"hash_code_{i:03d}",
        ))

    train_docs, val_docs, stats = split_documents_leakage_safe(docs, val_ratio=0.10, seed=123)

    # 1. Total split sizes
    assert len(train_docs) + len(val_docs) == 100
    assert len(val_docs) == 10  # 10% of 100
    assert len(train_docs) == 90

    # 2. Source counts in train: 70*0.9=63, 20*0.9=18, 10*0.9=9
    train_by_src = stats["train"].documents_by_source
    assert train_by_src["fineweb-edu"] == 63
    assert train_by_src["cosmopedia-v2"] == 18
    assert train_by_src["codeparrot-clean"] == 9

    # 3. Proportions in train split match 70/20/10
    prop = stats["train"].source_proportions_by_docs
    assert prop["fineweb-edu"] == 0.70
    assert prop["cosmopedia-v2"] == 0.20
    assert prop["codeparrot-clean"] == 0.10

    # 4. Zero leakage across splits
    train_hashes = {d.content_hash for d in train_docs}
    val_hashes = {d.content_hash for d in val_docs}
    assert len(train_hashes.intersection(val_hashes)) == 0


def test_three_source_ratio_validation_in_prepare_corpus(tmp_path):
    """Verify that invalid source ratio sums are rejected."""
    from data.pipeline import cmd_prepare_corpus

    with pytest.raises(ValueError, match="Source ratios must sum to 1.0"):
        cmd_prepare_corpus(
            output_dir=tmp_path / "corpus",
            fineweb_ratio=0.70,
            cosmo_ratio=0.20,
            code_ratio=0.20,  # Sum = 1.10
        )


def test_100m_optimizer_steps_and_scheduler_termination():
    """Verify planned step calculation and scheduler endpoints for 100M budget."""
    batch_size = 4
    context_length = 512
    gradient_accumulation_steps = 1
    tokens_per_step = batch_size * context_length * gradient_accumulation_steps  # 2,048

    budget = 100_000_000
    planned_steps = budget // tokens_per_step
    assert planned_steps == 48828
    assert planned_steps * tokens_per_step == 99_999_744

    warmup_steps = 1000
    base_lr = 4e-4
    min_lr = 4e-5

    # 1. Warmup initial step
    lr_0 = compute_learning_rate(step=0, schedule="cosine", warmup_steps=warmup_steps, decay_steps=planned_steps, base_lr=base_lr, min_lr=min_lr)
    assert math.isclose(lr_0, base_lr * (1 / warmup_steps), rel_tol=1e-5)

    # 2. Warmup peak
    lr_peak = compute_learning_rate(step=warmup_steps - 1, schedule="cosine", warmup_steps=warmup_steps, decay_steps=planned_steps, base_lr=base_lr, min_lr=min_lr)
    assert math.isclose(lr_peak, base_lr, rel_tol=1e-5)

    # 3. Decay continuity at warmup boundary
    lr_boundary = compute_learning_rate(step=warmup_steps, schedule="cosine", warmup_steps=warmup_steps, decay_steps=planned_steps, base_lr=base_lr, min_lr=min_lr)
    assert math.isclose(lr_boundary, base_lr, rel_tol=1e-5)

    # 4. Cosine termination at decay endpoint
    lr_end = compute_learning_rate(step=planned_steps, schedule="cosine", warmup_steps=warmup_steps, decay_steps=planned_steps, base_lr=base_lr, min_lr=min_lr)
    assert math.isclose(lr_end, min_lr, rel_tol=1e-5)

    # 5. Clamped beyond decay endpoint
    lr_beyond = compute_learning_rate(step=planned_steps + 1000, schedule="cosine", warmup_steps=warmup_steps, decay_steps=planned_steps, base_lr=base_lr, min_lr=min_lr)
    assert math.isclose(lr_beyond, min_lr, rel_tol=1e-5)
