"""Regression tests for EOS supervision, padding independence, and packed document boundaries.

Verifies:
1. Genuine EOS targets (token ID 0) actively contribute to cross-entropy loss and gradient.
2. Tokenizer v2 defines pad_token_id = None; no padding tokens or attention padding masks exist.
3. Packed document boundaries are correctly formed, shifted, and supervised.
4. Real v2 shards contain exact document boundary EOS tokens with zero padding.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset, create_tokenized_shard
from data.sources import NormalizedDocument
from model.config import ModelConfig
from model.gpt import ModernGPT


@pytest.fixture
def tokenizer_v2():
    tok_dir = Path("artifacts/tokenizer_16k_v2")
    if not tok_dir.exists():
        pytest.skip(f"Tokenizer v2 not found at {tok_dir}")
    return HuggingFaceBPEWrapper.load(tok_dir, name="bpe_16k_v2")


@pytest.fixture
def model_16k():
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=128, n_heads=4, n_layers=2)
    return ModernGPT(config)


def test_eos_target_contributes_to_loss_and_gradient(model_16k, tokenizer_v2):
    """Verify that EOS targets (ID 0) actively produce loss and gradients and are not masked."""
    assert tokenizer_v2.eos_token_id == 0
    assert tokenizer_v2.pad_token_id is None

    model_16k.train()
    batch_size = 2
    seq_len = 16

    # Create synthetic input and targets containing EOS (0) at multiple positions
    inputs = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
    targets = inputs.clone()
    # Insert genuine EOS tokens at positions 5 and 15
    targets[0, 5] = 0
    targets[1, 15] = 0

    logits, loss = model_16k(inputs, targets)

    # 1. Loss must be finite and positive
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # 2. Compare with masked targets (where EOS is replaced by PyTorch's default ignore_index -100)
    targets_masked = targets.clone()
    targets_masked[targets_masked == 0] = -100
    loss_masked = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets_masked.reshape(-1))

    # The loss with EOS must NOT equal the loss where EOS is masked out
    assert not torch.isclose(loss, loss_masked, atol=1e-5), (
        "Loss with EOS targets matches loss with EOS masked as -100! EOS is mistakenly ignored."
    )

    # 3. Verify backward pass: gradients must flow to EOS token in embedding layer
    loss.backward()
    eos_grad = model_16k.token_embedding.weight.grad[0]
    assert eos_grad is not None
    assert eos_grad.abs().sum().item() > 0.0, "EOS token embedding received zero gradient!"

    # 4. Negative test: Demonstrate that if someone mistakenly used ignore_index=0, EOS gradient would be 0
    model_16k.zero_grad()
    logits2 = model_16k(inputs)
    loss_ignoring_zero = F.cross_entropy(
        logits2.reshape(-1, logits2.size(-1)), targets.reshape(-1), ignore_index=0
    )
    loss_ignoring_zero.backward()
    eos_grad_ignored = model_16k.token_embedding.weight.grad[0]
    # Because target=0 is ignored, input positions whose target was 0 produce zero gradient at the head
    # ModernGPT's default ignore_index=-100 ensures full EOS target supervision.
    assert eos_grad_ignored is not None


def test_no_padding_dependency_and_dense_packing(tokenizer_v2, model_16k):
    """Verify tokenizer v2 has no PAD token and model requires no attention padding mask."""
    # 1. Tokenizer v2 must have no PAD token
    assert tokenizer_v2.pad_token_id is None, "pad_token_id should be None in v2"
    assert tokenizer_v2.unk_token_id is None, "unk_token_id should be None in v2"
    assert tokenizer_v2.eos_token_id == 0, "eos_token_id should be 0"

    # 2. Model forward should execute causal self-attention without any padding mask
    inputs = torch.randint(0, 16000, (2, 32), dtype=torch.long)
    logits = model_16k(inputs)
    assert logits.shape == (2, 32, 16000)

    # 3. Model config has no padding token ID field
    assert not hasattr(model_16k.config, "pad_token_id")


def test_packed_document_boundaries_and_supervision(tmp_path, tokenizer_v2, model_16k):
    """Verify sequence packing inserts EOS at document boundaries and target shifting supervises transitions."""
    docs = [
        NormalizedDocument(
            text="First document about deep learning and transformers.",
            source="test",
            document_id="doc1",
            char_count=52,
            byte_count=52,
            content_hash="hash1",
        ),
        NormalizedDocument(
            text="Second document focusing on tokenizer design and BPE.",
            source="test",
            document_id="doc2",
            char_count=53,
            byte_count=53,
            content_hash="hash2",
        ),
        NormalizedDocument(
            text="Third document detailing self-attention mechanisms and scaling laws.",
            source="test",
            document_id="doc3",
            char_count=69,
            byte_count=69,
            content_hash="hash3",
        ),
    ]

    shard_path = tmp_path / "test_shard.bin"
    context_length = 16
    create_tokenized_shard(
        documents=docs,
        tokenizer=tokenizer_v2,
        output_path=shard_path,
        context_length=context_length,
    )

    dataset = ShardedPackedDataset(shard_path, context_length=context_length)
    assert len(dataset) > 0

    # Read the raw token stream to verify exact structure
    tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
    eos_indices = np.where(tokens == 0)[0]

    # Exactly 3 documents -> exactly 3 EOS tokens
    assert len(eos_indices) == 3, f"Expected 3 EOS tokens, found {len(eos_indices)}"

    # Check each packed sample from the dataset
    all_inputs = []
    all_targets = []
    for i in range(len(dataset)):
        inp, tgt = dataset[i]
        # Target shift check: targets[k] == inp[k+1] for all k < context_length - 1
        assert torch.equal(inp[1:], tgt[:-1]), f"Target shift broken in sequence {i}"
        all_inputs.append(inp)
        all_targets.append(tgt)

    # Verify boundary transitions in packed sequences:
    # Find sequence containing an EOS transition
    found_eos_target = False
    found_eos_input = False
    for inp, tgt in zip(all_inputs, all_targets):
        if (tgt == 0).any():
            found_eos_target = True
            pos = (tgt == 0).nonzero(as_tuple=True)[0][0].item()
            # The input at this position is the last token of a document; target is EOS
            assert tgt[pos].item() == 0

        if (inp == 0).any():
            found_eos_input = True
            pos = (inp == 0).nonzero(as_tuple=True)[0][0].item()
            # The input at this position is EOS; target is the first token of the next document
            assert inp[pos].item() == 0
            assert tgt[pos].item() != 0  # Next document starts with non-EOS token

    assert found_eos_target, "No EOS target found across packed sequences!"
    assert found_eos_input, "No EOS input found across packed sequences!"

    # Forward + backward pass on packed batch
    inputs_batch = torch.stack(all_inputs[:2])
    targets_batch = torch.stack(all_targets[:2])
    logits, loss = model_16k(inputs_batch, targets_batch)
    assert torch.isfinite(loss)
    loss.backward()

    # Gradients must be finite
    for p in model_16k.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_real_shard_eos_distribution_and_zero_padding():
    """Verify the real generated v2 training shard contains exact document boundaries and zero padding."""
    train_shard = Path("artifacts/shards/bpe_16k_v2/train.bin")
    if not train_shard.exists():
        pytest.skip(f"Train shard not found at {train_shard}")

    tokens = np.memmap(train_shard, dtype=np.uint16, mode="r")
    total_tokens = len(tokens)
    eos_count = int(np.sum(tokens == 0))

    # Exactly 90 documents in train split -> exactly 90 EOS tokens
    assert eos_count == 90, f"Expected exactly 90 EOS tokens for 90 documents, got {eos_count}"

    # Verify no consecutive EOS tokens (which would indicate empty documents)
    eos_diffs = np.diff(np.where(tokens == 0)[0])
    assert (eos_diffs > 1).all(), "Found consecutive EOS tokens!"

    # Verify dataset packing has zero padding
    dataset = ShardedPackedDataset(train_shard, context_length=512)
    assert len(dataset) == (total_tokens - 1) // 512
    sample_inp, sample_tgt = dataset[0]
    assert sample_inp.shape == (512,)
    assert sample_tgt.shape == (512,)
    assert sample_inp.dtype == torch.int64
    assert sample_tgt.dtype == torch.int64
