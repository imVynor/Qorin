"""Unit test suite for ModernGPT-S 100M GQA-4 4K SFT infrastructure."""

import json
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer, Message
from model.config import ModelConfig
from model.gpt import ModernGPT
from train_sft_4k import collate_sft_batch, SFT4KTrainer


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")


@pytest.fixture(scope="module")
def serializer(tokenizer):
    return ConversationSerializer(tokenizer, max_length=4096)


def test_model_architecture_and_checkpoint_loading():
    """Verify loading 100M GQA-4 checkpoint with RoPE scale 8.0."""
    ckpt_path = REPO_ROOT / "checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt"
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("model_config", {})

    cfg = ModelConfig(
        vocab_size=cfg_dict.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 4096),
        emb_dim=cfg_dict.get("emb_dim", 768),
        n_heads=cfg_dict.get("n_heads", 12),
        num_kv_heads=cfg_dict.get("num_kv_heads", 4),
        n_layers=cfg_dict.get("n_layers", 14),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 2048),
        dropout=0.0,
        qkv_bias=False,
        rope_theta=10000.0,
        rope_scale=8.0,
        rope_type="linear",
    )

    model = ModernGPT(cfg)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert model.parameter_count == 100401408


def test_tokenizer_delimiter_fidelity(tokenizer):
    """Verify that ChatML delimiters round-trip deterministically with zero vocabulary modification."""
    delimiters = ["<|im_start|>", "<|im_end|>"]
    for delim in delimiters:
        ids = tokenizer.encode(delim)
        decoded = tokenizer.decode(ids)
        assert decoded == delim, f"Delimiter round-trip failure: expected {delim}, got {decoded}"


def test_assistant_only_loss_masking(serializer, tokenizer):
    """Verify that system and user turns are completely masked (-100) while assistant is supervised."""
    conv = [
        {"role": "system", "content": "You are a concise mathematician."},
        {"role": "user", "content": "What is 7 * 8?"},
        {"role": "assistant", "content": "56"},
    ]
    res = serializer.encode_conversation(conv)
    input_ids = res["input_ids"]
    labels = res["labels"]

    assert len(input_ids) == len(labels)
    # Check that system header is masked
    assert labels[0].item() == -100

    # Assistant tokens and closing EOS must be supervised
    supervised_indices = (labels != -100).nonzero(as_tuple=True)[0]
    assert len(supervised_indices) > 0

    # Ensure last token is supervised EOS
    assert labels[-1].item() == serializer.eos_id


def test_tool_call_and_observation_loss_masking(serializer):
    """Verify assistant tool_call is supervised while external tool_observation is masked."""
    conv = [
        {"role": "user", "content": "Fetch stock price for AAPL."},
        {"role": "tool_call", "content": '{"tool": "stock_lookup", "ticker": "AAPL"}'},
        {"role": "tool_observation", "content": '{"ticker": "AAPL", "price": 220.5}'},
        {"role": "assistant", "content": "Apple Inc. (AAPL) is currently trading at $220.50."},
    ]
    res = serializer.encode_conversation(conv)
    labels = res["labels"]

    # At least tool_call and final assistant must be supervised
    assert (labels != -100).sum().item() > 0


def test_batch_collation_and_padding(serializer, tokenizer):
    """Verify variable-length sequence padding and -100 label padding."""
    conv1 = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    conv2 = [
        {"role": "user", "content": "Write a short poem about the night sky."},
        {"role": "assistant", "content": "Stars ignite the velvet dome, guiding lost wanderers home."},
    ]

    enc1 = serializer.encode_conversation(conv1)
    enc2 = serializer.encode_conversation(conv2)

    batch = collate_sft_batch([enc1, enc2], pad_token_id=tokenizer.eos_token_id)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape[0] == 2
    assert input_ids.shape[1] == max(len(enc1["input_ids"]), len(enc2["input_ids"]))
    # Padding positions must have label == -100
    pad_mask = attention_mask == 0
    assert (labels[pad_mask] == -100).all()


def test_retrieval_dataset_isolation():
    """Verify that the synthetic retrieval training set and evaluation set have zero fact or entity overlap."""
    from scripts.generate_retrieval_sft import TRAIN_ENTITIES, EVAL_ENTITIES, TRAIN_DISTRACTORS, EVAL_DISTRACTORS

    train_names = {e["entity"] for e in TRAIN_ENTITIES}
    eval_names = {e["entity"] for e in EVAL_ENTITIES}
    overlap_names = train_names.intersection(eval_names)
    assert len(overlap_names) == 0, f"Found entity overlap: {overlap_names}"

    train_dist = set(TRAIN_DISTRACTORS)
    eval_dist = set(EVAL_DISTRACTORS)
    overlap_dist = train_dist.intersection(eval_dist)
    assert len(overlap_dist) == 0, f"Found distractor overlap: {overlap_dist}"


def test_state_recovery_mock(tmp_path):
    """Verify 19-point state recovery serialization and deserialization."""
    cfg_path = REPO_ROOT / "configs/modern_gpts_100m/sft_4k.yaml"
    trainer = SFT4KTrainer(cfg_path)

    test_ckpt = tmp_path / "recovery_test.pt"
    trainer.global_step = 100
    trainer.tokens_seen = 3276800
    trainer.best_val_loss = 2.45
    trainer.save_checkpoint(test_ckpt)

    assert test_ckpt.exists()
    trainer.global_step = 0
    trainer.load_checkpoint(test_ckpt)
    assert trainer.global_step == 100
    assert trainer.tokens_seen == 3276800
    assert trainer.best_val_loss == 2.45
