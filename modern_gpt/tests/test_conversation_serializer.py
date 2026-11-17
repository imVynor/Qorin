"""Unit tests for ConversationSerializer and assistant-only loss masking."""

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


@pytest.fixture(scope="module")
def tokenizer():
    return HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")


@pytest.fixture
def serializer(tokenizer):
    return ConversationSerializer(tokenizer, max_length=4096)


def test_single_turn_serialization(serializer):
    conv = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    res = serializer.encode_conversation(conv)
    input_ids = res["input_ids"]
    labels = res["labels"]

    assert len(input_ids) == len(labels)
    assert res["total_tokens"] > 0
    assert res["supervised_tokens"] > 0
    assert (labels == -100).sum() > 0

    # User and default system prompt should be masked (-100)
    # The first token is system header, must be -100
    assert labels[0].item() == -100

    # The last token is EOS, and should be supervised since assistant was last
    assert labels[-1].item() == serializer.eos_id


def test_multi_turn_serialization(serializer):
    conv = [
        {"role": "system", "content": "You are a concise tutor."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Give me a synonym for fast."},
        {"role": "assistant", "content": "Quick."},
    ]
    res = serializer.encode_conversation(conv)
    input_ids = res["input_ids"]
    labels = res["labels"]

    assert len(input_ids) == len(labels)
    assert res["supervised_tokens"] > 0
    # Both assistant turns must have supervised tokens
    assert (labels != -100).sum().item() == res["supervised_tokens"]


def test_tool_call_and_observation_masking(serializer):
    conv = [
        {"role": "user", "content": "Check weather in Tokyo"},
        {"role": "tool_call", "content": '{"call": "get_weather", "city": "Tokyo"}'},
        {"role": "tool_observation", "content": '{"temperature_c": 18, "condition": "Cloudy"}'},
        {"role": "assistant", "content": "The weather in Tokyo is cloudy and 18C."},
    ]
    res = serializer.encode_conversation(conv)
    labels = res["labels"]

    # Tool call should be supervised (assistant-authored)
    # Tool observation should be masked (-100)
    assert res["supervised_tokens"] > 0
    assert (labels == -100).sum() > 0


def test_malformed_conversation_handling(serializer):
    # Empty messages should raise ValueError
    with pytest.raises(ValueError):
        serializer.encode_conversation([])

    # No assistant turn should raise ValueError
    with pytest.raises(ValueError):
        serializer.encode_conversation([{"role": "user", "content": "Just talking to myself"}])

    # Invalid role should raise ValueError
    with pytest.raises(ValueError):
        serializer.encode_conversation([
            {"role": "alien", "content": "Greetings"},
            {"role": "assistant", "content": "Hello"}
        ])


def test_truncation(tokenizer):
    ser_short = ConversationSerializer(tokenizer, max_length=64)
    conv = [
        {"role": "user", "content": "Tell me a very long story about ancient history. " * 10},
        {"role": "assistant", "content": "Once upon a time in ancient Greece... " * 10},
    ]
    res = ser_short.encode_conversation(conv)
    assert len(res["input_ids"]) <= 64
    assert len(res["labels"]) <= 64


def test_causal_shift_loss_computation(serializer, tokenizer):
    conv = [
        {"role": "user", "content": "What is 1 + 1?"},
        {"role": "assistant", "content": "1 + 1 equals 2."},
    ]
    res = serializer.encode_conversation(conv)
    input_ids = res["input_ids"].unsqueeze(0)
    labels = res["labels"].unsqueeze(0)

    vocab_size = tokenizer.vocab_size
    dummy_logits = torch.randn(1, input_ids.size(1), vocab_size)

    shift_logits = dummy_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index=-100)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

