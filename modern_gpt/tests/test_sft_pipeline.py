"""Unit and regression tests for SFT training pipeline, loss masking, and evaluation suite."""

from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.evaluate_sft import extract_code_block, extract_json_block
from train_sft import SFTDataset, collate_sft_batch, compute_sft_loss, format_instruction_prompt


def test_prompt_formatting_structure():
    """Verify standard Alpaca format with and without input context."""
    p1 = format_instruction_prompt("Explain gravity.")
    assert "### Instruction:\nExplain gravity." in p1
    assert "### Response:\n" in p1
    assert "### Input:\n" not in p1

    p2 = format_instruction_prompt("Summarize this text.", "Some context here.")
    assert "### Instruction:\nSummarize this text." in p2
    assert "### Input:\nSome context here." in p2
    assert "### Response:\n" in p2


def test_assistant_only_loss_masking_and_eos_supervision(tmp_path):
    """Verify that prompt tokens are masked with -100 and response ends with EOS 0."""
    tokenizer = HuggingFaceBPEWrapper.load(root_dir / "artifacts/tokenizer_16k_v2")

    # Create dummy jsonl
    dummy_file = tmp_path / "dummy_sft.jsonl"
    dummy_file.write_text(
        '{"id": "test_1", "category": "qa", "instruction": "What is 2+2?", "input": "", "output": "4"}\n',
        encoding="utf-8",
    )

    dataset = SFTDataset(dummy_file, tokenizer, context_length=64)
    assert len(dataset) == 1

    item = dataset[0]
    input_ids = item["input_ids"]
    labels = item["labels"]

    assert len(input_ids) == len(labels)
    # The last token must be EOS token 0
    assert input_ids[-1] == 0
    assert labels[-1] == 0

    # Prompt tokens should be -100
    prompt_len = labels.index(tokenizer.encode("4")[0])
    assert all(l == -100 for l in labels[:prompt_len])
    # Response tokens must NOT be -100
    assert all(l != -100 for l in labels[prompt_len:])


def test_sft_collate_and_loss_backward():
    """Verify batch collation with padding and gradient backpropagation."""
    config = ModelConfig(vocab_size=16000, context_length=64, emb_dim=64, n_heads=2, n_layers=2)
    model = ModernGPT(config)
    model.train()

    batch = [
        {"input_ids": [10, 20, 30, 40], "labels": [-100, -100, 30, 40]},
        {"input_ids": [50, 60], "labels": [-100, 60]},
    ]
    inputs, targets = collate_sft_batch(batch, pad_token_id=0)

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    # Padded positions in targets must be -100
    assert targets[1, 2].item() == -100
    assert targets[1, 3].item() == -100

    loss = compute_sft_loss(model, inputs, targets)
    assert torch.isfinite(loss)
    loss.backward()

    # Gradients should exist and be finite
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()


def test_code_and_json_extractors():
    """Verify extraction of code and JSON blocks from markdown fences and XML tags."""
    # Markdown code
    md_code = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nEnjoy!"
    extracted_py = extract_code_block(md_code)
    assert "def add(a, b):" in extracted_py
    assert "Here is" not in extracted_py

    # XML tool call
    tool_call = "<tool_call>[{\"name\": \"search\", \"arguments\": {\"q\": \"apple\"}}]</tool_call>"
    extracted_js = extract_json_block(tool_call)
    assert extracted_js.startswith("[")
    assert extracted_js.endswith("]")

    # Raw JSON
    raw_js = '{"status": "ok", "count": 42}'
    assert extract_json_block(raw_js) == raw_js


def test_python_functional_tests_and_json_schema():
    """Verify execution of functional tests and schema validation on synthetic code and JSON."""
    from scripts.evaluate_sft import run_functional_test, validate_json_schema

    # Valid is_even
    code_is_even = "def is_even(n: int) -> bool:\n    return n % 2 == 0\n"
    ok, msg = run_functional_test("py_1", code_is_even)
    assert ok is True, f"Expected True, got {msg}"

    # Invalid is_even
    code_bad_even = "def is_even(n: int) -> bool:\n    return False\n"
    ok, msg = run_functional_test("py_1", code_bad_even)
    assert ok is False

    # Valid Counter
    code_counter = (
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.val = 0\n"
        "    def increment(self):\n"
        "        self.val += 1\n"
        "    def get_value(self):\n"
        "        return self.val\n"
    )
    ok, msg = run_functional_test("py_4", code_counter)
    assert ok is True

    # Valid JSON schema
    valid_data = {"name": "Alice", "age": 28, "city": "Seattle"}
    ok, msg = validate_json_schema("json_1", valid_data)
    assert ok is True

    # Missing key in JSON schema
    invalid_data = {"name": "Alice", "city": "Seattle"}
    ok, msg = validate_json_schema("json_1", invalid_data)
    assert ok is False
    assert "age" in msg

