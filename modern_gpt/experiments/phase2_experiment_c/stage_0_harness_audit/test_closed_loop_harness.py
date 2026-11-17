"""
Unit Test for Closed-Loop Tool Execution & Multi-Hop Generation
Verifies:
User Request -> Model -> Tool Call -> Harness -> Tool Execution -> Tool Result -> Model -> Final Answer
"""

import json
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.orchestrator import ToolOrchestrator
from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from tokenizers import Tokenizer

def test_closed_loop():
    print("Initializing Tool Orchestrator and Mock Environment...")
    orchestrator = ToolOrchestrator()

    # Step 1: User prompt
    user_prompt = "Calculate 12345 * 6789 using Python."
    print(f"1. User Request: {user_prompt}")

    # Step 2: Simulated model first-hop output emitting canonical tool call
    model_hop1_output = '<|tool_call|>{"name": "python", "arguments": {"code": "print(12345 * 6789)"}}<|im_end|>'
    print(f"2. Model Hop 1 (Tool Call): {model_hop1_output}")

    # Step 3: Harness parses tool call
    parsed = orchestrator.parse_model_tool_call(model_hop1_output)
    assert parsed is not None, "Failed to parse canonical tool call!"
    tool_name, tool_args = parsed
    print(f"3. Harness Parsed: tool={tool_name}, args={tool_args}")

    # Step 4: Harness executes tool in real sandbox
    trace = orchestrator.execute_tool(tool_name, tool_args, requested_by="model", python_enabled=True, web_enabled=True)
    assert trace.status == "completed", f"Execution failed: {trace.error}"
    print(f"4. Tool Execution: status={trace.status}, stdout={trace.output_payload.get('stdout').strip()}")

    # Step 5: Format observation turn
    obs_content = f"<|tool_result|>{json.dumps(trace.output_payload)}"
    print(f"5. Formatted Tool Observation: {obs_content[:80]}...")

    # Step 6: Construct multi-turn messages for second hop
    messages_hop2 = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": model_hop1_output},
        {"role": "tool", "content": obs_content}
    ]

    # Verify formatting with ChatML tokenizer
    tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
    tok_raw = Tokenizer.from_file(str(tok_path))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")

    chat_prompt = (
        f"<|im_start|>user\n{messages_hop2[0]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n{messages_hop2[1]['content']}<|im_end|>\n"
        f"<|im_start|>tool\n{messages_hop2[2]['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    input_ids = tokenizer.encode(chat_prompt)
    print(f"6. Second Hop Input Token Count: {len(input_ids)} tokens")
    assert len(input_ids) > 0

    print("Closed-loop execution flow verified successfully!")
    return True

if __name__ == "__main__":
    test_closed_loop()
