"""
Application-Level Harness Regression Test Suite
Verifies:
1. Standalone mode: no manifest, no execution.
2. Both tools enabled: correct manifest and complete closed loop.
3. Python only: Web Researcher unavailable.
4. Web Researcher only: Python unavailable.
5. No-tool prompt: direct answer, no dispatch.
6. Python-required prompt: actual execution followed by model synthesis.
7. Web-required prompt: actual retrieval followed by model synthesis.
8. Tool execution error: model receives the error and responds appropriately.
9. Malformed tool call: rejected safely.
10. Maximum tool-round limit: enforced.
"""

import json
from pathlib import Path
import sys
import unittest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import app, model_manager, get_compact_tool_manifest
from backend.models.base import GenerationConfig, ModelMetadata, ModelProvider, StreamChunk

class MockTwoHopModelProvider(ModelProvider):
    """Deterministic mock provider that simulates tool-call on hop 1 and synthesis on hop 2."""
    def __init__(self, tool_to_call="python", is_malformed=False, infinite_loop=False, error_recovery=False):
        self.tool_to_call = tool_to_call
        self.is_malformed = is_malformed
        self.infinite_loop = infinite_loop
        self.error_recovery = error_recovery
        self.captured_prompts = []

    def list_models(self):
        return [ModelMetadata(
            id="test-mock-model",
            name="Mock Model",
            provider="test",
            generation="500M",
            parameters=1,
            context_length=2048,
            architecture="Transformer",
            status="ready"
        )]

    def is_loaded(self, model_id):
        return True

    def load_model(self, model_id):
        pass

    def unload_model(self, model_id):
        pass

    def get_runtime_status(self, model_id: str):
        return {"status": "loaded", "device": "cpu", "vram_usage_mb": 0}

    async def generate_stream(self, model_id, messages, config):
        self.captured_prompts.append({"messages": list(messages), "config": config})
        last_msg = messages[-1] if messages else {}

        # If infinite loop testing
        if self.infinite_loop:
            yield StreamChunk(text='<|tool_call|>{"name": "python", "arguments": {"code": "print(1)"}}<|im_end|>', is_finished=True, token_count=12)
            return

        # Check if last message was a tool observation
        if last_msg.get("role") == "tool":
            obs_text = last_msg.get("content", "")
            if "<|tool_result|>" in obs_text:
                res_dict = json.loads(obs_text.replace("<|tool_result|>", ""))
                if "stdout" in res_dict:
                    out_val = res_dict["stdout"].strip()
                    yield StreamChunk(text=f"The calculated answer is {out_val}.", is_finished=True, token_count=8)
                    return
                elif "summary" in res_dict:
                    yield StreamChunk(text="According to verified web research, the information was confirmed.", is_finished=True, token_count=10)
                    return
            elif "<|tool_error|>" in obs_text:
                yield StreamChunk(text="I encountered a tool execution error and apologize for the failure.", is_finished=True, token_count=10)
                return

        # If malformed tool call
        if self.is_malformed:
            yield StreamChunk(text='<|tool_call|>{malformed: json code unquoted}<|im_end|>', is_finished=True, token_count=8)
            return

        # If normal direct prompt (no-tool test)
        if self.tool_to_call == "none":
            yield StreamChunk(text="Supervised learning uses labeled training data, whereas unsupervised learning finds patterns in unlabeled data.", is_finished=True, token_count=18)
            return

        # Emit tool call for Python or Web
        if self.tool_to_call == "python":
            code = "print(12345 * 6789)"
            if self.error_recovery:
                code = "10 / 0"  # will trigger ZeroDivisionError
            yield StreamChunk(text=f'<|tool_call|>{{"name": "python", "arguments": {{"code": "{code}"}}}}<|im_end|>', is_finished=True, token_count=15)
            return
        elif self.tool_to_call == "web_research":
            yield StreamChunk(text='<|tool_call|>{"name": "web_research", "arguments": {"query": "Nobel Prize in Physics 2024"}}<|im_end|>', is_finished=True, token_count=15)
            return

def parse_sse_events(response_text):
    events = []
    blocks = response_text.replace("\r\n", "\n").strip().split("\n\n")
    for block in blocks:
        if not block.strip():
            continue
        lines = block.splitlines()
        evt_type = None
        evt_data = None
        for l in lines:
            if l.startswith("event: "):
                evt_type = l[len("event: "):].strip()
            elif l.startswith("data: "):
                evt_data = l[len("data: "):].strip()
        if evt_type and evt_data:
            try:
                parsed_data = json.loads(evt_data)
            except Exception:
                parsed_data = evt_data
            events.append({"event": evt_type, "data": parsed_data})
    return events

class TestHarnessRegression(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.original_qorin = model_manager.qorin_provider

    def tearDown(self):
        model_manager.qorin_provider = self.original_qorin

    def test_01_standalone_mode(self):
        """1. Standalone mode: no manifest, no execution."""
        mock = MockTwoHopModelProvider(tool_to_call="none")
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "web_research_enabled": False,
            "python_sandbox_enabled": False,
            "retrieval_mode": "off"
        })
        self.assertEqual(resp.status_code, 200)
        events = parse_sse_events(resp.text)
        
        # Verify no manifest in config
        captured_cfg = mock.captured_prompts[0]["config"]
        self.assertIsNone(captured_cfg.system_prompt)
        
        # Verify no tool events
        tool_events = [e for e in events if e["event"] in ["tool_start", "tool_result"]]
        self.assertEqual(len(tool_events), 0)
        print("Test 01 (Standalone Mode): PASS")

    def test_02_both_tools_enabled_manifest(self):
        """2. Both tools enabled: correct manifest and complete closed loop."""
        mock = MockTwoHopModelProvider(tool_to_call="python")
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Calculate 12345 * 6789"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        self.assertEqual(resp.status_code, 200)
        events = parse_sse_events(resp.text)
        
        # Verify manifest contains both tools
        captured_cfg = mock.captured_prompts[0]["config"]
        self.assertIn("`python`", captured_cfg.system_prompt)
        self.assertIn("`web_research`", captured_cfg.system_prompt)
        
        # Verify complete closed loop: tool_start -> tool_result -> token (synthesized answer)
        event_types = [e["event"] for e in events]
        self.assertIn("tool_start", event_types)
        self.assertIn("tool_result", event_types)
        self.assertIn("token", event_types)
        self.assertIn("done", event_types)
        
        # Verify final answer synthesized
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        full_text = "".join(tokens)
        self.assertIn("83810205", full_text)
        print("Test 02 (Both Tools Manifest & Closed Loop): PASS")

    def test_03_python_only(self):
        """3. Python only: Web Researcher unavailable."""
        manifest = get_compact_tool_manifest(web_enabled=False, python_enabled=True)
        self.assertIn("`python`", manifest)
        self.assertNotIn("`web_research`", manifest)
        print("Test 03 (Python Only Manifest): PASS")

    def test_04_web_only(self):
        """4. Web Researcher only: Python unavailable."""
        manifest = get_compact_tool_manifest(web_enabled=True, python_enabled=False)
        self.assertIn("`web_research`", manifest)
        self.assertNotIn("`python`", manifest)
        print("Test 04 (Web Only Manifest): PASS")

    def test_05_no_tool_prompt(self):
        """5. No-tool prompt: direct answer, no dispatch."""
        mock = MockTwoHopModelProvider(tool_to_call="none")
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Explain supervised learning"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_events = [e for e in events if e["event"] in ["tool_start", "tool_result"]]
        self.assertEqual(len(tool_events), 0)
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        self.assertTrue(len(tokens) > 0)
        print("Test 05 (No-tool Prompt Direct Answer): PASS")

    def test_06_python_execution_and_synthesis(self):
        """6. Python-required prompt: actual execution followed by model synthesis."""
        mock = MockTwoHopModelProvider(tool_to_call="python")
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Compute 12345 * 6789"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_res = next(e["data"] for e in events if e["event"] == "tool_result")
        self.assertEqual(tool_res["status"], "completed")
        self.assertEqual(tool_res["output_payload"]["stdout"].strip(), "83810205")
        print("Test 06 (Python Execution and Synthesis): PASS")

    def test_07_web_execution_and_synthesis(self):
        """7. Web-required prompt: actual retrieval followed by model synthesis."""
        mock = MockTwoHopModelProvider(tool_to_call="web_research")
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Who won Nobel physics 2024?"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_res = next(e["data"] for e in events if e["event"] == "tool_result")
        self.assertEqual(tool_res["status"], "completed")
        self.assertTrue(len(tool_res["output_payload"]["sources"]) > 0)
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        self.assertIn("verified web research", "".join(tokens))
        print("Test 07 (Web Execution and Synthesis): PASS")

    def test_08_tool_error_recovery(self):
        """8. Tool execution error: model receives the error and responds appropriately."""
        mock = MockTwoHopModelProvider(tool_to_call="python", error_recovery=True)
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Divide by zero in python"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_res = next(e["data"] for e in events if e["event"] == "tool_result")
        self.assertEqual(tool_res["status"], "failed")
        self.assertEqual(tool_res["output_payload"]["exit_code"], 1)
        # Model second hop receives <|tool_error|> and responds with apology/diagnostic
        tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
        self.assertIn("encountered a tool execution error", "".join(tokens))
        print("Test 08 (Tool Error Handling): PASS")

    def test_09_malformed_tool_call(self):
        """9. Malformed tool call: rejected safely."""
        mock = MockTwoHopModelProvider(is_malformed=True)
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Trigger malformed"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_res = next(e["data"] for e in events if e["event"] == "tool_result")
        self.assertEqual(tool_res["status"], "failed")
        self.assertIn("Malformed", tool_res["error"])
        print("Test 09 (Malformed Tool Call Rejected Safely): PASS")

    def test_10_maximum_tool_round_limit(self):
        """10. Maximum tool-round limit: enforced."""
        mock = MockTwoHopModelProvider(infinite_loop=True)
        model_manager.qorin_provider = mock

        resp = self.client.post("/api/chat/stream", json={
            "model_id": "test-mock-model",
            "messages": [{"role": "user", "content": "Trigger infinite loop"}],
            "web_research_enabled": True,
            "python_sandbox_enabled": True,
            "retrieval_mode": "off"
        })
        events = parse_sse_events(resp.text)
        tool_starts = [e for e in events if e["event"] == "tool_start"]
        # Enforced limit: exactly 2 tool rounds max
        self.assertLessEqual(len(tool_starts), 2)
        done_event = next(e["data"] for e in events if e["event"] == "done")
        self.assertEqual(done_event["tools_executed"], 2)
        print("Test 10 (Maximum Tool Round Limit Enforced): PASS")

if __name__ == "__main__":
    unittest.main()
