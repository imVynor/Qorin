"""
QORIN Regression Test Suite: Stage 16A.2 Python Execution Isolation & Tool Policy
Tests:
1. Python execution returns its own stdout.
2. A second execution cannot return the first execution's stdout.
3. A failed execution cannot reuse a previous successful result.
4. A new conversation cannot inherit another conversation's execution result.
5. Concurrent executions preserve result ownership.
6. Streaming cannot display a previous execution's output.
7. Code-generation requests do not trigger Python execution unnecessarily.
8. Simple arithmetic can be answered directly.
9. Explicit Python execution returns the actual current result.
10. Tool-call IDs and execution IDs remain correctly paired through synthesis.
"""

import concurrent.futures
import json
import unittest
import uuid

from backend.tools.policy import ToolPolicyClassifier, UserIntent
from backend.tools.python_contract import PythonExecutionContract, PythonExecutionResult
from backend.tools.registry import CanonicalToolRegistry, CanonicalToolObservation
from backend.tools.output_protocol import OutputProtocolParser, OutputClassification


class TestPythonExecutionIsolation(unittest.TestCase):

    def setUp(self):
        self.registry = CanonicalToolRegistry()
        self.parser = OutputProtocolParser(registry=self.registry)

    # Test 1: Python execution returns its own stdout
    def test_01_execution_returns_own_stdout(self):
        sentinel = f"SENTINEL_OWN_{uuid.uuid4().hex[:8]}"
        code = f"print('{sentinel}')"
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        obs: CanonicalToolObservation = self.registry.execute_tool(
            "python",
            {"code": code},
            call_id="call_test_001",
            request_id=req_id,
            conversation_id="conv_test_001",
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn(sentinel, obs.stdout)
        self.assertEqual(obs.output_text, sentinel)
        self.assertTrue(obs.execution_id.startswith("exec_"))
        self.assertEqual(obs.request_id, req_id)

    # Test 2: A second execution cannot return the first execution's stdout
    def test_02_second_execution_does_not_return_first_stdout(self):
        sentinel_1 = f"SENTINEL_ALPHA_{uuid.uuid4().hex[:8]}"
        sentinel_2 = f"SENTINEL_BETA_{uuid.uuid4().hex[:8]}"

        obs_1 = self.registry.execute_tool(
            "python",
            {"code": f"print('{sentinel_1}')"},
            call_id="call_001",
            request_id="req_001",
            conversation_id="conv_001",
        )
        obs_2 = self.registry.execute_tool(
            "python",
            {"code": f"print('{sentinel_2}')"},
            call_id="call_002",
            request_id="req_002",
            conversation_id="conv_002",
        )

        self.assertIn(sentinel_1, obs_1.stdout)
        self.assertNotIn(sentinel_2, obs_1.stdout)

        self.assertIn(sentinel_2, obs_2.stdout)
        self.assertNotIn(sentinel_1, obs_2.stdout)

        self.assertNotEqual(obs_1.execution_id, obs_2.execution_id)

    # Test 3: A failed execution cannot reuse a previous successful result
    def test_03_failed_execution_cannot_reuse_previous_success(self):
        sentinel_success = f"SENTINEL_SUCC_{uuid.uuid4().hex[:8]}"
        obs_success = self.registry.execute_tool(
            "python",
            {"code": f"print('{sentinel_success}')"},
            call_id="call_succ",
            request_id="req_succ",
            conversation_id="conv_succ",
        )
        self.assertEqual(obs_success.status, "SUCCESS")
        self.assertIn(sentinel_success, obs_success.stdout)

        # Execute invalid code that causes NameError
        obs_fail = self.registry.execute_tool(
            "python",
            {"code": "undefined_symbol_should_crash()"},
            call_id="call_fail",
            request_id="req_fail",
            conversation_id="conv_fail",
        )
        self.assertEqual(obs_fail.status, "FAILED")
        self.assertNotIn(sentinel_success, obs_fail.stdout)
        self.assertNotIn(sentinel_success, obs_fail.output_text)
        self.assertIn("NameError", obs_fail.stderr)

    # Test 4: A new conversation cannot inherit another conversation's execution result
    def test_04_new_conversation_isolation(self):
        sentinel_conv_a = f"SENTINEL_CONV_A_{uuid.uuid4().hex[:8]}"
        obs_a = self.registry.execute_tool(
            "python",
            {"code": f"print('{sentinel_conv_a}')"},
            call_id="call_a",
            request_id="req_a",
            conversation_id="conv_a",
        )

        # Conversation B executes distinct code
        sentinel_conv_b = f"SENTINEL_CONV_B_{uuid.uuid4().hex[:8]}"
        obs_b = self.registry.execute_tool(
            "python",
            {"code": f"print('{sentinel_conv_b}')"},
            call_id="call_b",
            request_id="req_b",
            conversation_id="conv_b",
        )

        self.assertEqual(obs_a.conversation_id, "conv_a")
        self.assertEqual(obs_b.conversation_id, "conv_b")
        self.assertNotIn(sentinel_conv_a, obs_b.output_text)
        self.assertNotIn(sentinel_conv_b, obs_a.output_text)

    # Test 5: Concurrent executions preserve result ownership
    def test_05_concurrent_executions_preserve_ownership(self):
        num_threads = 5
        sentinels = [f"SENTINEL_THREAD_{i}_{uuid.uuid4().hex[:6]}" for i in range(num_threads)]

        def worker(idx):
            s = sentinels[idx]
            req_id = f"req_thr_{idx}"
            conv_id = f"conv_thr_{idx}"
            return self.registry.execute_tool(
                "python",
                {"code": f"print('{s}')"},
                call_id=f"call_{idx}",
                request_id=req_id,
                conversation_id=conv_id,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        for i, res in enumerate(results):
            expected = sentinels[i]
            self.assertEqual(res.status, "SUCCESS")
            self.assertEqual(res.stdout.strip(), expected)
            self.assertEqual(res.conversation_id, f"conv_thr_{i}")
            # Ensure no other sentinel exists in stdout
            for j, other_s in enumerate(sentinels):
                if i != j:
                    self.assertNotIn(other_s, res.stdout)

        # Ensure all execution IDs are strictly unique
        exec_ids = [r.execution_id for r in results]
        self.assertEqual(len(set(exec_ids)), num_threads)

    # Test 6: Streaming cannot display a previous execution's output
    def test_06_streaming_sanitization_and_isolation(self):
        # Verify OutputProtocolParser strips stale markers and disallows leaking
        raw_output_with_markers = (
            "The Python execution yielded: `Combinations: 0`.\n<code_string>\n<|tool_call|>"
        )
        cleaned = self.parser.sanitize_user_visible_text(raw_output_with_markers)
        self.assertNotIn("<code_string>", cleaned)
        self.assertNotIn("<|tool_call|>", cleaned)

        # Verify parsed output classification
        parsed = self.parser.parse_generation(
            '{"name": "python", "arguments": {"code": "print(10)"}}',
            call_index=1,
            request_id="req_stream_test"
        )
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        self.assertEqual(parsed.sanitized_text, "")  # Never show raw tool call to user

    # Test 7: Code-generation requests do not trigger Python execution unnecessarily
    def test_07_code_generation_intent_classification(self):
        code_prompts = [
            "Write a Python function that adds two numbers.",
            "rite a python function that adds two numbers",
            "write a python function to reverse a string",
            "implement a python function that checks for prime numbers",
            "how to write a python script to parse json",
        ]
        for prompt in code_prompts:
            intent, detail = ToolPolicyClassifier.classify_intent(prompt)
            self.assertEqual(
                intent,
                UserIntent.CODE_GENERATION,
                f"Failed to identify code generation for: '{prompt}'"
            )
            # Verify code synthesizer returns markdown code block, not tool call
            resp = ToolPolicyClassifier.synthesize_code_response(prompt)
            self.assertTrue(resp.startswith("```python\n"), f"Expected markdown code block, got: {resp}")
            self.assertIn("def ", resp)
            self.assertNotIn("<tool_call", resp)

    # Test 8: Simple arithmetic can be answered directly
    def test_08_simple_arithmetic_intent_and_direct_eval(self):
        arith_prompts = [
            ("whats 1+2?", "3"),
            ("what's 1+2?", "3"),
            ("what is 2 + 2", "4"),
            ("calculate 10 * 5", "50"),
            ("100 / 4", "25"),
            ("15 - 7", "8"),
        ]
        for prompt, expected_val in arith_prompts:
            intent, detail = ToolPolicyClassifier.classify_intent(prompt)
            self.assertEqual(
                intent,
                UserIntent.SIMPLE_ARITHMETIC,
                f"Failed to identify simple arithmetic for: '{prompt}'"
            )
            self.assertEqual(detail, expected_val)

    # Test 9: Explicit Python execution returns the actual current result
    def test_09_explicit_execution_returns_actual_result(self):
        exec_prompts = [
            "Run Python to calculate the average of these values: [10, 20, 30]",
            "Run python code to print FIRST_12345",
            "Execute python: print(2 ** 10)",
        ]
        for prompt in exec_prompts:
            intent, _ = ToolPolicyClassifier.classify_intent(prompt)
            self.assertEqual(
                intent,
                UserIntent.EXPLICIT_EXECUTION,
                f"Failed to identify explicit execution for: '{prompt}'"
            )

        # Actual execution check
        obs = self.registry.execute_tool(
            "python",
            {"code": "vals = [10, 20, 30]\nprint(sum(vals) / len(vals))"},
            call_id="call_avg",
            request_id="req_avg",
            conversation_id="conv_avg",
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.stdout.strip(), "20.0")

    # Test 10: Tool-call IDs and execution IDs remain correctly paired through synthesis
    def test_10_call_id_and_execution_id_pairing(self):
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        conv_id = f"conv_{uuid.uuid4().hex[:8]}"

        raw_gen = '<tool_call id="call_test">{"name": "python", "arguments": {"code": "print(42)"}}</tool_call>'
        parsed = self.parser.parse_generation(
            raw_gen,
            call_index=1,
            request_id=req_id,
            conversation_id=conv_id,
        )
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        tool_call = parsed.tool_call
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call.request_id, req_id)
        self.assertEqual(tool_call.conversation_id, conv_id)
        self.assertIn(req_id[:8], tool_call.call_id)

        obs = self.registry.execute_tool(
            tool_call.canonical_name,
            tool_call.arguments,
            call_id=tool_call.call_id,
            request_id=req_id,
            conversation_id=conv_id,
        )
        self.assertEqual(obs.call_id, tool_call.call_id)
        self.assertEqual(obs.request_id, req_id)
        self.assertEqual(obs.conversation_id, conv_id)
        self.assertTrue(obs.execution_id.startswith("exec_"))
        self.assertEqual(obs.raw_payload["execution_id"], obs.execution_id)
        self.assertEqual(obs.raw_payload["request_id"], req_id)


if __name__ == "__main__":
    unittest.main()
