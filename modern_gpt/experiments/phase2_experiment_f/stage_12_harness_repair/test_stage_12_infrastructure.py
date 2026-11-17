"""
Comprehensive Unit & Integration Test Suite for Stage 12 Infrastructure
Tests:
1. Orchestration State Machine (all 9 states, call IDs, retries, limits, conversation history)
2. Retrieval Providers & SQLite Cache (cache hit/miss, mock fixture, metadata preservation)
3. Python Contract (AST validation, math/statistics operations, error categorization, timeouts)
4. End-to-End Scripted Integration (Web -> Python multi-turn workflow)
"""

import json
import os
from pathlib import Path
import sys
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.evidence import EvidenceRecord, EvidenceBundle
from backend.tools.retrieval_providers import (
    BaseRetrievalProvider, SQLiteCacheManager, MockFixtureProvider, UnifiedWebResearcher
)
from backend.tools.python_contract import (
    PythonExecutionContract, PythonExecutionResult, PythonFailureCategory
)
from backend.tools.state_machine import (
    MultiTurnStateMachine, HarnessState, ToolCall, ToolObservation, SessionConfig, SessionTrace, ConversationTurn
)


class TestStage12Infrastructure(unittest.TestCase):

    def setUp(self):
        self.test_cache_db = Path(__file__).resolve().parent / "test_retrieval_cache.sqlite"
        self.cache_mgr = SQLiteCacheManager(self.test_cache_db)
        self.cache_mgr.clear()

        # Setup mock retrieval fixture
        self.mock_provider = MockFixtureProvider()
        self.mock_provider.register_fixture(
            "Apple and Microsoft FY23 revenue",
            [
                {"title": "Apple FY23 Earnings", "url": "https://apple.com/sec/fy23", "snippet": "Apple FY2023 revenue reached $383.29B."},
                {"title": "Microsoft FY23 Results", "url": "https://microsoft.com/sec/fy23", "snippet": "Microsoft FY2023 revenue reached $211.91B."}
            ]
        )
        self.web_researcher = UnifiedWebResearcher(
            provider=self.mock_provider,
            cache_manager=self.cache_mgr,
            enable_cache=True
        )
        self.python_contract = PythonExecutionContract(timeout_seconds=4.0)
        self.state_machine = MultiTurnStateMachine(
            web_researcher=self.web_researcher,
            python_contract=self.python_contract,
            config=SessionConfig(max_tool_calls=4, global_timeout_seconds=10.0)
        )

    def tearDown(self):
        try:
            self.cache_mgr.clear()
        except Exception:
            pass

    # =========================================================================
    # 1. RETRIEVAL & CACHE TESTS
    # =========================================================================

    def test_mock_fixture_retrieval_success(self):
        bundle = self.web_researcher.search("Apple and Microsoft FY23 revenue")
        self.assertTrue(bundle.has_evidence)
        self.assertEqual(len(bundle.records), 2)
        self.assertIn("383.29B", bundle.records[0].source_snippet)
        self.assertEqual(bundle.records[0].evidence_id, "SOURCE_1")
        self.assertFalse(bundle.records[0].is_cached)

    def test_sqlite_cache_hit_and_miss(self):
        q = "Apple and Microsoft FY23 revenue"
        # First query: Cache Miss, populated
        b1 = self.web_researcher.search(q)
        self.assertFalse(b1.records[0].is_cached)

        # Second query: Cache Hit
        b2 = self.web_researcher.search(q)
        self.assertTrue(b2.has_evidence)
        self.assertTrue(b2.records[0].is_cached)
        self.assertEqual(b2.records[0].source_url, "https://apple.com/sec/fy23")

    def test_empty_retrieval_handling(self):
        bundle = self.web_researcher.search("Unknown non-existent entity 123456")
        self.assertFalse(bundle.has_evidence)
        self.assertIsNotNone(bundle.error)

    # =========================================================================
    # 2. PYTHON EXECUTION CONTRACT TESTS
    # =========================================================================

    def test_python_valid_math_operations(self):
        code = """
import math
import statistics

r1 = math.isqrt(144)
r2 = math.comb(10, 3)
r3 = statistics.mean([10, 20, 30, 40])
print(f"isqrt={r1}, comb={r2}, mean={r3}")
"""
        res = self.python_contract.execute(code)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.exit_code, 0)
        self.assertIn("isqrt=12, comb=120, mean=25", res.stdout)
        self.assertEqual(res.failure_category, PythonFailureCategory.NONE)

    def test_python_syntax_error_pre_validation(self):
        code = "def broken_func(:\n    pass"
        res = self.python_contract.execute(code)
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, PythonFailureCategory.SYNTAX_ERROR)
        self.assertIn("SyntaxError", res.stderr)

    def test_python_unsupported_api_classification(self):
        code = "import math\nres = math.prime(17)\nprint(res)"
        res = self.python_contract.execute(code)
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, PythonFailureCategory.UNSUPPORTED_API)
        self.assertIn("has no attribute 'prime'", res.stderr)

    def test_python_runtime_exception_classification(self):
        code = "x = 100 / 0"
        res = self.python_contract.execute(code)
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, PythonFailureCategory.RUNTIME_EXCEPTION)
        self.assertIn("ZeroDivisionError", res.stderr)

    def test_python_security_disallowed_module(self):
        code = "import os\nos.listdir('.')"
        res = self.python_contract.execute(code)
        self.assertEqual(res.status, "FAILED")
        self.assertEqual(res.failure_category, PythonFailureCategory.SECURITY_VIOLATION)
        self.assertIn("SecurityViolation", res.stderr)

    def test_python_timeout_enforcement(self):
        contract = PythonExecutionContract(timeout_seconds=0.5)
        code = "import time\ntime.sleep(2.0)"
        res = contract.execute(code)
        self.assertEqual(res.status, "TIMEOUT")
        self.assertEqual(res.failure_category, PythonFailureCategory.TIMEOUT)

    # =========================================================================
    # 3. ORCHESTRATION STATE MACHINE TESTS (SCRIPTED INTEGRATION)
    # =========================================================================

    def test_direct_answer_workflow(self):
        def mock_direct(prompt, turns):
            return "The capital of France is Paris."

        trace = self.state_machine.execute_session(
            user_request="What is the capital of France?",
            system_prompt="Answer directly.",
            model_generator=mock_direct
        )
        self.assertEqual(trace.status, "COMPLETED")
        self.assertEqual(len(trace.tool_calls), 0)
        self.assertIn(HarnessState.RECEIVE_USER_REQUEST.value, trace.states_traversed)
        self.assertIn(HarnessState.MODEL_DECISION.value, trace.states_traversed)
        self.assertIn(HarnessState.FINAL_SYNTHESIS.value, trace.states_traversed)
        self.assertIn(HarnessState.COMPLETE.value, trace.states_traversed)
        self.assertEqual(trace.final_response, "The capital of France is Paris.")

    def test_single_web_workflow(self):
        turn_step = 0
        def mock_web(prompt, turns):
            nonlocal turn_step
            turn_step += 1
            if turn_step == 1:
                return '<|tool_call|>{"name": "web_research", "arguments": {"query": "Apple and Microsoft FY23 revenue"}}'
            return "Apple generated $383.29B while Microsoft generated $211.91B."

        trace = self.state_machine.execute_session(
            user_request="Look up Apple and Microsoft revenue.",
            system_prompt="Use web research when needed.",
            model_generator=mock_web
        )
        self.assertEqual(trace.status, "COMPLETED")
        self.assertEqual(len(trace.tool_calls), 1)
        self.assertEqual(trace.tool_calls[0].call_id, "call_001")
        self.assertEqual(len(trace.observations), 1)
        self.assertEqual(trace.observations[0].call_id, "call_001")
        self.assertEqual(trace.observations[0].status, "SUCCESS")
        self.assertIn("383.29B", trace.observations[0].output_text)
        self.assertIn(HarnessState.EXECUTE_TOOL.value, trace.states_traversed)
        self.assertIn(HarnessState.APPEND_TOOL_OBSERVATION.value, trace.states_traversed)

    def test_multi_turn_web_to_python_chaining(self):
        """Tests complete 3-turn workflow: Web -> Observation -> Python -> Observation -> Synthesis."""
        turn_step = 0
        def mock_multistep(prompt, turns):
            nonlocal turn_step
            turn_step += 1
            if turn_step == 1:
                # Turn 1: Model requests web research
                return '<|tool_call|>{"name": "web_research", "arguments": {"query": "Apple and Microsoft FY23 revenue"}}'
            elif turn_step == 2:
                # Turn 2: Model inspects web observation and calls Python to calculate difference
                last_turn = turns[-1]
                self.assertEqual(last_turn.role, "tool")
                self.assertEqual(last_turn.tool_call_id, "call_001")
                self.assertIn("383.29B", last_turn.content)
                return '<|tool_call|>{"name": "python", "arguments": {"code": "diff = 383.29 - 211.91\\nprint(f\'Difference: {diff:.2f}\')"}}'
            else:
                # Turn 3: Model receives calculation output and synthesizes grounded answer
                last_turn = turns[-1]
                self.assertEqual(last_turn.role, "tool")
                self.assertEqual(last_turn.tool_call_id, "call_002")
                self.assertIn("Difference: 171.38", last_turn.content)
                return "Apple exceeded Microsoft by $171.38 billion in FY2023 [call_001][call_002]."

        trace = self.state_machine.execute_session(
            user_request="Find Apple and Microsoft FY23 revenue, then calculate the difference.",
            system_prompt="Tool assistant.",
            model_generator=mock_multistep
        )
        self.assertEqual(trace.status, "COMPLETED")
        self.assertEqual(len(trace.tool_calls), 2)
        self.assertEqual(trace.tool_calls[0].call_id, "call_001")
        self.assertEqual(trace.tool_calls[0].tool_name, "web_research")
        self.assertEqual(trace.tool_calls[1].call_id, "call_002")
        self.assertEqual(trace.tool_calls[1].tool_name, "python")

        self.assertEqual(len(trace.observations), 2)
        self.assertEqual(trace.observations[0].call_id, "call_001")
        self.assertEqual(trace.observations[1].call_id, "call_002")

        self.assertIn("Difference: 171.38", trace.observations[1].output_text)
        self.assertIn("171.38 billion", trace.final_response)

        # Verify states traversed
        expected_states = [
            HarnessState.RECEIVE_USER_REQUEST.value,
            HarnessState.MODEL_DECISION.value,
            HarnessState.VALIDATE_TOOL_CALL.value,
            HarnessState.EXECUTE_TOOL.value,
            HarnessState.APPEND_TOOL_OBSERVATION.value,
            HarnessState.MODEL_FOLLOWUP_DECISION.value,
            HarnessState.VALIDATE_TOOL_CALL.value,
            HarnessState.EXECUTE_TOOL.value,
            HarnessState.APPEND_TOOL_OBSERVATION.value,
            HarnessState.MODEL_FOLLOWUP_DECISION.value,
            HarnessState.FINAL_SYNTHESIS.value,
            HarnessState.COMPLETE.value,
        ]
        self.assertEqual(trace.states_traversed, expected_states)

    def test_schema_error_recovery_retry(self):
        step = 0
        def mock_retry(prompt, turns):
            nonlocal step
            step += 1
            if step == 1:
                # Invalid Python code
                return '<|tool_call|>{"name": "python", "arguments": {"code": "def bad_syntax(:"}}'
            elif step == 2:
                # Model inspects error and self-corrects
                last_turn = turns[-1]
                self.assertEqual(last_turn.role, "tool")
                self.assertIn("SchemaValidationError", last_turn.content)
                return '<|tool_call|>{"name": "python", "arguments": {"code": "print(2 + 2)"}}'
            return "The answer is 4."

        trace = self.state_machine.execute_session(
            user_request="Calculate 2+2 with error retry.",
            system_prompt="Assistant.",
            model_generator=mock_retry
        )
        self.assertEqual(trace.status, "COMPLETED")
        self.assertEqual(len(trace.tool_calls), 2)
        self.assertEqual(trace.final_response, "The answer is 4.")

    def test_max_call_limit_enforcement(self):
        def infinite_loop_model(prompt, turns):
            return '<|tool_call|>{"name": "python", "arguments": {"code": "print(1)"}}'

        sm = MultiTurnStateMachine(
            web_researcher=self.web_researcher,
            python_contract=self.python_contract,
            config=SessionConfig(max_tool_calls=2)
        )
        trace = sm.execute_session(
            user_request="Trigger loop",
            system_prompt="Assistant",
            model_generator=infinite_loop_model
        )
        self.assertEqual(trace.status, "COMPLETED_BUDGET_REACHED")
        self.assertEqual(len(trace.tool_calls), 2)
        self.assertIn("Maximum tool call limit (2) reached", trace.final_response)


if __name__ == "__main__":
    unittest.main()
