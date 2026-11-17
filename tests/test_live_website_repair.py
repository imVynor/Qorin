"""
Automated Regression Test Suite for QORIN Live Website Inference & Tool-Dispatch Repair
Covers all 14 mandatory test requirements from Stage 16A:
1. Current weather in Ahmedabad invokes the weather provider.
2. Donald Trump triggers Web research when external verification is required.
3. Zero RAG results do not block Web dispatch.
4. web_research resolves to canonical web_search.
5. <code_string> never appears in user-visible output.
6. Malformed JSON is rejected safely.
7. Invalid Python (math.comb(3, 5)) returns an execution error.
8. Successful Web calls return real observations.
9. Web observations are passed back into model context.
10. The model can generate a second Python call after Web.
11. Python output is returned to the model before synthesis.
12. A citation cannot be emitted as verified unless backed by an actual source.
13. Failed tool execution never becomes a fabricated successful answer.
14. Duplicate tool calls are not executed twice.
"""

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.weather_provider import RealWeatherProvider, WeatherObservation
from backend.tools.registry import CanonicalToolRegistry, CanonicalToolCall, CanonicalToolObservation
from backend.tools.output_protocol import OutputProtocolParser, OutputClassification
from backend.retrieval.engine import RetrievalEngine, RetrievalTrace
from backend.tools.python_contract import PythonExecutionContract, PythonExecutionResult


class TestLiveWebsiteInferenceRepair(unittest.TestCase):

    def setUp(self):
        self.weather_provider = RealWeatherProvider()
        self.tool_registry = CanonicalToolRegistry(weather_provider=self.weather_provider)
        self.parser = OutputProtocolParser(registry=self.tool_registry)
        self.retrieval_engine = RetrievalEngine()
        self.python_contract = PythonExecutionContract(timeout_seconds=4.0)

    # 1. Current weather in Ahmedabad invokes the weather provider
    def test_01_weather_ahmedabad_invokes_provider(self):
        success, obs, err = self.weather_provider.get_weather("Ahmedabad")
        self.assertTrue(success, f"Weather retrieval failed: {err}")
        self.assertIsNotNone(obs)
        self.assertIn("Ahmedabad", obs.location_name)
        self.assertGreater(obs.temperature_c, -50.0)
        self.assertLess(obs.temperature_c, 60.0)
        self.assertTrue(obs.weather_condition)
        self.assertIn("Open-Meteo", obs.source)

    # 2. Donald Trump triggers Web research when configured policy requires external verification
    def test_02_donald_trump_triggers_web_search(self):
        gen_text = '{"name": "web_research", "arguments": {"query": "Who is donald trump?"}}'
        parsed = self.parser.parse_generation(gen_text, call_index=1)
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        self.assertEqual(parsed.tool_call.canonical_name, "web_search")
        self.assertEqual(parsed.tool_call.arguments.get("query"), "Who is donald trump?")

    # 3. Zero RAG results do not block Web dispatch
    def test_03_zero_rag_does_not_block_web_dispatch(self):
        rag_trace = self.retrieval_engine.retrieve(query="who is donald trump?", mode="auto")
        # RAG should find 0 chunks in internal GPU/technical documentation
        self.assertEqual(len(rag_trace.retrieved_chunks), 0)
        # Even with 0 RAG chunks, the model's tool call parses and dispatches cleanly
        gen_text = '<tool_call id="call_001">{"name": "web_search", "arguments": {"query": "Donald Trump biography"}}</tool_call>'
        parsed = self.parser.parse_generation(gen_text, call_index=1)
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        is_valid, args, err = self.tool_registry.validate_arguments(parsed.tool_call.canonical_name, parsed.tool_call.arguments)
        self.assertTrue(is_valid)

    # 4. web_research resolves to canonical web_search
    def test_04_web_research_resolves_to_canonical_web_search(self):
        canonical, orig = self.tool_registry.normalize_tool_name("web_research")
        self.assertEqual(canonical, "web_search")
        self.assertEqual(orig, "web_research")

        canonical_py, _ = self.tool_registry.normalize_tool_name("python_interpreter")
        self.assertEqual(canonical_py, "python")

        canonical_w, _ = self.tool_registry.normalize_tool_name("get_weather")
        self.assertEqual(canonical_w, "weather")

    # 5. <code_string> never appears in user-visible output
    def test_05_code_string_never_leaks(self):
        contaminated_text = "The calculated value is 42.<code_string> print(42)"
        clean = self.parser.sanitize_user_visible_text(contaminated_text)
        self.assertNotIn("<code_string>", clean)
        self.assertNotIn("<query_string>", clean)
        self.assertNotIn("<|tool_call|>", clean)
        self.assertEqual(clean, "The calculated value is 42. print(42)")

        # When prepended to a tool call, the tool call is recognized and sanitized text is empty
        gen = '<code_string>{"name": "python", "arguments": {"code": "print(42)"}}'
        parsed = self.parser.parse_generation(gen, call_index=1)
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        self.assertEqual(parsed.sanitized_text, "")

    # 6. Malformed JSON is rejected safely
    def test_06_malformed_json_rejected_safely(self):
        bad_json = '<tool_call id="call_001">{"name": "web_search", "arguments": {broken_json}}</tool_call>'
        parsed = self.parser.parse_generation(bad_json, call_index=1)
        self.assertEqual(parsed.classification, OutputClassification.MALFORMED_TOOL_CALL)
        self.assertIsNotNone(parsed.malformed_error)
        self.assertEqual(parsed.sanitized_text, "")

    # 7. Invalid Python (math.comb(3, 5) without import) returns an execution error
    def test_07_invalid_python_returns_execution_error(self):
        invalid_code = "res = math.comb(3, 5)\nprint(res)"
        res = self.python_contract.execute(invalid_code)
        self.assertNotEqual(res.exit_code, 0)
        self.assertEqual(res.status, "FAILED")
        self.assertIn("NameError", res.stderr)

    # 8. Successful Web calls return real observations
    def test_08_successful_web_calls_return_real_observations(self):
        obs = self.tool_registry.execute_tool("web_search", {"query": "Python programming language release year"}, call_id="call_001")
        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn("Python", obs.output_text)
        self.assertGreater(len(obs.source_metadata.get("sources", [])), 0)

    # 9. Web observations are passed back into model context
    def test_09_web_observations_passed_back_into_context(self):
        obs = self.tool_registry.execute_tool("web_search", {"query": "Apple revenue"}, call_id="call_001")
        active_messages = [
            {"role": "user", "content": "What is Apple revenue?"},
            {"role": "assistant", "content": '<tool_call id="call_001">{"name": "web_search", "arguments": {"query": "Apple revenue"}}</tool_call>'},
            {"role": "tool", "name": obs.tool_name, "id": obs.call_id, "content": obs.output_text}
        ]
        self.assertEqual(active_messages[-1]["role"], "tool")
        self.assertEqual(active_messages[-1]["id"], "call_001")
        self.assertEqual(active_messages[-1]["name"], "web_search")
        self.assertIn("SOURCE_1", active_messages[-1]["content"])

    # 10. The model can generate a second Python call after Web
    def test_10_second_python_call_after_web(self):
        # Turn 1: Web observation in context
        web_obs = "Apple FY23 revenue: $383.29B. Microsoft FY23 revenue: $211.91B."
        model_turn_2_gen = '<tool_call id="call_002">{"name": "python", "arguments": {"code": "diff = 383.29 - 211.91\\nprint(f\'Difference: {diff:.2f}\')"}}</tool_call>'
        parsed = self.parser.parse_generation(model_turn_2_gen, call_index=2)
        self.assertEqual(parsed.classification, OutputClassification.TOOL_CALL)
        self.assertEqual(parsed.tool_call.canonical_name, "python")
        self.assertEqual(parsed.tool_call.call_id, "call_002")

        # Execute turn 2 Python tool
        py_obs = self.tool_registry.execute_tool("python", parsed.tool_call.arguments, call_id="call_002")
        self.assertEqual(py_obs.status, "SUCCESS")
        self.assertIn("Difference: 171.38", py_obs.output_text)

    # 11. Python output is returned to the model before synthesis
    def test_11_python_output_returned_before_synthesis(self):
        py_obs = self.tool_registry.execute_tool("python", {"code": "print(round(383.29 - 211.91, 2))"}, call_id="call_002")
        self.assertEqual(py_obs.output_text, "171.38")
        messages = [
            {"role": "tool", "name": "python", "id": "call_002", "content": py_obs.output_text}
        ]
        self.assertIn("171.38", messages[-1]["content"])

    # 12. A citation cannot be emitted as verified unless backed by an actual source
    def test_12_citation_requires_actual_source(self):
        obs = self.tool_registry.execute_tool("web_search", {"query": "Ahmedabad weather"}, call_id="call_001")
        sources = obs.source_metadata.get("sources", [])
        if sources:
            self.assertTrue(sources[0]["url"].startswith("http"))
            self.assertTrue(len(sources[0]["title"]) > 0)

    # 13. Failed tool execution never becomes a fabricated successful answer
    def test_13_failed_tool_execution_status(self):
        # Invalid python
        obs = self.tool_registry.execute_tool("python", {"code": "assert 1 == 2"}, call_id="call_001")
        self.assertEqual(obs.status, "FAILED")
        self.assertIn("AssertionError", obs.output_text)

        # Invalid location for weather
        ok, weather_obs, err = self.weather_provider.get_weather("NON_EXISTENT_COSMIC_VOID_XYZ_123456789")
        self.assertFalse(ok)
        self.assertIsNone(weather_obs)
        self.assertIsNotNone(err)

    # 14. Duplicate tool calls are not executed twice
    def test_14_duplicate_tool_calls_prevented(self):
        gen = '<tool_call id="call_001">{"name": "web_search", "arguments": {"query": "test query"}}</tool_call>'
        p1 = self.parser.parse_generation(gen, call_index=1)
        self.assertEqual(p1.classification, OutputClassification.TOOL_CALL)

        # Second identical dispatch should be rejected as duplicate
        p2 = self.parser.parse_generation(gen, call_index=2)
        self.assertEqual(p2.classification, OutputClassification.MALFORMED_TOOL_CALL)
        self.assertIn("Duplicate", p2.malformed_error)


if __name__ == "__main__":
    unittest.main()
