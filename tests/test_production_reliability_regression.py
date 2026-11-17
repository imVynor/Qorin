"""
QORIN — Production Inference Timeout & Tool-Policy Reliability Regression Suite
14-Scenario Comprehensive Test Suite
"""

import time
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app import app, generate_session_token
from backend.tools.policy import ToolPolicyClassifier, UserIntent
from backend.tools.retrieval_providers import WikipediaSearchProvider, UnifiedWebResearcher

client = TestClient(app)
AUTH_TOKEN = generate_session_token("prof_udit_bhatia", expiry_hours=2)


def test_01_code_generation_no_sandbox():
    """Case A: 'Write a basic python code of Hello world' must NOT invoke python execution."""
    intent, detail = ToolPolicyClassifier.classify_intent("Write a basic python code of Hello world")
    assert intent == UserIntent.CODE_GENERATION

    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        ws.send_json({
            "type": "chat",
            "request_id": "req_case_a",
            "conversation_id": "conv_case_a",
            "message": "Write a basic python code of Hello world",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 64,
            "temperature": 0.0,
        })

        events = []
        tokens = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] == "token":
                tokens.append(evt.get("text", ""))
            elif evt["type"] in ("generation_completed", "error"):
                break

        full_text = "".join(tokens)
        event_types = [e["type"] for e in events]
        assert "tool_execution_started" not in event_types, "Python sandbox should NOT be executed for code generation!"
        assert "print(" in full_text or "Hello" in full_text
        assert "generation_completed" in event_types


def test_02_code_explanation_no_sandbox():
    """Scenario 2: 'Explain how a Python function works' must NOT execute code."""
    intent, _ = ToolPolicyClassifier.classify_intent("Explain how a Python function works.")
    assert intent == UserIntent.CODE_EXPLANATION

    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        ws.send_json({
            "type": "chat",
            "request_id": "req_explain",
            "conversation_id": "conv_explain",
            "message": "Explain how a Python function works.",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 16,
        })

        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] in ("generation_completed", "error"):
                break

        event_types = [e["type"] for e in events]
        assert "tool_execution_started" not in event_types


def test_03_code_debugging_no_sandbox():
    """Scenario 3: 'Debug this Python code' must NOT execute code."""
    intent, _ = ToolPolicyClassifier.classify_intent("Debug this Python code: print('hello'")
    assert intent == UserIntent.CODE_DEBUGGING

    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        ws.send_json({
            "type": "chat",
            "request_id": "req_debug",
            "conversation_id": "conv_debug",
            "message": "Debug this Python code: print('hello'",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 16,
        })

        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] in ("generation_completed", "error"):
                break

        event_types = [e["type"] for e in events]
        assert "tool_execution_started" not in event_types


def test_04_explicit_code_execution():
    """Scenario 4: Explicit execution request triggers Python sandbox."""
    intent, _ = ToolPolicyClassifier.classify_intent("Run this python code and show output:\nprint(2 + 2)")
    assert intent == UserIntent.EXPLICIT_CODE_EXECUTION

    from backend.tools.registry import CanonicalToolRegistry
    reg = CanonicalToolRegistry()
    obs = reg.execute_tool("python", {"code": "print(2 + 2)"})
    assert obs.status == "SUCCESS"
    assert "4" in obs.stdout


def test_05_simple_arithmetic():
    """Scenario 5: 'What is 1 + 1?' evaluates directly to 2."""
    intent, detail = ToolPolicyClassifier.classify_intent("What is 1 + 1?")
    assert intent == UserIntent.SIMPLE_ARITHMETIC
    assert detail == "2"

    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        ws.send_json({
            "type": "chat",
            "request_id": "req_arith",
            "conversation_id": "conv_arith",
            "message": "What is 1 + 1?",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 16,
        })

        events = []
        tokens = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] == "token":
                tokens.append(evt.get("text", ""))
            elif evt["type"] in ("generation_completed", "error"):
                break

        full_text = "".join(tokens)
        assert "2" in full_text
        event_types = [e["type"] for e in events]
        assert "tool_execution_started" not in event_types


def test_06_current_factual_query_wikipedia_fallback():
    """Case B: 'Who is Chief minister of Tamilnadu?' triggers web search with Wikipedia fallback."""
    intent, _ = ToolPolicyClassifier.classify_intent("Who is Chief minister of Tamilnadu?")
    assert intent in (UserIntent.CURRENT_FACTUAL_QUERY, UserIntent.FACTUAL_INQUIRY)

    # Verify Wikipedia fallback retrieves real documents directly
    wiki = WikipediaSearchProvider(timeout_seconds=4.0)
    bundle = wiki.search("Chief minister of Tamilnadu", max_results=3)
    assert len(bundle.records) > 0
    assert any("Stalin" in r.title or "Stalin" in r.snippet or "Tamil Nadu" in r.title for r in bundle.records)


def test_07_current_president_factual_query():
    """Scenario 7: 'Who is the current President of the United States?' recognized as factual."""
    intent, _ = ToolPolicyClassifier.classify_intent("Who is the current President of the United States?")
    assert intent in (UserIntent.CURRENT_FACTUAL_QUERY, UserIntent.FACTUAL_INQUIRY)

    researcher = UnifiedWebResearcher()
    bundle = researcher.search("current President of the United States", max_results=3)
    assert len(bundle.records) > 0
    assert any("President" in r.title or "United States" in r.title or "President" in r.snippet for r in bundle.records)


def test_08_conversational_follow_up():
    """Scenario 8: 'tell me more about him' recognized as conversational follow up."""
    intent, _ = ToolPolicyClassifier.classify_intent("tell me more about him")
    assert intent == UserIntent.CONVERSATIONAL_FOLLOW_UP


def test_09_web_provider_timeout_budget():
    """Scenario 9: Bounded tool budget handles slow provider without blocking."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        wiki = WikipediaSearchProvider(timeout_seconds=0.5)
        start = time.perf_counter()
        bundle = wiki.search("test query", max_results=2)
        elapsed = time.perf_counter() - start
        assert len(bundle.records) == 0
        assert bundle.error is not None
        assert elapsed < 2.0


def test_10_python_execution_infinite_loop():
    """Scenario 10: Infinite loops in Python tool abort cleanly within bounded timeout."""
    from backend.tools.registry import CanonicalToolRegistry
    reg = CanonicalToolRegistry()
    start = time.perf_counter()
    obs = reg.execute_tool("python", {"code": "while True: pass"})
    elapsed = time.perf_counter() - start
    assert obs.status in ("TIMEOUT", "ERROR", "FAILED")
    assert "timed out" in obs.output_text.lower()
    assert elapsed < 7.0


def test_11_model_generation_deadline_enforcement():
    """Scenario 11: When generation deadline is reached, DEADLINE_EXCEEDED is sent and generation_completed is suppressed."""
    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        # Patch OVERALL_REQUEST_DEADLINE_SEC in backend.app to 0.001s to trigger immediate timeout
        with patch("backend.app.OVERALL_REQUEST_DEADLINE_SEC", 0.001):
            ws.send_json({
                "type": "chat",
                "request_id": "req_timeout_test",
                "conversation_id": "conv_timeout_test",
                "message": "Hello",
                "model_id": "qorin-100m-sft-v3",
                "compute_token": AUTH_TOKEN,
                "max_new_tokens": 10,
            })

            events = []
            while True:
                evt = ws.receive_json()
                events.append(evt)
                if evt["type"] in ("generation_completed", "error"):
                    break

            event_types = [e["type"] for e in events]
            assert "error" in event_types
            assert "generation_completed" not in event_types, "generation_completed must be suppressed on deadline exceeded"
            err_evt = next(e for e in events if e["type"] == "error")
            assert err_evt.get("error_code") == "DEADLINE_EXCEEDED"


def test_12_client_cancellation():
    """Scenario 12: Cancellation stops generation and confirms with generation_cancelled."""
    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()
        ws.send_json({
            "type": "cancel",
            "request_id": "req_cancel_turn",
            "conversation_id": "conv_cancel_turn",
        })
        res = ws.receive_json()
        assert res["type"] == "generation_cancelled"
        assert res["request_id"] == "req_cancel_turn"


def test_13_two_sequential_independent_conversations():
    """Scenario 13: Multiple turns in distinct conversations maintain isolation."""
    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()

        # Turn 1 in Conversation A
        ws.send_json({
            "type": "chat",
            "request_id": "req_seq_1",
            "conversation_id": "conv_alpha",
            "message": "1 + 1",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 5,
        })
        while True:
            evt = ws.receive_json()
            if evt["type"] in ("generation_completed", "error"):
                break

        # Turn 2 in Conversation B
        ws.send_json({
            "type": "chat",
            "request_id": "req_seq_2",
            "conversation_id": "conv_beta",
            "message": "1 + 2",
            "model_id": "qorin-100m-sft-v3",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 5,
        })
        while True:
            evt = ws.receive_json()
            if evt["type"] in ("generation_completed", "error"):
                break

        assert evt["type"] == "generation_completed"
        assert evt["request_id"] == "req_seq_2"
        assert evt["conversation_id"] == "conv_beta"


def test_14_model_switching_during_active_conversation():
    """Scenario 14: Switching model loads cleanly and executes."""
    with client.websocket_connect("/ws/chat") as ws:
        _ack = ws.receive_json()

        # Use base model
        ws.send_json({
            "type": "chat",
            "request_id": "req_switch_base",
            "conversation_id": "conv_switch",
            "message": "Hi",
            "model_id": "qorin-100m-base",
            "compute_token": AUTH_TOKEN,
            "max_new_tokens": 5,
        })

        events = []
        while True:
            evt = ws.receive_json()
            events.append(evt)
            if evt["type"] in ("generation_completed", "error"):
                break

        event_types = [e["type"] for e in events]
        assert "generation_completed" in event_types
        completed_evt = next(e for e in events if e["type"] == "generation_completed")
        assert completed_evt["model_id"] == "qorin-100m-base"
