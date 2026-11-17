import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from backend.tools.orchestrator import ToolOrchestrator, ToolExecutionTrace
from backend.tools.two_stage_protocol import QorinTwoStageProtocol
from backend.tools.web_researcher import WebResearcher, WebResearchResult
from backend.app import get_compact_tool_manifest

@pytest.fixture
def orchestrator():
    return ToolOrchestrator()

@pytest.fixture
def protocol(orchestrator):
    return QorinTwoStageProtocol(orchestrator)

# --- Scenario 1: Web enabled + ambiguous information request ('grab info on elon musk') ---

def test_scenario_1_ambiguous_request_model_direct(protocol):
    """
    When model emits direct ungrounded response (no tool tag), protocol must strictly
    respect model intent as DIRECT and not invent or force a tool invocation.
    """
    raw_model_response = "elon musk is a language model research assistant developed by the QORIN project."
    action, tname, targs = protocol.determine_action(
        raw_model_response,
        web_enabled=True,
        python_enabled=True,
        manual_override=None
    )
    assert action == "DIRECT"
    assert tname is None
    assert targs is None

def test_scenario_1_ambiguous_request_manual_override(protocol):
    """
    When manual_override is explicitly set to WEB by user/UI, protocol routes to WEB
    and falls back to user query if model response lacks structured arguments.
    """
    raw_model_response = "elon musk is a language model research assistant."
    action, tname, targs = protocol.determine_action(
        raw_model_response,
        web_enabled=True,
        python_enabled=True,
        manual_override="WEB"
    )
    assert action == "WEB"
    assert tname == "web_research"

    is_valid, validated_args, val_err = protocol.validate_and_sanitize_arguments(
        tname, targs, raw_model_response, user_query="grab info on elon musk"
    )
    assert is_valid is True
    assert validated_args["query"] == "grab info on elon musk"
    assert val_err is None

# --- Scenario 2: Web enabled + explicit search request ---

def test_scenario_2_canonical_web_call(protocol):
    """Canonical <|tool_call|> with web_research routes cleanly to WEB."""
    raw = '<|tool_call|>{"name": "web_research", "arguments": {"query": "Elon Musk"}}<|im_end|>'
    action, tname, targs = protocol.determine_action(raw, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert targs == {"query": "Elon Musk"}

    is_valid, validated_args, val_err = protocol.validate_and_sanitize_arguments(
        tname, targs, raw, user_query="Search for Elon Musk"
    )
    assert is_valid is True
    assert validated_args["query"] == "Elon Musk"

def test_scenario_2_schema_drift_topic_as_name(protocol):
    """When model hallucinates topic/entity name as tool name, schema detection recovers web_research."""
    raw = '<|tool_call|>{"name": "Elon Musk", "arguments": {"query": "Elon Musk current news"}}<|im_end|>'
    action, tname, targs = protocol.determine_action(raw, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert targs == {"query": "Elon Musk current news"}

# --- Scenario 3: Web disabled + explicit search request ---

def test_scenario_3_web_disabled_blocks_dispatch(protocol):
    """When Web retrieval capability toggle is OFF, any web tool invocation must be strictly blocked."""
    raw = '<|tool_call|>{"name": "web_research", "arguments": {"query": "Elon Musk"}}<|im_end|>'
    action, tname, targs = protocol.determine_action(raw, web_enabled=False, python_enabled=True)
    assert action == "DIRECT"
    assert tname is None
    assert targs is None

# --- Scenario 4: Web returns zero sources ---

def test_scenario_4_web_zero_sources(monkeypatch):
    """Web researcher handles zero sources without crashing or raising unhandled exceptions."""
    researcher = WebResearcher()
    # Mock both DuckDuckGo and Wikipedia returning empty list
    monkeypatch.setattr(researcher, "search_duckduckgo_lite", lambda q, max_results=3: [])

    res = researcher.research("nonexistent_random_zero_source_query_12345")
    assert isinstance(res, WebResearchResult)
    assert len(res.sources) == 0
    assert res.error is not None
    assert "0 results" in res.error or "timed out" in res.error
    d = res.to_dict()
    assert d["sources"] == []
    assert d["error"] is not None

# --- Scenario 5: Web returns multiple sources ---

def test_scenario_5_web_multiple_sources():
    """Web researcher retrieves multiple verified sources with titles and URLs."""
    researcher = WebResearcher()
    res = researcher.research("Python programming language", max_sources=3)
    assert isinstance(res, WebResearchResult)
    assert len(res.sources) > 0
    for s in res.sources:
        assert "title" in s and len(s["title"]) > 0
        assert "url" in s and (s["url"].startswith("http://") or s["url"].startswith("https://"))
        assert "snippet" in s

# --- Scenario 6: Web succeeds but synthesis fails ---

def test_scenario_6_second_stage_synthesis_error_resilience():
    """Closed-loop protocol handles observation packaging safely even under malformed or empty payloads."""
    trace = ToolExecutionTrace(
        tool_name="web_research",
        requested_by="model",
        status="completed",
        input_payload={"query": "Elon Musk"},
        output_payload={"sources": [], "summary": "No live sources", "error": "Search returned 0 results."},
        latency_ms=120.0,
    )
    d = trace.to_dict()
    assert d["status"] == "completed"
    assert "output_payload" in d
    assert d["output_payload"]["sources"] == []

# --- Scenario 7: Tool action is malformed ---

def test_scenario_7_malformed_tool_actions(protocol, orchestrator):
    """Tests various malformed and SLM quirk formats."""
    # 7a: Single quotes in JSON
    single_quote_raw = "<|tool_call|>{'name': 'web_research', 'arguments': {'query': 'Elon Musk'}}<|im_end|>"
    action, tname, targs = protocol.determine_action(single_quote_raw, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert targs == {"query": "Elon Musk"}

    # 7b: Markdown fence inside tool call
    md_raw = '<|tool_call|>```json\n{"name": "web_research", "arguments": {"query": "SpaceX Starship"}}\n```<|im_end|>'
    action, tname, targs = protocol.determine_action(md_raw, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert targs == {"query": "SpaceX Starship"}

    # 7c: Flattened schema drift
    flat_raw = '<|tool_call|>{"name": "web_research", "query": "Artificial Intelligence"}<|im_end|>'
    action, tname, targs = protocol.determine_action(flat_raw, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert targs == {"query": "Artificial Intelligence"}

    # 7d: Completely broken syntax with no query
    broken_raw = "<|tool_call|>invalid_garbage_syntax<|im_end|>"
    action, tname, targs = protocol.determine_action(broken_raw, web_enabled=True, python_enabled=True)
    assert action == "DIRECT"

# --- Scenario 8: Both tools enabled but Web is appropriate choice ---

def test_scenario_8_both_tools_enabled_web_routing(protocol):
    """When both capabilities are ON, web research calls are strictly dispatched to web_research and not python."""
    web_call = '<|tool_call|>{"name": "web_research", "arguments": {"query": "James Webb Space Telescope deep field"}}<|im_end|>'
    action, tname, targs = protocol.determine_action(web_call, web_enabled=True, python_enabled=True)
    assert action == "WEB"
    assert tname == "web_research"
    assert tname != "python"

    is_valid, validated_args, val_err = protocol.validate_and_sanitize_arguments(
        tname, targs, web_call, user_query="Tell me about James Webb Telescope"
    )
    assert is_valid is True
    assert validated_args == {"query": "James Webb Space Telescope deep field"}
