import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import time
from backend.app import get_compact_tool_manifest
from backend.tools.orchestrator import ToolOrchestrator
from backend.tools.two_stage_protocol import QorinTwoStageProtocol
from backend.tools.python_sandbox import PythonSandbox
from backend.tools.web_researcher import WebResearcher

@pytest.fixture
def orchestrator():
    return ToolOrchestrator()

@pytest.fixture
def protocol(orchestrator):
    return QorinTwoStageProtocol(orchestrator)

# --- 1. Capability Configurations & Manifest Tests ---

def test_four_capability_configurations_manifest():
    # 1. Neither tool (OFF, OFF)
    m_none = get_compact_tool_manifest(web_enabled=False, python_enabled=False)
    assert m_none is None, "Expected None when both tools are disabled"

    # 2. Python only (OFF, ON)
    m_py = get_compact_tool_manifest(web_enabled=False, python_enabled=True)
    assert m_py is not None
    assert "python" in m_py
    assert "web_research" not in m_py

    # 3. Web only (ON, OFF)
    m_web = get_compact_tool_manifest(web_enabled=True, python_enabled=False)
    assert m_web is not None
    assert "web_research" in m_web
    assert "python" not in m_web

    # 4. Both tools (ON, ON)
    m_both = get_compact_tool_manifest(web_enabled=True, python_enabled=True)
    assert m_both is not None
    assert "python" in m_both
    assert "web_research" in m_both

# --- 2. Disabled Tool Dispatch Blocking Tests ---

def test_disabled_tool_cannot_be_dispatched_via_protocol(protocol):
    py_call = '<|tool_call|>{"name": "python", "arguments": {"code": "print(2 + 2)"}}'
    web_call = '<|tool_call|>{"name": "web_research", "arguments": {"query": "QORIN architecture"}}'

    # Configuration 1: OFF, OFF -> neither tool allowed
    action, tname, _ = protocol.determine_action(py_call, web_enabled=False, python_enabled=False)
    assert action == "DIRECT"
    assert tname is None

    action, tname, _ = protocol.determine_action(web_call, web_enabled=False, python_enabled=False)
    assert action == "DIRECT"
    assert tname is None

    # Configuration 2: OFF, ON (Python only) -> Web call blocked, NOT rerouted
    action, tname, _ = protocol.determine_action(web_call, web_enabled=False, python_enabled=True)
    assert action == "DIRECT", "Disabled Web call must not be routed or converted"
    assert tname is None

    action, tname, args = protocol.determine_action(py_call, web_enabled=False, python_enabled=True)
    assert action == "PYTHON"
    assert tname == "python"
    assert args == {"code": "print(2 + 2)"}

    # Configuration 3: ON, OFF (Web only) -> Python call blocked, NOT rerouted
    action, tname, _ = protocol.determine_action(py_call, web_enabled=True, python_enabled=False)
    assert action == "DIRECT", "Disabled Python call must not be routed or converted"
    assert tname is None

    action, tname, args = protocol.determine_action(web_call, web_enabled=True, python_enabled=False)
    assert action == "WEB"
    assert tname == "web_research"
    assert args == {"query": "QORIN architecture"}

def test_disabled_tool_blocked_at_orchestrator(orchestrator):
    # Direct orchestrator call with python disabled
    trace_py = orchestrator.execute_tool(
        "python",
        {"code": "print(10)"},
        python_enabled=False
    )
    assert trace_py.status == "disabled"
    assert "disabled" in trace_py.output_payload["error"].lower()

    # Direct orchestrator call with web disabled
    trace_web = orchestrator.execute_tool(
        "web_research",
        {"query": "ES 335"},
        web_enabled=False
    )
    assert trace_web.status == "disabled"
    assert "disabled" in trace_web.output_payload["error"].lower()

# --- 3. Argument Validation & Sanitization Tests ---

def test_python_argument_validation(protocol):
    # Valid syntax
    valid, args, err = protocol.validate_and_sanitize_arguments("python", {"code": "x = 42\nprint(x * 2)"}, "")
    assert valid is True
    assert args == {"code": "x = 42\nprint(x * 2)"}
    assert err is None

    # Syntax error
    valid, args, err = protocol.validate_and_sanitize_arguments("python", {"code": "def bad_func(: print(1)"}, "")
    assert valid is False
    assert "syntax error" in err.lower()

    # Missing code argument
    valid, args, err = protocol.validate_and_sanitize_arguments("python", {}, "")
    assert valid is False
    assert "missing 'code'" in err.lower()

def test_web_argument_validation(protocol):
    # Valid query
    valid, args, err = protocol.validate_and_sanitize_arguments("web_research", {"query": "Quantum computing basics"}, "")
    assert valid is True
    assert args == {"query": "Quantum computing basics"}
    assert err is None

    # Empty query
    valid, args, err = protocol.validate_and_sanitize_arguments("web_research", {"query": "   "}, "")
    assert valid is False
    assert "missing non-empty 'query'" in err.lower()

# --- 4. Python Sandbox Isolation & Security Tests ---

def test_python_sandbox_security_rejections():
    sandbox = PythonSandbox()

    # Test forbidden system primitives
    dangerous_snippets = [
        "import os; os.system('echo hacked')",
        "import os; os.popen('dir')",
        "import subprocess; subprocess.run(['calc.exe'])",
        "__import__('subprocess').call(['ls'])",
    ]

    for snippet in dangerous_snippets:
        res = sandbox.execute(snippet)
        assert res.exit_code != 0
        assert "SecurityViolation" in res.stderr
        assert res.error == "Security violation detected."

def test_python_sandbox_execution_correctness():
    sandbox = PythonSandbox()
    res = sandbox.execute("import math\nprint(math.gcd(48, 18))\nprint(math.comb(10, 3))")
    assert res.exit_code == 0
    assert not res.timed_out
    assert "6" in res.stdout
    assert "120" in res.stdout

def test_python_sandbox_timeout_enforcement():
    sandbox = PythonSandbox()
    start = time.perf_counter()
    res = sandbox.execute("import time\ntime.sleep(10)", timeout=1.0)
    duration = time.perf_counter() - start
    assert res.timed_out is True
    assert res.exit_code == 124
    assert duration < 3.0, f"Timeout took too long: {duration}s"
    assert "TimeoutError" in res.stderr

# --- 5. Web Retrieval Handling Tests ---

def test_web_researcher_empty_query():
    researcher = WebResearcher()
    res = researcher.research("")
    assert res.error is not None
    assert len(res.sources) == 0
    assert "No live web sources" in res.summary
