import ast
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from backend.tools.orchestrator import ToolOrchestrator, ToolExecutionTrace

class QorinTwoStageProtocol:
    """
    Two-Stage Action and Argument Protocol for QORIN:
    Stage A: Action Selection (DIRECT, WEB, or PYTHON) based on active capability toggles and model intent.
    Stage B: Structured Argument Generation, Schema Validation, Controlled Dispatch, and Result Interpretation.
    """

    def __init__(self, orchestrator: ToolOrchestrator):
        self.orchestrator = orchestrator

    def determine_action(
        self,
        first_round_text: str,
        web_enabled: bool,
        python_enabled: bool,
        manual_override: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
        """
        Determines the Stage A action: 'DIRECT', 'WEB', or 'PYTHON'.
        Returns (action, tool_name, parsed_arguments).
        """
        # Manual override by user takes precedence if explicitly supplied
        if manual_override and manual_override in ["PYTHON", "WEB", "DIRECT"]:
            if manual_override == "PYTHON" and python_enabled:
                return "PYTHON", "python", None
            elif manual_override == "WEB" and web_enabled:
                return "WEB", "web_research", None
            return "DIRECT", None, None

        # If neither capability is active, strict standalone inference
        if not web_enabled and not python_enabled:
            return "DIRECT", None, None

        # Check if model emitted tool-call intent
        is_tool_call = ("<tool_call" in first_round_text or "<|tool_call|>" in first_round_text)
        if not is_tool_call:
            return "DIRECT", None, None

        # Parse tool call if available
        parsed = self.orchestrator.parse_model_tool_call(first_round_text)
        if not parsed:
            # Model emitted tag but malformed JSON; attempt regex recovery
            name_match = re.search(r'["\']name["\']\s*:\s*["\'](python|web_research|web_search|search)["\']', first_round_text)
            tool_candidate = name_match.group(1) if name_match else None
        else:
            tool_candidate = parsed[0]

        # Route tool candidate with capability verification
        if tool_candidate in ["web_research", "web_search", "search"]:
            if web_enabled:
                return "WEB", "web_research", parsed[1] if parsed else None
            else:
                # Requested tool is disabled: block execution, do not dispatch or morph into another tool
                return "DIRECT", None, None
        elif tool_candidate in ["python", "python_interpreter", "run_code"]:
            if python_enabled:
                return "PYTHON", "python", parsed[1] if parsed else None
            else:
                # Requested tool is disabled: block execution, do not dispatch or morph into another tool
                return "DIRECT", None, None
        elif parsed and isinstance(parsed[1], dict):
            # Model emitted tool call with schema drift (e.g. topic/entity name as tool name)
            args = parsed[1]
            if ("query" in args or "search_query" in args) and "code" not in args:
                if web_enabled:
                    return "WEB", "web_research", args
                else:
                    return "DIRECT", None, None
            elif "code" in args and "query" not in args:
                if python_enabled:
                    return "PYTHON", "python", args
                else:
                    return "DIRECT", None, None

        # Fallback to DIRECT if permissions disallow or tool unrecognized
        return "DIRECT", None, None

    def validate_and_sanitize_arguments(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        raw_text: str,
        user_query: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Validates Stage B arguments against tool schema.
        Returns (is_valid, validated_args, error_message).
        """
        if arguments is None:
            arguments = {}

        if tool_name == "python":
            code = arguments.get("code")
            if not code or not isinstance(code, str) or not code.strip():
                # Attempt to extract code from markdown block if json was malformed
                if "```python" in raw_text:
                    code = raw_text.split("```python")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    code = raw_text.split("```")[1].split("```")[0].strip()

            if not code or not code.strip():
                return False, {}, "Python execution payload missing 'code' argument."

            # Verify Python syntax via AST
            try:
                ast.parse(code)
            except SyntaxError as syn_err:
                return False, {"code": code}, f"Python syntax error in generated code: {syn_err.msg} at line {syn_err.lineno}"

            return True, {"code": code}, None

        elif tool_name == "web_research":
            query = arguments.get("query") or arguments.get("q") or arguments.get("search_query")
            # Handle schema drift where argument key was emitted as "code" containing text query
            if not query and "code" in arguments and isinstance(arguments["code"], str):
                code_val = arguments["code"].strip()
                if "\n" not in code_val and not code_val.startswith("import ") and not code_val.startswith("def "):
                    query = code_val
            if not query or not isinstance(query, str) or not query.strip():
                # Extract query from raw string
                m = re.search(r'["\'](?:query|search_query)["\']\s*:\s*["\']([^"\']+)["\']', raw_text)
                if m:
                    query = m.group(1).strip()
                elif user_query and user_query.strip():
                    query = user_query.strip()

            if not query or not query.strip():
                return False, {}, "Web research payload missing non-empty 'query' argument."

            # Clean query
            clean_query = query.strip().strip("'\"`")
            return True, {"query": clean_query}, None

        return False, {}, f"Unsupported tool identifier: {tool_name}"
