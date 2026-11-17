"""
QORIN Canonical Tool Registry & Schema Validator
Enforces canonical tool names ('web_search', 'python', 'weather'), alias normalization,
strict argument schema validation, dispatch recording, and execution integration.
"""

from dataclasses import dataclass, field, asdict
import json
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.tools.evidence import EvidenceBundle
from backend.tools.python_contract import PythonExecutionContract, PythonExecutionResult
from backend.tools.retrieval_providers import UnifiedWebResearcher
from backend.tools.weather_provider import RealWeatherProvider, WeatherObservation


CANONICAL_TOOLS: Set[str] = {"web_search", "python", "weather"}

TOOL_ALIASES: Dict[str, str] = {
    # Web research aliases
    "web_research": "web_search",
    "web-search": "web_search",
    "search": "web_search",
    "google": "web_search",
    "bing": "web_search",
    "web": "web_search",
    # Python interpreter aliases
    "python_interpreter": "python",
    "python-interpreter": "python",
    "py": "python",
    "run_code": "python",
    "execute_python": "python",
    "python_sandbox": "python",
    # Weather tool aliases
    "get_weather": "weather",
    "weather_lookup": "weather",
    "weather_report": "weather",
    "weather_forecast": "weather",
    "fetch_weather": "weather",
}


@dataclass
class CanonicalToolCall:
    call_id: str
    original_name: str
    canonical_name: str
    arguments: Dict[str, Any]
    raw_snippet: str
    request_id: str = ""
    conversation_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalToolObservation:
    call_id: str
    tool_name: str
    original_tool_name: str
    output_text: str
    status: str  # "SUCCESS" | "FAILED" | "UNAVAILABLE"
    duration_ms: float
    execution_id: str = ""
    request_id: str = ""
    conversation_id: str = ""
    stdout: str = ""
    stderr: str = ""
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CanonicalToolRegistry:
    """Central registry and executor for all QORIN tool operations."""

    def __init__(
        self,
        web_researcher: Optional[UnifiedWebResearcher] = None,
        python_contract: Optional[PythonExecutionContract] = None,
        weather_provider: Optional[RealWeatherProvider] = None,
    ):
        self.web_researcher = web_researcher or UnifiedWebResearcher()
        self.python_contract = python_contract or PythonExecutionContract()
        self.weather_provider = weather_provider or RealWeatherProvider()

    def normalize_tool_name(self, raw_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Normalizes any tool name or alias to canonical form.
        Returns: (canonical_name: Optional[str], original_normalized: Optional[str])
        """
        if not raw_name or not isinstance(raw_name, str):
            return None, None

        cleaned = raw_name.strip().lower()
        # Direct canonical match
        if cleaned in CANONICAL_TOOLS:
            return cleaned, cleaned

        # Alias lookup
        if cleaned in TOOL_ALIASES:
            return TOOL_ALIASES[cleaned], cleaned

        return None, cleaned

    def validate_arguments(
        self,
        canonical_name: str,
        arguments: Any,
        enabled_tools: Optional[Set[str]] = None
    ) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """
        Strictly validates schema for canonical tool.
        Returns: (is_valid: bool, validated_arguments: dict, error_message: Optional[str])
        """
        if enabled_tools is not None and canonical_name not in enabled_tools:
            return False, {}, f"Tool '{canonical_name}' is currently disabled in session configuration."

        if not isinstance(arguments, dict):
            return False, {}, f"Tool arguments must be a dictionary, got {type(arguments).__name__}."

        if canonical_name == "web_search":
            q = arguments.get("query") or arguments.get("q") or arguments.get("search_query")
            if not q or not isinstance(q, str) or len(q.strip()) < 2:
                return False, {}, "Missing or invalid 'query' argument for web_search (must be a non-empty string)."
            return True, {"query": q.strip()}, None

        elif canonical_name == "python":
            code = arguments.get("code") or arguments.get("command") or arguments.get("script")
            if not code or not isinstance(code, str) or not code.strip():
                return False, {}, "Missing or invalid 'code' argument for python execution (must be non-empty code)."
            # Perform AST syntax pre-validation
            is_valid_ast, fail_cat, err_msg = self.python_contract.pre_validate_ast(code)
            if not is_valid_ast:
                return False, {"code": code}, f"Invalid Python syntax: {err_msg}"
            return True, {"code": code}, None

        elif canonical_name == "weather":
            loc = arguments.get("location") or arguments.get("city") or arguments.get("place")
            if not loc or not isinstance(loc, str) or len(loc.strip()) < 2:
                return False, {}, "Missing or invalid 'location' argument for weather (must be a non-empty location string)."
            days = arguments.get("duration_days") or arguments.get("days") or 1
            try:
                days = max(1, min(7, int(days)))
            except Exception:
                days = 1
            return True, {"location": loc.strip(), "duration_days": days}, None

        return False, {}, f"Unrecognized canonical tool name: '{canonical_name}'."

    def execute_tool(
        self,
        canonical_name: str,
        validated_args: Dict[str, Any],
        call_id: str = "call_default",
        original_name: Optional[str] = None,
        request_id: str = "",
        conversation_id: str = "",
    ) -> CanonicalToolObservation:
        """
        Executes the canonical tool in complete isolation and produces a structured observation.
        Every execution is assigned a globally unique execution_id.
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        orig_name = original_name or canonical_name

        if canonical_name == "web_search":
            query = validated_args["query"]
            clean_q = query.strip()

            # If web search query is specifically for live weather, resolve via authentic weather provider
            weather_match = re.search(r"weather\s+(?:in|of|at|for)\s+([A-Za-z\s]+)", clean_q, re.IGNORECASE)
            if weather_match:
                loc = weather_match.group(1).strip()
                w_ok, w_obs, _ = self.weather_provider.get_weather(loc)
                if w_ok and w_obs:
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    source_meta = {
                        "provider": w_obs.source,
                        "url": w_obs.source_url,
                        "location": f"{w_obs.location_name}, {w_obs.admin1}, {w_obs.country}".strip(", "),
                        "coordinates": f"{w_obs.latitude:.4f}, {w_obs.longitude:.4f}",
                        "observation_time": w_obs.observation_time
                    }
                    payload = w_obs.to_dict()
                    payload["sources"] = [{"title": w_obs.source, "url": w_obs.source_url}]
                    payload["execution_id"] = execution_id
                    payload["request_id"] = request_id
                    payload["conversation_id"] = conversation_id
                    return CanonicalToolObservation(
                        call_id=call_id,
                        tool_name="weather",
                        original_tool_name=orig_name,
                        output_text=w_obs.format_for_model(),
                        status="SUCCESS",
                        duration_ms=duration_ms,
                        execution_id=execution_id,
                        request_id=request_id,
                        conversation_id=conversation_id,
                        stdout=w_obs.format_for_model(),
                        stderr="",
                        raw_payload=payload,
                        source_metadata=source_meta
                    )

            bundle: EvidenceBundle = self.web_researcher.search(query)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            status = "SUCCESS" if bundle.has_evidence else "FAILED"
            source_meta = {
                "source_count": len(bundle.records),
                "sources": [
                    {"title": r.source_title, "url": r.source_url}
                    for r in bundle.records
                ]
            }
            payload = bundle.to_dict()
            payload["sources"] = source_meta["sources"]
            payload["execution_id"] = execution_id
            payload["request_id"] = request_id
            payload["conversation_id"] = conversation_id
            return CanonicalToolObservation(
                call_id=call_id,
                tool_name="web_search",
                original_tool_name=orig_name,
                output_text=bundle.format_for_model(),
                status=status,
                duration_ms=duration_ms,
                execution_id=execution_id,
                request_id=request_id,
                conversation_id=conversation_id,
                stdout=bundle.format_for_model(),
                stderr="",
                raw_payload=payload,
                source_metadata=source_meta
            )

        elif canonical_name == "python":
            code = validated_args["code"]
            py_res: PythonExecutionResult = self.python_contract.execute(code)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            out_str = py_res.stdout.strip() if py_res.stdout.strip() else py_res.stderr.strip()
            status = "SUCCESS" if py_res.status == "SUCCESS" else "FAILED"
            py_dict = py_res.to_dict()
            py_dict["execution_id"] = execution_id
            py_dict["request_id"] = request_id
            py_dict["conversation_id"] = conversation_id
            py_dict["stdout"] = py_res.stdout
            py_dict["stderr"] = py_res.stderr
            return CanonicalToolObservation(
                call_id=call_id,
                tool_name="python",
                original_tool_name=orig_name,
                output_text=out_str,
                status=status,
                duration_ms=duration_ms,
                execution_id=execution_id,
                request_id=request_id,
                conversation_id=conversation_id,
                stdout=py_res.stdout,
                stderr=py_res.stderr,
                raw_payload=py_dict,
                source_metadata={"exit_code": py_res.exit_code, "status": py_res.status, "failure_category": py_res.failure_category.value}
            )

        elif canonical_name == "weather":
            location = validated_args["location"]
            days = validated_args.get("duration_days", 1)
            ok, obs, err = self.weather_provider.get_weather(location, duration_days=days)
            duration_ms = (time.perf_counter() - t0) * 1000.0

            if ok and obs:
                source_meta = {
                    "provider": obs.source,
                    "url": obs.source_url,
                    "location": f"{obs.location_name}, {obs.admin1}, {obs.country}".strip(", "),
                    "coordinates": f"{obs.latitude:.4f}, {obs.longitude:.4f}",
                    "observation_time": obs.observation_time
                }
                w_dict = obs.to_dict()
                w_dict["execution_id"] = execution_id
                w_dict["request_id"] = request_id
                w_dict["conversation_id"] = conversation_id
                return CanonicalToolObservation(
                    call_id=call_id,
                    tool_name="weather",
                    original_tool_name=orig_name,
                    output_text=obs.format_for_model(),
                    status="SUCCESS",
                    duration_ms=duration_ms,
                    execution_id=execution_id,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    stdout=obs.format_for_model(),
                    stderr="",
                    raw_payload=w_dict,
                    source_metadata=source_meta
                )
            else:
                fail_msg = f"WeatherUnavailableError: {err or 'Live weather conditions could not be retrieved at this time.'}"
                fail_payload = {"error": err, "location": location, "execution_id": execution_id, "request_id": request_id, "conversation_id": conversation_id}
                return CanonicalToolObservation(
                    call_id=call_id,
                    tool_name="weather",
                    original_tool_name=orig_name,
                    output_text=fail_msg,
                    status="UNAVAILABLE",
                    duration_ms=duration_ms,
                    execution_id=execution_id,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    stdout="",
                    stderr=fail_msg,
                    raw_payload=fail_payload,
                    source_metadata={"provider": "Open-Meteo Weather API", "error": err}
                )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return CanonicalToolObservation(
            call_id=call_id,
            tool_name=canonical_name,
            original_tool_name=orig_name,
            output_text=f"Unknown tool '{canonical_name}'",
            status="FAILED",
            duration_ms=duration_ms,
            execution_id=execution_id,
            request_id=request_id,
            conversation_id=conversation_id,
            stdout="",
            stderr=f"Unknown tool '{canonical_name}'"
        )
