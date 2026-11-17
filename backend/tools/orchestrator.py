import ast
from dataclasses import dataclass, field
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from backend.tools.python_sandbox import PythonSandbox, PythonSandboxResult
from backend.tools.web_researcher import WebResearcher, WebResearchResult


@dataclass
class ToolExecutionTrace:
    tool_name: str
    requested_by: str  # "model" | "backend_router"
    status: str  # "completed" | "failed" | "disabled" | "timeout"
    input_payload: Dict[str, Any]
    output_payload: Dict[str, Any]
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "requested_by": self.requested_by,
            "status": self.status,
            "input_payload": self.input_payload,
            "output_payload": self.output_payload,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


class ToolOrchestrator:
    def __init__(self):
        self.sandbox = PythonSandbox()
        self.web_researcher = WebResearcher()

    def parse_model_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Parses structured tool calls emitted by model (canonical <|tool_call|> or legacy <tool_call>)."""
        raw_snippet = None

        # 1. Canonical token format: <|tool_call|>{...}(<|im_end|> or </tool_call> or <|endoftext|> or EOF)
        match_canonical = re.search(r"<\|tool_call\|>\s*(.*?)\s*(?:<\|im_end\|>|<\|endoftext\|>|</tool_call>|$)", text, re.DOTALL)
        if match_canonical and match_canonical.group(1):
            raw_snippet = match_canonical.group(1).strip()

        # 2. XML tag format: <tool_call>{...}</tool_call> or <tool_call id="...">...</tool_call>
        if not raw_snippet:
            match_legacy = re.search(r"<tool_call[^>]*>(.*?)</tool_call>", text, re.DOTALL)
            if match_legacy:
                raw_snippet = match_legacy.group(1).strip()

        if not raw_snippet:
            return None

        # Clean markdown code fences if present (e.g. ```json ... ```)
        if "```" in raw_snippet:
            clean_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_snippet)
            if clean_match:
                raw_snippet = clean_match.group(1).strip()

        data = None
        # Try standard JSON parsing
        try:
            data = json.loads(raw_snippet, strict=False)
        except Exception:
            # Fallback to ast.literal_eval for single-quoted Python dict representations
            try:
                data = ast.literal_eval(raw_snippet)
            except Exception:
                pass

        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if isinstance(data, dict):
            name = data.get("name")
            args = data.get("arguments")
            if not isinstance(args, dict):
                # Handle flattened schema drift: {"name": "...", "query": "..."} or {"name": "...", "code": "..."}
                args = {k: v for k, v in data.items() if k not in ["name", "tool", "action"]}
            if name:
                return str(name).strip(), args
            elif "query" in args or "code" in args:
                # Missing explicit tool name: infer from argument presence
                inferred_name = "web_research" if "query" in args else "python"
                return inferred_name, args

        return None

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        requested_by: str = "model",
        web_enabled: bool = True,
        python_enabled: bool = True,
    ) -> ToolExecutionTrace:
        start_t = time.perf_counter()

        if tool_name in ["python", "python_interpreter", "run_code"]:
            if not python_enabled:
                return ToolExecutionTrace(
                    tool_name="python",
                    requested_by=requested_by,
                    status="disabled",
                    input_payload=arguments,
                    output_payload={"error": "Python execution is disabled by user setting."},
                    latency_ms=0.0,
                    error="Tool execution blocked by permission control.",
                )

            code = arguments.get("code", "")
            res: PythonSandboxResult = self.sandbox.execute(code)
            elapsed = (time.perf_counter() - start_t) * 1000.0

            return ToolExecutionTrace(
                tool_name="python",
                requested_by=requested_by,
                status="failed" if (res.exit_code != 0 or res.timed_out) else "completed",
                input_payload={"code": code},
                output_payload=res.to_dict(),
                latency_ms=elapsed,
                error=res.error,
            )

        elif tool_name in ["web_research", "web_search", "search"]:
            if not web_enabled:
                return ToolExecutionTrace(
                    tool_name="web_research",
                    requested_by=requested_by,
                    status="disabled",
                    input_payload=arguments,
                    output_payload={"error": "Web research is disabled by user setting."},
                    latency_ms=0.0,
                    error="Tool execution blocked by permission control.",
                )

            query = arguments.get("query", "")
            res: WebResearchResult = self.web_researcher.research(query)
            elapsed = (time.perf_counter() - start_t) * 1000.0

            return ToolExecutionTrace(
                tool_name="web_research",
                requested_by=requested_by,
                status="failed" if res.error else "completed",
                input_payload={"query": query},
                output_payload=res.to_dict(),
                latency_ms=elapsed,
                error=res.error,
            )

        else:
            return ToolExecutionTrace(
                tool_name=tool_name,
                requested_by=requested_by,
                status="failed",
                input_payload=arguments,
                output_payload={"error": f"Unknown tool: {tool_name}"},
                latency_ms=0.0,
                error=f"Unrecognized tool identifier: {tool_name}",
            )

