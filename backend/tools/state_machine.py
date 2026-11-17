"""
QORIN Bounded Asynchronous Multi-Turn Orchestration State Machine
Implements the formal 9-state orchestration lifecycle with tool-call ID tracking,
bounded retries, global timeout enforcement, and rolling conversation history preservation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.tools.evidence import EvidenceBundle
from backend.tools.retrieval_providers import UnifiedWebResearcher
from backend.tools.python_contract import PythonExecutionContract, PythonExecutionResult, PythonFailureCategory


class HarnessState(str, Enum):
    RECEIVE_USER_REQUEST = "RECEIVE_USER_REQUEST"
    MODEL_DECISION = "MODEL_DECISION"
    VALIDATE_TOOL_CALL = "VALIDATE_TOOL_CALL"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    APPEND_TOOL_OBSERVATION = "APPEND_TOOL_OBSERVATION"
    MODEL_FOLLOWUP_DECISION = "MODEL_FOLLOWUP_DECISION"
    FINAL_SYNTHESIS = "FINAL_SYNTHESIS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    raw_snippet: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolObservation:
    call_id: str
    tool_name: str
    output_text: str
    status: str  # "SUCCESS" | "FAILED" | "TIMEOUT"
    duration_ms: float
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionConfig:
    max_tool_calls: int = 4
    max_transient_retries: int = 2
    max_schema_retries: int = 1
    global_timeout_seconds: float = 30.0
    enable_web: bool = True
    enable_python: bool = True


@dataclass
class ConversationTurn:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionTrace:
    session_id: str
    user_request: str
    states_traversed: List[str] = field(default_factory=list)
    turns: List[ConversationTurn] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[ToolObservation] = field(default_factory=list)
    final_response: Optional[str] = None
    total_duration_ms: float = 0.0
    status: str = "INITIALIZED"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_request": self.user_request,
            "states_traversed": self.states_traversed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "status": self.status,
            "error_message": self.error_message,
            "final_response": self.final_response,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "observations": [o.to_dict() for o in self.observations],
            "turns": [t.to_dict() for t in self.turns]
        }


class MultiTurnStateMachine:
    """Orchestrates bounded multi-turn tool workflows across 9 explicit states."""

    def __init__(
        self,
        web_researcher: Optional[UnifiedWebResearcher] = None,
        python_contract: Optional[PythonExecutionContract] = None,
        config: Optional[SessionConfig] = None
    ):
        self.web_researcher = web_researcher or UnifiedWebResearcher()
        self.python_contract = python_contract or PythonExecutionContract()
        self.config = config or SessionConfig()

    @staticmethod
    def parse_tool_call(text: str, call_index: int) -> Optional[ToolCall]:
        """Parses model-generated tool calls with canonical and legacy syntax, assigning unique call ID."""
        patterns = [
            r"<\|tool_call\|>(.*?)(?:<\|im_end\|>|$)",
            r"<tool_call[^>]*>(.*?)</tool_call>",
            r"<\|tool_call\|>(.*)",
            r"<tool_call[^>]*>(.*)"
        ]
        raw_json_str = None
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                raw_json_str = m.group(1).strip()
                break

        if not raw_json_str:
            # Check bare json containing "name" and "arguments"
            m_bare = re.search(r'(\{"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*\}\})', text, re.DOTALL)
            if m_bare:
                raw_json_str = m_bare.group(1).strip()
            else:
                m_bare2 = re.search(r'(\{"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\})', text, re.DOTALL)
                if m_bare2:
                    raw_json_str = m_bare2.group(1).strip()

        if not raw_json_str:
            return None

        try:
            data = json.loads(raw_json_str, strict=False)
            tool_name = data.get("name")
            args = data.get("arguments", {})
            if isinstance(tool_name, str) and isinstance(args, dict):
                return ToolCall(
                    call_id=f"call_{call_index:03d}",
                    tool_name=tool_name.strip().lower(),
                    arguments=args,
                    raw_snippet=raw_json_str
                )
        except Exception:
            pass

        return None

    def execute_session(
        self,
        user_request: str,
        system_prompt: str,
        model_generator: Callable[[str, List[ConversationTurn]], str],
        session_id: Optional[str] = None
    ) -> SessionTrace:
        """
        Executes complete multi-turn conversation session driven by model generator callback.
        model_generator: func(current_prompt_text, conversation_turns) -> model_output_string.
        """
        t0_session = time.perf_counter()
        sid = session_id or f"sess_{int(t0_session * 1000)}"
        trace = SessionTrace(session_id=sid, user_request=user_request)

        # 1. State: RECEIVE_USER_REQUEST
        state = HarnessState.RECEIVE_USER_REQUEST
        trace.states_traversed.append(state.value)

        trace.turns.append(ConversationTurn(role="system", content=system_prompt))
        trace.turns.append(ConversationTurn(role="user", content=user_request))

        call_counter = 0
        consecutive_schema_retries = 0

        # Loop controller
        while True:
            # Check global timeout
            elapsed = time.perf_counter() - t0_session
            if elapsed >= self.config.global_timeout_seconds:
                state = HarnessState.FAILED
                trace.states_traversed.append(state.value)
                trace.status = "TIMEOUT"
                trace.error_message = f"Global execution timeout ({self.config.global_timeout_seconds}s) exceeded."
                break

            # 2. State: MODEL_DECISION or MODEL_FOLLOWUP_DECISION
            if call_counter == 0:
                state = HarnessState.MODEL_DECISION
            else:
                state = HarnessState.MODEL_FOLLOWUP_DECISION
            trace.states_traversed.append(state.value)

            # Build prompt string from turns
            prompt_str = self._render_chatml(trace.turns)
            model_output = model_generator(prompt_str, trace.turns)

            # Check if model emitted a tool call
            parsed_call = self.parse_tool_call(model_output, call_counter + 1)

            # If no tool call, transition to FINAL_SYNTHESIS / COMPLETE
            if not parsed_call:
                state = HarnessState.FINAL_SYNTHESIS
                trace.states_traversed.append(state.value)
                trace.final_response = model_output
                trace.turns.append(ConversationTurn(role="assistant", content=model_output))
                state = HarnessState.COMPLETE
                trace.states_traversed.append(state.value)
                trace.status = "COMPLETED"
                break

            # Check max tool calls budget
            if call_counter >= self.config.max_tool_calls:
                # Force final synthesis
                state = HarnessState.FINAL_SYNTHESIS
                trace.states_traversed.append(state.value)
                trace.final_response = f"Maximum tool call limit ({self.config.max_tool_calls}) reached. Final response: {model_output}"
                trace.turns.append(ConversationTurn(role="assistant", content=trace.final_response))
                state = HarnessState.COMPLETE
                trace.states_traversed.append(state.value)
                trace.status = "COMPLETED_BUDGET_REACHED"
                break

            # 3. State: VALIDATE_TOOL_CALL
            state = HarnessState.VALIDATE_TOOL_CALL
            trace.states_traversed.append(state.value)
            trace.tool_calls.append(parsed_call)
            call_counter += 1

            trace.turns.append(ConversationTurn(
                role="assistant",
                content=f"<|tool_call|>{json.dumps({'name': parsed_call.tool_name, 'arguments': parsed_call.arguments})}<|im_end|>",
                tool_call_id=parsed_call.call_id,
                tool_name=parsed_call.tool_name
            ))

            is_valid_args, validated_args, schema_err = self._validate_arguments(parsed_call)
            if not is_valid_args:
                if consecutive_schema_retries < self.config.max_schema_retries:
                    consecutive_schema_retries += 1
                    # Append error as tool observation for model self-correction
                    obs_err = ToolObservation(
                        call_id=parsed_call.call_id,
                        tool_name=parsed_call.tool_name,
                        output_text=f"SchemaValidationError: {schema_err}",
                        status="FAILED",
                        duration_ms=0.0
                    )
                    trace.observations.append(obs_err)
                    trace.turns.append(ConversationTurn(
                        role="tool",
                        content=f"<|tool_result|>\n{obs_err.output_text}\n<|end_tool_result|>",
                        tool_call_id=parsed_call.call_id,
                        tool_name=parsed_call.tool_name
                    ))
                    continue
                else:
                    state = HarnessState.FAILED
                    trace.states_traversed.append(state.value)
                    trace.status = "SCHEMA_ERROR"
                    trace.error_message = f"Exhausted schema retries: {schema_err}"
                    break

            consecutive_schema_retries = 0

            # 4. State: EXECUTE_TOOL
            state = HarnessState.EXECUTE_TOOL
            trace.states_traversed.append(state.value)

            observation = self._execute_tool(parsed_call.tool_name, validated_args, parsed_call.call_id)
            trace.observations.append(observation)

            # 5. State: APPEND_TOOL_OBSERVATION
            state = HarnessState.APPEND_TOOL_OBSERVATION
            trace.states_traversed.append(state.value)

            trace.turns.append(ConversationTurn(
                role="tool",
                content=f"<|tool_result|>\n{observation.output_text}\n<|end_tool_result|>",
                tool_call_id=observation.call_id,
                tool_name=observation.tool_name
            ))

        trace.total_duration_ms = (time.perf_counter() - t0_session) * 1000.0
        return trace

    def _render_chatml(self, turns: List[ConversationTurn]) -> str:
        pieces = []
        for t in turns:
            content = t.content
            if content.endswith("<|im_end|>"):
                content = content[:-10].rstrip()
            pieces.append(f"<|im_start|>{t.role}\n{content}<|im_end|>\n")
        pieces.append("<|im_start|>assistant\n")
        return "".join(pieces)

    def _validate_arguments(self, call: ToolCall) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        if call.tool_name in ("web_research", "web_search"):
            if not self.config.enable_web:
                return False, {}, "Web research tool is disabled."
            q = call.arguments.get("query") or call.arguments.get("q")
            if not q or not isinstance(q, str) or len(q.strip()) < 2:
                return False, {}, "Missing or invalid 'query' argument for web_research."
            return True, {"query": q.strip()}, None

        elif call.tool_name in ("python", "python_interpreter"):
            if not self.config.enable_python:
                return False, {}, "Python execution sandbox is disabled."
            code = call.arguments.get("code")
            if not code or not isinstance(code, str) or not code.strip():
                return False, {}, "Missing or invalid 'code' argument for python execution."
            # Pre-validate AST syntax
            valid_ast, fail_cat, err_msg = self.python_contract.pre_validate_ast(code)
            if not valid_ast:
                return False, {}, f"Invalid Python code: {err_msg}"
            return True, {"code": code}, None

        return False, {}, f"Unknown tool: '{call.tool_name}'."

    def _execute_tool(self, tool_name: str, args: Dict[str, Any], call_id: str) -> ToolObservation:
        t0 = time.perf_counter()
        if tool_name in ("web_research", "web_search"):
            query = args["query"]
            bundle: EvidenceBundle = self.web_researcher.search(query)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            status = "SUCCESS" if bundle.has_evidence else "FAILED"
            return ToolObservation(
                call_id=call_id,
                tool_name=tool_name,
                output_text=bundle.format_for_model(),
                status=status,
                duration_ms=duration_ms,
                raw_payload=bundle.to_dict()
            )

        elif tool_name in ("python", "python_interpreter"):
            code = args["code"]
            py_res: PythonExecutionResult = self.python_contract.execute(code)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            out_str = py_res.stdout.strip() if py_res.stdout.strip() else py_res.stderr.strip()
            return ToolObservation(
                call_id=call_id,
                tool_name=tool_name,
                output_text=out_str,
                status=py_res.status,
                duration_ms=duration_ms,
                raw_payload=py_res.to_dict()
            )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return ToolObservation(
            call_id=call_id,
            tool_name=tool_name,
            output_text=f"Unknown tool: '{tool_name}'",
            status="FAILED",
            duration_ms=duration_ms
        )
