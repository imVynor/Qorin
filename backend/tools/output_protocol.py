"""
QORIN Canonical Output Protocol & Robust Parser
Distinguishes:
A. Normal assistant response.
B. Tool-call response.
C. Malformed or incomplete tool-call response.
D. Internal reasoning / intermediate protocol markers (<code_string>, raw JSON, etc.).

Safely strips internal markers so raw syntax never leaks to the frontend.
"""

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.tools.registry import CanonicalToolCall, CanonicalToolRegistry


class OutputClassification(str, Enum):
    NORMAL_ASSISTANT = "NORMAL_ASSISTANT"
    TOOL_CALL = "TOOL_CALL"
    MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
    INTERNAL_REASONING = "INTERNAL_REASONING"


# Disallowed protocol markers that must NEVER leak to user UI
DISALLOWED_MARKERS = [
    "<code_string>",
    "<query_string>",
    "<location_string>",
    "<|tool_call|>",
    "<|tool_result|>",
    "<|tool_error|>",
    "<|im_start|>",
    "<|im_end|>",
    "</tool_call>",
    "<tool_call>",
]


@dataclass
class ParsedOutput:
    classification: OutputClassification
    sanitized_text: str
    tool_call: Optional[CanonicalToolCall] = None
    malformed_error: Optional[str] = None
    raw_generation: str = ""


class OutputProtocolParser:
    """Robust parser for model generation output."""

    def __init__(self, registry: Optional[CanonicalToolRegistry] = None):
        self.registry = registry or CanonicalToolRegistry()
        self._dispatched_calls: Set[str] = set()

    def reset_session(self):
        self._dispatched_calls.clear()

    @staticmethod
    def sanitize_user_visible_text(text: str) -> str:
        """Strips all internal protocol tokens, tags, and markers from visible text."""
        cleaned = text
        for marker in DISALLOWED_MARKERS:
            cleaned = cleaned.replace(marker, "")

        # Also strip any lingering <tool_call ...> opening tags
        cleaned = re.sub(r"<tool_call[^>]*>", "", cleaned)
        cleaned = re.sub(r"<\|im_start\|>[a-zA-Z0-9_\-]*\n?", "", cleaned)

        # Clean trailing and redundant whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def parse_generation(
        self,
        generation_text: str,
        call_index: int = 1,
        enabled_tools: Optional[Set[str]] = None,
        request_id: str = "",
        conversation_id: str = "",
        dispatched_calls: Optional[Set[str]] = None
    ) -> ParsedOutput:
        """
        Classifies generation and extracts canonical tool call if present.
        Ensures call_id uniqueness and isolates dispatched call tracking per request.
        """
        active_dispatched = dispatched_calls if dispatched_calls is not None else self._dispatched_calls
        raw = generation_text.strip()
        if not raw:
            return ParsedOutput(
                classification=OutputClassification.NORMAL_ASSISTANT,
                sanitized_text="",
                raw_generation=raw
            )

        # 1. Check for explicit tool-call patterns
        patterns = [
            r"<\|tool_call\|>(.*?)(?:<\|im_end\|>|$)",
            r"<tool_call[^>]*>(.*?)</tool_call>",
            r"<tool_call[^>]*>(.*)",
            r"<\|tool_call\|>(.*)",
        ]

        candidate_json_str: Optional[str] = None
        for pat in patterns:
            m = re.search(pat, raw, re.DOTALL)
            if m:
                candidate_json_str = m.group(1).strip()
                break

        # 2. Check for bare JSON containing tool signature ("name" and "arguments")
        if not candidate_json_str:
            # Check for bare JSON like {"name": "...", "arguments": {...}}
            bare_m = re.search(r'(\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', raw, re.DOTALL)
            if bare_m:
                candidate_json_str = bare_m.group(1).strip()
            else:
                bare_m2 = re.search(r'(\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\})', raw, re.DOTALL)
                if bare_m2:
                    candidate_json_str = bare_m2.group(1).strip()

        # 3. If model emitted a tool snippet or placeholder token
        has_tool_indicators = (
            "<tool_call" in raw or
            "<|tool_call|>" in raw or
            '"name":' in raw or
            "<code_string>" in raw or
            "web_search" in raw or
            "web_research" in raw or
            "python_interpreter" in raw or
            '"arguments":' in raw
        )

        if candidate_json_str:
            # Clean possible markdown formatting (e.g. ```json ... ```)
            clean_str = candidate_json_str
            clean_str = re.sub(r"^```(?:json)?", "", clean_str, flags=re.IGNORECASE).strip()
            clean_str = re.sub(r"```$", "", clean_str).strip()

            # Clean placeholder markers that might have leaked into string
            clean_str = clean_str.replace("<code_string>", "").replace("<query_string>", "")

            try:
                data = json.loads(clean_str, strict=False)
            except Exception as json_err:
                # Try aggressive regex extraction of name and arguments
                data = self._regex_recover_json(clean_str)
                if not data:
                    return ParsedOutput(
                        classification=OutputClassification.MALFORMED_TOOL_CALL,
                        sanitized_text="",
                        malformed_error=f"JSONDecodeError: {json_err}",
                        raw_generation=raw
                    )

            if isinstance(data, dict):
                raw_tool_name = data.get("name")
                raw_args = data.get("arguments")

                if not raw_tool_name or not isinstance(raw_tool_name, str):
                    return ParsedOutput(
                        classification=OutputClassification.MALFORMED_TOOL_CALL,
                        sanitized_text="",
                        malformed_error="Tool call JSON is missing required 'name' field.",
                        raw_generation=raw
                    )

                canonical_name, orig_norm = self.registry.normalize_tool_name(raw_tool_name)
                if not canonical_name:
                    return ParsedOutput(
                        classification=OutputClassification.MALFORMED_TOOL_CALL,
                        sanitized_text="",
                        malformed_error=f"Unrecognized tool name '{raw_tool_name}'. Allowed tools: {list(self.registry.normalize_tool_name(t)[0] for t in ['web_search', 'python', 'weather'])}",
                        raw_generation=raw
                    )

                if not isinstance(raw_args, dict):
                    # Schema drift: flat arguments like {"name": "web_search", "query": "..."}
                    raw_args = {k: v for k, v in data.items() if k not in ["name", "tool", "call_id"]}

                # Deduplication check
                call_sig = f"{canonical_name}:{json.dumps(raw_args, sort_keys=True)}"
                if call_sig in active_dispatched:
                    return ParsedOutput(
                        classification=OutputClassification.MALFORMED_TOOL_CALL,
                        sanitized_text="",
                        malformed_error=f"Duplicate tool call detected ({canonical_name}). Duplicate dispatch rejected.",
                        raw_generation=raw
                    )

                active_dispatched.add(call_sig)
                call_id = f"call_{request_id[:8]}_{call_index:03d}" if request_id else f"call_{call_index:03d}"
                tool_call = CanonicalToolCall(
                    call_id=call_id,
                    original_name=orig_norm or raw_tool_name,
                    canonical_name=canonical_name,
                    arguments=raw_args,
                    raw_snippet=clean_str,
                    request_id=request_id,
                    conversation_id=conversation_id,
                )
                return ParsedOutput(
                    classification=OutputClassification.TOOL_CALL,
                    sanitized_text="",  # Never show raw tool call to user!
                    tool_call=tool_call,
                    raw_generation=raw
                )

        # If it had strong tool indicators but failed to extract valid JSON
        if has_tool_indicators and ('"name"' in raw or '"arguments"' in raw):
            return ParsedOutput(
                classification=OutputClassification.MALFORMED_TOOL_CALL,
                sanitized_text="",
                malformed_error="Malformed tool-call detected: unparseable syntax or schema structure.",
                raw_generation=raw
            )

        # Normal assistant response: sanitize user-visible text
        sanitized = self.sanitize_user_visible_text(raw)
        return ParsedOutput(
            classification=OutputClassification.NORMAL_ASSISTANT,
            sanitized_text=sanitized,
            raw_generation=raw
        )

    def _regex_recover_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Recovers tool name and arguments via regex when simple JSON quotes/newlines are broken."""
        name_m = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', text)
        if not name_m:
            return None
        tool_name = name_m.group(1).strip()

        # Query argument
        q_m = re.search(r'["\'](?:query|search_query|q)["\']\s*:\s*["\']([^"\']+)["\']', text)
        if q_m:
            return {"name": tool_name, "arguments": {"query": q_m.group(1).strip()}}

        # Location argument
        loc_m = re.search(r'["\'](?:location|city|place)["\']\s*:\s*["\']([^"\']+)["\']', text)
        if loc_m:
            return {"name": tool_name, "arguments": {"location": loc_m.group(1).strip()}}

        # Code argument
        code_m = re.search(r'["\']code["\']\s*:\s*["\']([\s\S]*?)["\']\s*\}', text)
        if code_m:
            return {"name": tool_name, "arguments": {"code": code_m.group(1).strip()}}

        return None
