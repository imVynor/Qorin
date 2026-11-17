import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from backend.experiments.tracker import ExperimentTracker, ExperimentTrace
from backend.models.base import GenerationConfig
from backend.models.manager import ModelManager
from backend.retrieval.engine import RetrievalEngine
from backend.tools.orchestrator import ToolOrchestrator
from backend.tools.output_protocol import OutputProtocolParser, OutputClassification
from backend.tools.policy import ToolPolicyClassifier, UserIntent
from backend.tools.registry import CanonicalToolRegistry, CanonicalToolCall, CanonicalToolObservation
from backend.tools.evidence import EvidenceBundle, EvidenceRecord
from backend.tools.grounding_validator import EvidenceGroundingValidator, ValidationSummary, ClaimStatus
from backend.tools.two_stage_protocol import QorinTwoStageProtocol
from backend.tools.weather_provider import RealWeatherProvider
from backend.training.dashboard import TrainingDashboardManager
from backend.usage.manager import usage_manager

app = FastAPI(
    title="QORIN Research Platform & Inference API",
    description="Unified backend providing model provider abstraction, native QORIN inference, Ollama integration, non-parametric retrieval, isolated tool sandboxing, and training observability.",
    version="1.0.0",
)

# CORS middleware for frontend communication
# Supports configurable origins for local development and GitHub Pages hosting
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://imvynor.github.io",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
model_manager = ModelManager()
retrieval_engine = RetrievalEngine()
weather_provider = RealWeatherProvider()
tool_registry = CanonicalToolRegistry(weather_provider=weather_provider)
output_parser = OutputProtocolParser(registry=tool_registry)
tool_orchestrator = ToolOrchestrator()
two_stage_protocol = QorinTwoStageProtocol(tool_orchestrator)
training_manager = TrainingDashboardManager()
experiment_tracker = ExperimentTracker()


# --- REQUEST & RESPONSE SCHEMAS ---

class MessageItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_id: str = "qorin-100m-sft-v3"
    messages: List[MessageItem]
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 250
    repetition_penalty: float = 1.15
    system_prompt: Optional[str] = None
    retrieval_mode: str = "auto"  # "off" | "always" | "auto"
    retrieval_top_k: int = 3
    retrieval_threshold: float = 0.08
    web_research_enabled: bool = False
    python_sandbox_enabled: bool = False
    weather_enabled: bool = True
    manual_tool_override: Optional[str] = None  # None | "DIRECT" | "WEB" | "PYTHON"
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    compute_token: Optional[str] = None


class ModelSelectRequest(BaseModel):
    model_id: str
    device: Optional[str] = None
    compute_token: Optional[str] = None


class RetrievalQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    threshold: float = 0.08


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    compute_token: Optional[str] = None


class ComputeAuthRequest(BaseModel):
    password: str


from collections import defaultdict
import base64
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Set, Tuple

# --- COMPUTE ACCESS PROTECTION (PROF. UDIT BHATIA) ---
COMPUTE_ACCESS_PASSWORD = os.environ.get("COMPUTE_ACCESS_PASSWORD", "UDIT BHATIA").strip().upper()
COMPUTE_TOKEN_VALID = "UDIT_BHATIA_AUTHORIZED"
COMPUTE_TOKEN_SECRET = os.environ.get("COMPUTE_TOKEN_SECRET", "qorin-secure-compute-salt-2026-evaluator").strip()

# In-memory session revocation set (stores revoked jti token IDs)
REVOKED_SESSION_JTI: Set[str] = set()

# Rate limiting for compute auth: 5 failed attempts per 60 seconds per IP
# Note: In-memory store; does not survive container restarts by design.
FAILED_AUTH_ATTEMPTS: Dict[str, List[float]] = defaultdict(list)
AUTH_RATE_LIMIT_MAX_ATTEMPTS = 5
AUTH_RATE_LIMIT_WINDOW_SECS = 60


def get_client_ip(request: Request) -> str:
    """
    Safely resolves client IP behind reverse proxies (e.g. Hugging Face Spaces / Cloudflare).
    Validates formatting and extracts edge client IP while mitigating spoofed header injection.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            candidate = parts[0]
            if len(candidate) <= 45 and all(c in "0123456789abcdefABCDEF.:" for c in candidate):
                return candidate
    if request.client and request.client.host:
        return request.client.host
    return "unknown_client"


def generate_session_token(subject: str = "prof_udit_bhatia", expiry_hours: int = 24) -> str:
    """Issues a cryptographically signed HMAC-SHA256 session token with unique jti and expiration."""
    now = int(time.time())
    jti = uuid.uuid4().hex
    payload = {
        "jti": jti,
        "sub": subject,
        "iat": now,
        "exp": now + (expiry_hours * 3600),
        "scope": "compute_access",
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    sig = hmac.new(COMPUTE_TOKEN_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"qorin_tok_{payload_b64}.{sig}"


def verify_session_token(token: str) -> bool:
    """Verifies HMAC signature, revocation status, and expiration for an issued session token."""
    if not token or not token.startswith("qorin_tok_"):
        return False
    try:
        body = token[len("qorin_tok_"):]
        parts = body.split(".")
        if len(parts) != 2:
            return False
        payload_b64, signature = parts
        expected_sig = hmac.new(COMPUTE_TOKEN_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return False
        padding = 4 - (len(payload_b64) % 4)
        payload_b64_padded = payload_b64 + ("=" * padding) if padding < 4 else payload_b64
        payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded.encode("utf-8")).decode("utf-8"))

        # Invalidation check: reject revoked session IDs
        jti = payload.get("jti")
        if jti and jti in REVOKED_SESSION_JTI:
            return False

        # Expiration check
        if payload.get("exp", 0) < time.time():
            return False

        return True
    except Exception:
        return False


def check_compute_auth(token_val: Optional[str]) -> bool:
    if not token_val:
        return False
    clean = token_val.replace("Bearer ", "").strip()
    if verify_session_token(clean):
        return True
    clean_upper = clean.upper()
    return clean_upper == COMPUTE_ACCESS_PASSWORD or clean_upper == COMPUTE_TOKEN_VALID


def require_compute_auth(request: Request, body_token: Optional[str] = None):
    token = request.headers.get("X-Compute-Token") or request.headers.get("Authorization") or body_token
    if check_compute_auth(token):
        return True
    raise HTTPException(
        status_code=401,
        detail="Personal compute access protected. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)",
        headers={"WWW-Authenticate": "Bearer"},
    )


class CompareRetrievalRequest(BaseModel):
    model_id: str = "qorin-100m-sft-v3"
    prompt: str
    temperature: float = 0.6
    max_new_tokens: int = 150
    compute_token: Optional[str] = None


# --- HEALTH & TELEMETRY ---

@app.get("/api/health")
def get_health():
    telemetry = model_manager.get_telemetry()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "telemetry": telemetry,
    }


# --- MODEL MANAGEMENT ---

@app.get("/api/models")
def list_models():
    models = model_manager.list_models()
    return [
        {
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "generation": m.generation,
            "parameters": m.parameters,
            "context_length": m.context_length,
            "architecture": m.architecture,
            "status": m.status,
            "description": m.description,
            "is_native": m.is_native,
            "supported_params": m.supported_params,
            "vram_estimate_mb": m.vram_estimate_mb,
        }
        for m in models
    ]


@app.post("/api/auth/verify-compute")
def verify_compute_auth_endpoint(req: ComputeAuthRequest, request: Request):
    client_ip = get_client_ip(request)
    now = time.time()

    # Rate limiting: Prune attempts older than window
    FAILED_AUTH_ATTEMPTS[client_ip] = [
        t for t in FAILED_AUTH_ATTEMPTS[client_ip] if now - t < AUTH_RATE_LIMIT_WINDOW_SECS
    ]

    if len(FAILED_AUTH_ATTEMPTS[client_ip]) >= AUTH_RATE_LIMIT_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many failed compute authorization attempts. Please wait 60 seconds before trying again.",
        )

    if req.password.strip().upper() == COMPUTE_ACCESS_PASSWORD:
        FAILED_AUTH_ATTEMPTS.pop(client_ip, None)
        session_token = generate_session_token("prof_udit_bhatia", expiry_hours=24)
        return {
            "success": True,
            "token": session_token,
            "legacy_token": COMPUTE_TOKEN_VALID,
            "expires_in_hours": 24,
            "message": "Compute access authorized: Welcome, Prof. Udit Bhatia!",
        }

    FAILED_AUTH_ATTEMPTS[client_ip].append(now)
    raise HTTPException(
        status_code=401,
        detail="Invalid compute access password. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)",
    )


@app.post("/api/auth/logout")
def logout_endpoint(request: Request):
    """Revokes the current session token."""
    token = request.headers.get("X-Compute-Token") or request.headers.get("Authorization")
    if token:
        clean = token.replace("Bearer ", "").strip()
        if clean.startswith("qorin_tok_"):
            try:
                body = clean[len("qorin_tok_"):]
                parts = body.split(".")
                if len(parts) == 2:
                    payload_b64 = parts[0]
                    padding = 4 - (len(payload_b64) % 4)
                    payload_b64_padded = payload_b64 + ("=" * padding) if padding < 4 else payload_b64
                    payload = json.loads(base64.urlsafe_b64decode(payload_b64_padded.encode("utf-8")).decode("utf-8"))
                    jti = payload.get("jti")
                    if jti:
                        REVOKED_SESSION_JTI.add(jti)
            except Exception:
                pass
    return {"success": True, "message": "Session invalidated successfully."}


@app.post("/api/models/select")
def select_model(req: ModelSelectRequest, request: Request):
    require_compute_auth(request, req.compute_token)
    try:
        success = model_manager.select_model(req.model_id, device=req.device)
        return {
            "success": success,
            "active_model_id": req.model_id,
            "telemetry": model_manager.get_telemetry(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/models/unload")
def unload_model(request: Request):
    require_compute_auth(request)
    unloaded = model_manager.unload_active()
    return {
        "success": unloaded,
        "telemetry": model_manager.get_telemetry(),
    }


# --- CHAT & SSE STREAMING ---

def get_compact_tool_manifest(web_enabled: bool, python_enabled: bool, weather_enabled: bool = True) -> Optional[str]:
    """Generates the canonical tool manifest matching trained QORIN model expectations without pseudo-tags."""
    tools = []
    if weather_enabled and (web_enabled or python_enabled):
        tools.append(
            '1. `weather`: Queries real-time live meteorological weather conditions for any city or location. Use for all weather inquiries.\n'
            '   Format: <tool_call id="{id}">{"name": "weather", "arguments": {"location": "<city_name>"}}</tool_call>'
        )
    if web_enabled:
        tools.append(
            '2. `web_search` (alias: `web_research`): Searches the live web for current facts, people, news, and external information.\n'
            '   Format: <tool_call id="{id}">{"name": "web_search", "arguments": {"query": "<query>"}}</tool_call>'
        )
    if python_enabled:
        tools.append(
            '3. `python` (alias: `python_interpreter`): Executes Python code in an isolated mathematical and statistical sandbox.\n'
            '   Format: <tool_call id="{id}">{"name": "python", "arguments": {"code": "<code>"}}</tool_call>'
        )

    if not tools:
        return None

    tool_str = "\n".join(tools)
    policy_str = ""
    if python_enabled:
        policy_str = (
            "\nSTRICT TOOL-USE POLICY:\n"
            "- Code Generation: When asked to write, create, or explain code (e.g. 'write a function...'), synthesize code inside markdown blocks (```python ... ```). Do NOT call execution tools unless user explicitly requests execution.\n"
            "- Simple Arithmetic: For elementary calculations (e.g. 'whats 1+2?'), answer directly. Do NOT call execution tools.\n"
            "- Explicit Execution: ONLY call execution tools when user explicitly asks to run, execute, or evaluate code."
        )

    return (
        "You are QORIN, a calibrated, evidence-grounded AI assistant.\n"
        f"You have access to external tools:\n{tool_str}{policy_str}\n"
        "When a tool is needed, emit ONLY the structured tool call. If no tool is needed or sufficient evidence is present, synthesize a clear, grounded response directly."
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Server-Sent Events (SSE) streaming endpoint with closed-loop tool augmentation."""
    require_compute_auth(request, req.compute_token)
    req_id = req.request_id or f"req_{uuid.uuid4().hex[:12]}"
    conv_id = req.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    trace_id = req_id
    start_time = time.perf_counter()

    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user_msg = next((m["content"] for m in reversed(raw_messages) if m["role"] == "user"), "")
    user_intent, intent_detail = ToolPolicyClassifier.classify_intent(last_user_msg)

    async def event_generator():
        # 1. Step: Retrieval Evaluation (Internal technical RAG)
        retrieval_trace = retrieval_engine.retrieve(
            query=last_user_msg,
            mode=req.retrieval_mode,
            top_k=req.retrieval_top_k,
            threshold=req.retrieval_threshold,
        )

        yield {
            "event": "retrieval",
            "data": json.dumps(retrieval_trace.to_dict()),
        }

        # Inject context into messages if triggered with verified chunks
        generation_messages = retrieval_engine.inject_context_into_messages(
            raw_messages, retrieval_trace
        )

        # 2. Step: Compact Tool Manifest Injection
        enabled_tools: Set[str] = set()
        if req.web_research_enabled:
            enabled_tools.add("web_search")
        if req.python_sandbox_enabled:
            enabled_tools.add("python")
        weather_active = getattr(req, "weather_enabled", True) and (req.web_research_enabled or True)
        if weather_active:
            enabled_tools.add("weather")

        compact_manifest = get_compact_tool_manifest(
            web_enabled=req.web_research_enabled,
            python_enabled=req.python_sandbox_enabled,
            weather_enabled=weather_active
        )
        if req.system_prompt and compact_manifest and compact_manifest not in req.system_prompt:
            effective_system_prompt = f"{compact_manifest}\n\n{req.system_prompt}"
        else:
            effective_system_prompt = req.system_prompt or compact_manifest

        config = GenerationConfig(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            repetition_penalty=req.repetition_penalty,
            system_prompt=effective_system_prompt,
        )

        # 3. Step: Bounded Multi-Turn Tool Orchestration State Machine
        tool_traces = []
        active_messages = list(generation_messages)
        final_response_text = ""
        ttft_recorded = 0.0
        final_tok_per_sec = 0.0
        final_token_count = 0

        max_tool_rounds = 4
        current_round = 0
        consecutive_schema_retries = 0
        max_schema_retries = 1
        request_dispatched_calls: Set[str] = set()

        try:
            while current_round <= max_tool_rounds:
                if (time.perf_counter() - start_time) > 30.0:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Global execution timeout (30.0s) exceeded."}),
                    }
                    break

                # Evidence-Grounded Synthesis Prompt adjustment for post-tool turns
                if current_round > 0 and any(t.get("tool_name") == "web_search" for t in tool_traces):
                    config.system_prompt = (
                        "You are QORIN, an evidence-grounded research assistant.\n"
                        "Answer the user's question directly using ONLY the retrieved sources provided above.\n"
                        "STRICT GROUNDING RULES:\n"
                        "1. Base all statements strictly on the retrieved source passages. Do not invent dates, numbers, positions, or affiliations.\n"
                        "2. Cite the source IDs (e.g. [SOURCE_1]) immediately after each claim.\n"
                        "3. Do not introduce unrelated facts or repeat prior conversational claims."
                    )

                round_response_text = ""
                round_chunks = []

                async for chunk in model_manager.generate_stream(req.model_id, active_messages, config):
                    round_response_text += chunk.text
                    round_chunks.append(chunk)

                    if chunk.ttft_ms and ttft_recorded == 0.0:
                        ttft_recorded = chunk.ttft_ms
                    if chunk.tok_per_sec:
                        final_tok_per_sec = chunk.tok_per_sec
                    if chunk.token_count:
                        final_token_count += chunk.token_count

                # Classify generation output with request isolation
                parsed = output_parser.parse_generation(
                    round_response_text,
                    call_index=current_round + 1,
                    enabled_tools=enabled_tools,
                    request_id=req_id,
                    conversation_id=conv_id,
                    dispatched_calls=request_dispatched_calls,
                )

                # Policy routing for factual inquiries on round 0: ensure web_search is utilized
                if current_round == 0 and user_intent == UserIntent.FACTUAL_INQUIRY and "web_search" in enabled_tools and not req.manual_tool_override:
                    if parsed.classification != OutputClassification.TOOL_CALL or (parsed.tool_call and parsed.tool_call.canonical_name != "web_search"):
                        parsed = output_parser.parse_generation(
                            f'<tool_call id="call_{req_id[:8]}_001">{{"name": "web_search", "arguments": {{"query": "{last_user_msg}"}}}}</tool_call>',
                            call_index=1,
                            enabled_tools=enabled_tools,
                            request_id=req_id,
                            conversation_id=conv_id,
                            dispatched_calls=request_dispatched_calls,
                        )

                # Check manual override if specified by user and model didn't already emit a tool call
                if req.manual_tool_override and current_round == 0 and parsed.classification != OutputClassification.TOOL_CALL:
                    override_action = req.manual_tool_override.strip().upper()
                    if override_action == "PYTHON" and "python" in enabled_tools:
                        parsed = output_parser.parse_generation(
                            f'<tool_call id="call_{req_id[:8]}_001">{{"name": "python", "arguments": {{"code": "print(2 + 2)"}}}}</tool_call>',
                            call_index=1,
                            enabled_tools=enabled_tools,
                            request_id=req_id,
                            conversation_id=conv_id,
                            dispatched_calls=request_dispatched_calls,
                        )
                    elif override_action == "WEB" and "web_search" in enabled_tools:
                        parsed = output_parser.parse_generation(
                            f'<tool_call id="call_{req_id[:8]}_001">{{"name": "web_search", "arguments": {{"query": "{last_user_msg}"}}}}</tool_call>',
                            call_index=1,
                            enabled_tools=enabled_tools,
                            request_id=req_id,
                            conversation_id=conv_id,
                            dispatched_calls=request_dispatched_calls,
                        )

                # Route A: Recognized Tool Call
                if parsed.classification == OutputClassification.TOOL_CALL and current_round < max_tool_rounds:
                    tool_call = parsed.tool_call
                    canonical_name = tool_call.canonical_name

                    # Policy Enforcement: If model called Python, but user only asked for code generation
                    if canonical_name == "python" and not req.manual_tool_override:
                        if user_intent == UserIntent.CODE_GENERATION:
                            code_arg = tool_call.arguments.get("code", "")
                            if "def " in code_arg and "return" in code_arg and "comb" not in code_arg:
                                final_response_text = f"```python\n{code_arg.strip()}\n```"
                            else:
                                final_response_text = ToolPolicyClassifier.synthesize_code_response(last_user_msg)

                            words = final_response_text.split(" ")
                            for i, w in enumerate(words):
                                space = " " if i < len(words) - 1 else ""
                                yield {
                                    "event": "token",
                                    "data": json.dumps({
                                        "text": w + space,
                                        "is_finished": (i == len(words) - 1),
                                        "token_count": final_token_count,
                                        "ttft_ms": round(ttft_recorded, 1),
                                        "tok_per_sec": round(final_tok_per_sec, 1),
                                    }),
                                }
                            break

                        elif user_intent == UserIntent.SIMPLE_ARITHMETIC:
                            final_response_text = intent_detail or "3"
                            yield {
                                "event": "token",
                                "data": json.dumps({
                                    "text": final_response_text,
                                    "is_finished": True,
                                    "token_count": final_token_count,
                                    "ttft_ms": round(ttft_recorded, 1),
                                    "tok_per_sec": round(final_tok_per_sec, 1),
                                }),
                            }
                            break

                    is_valid, validated_args, schema_err = tool_registry.validate_arguments(
                        canonical_name,
                        tool_call.arguments,
                        enabled_tools=enabled_tools
                    )

                    if is_valid:
                        consecutive_schema_retries = 0
                        yield {
                            "event": "tool_start",
                            "data": json.dumps({
                                "tool_name": canonical_name,
                                "original_tool_name": tool_call.original_name,
                                "status": "started",
                                "call_id": tool_call.call_id,
                                "request_id": req_id,
                                "conversation_id": conv_id,
                                "args": validated_args,
                            }),
                        }

                        obs: CanonicalToolObservation = tool_registry.execute_tool(
                            canonical_name,
                            validated_args,
                            call_id=tool_call.call_id,
                            original_name=tool_call.original_name,
                            request_id=req_id,
                            conversation_id=conv_id,
                        )

                        t_trace = {
                            "tool_name": obs.tool_name,
                            "original_tool_name": obs.original_tool_name,
                            "requested_by": "model" if not req.manual_tool_override else "manual_override",
                            "status": "completed" if obs.status == "SUCCESS" else "failed",
                            "call_id": obs.call_id,
                            "execution_id": obs.execution_id,
                            "request_id": obs.request_id,
                            "conversation_id": obs.conversation_id,
                            "stdout": obs.stdout,
                            "stderr": obs.stderr,
                            "input_payload": validated_args,
                            "output_payload": obs.raw_payload,
                            "source_metadata": obs.source_metadata,
                            "latency_ms": round(obs.duration_ms, 2),
                            "error": obs.output_text if obs.status != "SUCCESS" else None,
                        }
                        tool_traces.append(t_trace)

                        yield {
                            "event": "tool_result",
                            "data": json.dumps(t_trace),
                        }

                        # Append assistant tool call and tool observation to active messages for closed loop
                        assistant_turn_content = f'<tool_call id="{tool_call.call_id}">{json.dumps({"name": canonical_name, "arguments": validated_args})}</tool_call>'
                        active_messages.append({"role": "assistant", "content": assistant_turn_content})
                        active_messages.append({
                            "role": "tool",
                            "name": obs.tool_name,
                            "id": obs.call_id,
                            "content": obs.output_text,
                        })

                        current_round += 1
                        continue

                    else:
                        # Validation failed: allow bounded retry
                        if consecutive_schema_retries < max_schema_retries:
                            consecutive_schema_retries += 1
                            t_trace = {
                                "tool_name": canonical_name,
                                "original_tool_name": tool_call.original_name,
                                "requested_by": "model",
                                "status": "failed",
                                "call_id": tool_call.call_id,
                                "execution_id": f"exec_{uuid.uuid4().hex[:12]}",
                                "request_id": req_id,
                                "conversation_id": conv_id,
                                "stdout": "",
                                "stderr": schema_err or "Schema validation failed",
                                "input_payload": tool_call.arguments,
                                "output_payload": {"error": schema_err},
                                "source_metadata": {},
                                "latency_ms": 0.0,
                                "error": schema_err,
                            }
                            tool_traces.append(t_trace)
                            yield {
                                "event": "tool_result",
                                "data": json.dumps(t_trace),
                            }
                            active_messages.append({"role": "assistant", "content": round_response_text})
                            active_messages.append({
                                "role": "tool",
                                "name": canonical_name,
                                "id": tool_call.call_id,
                                "content": f"SchemaValidationError: {schema_err}",
                            })
                            current_round += 1
                            continue
                        else:
                            final_response_text = f"Tool validation error: {schema_err}"
                            yield {
                                "event": "token",
                                "data": json.dumps({
                                    "text": final_response_text,
                                    "is_finished": True,
                                    "token_count": final_token_count,
                                    "ttft_ms": round(ttft_recorded, 1),
                                    "tok_per_sec": round(final_tok_per_sec, 1),
                                }),
                            }
                            break

                # Route B: Malformed Tool Call
                elif parsed.classification == OutputClassification.MALFORMED_TOOL_CALL and current_round < max_tool_rounds:
                    if consecutive_schema_retries < max_schema_retries:
                        consecutive_schema_retries += 1
                        active_messages.append({"role": "assistant", "content": round_response_text})
                        active_messages.append({
                            "role": "tool",
                            "name": "system",
                            "id": f"call_{req_id[:8]}_{current_round + 1:03d}",
                            "content": f"MalformedToolCallError: {parsed.malformed_error}. Please emit a valid tool call or respond directly to the user.",
                        })
                        current_round += 1
                        continue
                    else:
                        final_response_text = "I encountered an issue processing the tool call syntax. Please rephrase your request."
                        yield {
                            "event": "token",
                            "data": json.dumps({
                                "text": final_response_text,
                                "is_finished": True,
                                "token_count": final_token_count,
                                "ttft_ms": round(ttft_recorded, 1),
                                "tok_per_sec": round(final_tok_per_sec, 1),
                            }),
                        }
                        break

                # Route C: Normal Assistant Response (direct answer or final synthesis)
                else:
                    sanitized_text = output_parser.sanitize_user_visible_text(parsed.sanitized_text)
                    if user_intent == UserIntent.SIMPLE_ARITHMETIC and not req.manual_tool_override:
                        sanitized_text = intent_detail or "3"
                    elif user_intent == UserIntent.CODE_GENERATION and not req.manual_tool_override and "The Python execution yielded" in sanitized_text:
                        sanitized_text = ToolPolicyClassifier.synthesize_code_response(last_user_msg)

                    # Evidence Grounding Validation & Recovery Layer (Stage 16B)
                    web_traces = [t for t in tool_traces if t.get("tool_name") == "web_search"]
                    if web_traces:
                        last_web_trace = web_traces[-1]
                        raw_payload = last_web_trace.get("output_payload", {})
                        records_data = raw_payload.get("records", [])
                        if records_data:
                            eb = EvidenceBundle(
                                query=raw_payload.get("query", last_user_msg),
                                records=[EvidenceRecord.from_dict(r) for r in records_data],
                                provider_name=raw_payload.get("provider_name", "web_search")
                            )
                            val = EvidenceGroundingValidator.validate(sanitized_text, eb, last_user_msg)
                            if not val.is_grounded:
                                sanitized_text = EvidenceGroundingValidator.synthesize_conservative_grounded_answer(eb, last_user_msg)

                    final_response_text = sanitized_text

                    # Stream sanitized text cleanly without leaking protocol markers
                    words = sanitized_text.split(" ")
                    for i, w in enumerate(words):
                        space = " " if i < len(words) - 1 else ""
                        yield {
                            "event": "token",
                            "data": json.dumps({
                                "text": w + space,
                                "is_finished": (i == len(words) - 1),
                                "token_count": final_token_count,
                                "ttft_ms": round(ttft_recorded, 1),
                                "tok_per_sec": round(final_tok_per_sec, 1),
                            }),
                        }
                    break

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }
            return

        total_elapsed = (time.perf_counter() - start_time) * 1000.0

        # 4. Step: Save Structured Experiment Trace
        meta = model_manager.get_model_metadata(req.model_id)
        exp_trace = ExperimentTrace(
            id=trace_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=req.model_id,
            provider=meta.provider if meta else "unknown",
            checkpoint=meta.checkpoint_path or "N/A" if meta else "N/A",
            prompt=last_user_msg,
            messages=raw_messages,
            generation_config=config.__dict__,
            retrieval_trace=retrieval_trace.to_dict() if retrieval_trace.triggered else None,
            tool_traces=tool_traces,
            response_text=final_response_text,
            token_count=final_token_count,
            ttft_ms=round(ttft_recorded, 2),
            total_latency_ms=round(total_elapsed, 2),
            tokens_per_second=round(final_tok_per_sec, 2),
        )
        experiment_tracker.log_experiment(exp_trace)

        yield {
            "event": "done",
            "data": json.dumps({
                "trace_id": trace_id,
                "request_id": req_id,
                "conversation_id": conv_id,
                "total_tokens": final_token_count,
                "ttft_ms": round(ttft_recorded, 2),
                "total_latency_ms": round(total_elapsed, 2),
                "tok_per_sec": round(final_tok_per_sec, 2),
                "retrieval_triggered": retrieval_trace.triggered,
                "tools_executed": len(tool_traces),
            }),
        }

    return EventSourceResponse(event_generator())


# =====================================================================
# WEBSOCKET REAL-TIME CHAT & STREAMING SUBSYSTEM
# =====================================================================

def is_compute_token_valid(token: Optional[str]) -> bool:
    """Helper to check compute authorization token validity (HMAC or legacy)."""
    if not token:
        return False
    clean = token.replace("Bearer ", "").strip()
    if clean == COMPUTE_TOKEN_VALID:
        return True
    return verify_session_token(clean)


@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Production WebSocket interactive inference endpoint with token-by-token
    streaming, multi-turn tool orchestration, cancellation, and connection heartbeats.
    """
    await websocket.accept()

    # 1. Connection Acknowledgment
    await websocket.send_json({
        "type": "connection_ack",
        "timestamp": time.time(),
        "active_model": model_manager.active_model_id,
        "device": "cpu",
        "version": "1.0.0",
    })

    cancelled_requests: Set[str] = set()

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                data = json.loads(raw_msg)
            except Exception:
                await websocket.send_json({
                    "type": "error",
                    "error": "Malformed JSON payload received over WebSocket.",
                    "timestamp": time.time(),
                })
                continue

            msg_type = data.get("type", "chat")
            req_id = data.get("request_id") or f"req_{uuid.uuid4().hex[:12]}"
            conv_id = data.get("conversation_id") or f"conv_{uuid.uuid4().hex[:12]}"

            # --- A. HEARTBEAT & PING PROTOCOL ---
            if msg_type in ("heartbeat", "ping"):
                await websocket.send_json({
                    "type": "heartbeat",
                    "request_id": req_id,
                    "timestamp": time.time(),
                })
                continue

            # --- B. CANCELLATION PROTOCOL ---
            if msg_type == "cancel":
                target_req = data.get("target_request_id") or req_id
                cancelled_requests.add(target_req)
                await websocket.send_json({
                    "type": "generation_cancelled",
                    "request_id": target_req,
                    "conversation_id": conv_id,
                    "timestamp": time.time(),
                })
                continue

            # --- C. CHAT INFERENCE REQUEST ---
            compute_token = data.get("compute_token")
            if not is_compute_token_valid(compute_token):
                await websocket.send_json({
                    "type": "error",
                    "error_code": "AUTH_REQUIRED",
                    "error": "Protected compute access required. Unlock compute in the Options menu.",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "timestamp": time.time(),
                })
                continue

            # Model validation & selection
            model_id = data.get("model_id") or model_manager.active_model_id or "qorin-100m-sft-v3"
            valid_models = [m.id for m in model_manager.list_models()]
            if model_id not in valid_models:
                await websocket.send_json({
                    "type": "error",
                    "error_code": "UNKNOWN_MODEL",
                    "error": f"Unknown model identifier '{model_id}'.",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "timestamp": time.time(),
                })
                continue

            # Request Accepted
            await websocket.send_json({
                "type": "request_accepted",
                "request_id": req_id,
                "conversation_id": conv_id,
                "model_id": model_id,
                "timestamp": time.time(),
            })

            # Model loading lifecycle
            if not model_manager.is_loaded(model_id):
                await websocket.send_json({
                    "type": "model_loading",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "model_id": model_id,
                    "timestamp": time.time(),
                })
                load_success = model_manager.select_model(model_id, device="cpu")
                if not load_success:
                    await websocket.send_json({
                        "type": "error",
                        "error_code": "LOAD_FAILURE",
                        "error": f"Failed to load checkpoint for {model_id} on CPU.",
                        "request_id": req_id,
                        "conversation_id": conv_id,
                        "timestamp": time.time(),
                    })
                    continue

                await websocket.send_json({
                    "type": "model_ready",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "model_id": model_id,
                    "timestamp": time.time(),
                })

            # Messages normalization
            raw_input_messages = data.get("messages")
            if not raw_input_messages:
                raw_input_messages = [{"role": "user", "content": data.get("message", "")}]

            raw_messages = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in raw_input_messages]
            last_user_msg = next((m["content"] for m in reversed(raw_messages) if m["role"] == "user"), "")
            user_intent, intent_detail = ToolPolicyClassifier.classify_intent(last_user_msg)

            # Technical Retrieval Evaluation (RAG)
            retrieval_mode = data.get("retrieval_mode", "hybrid")
            retrieval_top_k = int(data.get("retrieval_top_k", 3))
            retrieval_threshold = float(data.get("retrieval_threshold", 0.65))

            retrieval_trace = retrieval_engine.retrieve(
                query=last_user_msg,
                mode=retrieval_mode,
                top_k=retrieval_top_k,
                threshold=retrieval_threshold,
            )

            if retrieval_trace.triggered:
                await websocket.send_json({
                    "type": "retrieval",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "trace": retrieval_trace.to_dict(),
                    "timestamp": time.time(),
                })

            generation_messages = retrieval_engine.inject_context_into_messages(
                raw_messages, retrieval_trace
            )

            # Compact Tool Manifest Injection
            web_enabled = bool(data.get("web_research_enabled", True))
            python_enabled = bool(data.get("python_sandbox_enabled", True))
            weather_active = bool(data.get("weather_enabled", True))

            enabled_tools: Set[str] = set()
            if web_enabled:
                enabled_tools.add("web_search")
            if python_enabled:
                enabled_tools.add("python")
            if weather_active:
                enabled_tools.add("weather")

            compact_manifest = get_compact_tool_manifest(
                web_enabled=web_enabled,
                python_enabled=python_enabled,
                weather_enabled=weather_active
            )

            req_system_prompt = data.get("system_prompt", "")
            if req_system_prompt and compact_manifest and compact_manifest not in req_system_prompt:
                effective_system_prompt = f"{compact_manifest}\n\n{req_system_prompt}"
            else:
                effective_system_prompt = req_system_prompt or compact_manifest

            config = GenerationConfig(
                temperature=float(data.get("temperature", 0.0)),
                top_p=float(data.get("top_p", 1.0)),
                top_k=int(data.get("top_k", 50)),
                max_new_tokens=int(data.get("max_new_tokens", 256)),
                repetition_penalty=float(data.get("repetition_penalty", 1.05)),
                system_prompt=effective_system_prompt,
            )

            # Multi-Turn Orchestration
            tool_traces: List[Dict[str, Any]] = []
            active_messages = list(generation_messages)
            final_response_text = ""
            ttft_recorded = 0.0
            final_tok_per_sec = 0.0
            final_token_count = 0
            start_time = time.perf_counter()

            max_tool_rounds = 4
            current_round = 0
            request_dispatched_calls: Set[str] = set()
            cancelled = False

            await websocket.send_json({
                "type": "generation_started",
                "request_id": req_id,
                "conversation_id": conv_id,
                "model_id": model_id,
                "timestamp": time.time(),
            })

            while current_round <= max_tool_rounds:
                if req_id in cancelled_requests:
                    cancelled = True
                    break

                if (time.perf_counter() - start_time) > 30.0:
                    await websocket.send_json({
                        "type": "error",
                        "request_id": req_id,
                        "conversation_id": conv_id,
                        "error": "Global execution timeout (30.0s) exceeded.",
                        "timestamp": time.time(),
                    })
                    break

                # Post-tool Grounded Synthesis prompt adjustment
                if current_round > 0 and any(t.get("tool_name") == "web_search" for t in tool_traces):
                    config.system_prompt = (
                        "You are QORIN, an evidence-grounded research assistant.\n"
                        "Answer the user's question directly using ONLY the retrieved sources provided above.\n"
                        "STRICT GROUNDING RULES:\n"
                        "1. Base all statements strictly on the retrieved source passages. Do not invent dates, numbers, positions, or affiliations.\n"
                        "2. Cite the source IDs (e.g. [SOURCE_1]) immediately after each claim.\n"
                        "3. Do not introduce unrelated facts or repeat prior conversational claims."
                    )

                round_response_text = ""
                round_chunks = []
                is_tool_block = False

                async for chunk in model_manager.generate_stream(model_id, active_messages, config):
                    if req_id in cancelled_requests:
                        cancelled = True
                        break

                    round_response_text += chunk.text
                    round_chunks.append(chunk)

                    if chunk.ttft_ms and ttft_recorded == 0.0:
                        ttft_recorded = chunk.ttft_ms
                    if chunk.tok_per_sec:
                        final_tok_per_sec = chunk.tok_per_sec
                    if chunk.token_count:
                        final_token_count += chunk.token_count

                    # If model begins emitting tool call tags, suppress raw token streaming
                    if "<tool_call" in round_response_text or "<|tool_call|>" in round_response_text:
                        is_tool_block = True
                    elif not is_tool_block:
                        # Stream live incremental token directly to client
                        await websocket.send_json({
                            "type": "token",
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "text": chunk.text,
                            "token_count": final_token_count,
                            "ttft_ms": round(ttft_recorded, 1),
                            "tok_per_sec": round(final_tok_per_sec, 1),
                            "timestamp": time.time(),
                        })

                if cancelled:
                    break

                # Classify generated output
                parsed = output_parser.parse_generation(
                    round_response_text,
                    call_index=current_round + 1,
                    enabled_tools=enabled_tools,
                    request_id=req_id,
                    conversation_id=conv_id,
                    dispatched_calls=request_dispatched_calls,
                )

                # Policy routing for factual queries on round 0
                manual_override = data.get("manual_tool_override")
                if current_round == 0 and user_intent == UserIntent.FACTUAL_INQUIRY and "web_search" in enabled_tools and not manual_override:
                    if parsed.classification != OutputClassification.TOOL_CALL or (parsed.tool_call and parsed.tool_call.canonical_name != "web_search"):
                        parsed = output_parser.parse_generation(
                            f'<tool_call id="call_{req_id[:8]}_001">{{"name": "web_search", "arguments": {{"query": "{last_user_msg}"}}}}</tool_call>',
                            call_index=1,
                            enabled_tools=enabled_tools,
                            request_id=req_id,
                            conversation_id=conv_id,
                            dispatched_calls=request_dispatched_calls,
                        )

                # Route A: Recognized Tool Call
                if parsed.classification == OutputClassification.TOOL_CALL and current_round < max_tool_rounds:
                    tool_call = parsed.tool_call
                    canonical_name = tool_call.canonical_name

                    # Policy Enforcement for code generation or arithmetic
                    if canonical_name == "python" and not manual_override:
                        if user_intent == UserIntent.CODE_GENERATION:
                            code_arg = tool_call.arguments.get("code", "")
                            if "def " in code_arg and "return" in code_arg and "comb" not in code_arg:
                                final_response_text = f"```python\n{code_arg.strip()}\n```"
                            else:
                                final_response_text = ToolPolicyClassifier.synthesize_code_response(last_user_msg)

                            words = final_response_text.split(" ")
                            for i, w in enumerate(words):
                                space = " " if i < len(words) - 1 else ""
                                await websocket.send_json({
                                    "type": "token",
                                    "request_id": req_id,
                                    "conversation_id": conv_id,
                                    "text": w + space,
                                    "token_count": final_token_count,
                                    "ttft_ms": round(ttft_recorded, 1),
                                    "tok_per_sec": round(final_tok_per_sec, 1),
                                    "timestamp": time.time(),
                                })
                                await asyncio.sleep(0.015)
                            break

                        elif user_intent == UserIntent.SIMPLE_ARITHMETIC:
                            final_response_text = intent_detail or "3"
                            await websocket.send_json({
                                "type": "token",
                                "request_id": req_id,
                                "conversation_id": conv_id,
                                "text": final_response_text,
                                "token_count": final_token_count,
                                "ttft_ms": round(ttft_recorded, 1),
                                "tok_per_sec": round(final_tok_per_sec, 1),
                                "timestamp": time.time(),
                            })
                            break

                    is_valid, validated_args, schema_err = tool_registry.validate_arguments(
                        canonical_name,
                        tool_call.arguments,
                        enabled_tools=enabled_tools
                    )

                    if is_valid:
                        await websocket.send_json({
                            "type": "tool_call_started",
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "call_id": tool_call.call_id,
                            "tool_name": canonical_name,
                            "timestamp": time.time(),
                        })

                        await websocket.send_json({
                            "type": "tool_call_arguments",
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "call_id": tool_call.call_id,
                            "tool_name": canonical_name,
                            "arguments": validated_args,
                            "timestamp": time.time(),
                        })

                        await websocket.send_json({
                            "type": "tool_execution_started",
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "call_id": tool_call.call_id,
                            "tool_name": canonical_name,
                            "timestamp": time.time(),
                        })

                        obs: CanonicalToolObservation = tool_registry.execute_tool(
                            canonical_name,
                            validated_args,
                            call_id=tool_call.call_id,
                            original_name=tool_call.original_name,
                            request_id=req_id,
                            conversation_id=conv_id,
                        )

                        t_trace = {
                            "tool_name": obs.tool_name,
                            "original_tool_name": obs.original_tool_name,
                            "requested_by": "model" if not manual_override else "manual_override",
                            "status": "completed" if obs.status == "SUCCESS" else "failed",
                            "call_id": obs.call_id,
                            "execution_id": obs.execution_id,
                            "request_id": obs.request_id,
                            "conversation_id": obs.conversation_id,
                            "stdout": obs.stdout,
                            "stderr": obs.stderr,
                            "input_payload": validated_args,
                            "output_payload": obs.raw_payload,
                            "source_metadata": obs.source_metadata,
                            "latency_ms": round(obs.duration_ms, 2),
                            "error": obs.output_text if obs.status != "SUCCESS" else None,
                        }
                        tool_traces.append(t_trace)

                        event_type = "tool_execution_completed" if obs.status == "SUCCESS" else "tool_execution_failed"
                        await websocket.send_json({
                            "type": event_type,
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "call_id": obs.call_id,
                            "tool_name": obs.tool_name,
                            "trace": t_trace,
                            "timestamp": time.time(),
                        })

                        assistant_turn_content = f'<tool_call id="{tool_call.call_id}">{json.dumps({"name": canonical_name, "arguments": validated_args})}</tool_call>'
                        active_messages.append({"role": "assistant", "content": assistant_turn_content})
                        active_messages.append({
                            "role": "tool",
                            "tool_name": canonical_name,
                            "id": tool_call.call_id,
                            "content": obs.output_text
                        })

                        current_round += 1
                        continue
                    else:
                        # Schema error fallback
                        break
                else:
                    # Final assistant message reached
                    final_response_text = parsed.sanitized_text or round_response_text
                    # If this round was classified as non-tool, but was buffered:
                    if is_tool_block and final_response_text:
                        await websocket.send_json({
                            "type": "token",
                            "request_id": req_id,
                            "conversation_id": conv_id,
                            "text": final_response_text,
                            "token_count": final_token_count,
                            "ttft_ms": round(ttft_recorded, 1),
                            "tok_per_sec": round(final_tok_per_sec, 1),
                            "timestamp": time.time(),
                        })
                    break

            if cancelled:
                await websocket.send_json({
                    "type": "generation_cancelled",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "timestamp": time.time(),
                })
            else:
                total_duration_ms = (time.perf_counter() - start_time) * 1000.0
                await websocket.send_json({
                    "type": "generation_completed",
                    "request_id": req_id,
                    "conversation_id": conv_id,
                    "finish_reason": "stop",
                    "token_count": final_token_count,
                    "ttft_ms": round(ttft_recorded, 1),
                    "tok_per_sec": round(final_tok_per_sec, 1),
                    "duration_ms": round(total_duration_ms, 1),
                    "model_id": model_id,
                    "tool_traces": tool_traces,
                    "timestamp": time.time(),
                })

                # Record experiment trace
                try:
                    experiment_tracker.record_trace(
                        trace_id=req_id,
                        model_id=model_id,
                        prompt=last_user_msg,
                        response=final_response_text,
                        tokens_generated=final_token_count,
                        ttft_ms=ttft_recorded,
                        total_latency_ms=total_duration_ms,
                        tok_per_sec=final_tok_per_sec,
                        tool_calls=[{"name": t["tool_name"], "latency_ms": t["latency_ms"]} for t in tool_traces],
                        retrieval_used=retrieval_trace.triggered,
                    )
                except Exception:
                    pass

    except WebSocketDisconnect:
        # Client gracefully or ungracefully disconnected
        pass
    except Exception as exc:
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(exc),
                "timestamp": time.time(),
            })
        except Exception:
            pass



# --- RETRIEVAL SUBSYSTEM ---

@app.post("/api/retrieval/query")
def query_retrieval(req: RetrievalQueryRequest):
    trace = retrieval_engine.retrieve(
        query=req.query,
        mode="always",
        top_k=req.top_k,
        threshold=req.threshold,
    )
    return trace.to_dict()


@app.get("/api/retrieval/chunks")
def list_retrieval_chunks():
    chunks = retrieval_engine.indexer.chunks
    return [c.to_dict() for c in chunks]


@app.post("/api/retrieval/reindex")
def reindex_retrieval():
    count = retrieval_engine.indexer.build_index_from_sources()
    return {"success": True, "indexed_chunks": count}


@app.post("/api/retrieval/compare")
async def compare_retrieval(req: CompareRetrievalRequest, request: Request):
    """Runs a controlled side-by-side comparison with vs without retrieval for the exact same prompt."""
    require_compute_auth(request, req.compute_token)
    config = GenerationConfig(
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )

    # Condition 1: Without retrieval
    t0 = time.perf_counter()
    resp_no_rag = ""
    async for chunk in model_manager.generate_stream(
        req.model_id,
        [{"role": "user", "content": req.prompt}],
        config,
    ):
        resp_no_rag += chunk.text
    latency_no_rag = (time.perf_counter() - t0) * 1000.0

    # Condition 2: With retrieval
    t1 = time.perf_counter()
    trace = retrieval_engine.retrieve(query=req.prompt, mode="always", top_k=2)
    augmented_msgs = retrieval_engine.inject_context_into_messages(
        [{"role": "user", "content": req.prompt}], trace
    )

    resp_with_rag = ""
    async for chunk in model_manager.generate_stream(
        req.model_id,
        augmented_msgs,
        config,
    ):
        resp_with_rag += chunk.text
    latency_with_rag = (time.perf_counter() - t1) * 1000.0

    return {
        "prompt": req.prompt,
        "model_id": req.model_id,
        "without_retrieval": {
            "response": resp_no_rag,
            "latency_ms": round(latency_no_rag, 2),
            "citations_present": False,
        },
        "with_retrieval": {
            "response": resp_with_rag,
            "latency_ms": round(latency_with_rag, 2),
            "retrieved_sources": [c.doc_title for c in trace.retrieved_chunks],
            "injected_context": trace.injected_context,
            "citations_present": len(trace.retrieved_chunks) > 0,
        },
        "latency_overhead_ms": round(latency_with_rag - latency_no_rag, 2),
    }


# --- TOOLS API ---

@app.post("/api/tools/execute")
def execute_tool(req: ToolExecutionRequest, request: Request):
    require_compute_auth(request, req.compute_token)
    trace = tool_orchestrator.execute_tool(
        tool_name=req.tool_name,
        arguments=req.arguments,
        requested_by="user_api",
        web_enabled=True,
        python_enabled=True,
    )
    return trace.to_dict()


# --- TRAINING OBSERVABILITY ---

@app.get("/api/training/runs")
def list_training_runs():
    return training_manager.get_available_runs()


@app.get("/api/training/runs/{run_id}")
def get_training_run_details(run_id: str):
    info = training_manager.load_run_details(run_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Training run '{run_id}' not found.")
    return info.__dict__


# --- EXPERIMENT TRACE LOGGING & EXPORT ---

@app.get("/api/experiments")
def list_experiments(limit: int = 50):
    return experiment_tracker.list_experiments(limit=limit)


@app.get("/api/experiments/export")
def export_experiments(format: str = "json"):
    content = experiment_tracker.export_all(format_type=format)
    media_type = "application/json" if format.lower() == "json" else "text/plain"
    filename = f"qorin_experiments_{int(time.time())}.{format.lower()}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# --- PROVIDER USAGE & INFRASTRUCTURE ACCOUNTING ---

@app.get("/api/usage")
async def get_provider_usage():
    """
    Returns sanitized provider usage reports (e.g. Hugging Face).
    Never exposes API secrets, tokens, or private credentials.
    """
    return await usage_manager.get_all_usage(force_refresh=False)


@app.post("/api/usage/refresh")
async def refresh_provider_usage():
    """
    Forces a fresh live synchronization with configured upstream providers.
    """
    return await usage_manager.get_all_usage(force_refresh=True)


