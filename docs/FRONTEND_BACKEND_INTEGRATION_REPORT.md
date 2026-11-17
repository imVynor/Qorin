# QORIN — Full Frontend–Backend Integration & Live WebSocket Streaming Report

**Repository**: `https://github.com/imVynor/Qorin`  
**Frontend Deployment**: GitHub Pages (`https://imvynor.github.io/Qorin/`)  
**Backend Deployment**: Hugging Face Spaces (`https://vynor-stt4.hf.space`, `wss://vynor-stt4.hf.space`)  
**Audit & Implementation Date**: November 2026  
**Status**: Production Verified  

---

## 1. Executive Summary & Architecture Overview

The QORIN research interface has been upgraded from a hybrid REST/SSE prototype to a persistent, bidirectional WebSocket-powered interactive language model harness. The architecture bridges the modern React 19 / TypeScript / Vite frontend client with the FastAPI / Uvicorn Python backend running on CPU execution environments.

```
+-----------------------------------------------------------------------------------+
|                            QORIN Interactive Frontend                             |
|                                                                                   |
|  +------------------------+  +------------------------+  +---------------------+  |
|  |     Harness Page       |  |     ChatArea UI        |  |  Inspection Panel   |  |
|  +-----------+------------+  +-----------+------------+  +----------+----------+  |
|              |                           |                          |             |
|              +---------------------------+--------------------------+             |
|                                          |                                        |
|                          +---------------+---------------+                        |
|                          |    useStreamingChat Hook      |                        |
|                          +---------------+---------------+                        |
|                                          |                                        |
|                          +---------------+---------------+                        |
|                          |    useQorinConnection Hook    |                        |
|                          +---------------+---------------+                        |
|                                          |                                        |
|                 +------------------------+------------------------+               |
|                 |                                                 |               |
|  +--------------v---------------+                 +---------------v------------+  |
|  |   connectionManager.ts       |                 |   websocketClient.ts       |  |
|  |   - Health & telemetry poll  |                 |   - WSS socket lifecycle   |  |
|  |   - Auth token state (HMAC)  |                 |   - 15 typed protocol evts |  |
|  |   - Space waking detection   |                 |   - Exponential backoff    |  |
|  +--------------+---------------+                 +---------------+------------+  |
+-----------------|-------------------------------------------------|---------------+
                  | HTTP/HTTPS (REST)                               | WSS (WebSocket)
                  | /api/health, /api/models                        | /ws/chat
+-----------------v-------------------------------------------------v---------------+
|                        FastAPI / Uvicorn Backend Server                           |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                            /ws/chat Endpoint                                |  |
|  |  - Connection Ack & Heartbeats (15s interval)                               |  |
|  |  - HMAC Session Authorization & Rate Limiting                               |  |
|  |  - Asynchronous Request Cancellation Map                                    |  |
|  |  - Incremental Token Yielding (stream_chunks)                               |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        |                                          |
|                +-----------------------+-----------------------+                  |
|                |                                               |                  |
|  +-------------v-------------+                   +-------------v---------------+  |
|  |    ModelManager           |                   |    ToolRegistry             |  |
|  |  - Lazy weight loading    |                   |  - Web search (DuckDuckGo)  |  |
|  |  - Float32 CPU inference  |                   |  - Python Sandbox (Isolated)|  |
|  |  - 4 canonical models     |                   |  - Weather (Open-Meteo)     |  |
|  +---------------------------+                   +-----------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. WebSocket Communication Protocol Specification

The WebSocket endpoint is mounted at `/ws/chat` on the FastAPI application (`backend/app.py`). It adheres to a strictly defined JSON event schema supporting bidirectional communication, heartbeats, and client-side cancellation.

### 2.1 Connection Lifecycle & Protocol Events

| Event Type | Direction | Payload Schema | Purpose |
|---|---|---|---|
| `connection_ack` | Server &rarr; Client | `{"type": "connection_ack", "timestamp": float, "active_model": str, "device": "cpu", "version": "1.0.0"}` | Acknowledges socket handshake and declares runtime device. |
| `ping` | Client &rarr; Server | `{"type": "ping", "timestamp": int}` | Keep-alive heartbeat packet sent every 15 seconds. |
| `heartbeat` | Server &rarr; Client | `{"type": "heartbeat", "timestamp": float}` | Server heartbeat response confirming active socket state. |
| `chat` | Client &rarr; Server | `{"type": "chat", "request_id": str, "conversation_id": str, "message": str, "model_id": str, "compute_token": str, ...}` | Initiates model generation request. |
| `request_accepted` | Server &rarr; Client | `{"type": "request_accepted", "request_id": str, "conversation_id": str, "model_id": str, "timestamp": float}` | Confirms acceptance and begins pipeline orchestration. |
| `model_loading` | Server &rarr; Client | `{"type": "model_loading", "model_id": str, "timestamp": float}` | Notifies UI that model weights are loading into RAM. |
| `model_ready` | Server &rarr; Client | `{"type": "model_ready", "model_id": str, "timestamp": float}` | Notifies UI that model weights are fully initialized. |
| `retrieval` | Server &rarr; Client | `{"type": "retrieval", "request_id": str, "conversation_id": str, "trace": RetrievalTrace, "timestamp": float}` | Emits non-parametric knowledge retrieval passages and scores. |
| `generation_started` | Server &rarr; Client | `{"type": "generation_started", "request_id": str, "conversation_id": str, "model_id": str, "timestamp": float}` | Signals start of token decoding. |
| `token` | Server &rarr; Client | `{"type": "token", "request_id": str, "conversation_id": str, "text": str, "token_count": int, "ttft_ms": float, "tok_per_sec": float, "timestamp": float}` | Streams incremental text chunks in real time. |
| `tool_call_started` | Server &rarr; Client | `{"type": "tool_call_started", "request_id": str, "call_id": str, "tool_name": str, "timestamp": float}` | Alerts UI that model has emitted a tool invocation. |
| `tool_call_arguments` | Server &rarr; Client | `{"type": "tool_call_arguments", "request_id": str, "call_id": str, "tool_name": str, "arguments": dict, "timestamp": float}` | Streams validated tool input arguments. |
| `tool_execution_started` | Server &rarr; Client | `{"type": "tool_execution_started", "request_id": str, "call_id": str, "tool_name": str, "timestamp": float}` | Alerts UI that external tool execution has begun. |
| `tool_execution_completed` | Server &rarr; Client | `{"type": "tool_execution_completed", "request_id": str, "call_id": str, "tool_name": str, "trace": ToolTrace, "timestamp": float}` | Emits successful tool outputs, stdout, and execution latency. |
| `tool_execution_failed` | Server &rarr; Client | `{"type": "tool_execution_failed", "request_id": str, "call_id": str, "tool_name": str, "trace": ToolTrace, "timestamp": float}` | Emits execution failure, traceback, or timeout error. |
| `cancel` | Client &rarr; Server | `{"type": "cancel", "request_id": str, "conversation_id": str}` | Signals server to abort active token generation. |
| `generation_cancelled` | Server &rarr; Client | `{"type": "generation_cancelled", "request_id": str, "conversation_id": str, "timestamp": float}` | Confirms generation cancellation to client. |
| `generation_completed` | Server &rarr; Client | `{"type": "generation_completed", "request_id": str, "conversation_id": str, "trace_id": str, "total_tokens": int, "ttft_ms": float, "total_latency_ms": float, "tok_per_sec": float, "tools_executed": int, "tool_traces": list, "timestamp": float}` | Signals final completion and records comprehensive telemetry. |
| `error` | Server &rarr; Client | `{"type": "error", "request_id": str, "error": str, "code": str, "timestamp": float}` | Emits structured error information. |

### 2.2 Reconnection & Resilience Logic

The client (`web/src/services/websocketClient.ts`) implements exponential backoff reconnection:
- **Base delay**: 1,000 ms
- **Multiplier**: 2.0x per failure
- **Max delay**: 15,000 ms
- **Jitter**: Randomized +0–500 ms to prevent thundering herd spikes.
- **Heartbeat ping**: Dispatched every 15,000 ms; closes stale connections if two consecutive cycles receive no socket response.

---

## 3. Compute Protection & Authorization Architecture

Live model inference and external tool execution consume dedicated CPU and server resources. To protect against denial-of-service and runaway cloud compute charges, execution requires authenticated session authorization.

### 3.1 Security Principles
1. **Zero Secret Leakage**: No passwords, master keys, or private salts are hardcoded in the frontend repository or emitted in client-side bundles.
2. **HMAC-SHA256 Signed Tokens**: Successful verification issues a cryptographically signed session token:
   ```
   qorin_tok_<base64_payload>.<hex_hmac_signature>
   ```
   Payload structure:
   ```json
   {
     "sub": "evaluator_session",
     "exp": 1795000000,
     "jti": "8f3a9e2c1b4d..."
   }
   ```
3. **Session Revocation**: A dedicated `/api/auth/logout` endpoint records `jti` identifiers into a server-side `REVOKED_SESSION_JTI` blacklist, instantly invalidating logged-out tokens.
4. **Polite Client Guidance**: When an unauthenticated client interacts with the harness, the backend emits `{"type": "error", "code": "AUTH_REQUIRED"}`. The frontend displays a non-intrusive banner:
   > *"Protected Compute Locked: Live model inference requires compute authorization. Open Options → Unlock Compute to activate research access."*
   An **"Unlock Compute"** button opens the modal without interrupting or crashing the user session.

---

## 4. Token-by-Token Streaming & UI Engine

### 4.1 Incremental Text Rendering
The streaming subsystem in `web/src/hooks/useStreamingChat.ts` avoids full-array re-renders during inference:
- Incoming `token` events deliver discrete chunks (`event.text`).
- Tokens are appended directly to the active live stream buffer (`accumulatedText += event.text`).
- Live metrics (`ttft_ms`, `tok_per_sec`, `token_count`) update dynamically in a mono-spaced stats ticker below the streaming bubble.
- Once generation finishes (`generation_completed`), the accumulated string is promoted to the persistent message array as a frozen `ChatMessage` object, and the active stream buffer is cleared.

### 4.2 Auto-Scroll and Scroll Preservation
The scroll management in `web/src/components/harness/ChatArea.tsx` adheres strictly to user ergonomics:
- **Scroll Lock Threshold**: If the user's viewport is within 60px of the bottom (`distanceToBottom <= 60`), `isAutoScrollLocked` remains `true`, and new tokens smoothly scroll the view.
- **Scroll Freezing**: If the user scrolls up to review prior answers, code snippets, or knowledge chunks, `isAutoScrollLocked` immediately toggles to `false`.
- **"Jump to latest" Button**: When scrolled away from the bottom during generation or reading, a floating pill button appears at the bottom-right of the chat window:
  ```tsx
  <button className={styles.jumpToBottomBtn} onClick={scrollToBottom}>
    <ArrowDown size={13} />
    <span>Jump to latest</span>
  </button>
  ```
  Clicking the button scrolls smoothly to the bottom and re-engages the auto-scroll lock.

---

## 5. Tool Orchestration & Multi-Turn Execution

The WebSocket streaming pipeline fully supports multi-turn tool calling and grounded evidence synthesis:

1. **Tool Invocation Parsing**: As the model generates text, raw `<tool_call>` tags are suppressed from client display.
2. **Event Emission**:
   - `tool_call_started` declares the tool name (`web_search`, `python`, `weather`).
   - `tool_call_arguments` delivers the validated JSON arguments.
   - `tool_execution_started` shows active progress in the UI.
   - `tool_execution_completed` emits latency and structured payloads.
3. **Interactive UI Cards**: Tool results are rendered inline above the model's final response:
   - **Python**: Shows stdout/stderr formatted in code blocks.
   - **Web Search**: Shows titled source links with domain attribution.
   - **Weather**: Displays verified temperature, humidity, wind speed, and observation timestamps.
4. **Grounded Synthesis**: Once observations are collected, the backend injects the tool outputs into the conversation context and prompts the model to generate a cited, evidence-grounded answer.

---

## 6. Model Compatibility & Matrix

All four canonical QORIN configurations are supported across the WebSocket protocol on CPU:

| Model ID | Public Name | Architecture | Checkpoint Source | CPU VRAM / RAM | Status |
|---|---|---|---|---|---|
| `qorin-100m-sft-v3` | QORIN-100M SFT v3 (4K Context) | 14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE s=8.0 | `vynor/qorin-models/modern_gpts_100m_sft_4k_v3` | ~650 MB | Verified Active |
| `qorin-100m-base` | QORIN-100M Baseline Pretrained | 14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE base | `vynor/qorin-models/modern_gpts_100m` | ~550 MB | Verified Active |
| `qorin-500m-sft-exp-f` | QORIN-500M SFT (Autonomous Multi-Turn) | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm | `vynor/qorin-models/modern_gpts_500m_sft_exp_f` | ~2,100 MB | Verified Active |
| `qorin-500m-pretrain-pilot` | QORIN-500M Pretrain Pilot | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm | `vynor/qorin-models/modern_gpts_500m_pilot` | ~2,100 MB | Verified Active |

---

## 7. Verification Evidence & Test Results

### 7.1 Backend Test Results (`pytest`)
All 10 integration and unit tests passed in 10.51s:

```
tests/test_websocket_chat.py::test_websocket_connection_ack PASSED       [ 10%]
tests/test_websocket_chat.py::test_websocket_heartbeat PASSED            [ 20%]
tests/test_websocket_chat.py::test_websocket_unauthorized_chat PASSED    [ 30%]
tests/test_websocket_chat.py::test_websocket_authorized_chat_streaming PASSED [ 40%]
tests/test_websocket_chat.py::test_websocket_cancellation PASSED         [ 50%]
tests/test_websocket_chat.py::test_websocket_tool_execution_events PASSED [ 60%]
backend/test_backend.py::test_health PASSED                              [ 70%]
backend/test_backend.py::test_models_list PASSED                         [ 80%]
backend/test_backend.py::test_training_runs PASSED                       [ 90%]
backend/test_backend.py::test_retrieval_query PASSED                     [100%]

======================= 10 passed, 1 warning in 10.51s ========================
```

### 7.2 Frontend Build & Lint Verification
- **Production Build**: `npm run build:pages` completed with `✓ built in 451ms`, zero errors.
- **Code Linter**: `oxlint` completed with 0 errors across all 60 files.

### 7.3 Remote Container Health Check
- Endpoint: `https://vynor-stt4.hf.space/api/health`
- Response: `{"status": "healthy", "device": "cpu", "version": "1.0.0"}`
- ZeroGPU status: 0.0s (strictly standard CPU container, fully avoiding ZeroGPU quotas).

---

## 8. Summary of Created & Updated Files

1. `backend/app.py`: Implemented `/ws/chat` endpoint with 15 events, heartbeat ping-pong, cancellation map, and tool chaining.
2. `backend/models/manager.py`: Added `is_loaded(model_id)` delegation helper.
3. `tests/test_websocket_chat.py`: Comprehensive test suite verifying connection handshake, authorization, streaming, cancellation, and tool execution events.
4. `web/src/services/websocketClient.ts`: Strongly typed WebSocket client with exponential backoff and protocol dispatcher.
5. `web/src/services/connectionManager.ts`: Centralized singleton tracking REST health, WebSocket state, and HMAC auth.
6. `web/src/hooks/useQorinConnection.ts`: React hook exposing connection status and telemetry to UI components.
7. `web/src/hooks/useStreamingChat.ts`: React hook managing token-by-token streaming, active tool traces, stats, and cancellation.
8. `web/src/components/harness/ChatArea.tsx`: Re-architected chat area with WebSocket streaming, polite compute lock banner, and auto-scroll preservation.
9. `web/src/components/harness/Harness.module.css`: Added styles for compute lock banners, unlock buttons, and loading indicators.
10. `web/src/pages/Harness.tsx`: Updated canonical model list and telemetry synchronization.
11. `web/vite.config.ts`: Added `/ws` proxy rule for seamless local development.
12. `modern_gpt/deployment/huggingface_cpu_deployment/space_deploy/`: Synchronized updated backend and requirements (`websockets>=13.0`) for deployment.
