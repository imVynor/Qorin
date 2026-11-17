# Backend Architecture Inventory & Technical Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/BACKEND_ARCHITECTURE.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Executive Summary & Entry Point

The QORIN inference backend is an asynchronous Python service built on **FastAPI** (`v0.162.5`) and **Starlette** (`v1.10.0`), running on **Uvicorn** (`v0.54.0`). It serves as the unified orchestration layer connecting:
1. Native PyTorch small language model checkpoints (`ModernGPT` in 16M, 50M, 100M, and 500M scales).
2. Non-parametric TF-IDF knowledge retrieval with cosine similarity filtering.
3. Subprocess-isolated mathematical and algorithmic Python execution sandboxing.
4. Live web research via DuckDuckGo Lite and Wikipedia with SQLite query caching.
5. Multi-turn closed-loop tool decision and grounded synthesis state machines.
6. Training telemetry observability and JSONL experiment trace logging.
7. Evaluator compute protection gating interactive execution behind course professor authorization (`UDIT BHATIA`).

### Entry Point & Runtime Invocation
- **Primary Entry Point:** `backend/app.py`
- **Application Object:** `app = FastAPI(...)`
- **CLI Startup Command:**
  ```bash
  .venv/Scripts/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
  ```
- **Process Working Directory:** Repository root (`REPO_ROOT`), enforced via dynamic `sys.path.insert(0, str(REPO_ROOT))`.

---

## 2. Framework & Singleton Initializations

Upon module load, `backend/app.py` instantiates nine core singleton managers:

| Singleton | Instantiated Class | Source File | Responsibilities |
| :--- | :--- | :--- | :--- |
| `model_manager` | `ModelManager` | `backend/models/manager.py` | Multiplexes native `QorinModelProvider` and external `OllamaModelProvider`. Manages VRAM allocation, model loading/unloading, and asynchronous token streaming. |
| `retrieval_engine` | `RetrievalEngine` | `backend/retrieval/engine.py` | Evaluates incoming queries, computes TF-IDF vector similarity over indexed Markdown chunks, and injects context into message history. |
| `weather_provider` | `RealWeatherProvider` | `backend/tools/weather_provider.py` | Queries live meteorological conditions via Open-Meteo REST API. |
| `tool_registry` | `CanonicalToolRegistry` | `backend/tools/registry.py` | Maintains canonical definitions, schemas, aliases, and argument validators for `web_search`, `python`, and `weather`. |
| `output_parser` | `OutputProtocolParser` | `backend/tools/output_protocol.py` | Regex and AST parser that extracts `<tool_call id="...">{"name": "...", "arguments": {...}}</tool_call>` structures and handles malformed tool calls. |
| `tool_orchestrator` | `ToolOrchestrator` | `backend/tools/orchestrator.py` | Dispatches validated tool invocations to either the subprocess sandbox (`PythonSandbox`) or web researcher (`UnifiedWebResearcher`). |
| `two_stage_protocol` | `QorinTwoStageProtocol` | `backend/tools/two_stage_protocol.py` | Enforces Turn 1 (Tool Call) -> Turn 2 (Tool Observation) -> Turn 3 (Grounded Synthesis) multi-turn lifecycle. |
| `training_manager` | `TrainingDashboardManager` | `backend/training/dashboard.py` | Aggregates training loss logs, downsamples metric series, and reports thermal/duty cycle metrics. |
| `experiment_tracker` | `ExperimentTracker` | `backend/experiments/tracker.py` | Appends structured experiment traces to `backend/data/experiment_traces.jsonl`. |
| `usage_manager` | `UsageManager` | `backend/usage/manager.py` | Manages extensible provider usage accounting, queries Hugging Face WhoAmI API, and reports quotas. |

---

## 3. Comprehensive Endpoint Inventory

The backend exposes **17 REST and Server-Sent Event (SSE) endpoints**.

| Endpoint | Method | Purpose | Authentication | Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Reports backend operational status, active model ID, and CUDA/VRAM telemetry. | Public | `model_manager` |
| `/api/models` | `GET` | Lists all cataloged native QORIN checkpoints and available external Ollama models with metadata. | Public | `model_manager`, `qorin_provider`, `ollama_provider` |
| `/api/auth/verify-compute` | `POST` | Validates evaluator password (`UDIT BHATIA`) and returns authorization token (`UDIT_BHATIA_AUTHORIZED`). | Public (Gatekeeper) | `COMPUTE_ACCESS_PASSWORD` |
| `/api/models/select` | `POST` | Loads selected model checkpoint into GPU VRAM (or CPU RAM) and unloads previous model. | **Required** (`X-Compute-Token` or `password`) | `model_manager`, PyTorch, CUDA |
| `/api/models/unload` | `POST` | Unloads current model from VRAM and triggers garbage collection (`gc.collect()`, `torch.cuda.empty_cache()`). | **Required** (`X-Compute-Token` or `password`) | `model_manager` |
| `/api/chat/stream` | `POST` | Multi-turn SSE token streaming endpoint with closed-loop tool augmentation and retrieval injection. | **Required** (`X-Compute-Token` or `password`) | `model_manager`, `retrieval_engine`, `output_parser`, `tool_orchestrator`, `experiment_tracker` |
| `/api/retrieval/query` | `POST` | Queries TF-IDF index directly with a text prompt and returns ranked chunk scores. | Public | `retrieval_engine` |
| `/api/retrieval/chunks` | `GET` | Returns all pre-indexed document chunks with titles, source paths, and token counts. | Public | `retrieval_engine.indexer` |
| `/api/retrieval/reindex` | `POST` | Re-indexes all Markdown files in the knowledge base and regenerates `knowledge_index.json`. | Public | `retrieval_engine.indexer` |
| `/api/retrieval/compare` | `POST` | Runs a side-by-side generation test: Condition A (parametric only) vs Condition B (retrieval-augmented). | **Required** (`X-Compute-Token` or `password`) | `model_manager`, `retrieval_engine` |
| `/api/tools/execute` | `POST` | Direct tool invocation endpoint for executing isolated Python scripts or web research. | **Required** (`X-Compute-Token` or `password`) | `tool_orchestrator`, `PythonSandbox`, `UnifiedWebResearcher` |
| `/api/training/runs` | `GET` | Lists available pretraining and SFT run summaries (500M, 100M, 50M). | Public | `training_manager` |
| `/api/training/runs/{run_id}` | `GET` | Returns detailed downsampled training metrics, loss curves, and thermal duty cycle data for a run. | Public | `training_manager` |
| `/api/experiments` | `GET` | Lists recent structured inference traces from `backend/data/experiment_traces.jsonl`. | Public | `experiment_tracker` |
| `/api/experiments/export` | `GET` | Exports complete experiment trace history in JSON or JSONL format for download. | Public | `experiment_tracker` |
| `/api/usage` | `GET` | Returns live and cached provider usage accounting (Hugging Face WhoAmI status, tier, plan). | Public | `usage_manager` |
| `/api/usage/refresh` | `POST` | Forces cache bypass and queries upstream Hugging Face API for updated provider status. | Public | `usage_manager` |

---

## 4. Request & Response Schemas

All schemas inherit from Pydantic `BaseModel` (`pydantic v2.14.1`):

1. **`MessageItem`**:
   - `role: str` ("system" | "user" | "assistant" | "tool")
   - `content: str`
2. **`ChatRequest`**:
   - `model_id: str` (default: `"qorin-100m-sft-v3"`)
   - `messages: List[MessageItem]`
   - `temperature: float = 0.6`, `top_p: float = 0.9`, `top_k: int = 50`
   - `max_new_tokens: int = 250`, `repetition_penalty: float = 1.15`
   - `system_prompt: Optional[str] = None`
   - `retrieval_mode: str = "auto"` ("off" | "always" | "auto")
   - `retrieval_top_k: int = 3`, `retrieval_threshold: float = 0.08`
   - `web_research_enabled: bool = False`, `python_sandbox_enabled: bool = False`, `weather_enabled: bool = True`
   - `manual_tool_override: Optional[str] = None`
   - `request_id: Optional[str] = None`, `conversation_id: Optional[str] = None`
   - `compute_token: Optional[str] = None`
3. **`ModelSelectRequest`**:
   - `model_id: str`, `device: Optional[str] = None`, `compute_token: Optional[str] = None`
4. **`ToolExecutionRequest`**:
   - `tool_name: str`, `arguments: Dict[str, Any]`, `compute_token: Optional[str] = None`
5. **`ComputeAuthRequest`**:
   - `password: str`
6. **`CompareRetrievalRequest`**:
   - `model_id: str`, `prompt: str`, `temperature: float = 0.6`, `max_new_tokens: int = 150`, `compute_token: Optional[str] = None`

---

## 5. Middleware & CORS Configuration

CORS is managed through Starlette's `CORSMiddleware`:
```python
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
```
- **Allow Headers:** `["*"]` ensures custom headers `X-Compute-Token` and `Authorization` pass preflight inspection.
- **Allow Credentials:** `True` enables credentialed cross-origin fetches.
- **Origins:** Covers Vite local dev server (`5173`), Vite preview (`4173`), and production GitHub Pages (`https://imvynor.github.io`).

---

## 6. Authentication & Authorization Architecture

- **Protected Compute Gate:** Restricts access to the author's local workstation GPU and Python subprocess sandbox.
- **Target Password:** `UDIT BHATIA` (normalized via `clean = token_val.strip().upper()`).
- **Valid Bearer Token:** `UDIT_BHATIA_AUTHORIZED`.
- **Enforcement Helper:**
  ```python
  def check_compute_auth(token_val: Optional[str]) -> bool:
      if not token_val:
          return False
      clean = token_val.replace("Bearer ", "").strip().upper()
      return clean == COMPUTE_ACCESS_PASSWORD or clean == COMPUTE_TOKEN_VALID

  def require_compute_auth(request: Request, body_token: Optional[str] = None):
      token = request.headers.get("X-Compute-Token") or request.headers.get("Authorization") or body_token
      if check_compute_auth(token):
          return True
      raise HTTPException(
          status_code=401,
          detail="Personal compute access protected. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)",
          headers={"WWW-Authenticate": "Bearer"},
      )
  ```
- **Public vs Protected:**
  - *Public (Zero Auth Required):* Documentation, health checks, model catalog, retrieval chunk inspection, training metrics, experiment archives, usage dashboards.
  - *Protected (Auth Required):* Interactive model selection, active model unloading, live SSE chat streaming, retrieval comparison generation, and Python sandbox / web research tool execution.

---

## 7. Streaming & Protocol Implementation

- **Library:** `sse_starlette.sse.EventSourceResponse`
- **Wire Format:** Standard Server-Sent Events (`text/event-stream`).
- **SSE Event Taxonomy:**
  1. `event: retrieval`: Emitted before generation if TF-IDF retrieval triggers; payload contains chunks, similarity scores, and injected context.
  2. `event: tool_start`: Emitted when model or policy issues a tool call (`tool_name`, `arguments`, `call_id`).
  3. `event: tool_result`: Emitted when tool execution finishes; payload contains execution status, latency, stdout/sources, and evidence records.
  4. `event: token`: Emitted per decoded token chunk (`text`, `is_finished`, `token_count`, `ttft_ms`, `tok_per_sec`).
  5. `event: done`: Emitted at stream conclusion with overall generation metrics (`total_tokens`, `total_latency_ms`, `tokens_per_second`).
  6. `event: error`: Emitted if an unhandled exception or 30-second timeout occurs.

---

## 8. Inventory of Unused, Legacy & Experimental Components

1. **`backend/app.py.bak_stage0` (15.7 KB):**
   - *Status:* Legacy backup file from initial backend implementation prior to multi-turn tool orchestration. Unused by runtime.
2. **`backend/apply_app_repairs.py` (10.9 KB):**
   - *Status:* One-off repair script from Stage 16B. Unused by runtime.
3. **`backend/tools/two_stage_protocol.py` (6.4 KB):**
   - *Status:* Legacy two-stage protocol class instantiated at `app.py:71` (`two_stage_protocol = QorinTwoStageProtocol(...)`), but largely superseded by inline bounded state machine in `chat_stream()` and `backend/tools/state_machine.py`.
4. **`backend/models/ollama_provider.py` (168 lines):**
   - *Status:* Experimental test-model adapter. Functions only if an external Ollama daemon is listening on `127.0.0.1:11434`. Does not participate in native QORIN inference.
5. **`backend/data/sandbox/run_*` directories:**
   - *Status:* Transient sandbox run directories left from prior test runs. The sandbox cleans up after runs, but aborted processes can leave empty folders.
