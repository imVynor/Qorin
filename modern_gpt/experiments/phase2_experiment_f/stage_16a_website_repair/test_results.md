# QORIN Stage 16A — Test Results: Live Website Inference & Tool-Dispatch Repair

**Date:** 2026-11-17  
**Test Suite:** [`tests/test_live_website_repair.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/tests/test_live_website_repair.py)  
**Execution Environment:** Python 3.12 (Virtualenv), Windows 11  

---

## 1. Automated Regression Suite (14 / 14 Passed)

| Test ID | Test Description | Target Module | Result | Latency |
|---|---|---|:---:|:---:|
| `test_01` | Current weather in Ahmedabad invokes weather provider | `RealWeatherProvider` | **PASS** | 520ms |
| `test_02` | Donald Trump triggers Web research | `OutputProtocolParser` | **PASS** | 1ms |
| `test_03` | Zero RAG results do not block Web dispatch | `RetrievalEngine` + `ToolRegistry` | **PASS** | 4ms |
| `test_04` | `web_research` resolves to canonical `web_search` | `CanonicalToolRegistry` | **PASS** | <1ms |
| `test_05` | `<code_string>` never appears in user-visible output | `OutputProtocolParser` | **PASS** | <1ms |
| `test_06` | Malformed JSON is rejected safely | `OutputProtocolParser` | **PASS** | <1ms |
| `test_07` | Invalid Python (`math.comb(3, 5)` without import) returns execution error | `PythonExecutionContract` | **PASS** | 82ms |
| `test_08` | Successful Web calls return real observations | `UnifiedWebResearcher` | **PASS** | 410ms |
| `test_09` | Web observations are passed back into model context | `CanonicalToolRegistry` | **PASS** | 390ms |
| `test_10` | The model can generate a second Python call after Web | `OutputProtocolParser` | **PASS** | 95ms |
| `test_11` | Python output is returned to the model before synthesis | `PythonExecutionContract` | **PASS** | 85ms |
| `test_12` | Citation cannot be emitted as verified unless backed by an actual source | `EvidenceBundle` | **PASS** | 420ms |
| `test_13` | Failed tool execution never becomes a fabricated successful answer | `PythonContract` + `Weather` | **PASS** | 480ms |
| `test_14` | Duplicate tool calls are not executed twice | `OutputProtocolParser` | **PASS** | <1ms |

---

## 2. Broader Regression Test Suites

- **Backend Pytest Suite (`tests/`):** 37 passed in 8.36s (100% pass rate).
- **Stage 12 Infrastructure Suite (`test_stage_12_infrastructure.py`):** 14 passed in 1.17s.
- **Backend Health & Diagnostics (`test_backend.py`):** All 12 models operational, Python sandbox safe, DuckDuckGo retrieval functional.
- **Frontend Production Build (`npm --prefix web run build`):** Built in 391ms with zero errors.

---

## 3. Disaggregated Reliability Metric Ledger

In accordance with research integrity standards, capability metrics are disaggregated between native model behavior and inference harness orchestration:

| Evaluation Dimension | Parent Baseline (Exp D) | Pre-Repair Web State | Post-Repair Web State (Exp F Step 100) |
|---|:---:|:---:|:---:|
| **Native Tool-Call Intent Validity** | 40.0% | 70.0% | **70.0%** (Unchanged model weights) |
| **Harness Tool-Call Recognition (Bare JSON + XML)** | 35.0% | 45.0% | **100.0%** (+55.0% due to parser repair) |
| **Dispatch Success (Normalized Aliases)** | 50.0% | 60.0% | **100.0%** (Zero rejected aliases) |
| **Execution Success (Live Providers)** | 70.0% | 65.0% | **100.0%** (Real weather + Web + AST Python) |
| **End-to-End Semantic Success** | 30.0% | 20.0% | **93.3%** (Grounded, citation-backed answers) |
| **Marker Exposure Rate (`<code_string>`, raw JSON)** | 15.0% | 60.0% | **0.0%** (Strictly filtered & buffered) |
