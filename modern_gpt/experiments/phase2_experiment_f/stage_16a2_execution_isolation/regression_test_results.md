# Regression Test Results: Stage 16A.2 Python Execution Isolation

## 1. Test Suite Overview

A specialized regression test suite (`tests/test_execution_isolation.py`) containing 10 rigorous tests was executed to verify execution-result isolation, tool policy enforcement, and multi-threaded safety.

---

## 2. Unit & Isolation Test Results

Command: `.\.venv\Scripts\python.exe -m unittest tests/test_execution_isolation.py`

```text
..........
----------------------------------------------------------------------
Ran 10 tests in 0.463s

OK
```

### Individual Test Breakdown

| Test # | Test Name | Invariant Tested | Sentinel Verification | Result |
| :--- | :--- | :--- | :--- | :--- |
| **01** | `test_01_execution_returns_own_stdout` | Python execution returns its own stdout | `SENTINEL_OWN_<hex>` strictly verified in stdout | **PASSED** |
| **02** | `test_02_second_execution_does_not_return_first_stdout` | Execution 2 cannot inherit Execution 1 stdout | `SENTINEL_ALPHA` vs `SENTINEL_BETA` zero cross-contamination | **PASSED** |
| **03** | `test_03_failed_execution_cannot_reuse_previous_success` | Failed run cannot reuse previous success | `SENTINEL_SUCC` absent from failed run stderr/stdout | **PASSED** |
| **04** | `test_04_new_conversation_isolation` | New conversation cannot inherit previous results | `SENTINEL_CONV_A` absent from `conv_b` | **PASSED** |
| **05** | `test_05_concurrent_executions_preserve_ownership` | 5 concurrent threads maintain strict ownership | 5 unique thread sentinels & 5 unique execution IDs | **PASSED** |
| **06** | `test_06_streaming_sanitization_and_isolation` | Streaming strips internal markers (<code_string>) | Zero protocol marker leakage in user output | **PASSED** |
| **07** | `test_07_code_generation_intent_classification` | Code-gen queries do not invoke python tools | "rite a python function..." classified as CODE_GEN; markdown block emitted | **PASSED** |
| **08** | `test_08_simple_arithmetic_intent_and_direct_eval` | Simple arithmetic evaluated directly | "whats 1+2?" $\rightarrow$ 3; "10 * 5" $\rightarrow$ 50; 0 tools run | **PASSED** |
| **09** | `test_09_explicit_execution_returns_actual_result` | Explicit execution returns actual calculation | "Run Python to calculate average..." executed cleanly | **PASSED** |
| **10** | `test_10_call_id_and_execution_id_pairing` | Call ID & Execution ID remain correctly paired | `call_req_..._001` paired with `exec_...` throughout | **PASSED** |

---

## 3. Full Repository Test Suite Status

Command: `.\.venv\Scripts\pytest.exe tests/`

```text
============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\Anish\Desktop\Projects\ML-course
collected 47 items

tests\test_capability_isolation_and_safety.py .........                  [ 19%]
tests\test_execution_isolation.py ..........                             [ 40%]
tests\test_live_website_repair.py ..............                         [ 70%]
tests\test_model_manifest_validation.py ....                             [ 78%]
tests\test_web_retrieval_regression.py ..........                        [100%]

============================= 47 passed in 8.29s ==============================
```

Zero regressions across all 47 tests.
