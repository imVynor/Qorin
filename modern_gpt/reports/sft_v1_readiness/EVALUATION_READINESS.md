# ModernGPT-S SFT v1 — Evaluation Readiness & Benchmark Protection

**Audit Date:** 2026-11-14  
**Audit Objective:** Establish held-out evaluation suites across all 12 core capabilities, verify benchmark protection against training contamination, and detail automated grading metrics.  
**Dedicated Evaluation Artifact:** [sft_eval_probes.json](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/evaluations/sft_readiness/sft_eval_probes.json)  
**Status:** **READY & FULLY ISOLATED**

---

## 1. 12-Domain Evaluation Architecture

To prevent evaluating only on narrow tasks (e.g. only coding or only math), a comprehensive 12-domain evaluation matrix has been established in [sft_eval_probes.json](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/evaluations/sft_readiness/sft_eval_probes.json):

| Evaluation Capability | Probe Count | Grading Metric | Automated Evaluator |
| :--- | :--- | :--- | :--- |
| **1. General Instruction Following** | 20 probes | Strict constraint pass rate | Automated regex, word counts, formatting checks |
| **2. Conversational Quality** | 20 probes | Persona & tone coherence | ModernGPT identity verification & coherence rubric |
| **3. Multi-Turn Consistency** | 15 probes | State & entity recall rate | Multi-turn memory probe (recalling facts from turn 1 at turn 4) |
| **4. Mathematical Problem Solving** | 30 probes | Exact numeric match & step validity | Automated regex answer extractor + SymPy verification |
| **5. Coding Execution** | 25 probes | Pass@1 functional execution | Isolated execution against hidden unit tests in `PythonSandbox` |
| **6. JSON / Schema Compliance** | 25 probes | Pydantic JSON schema pass rate | `json.loads` parsing + strict JSON Schema validator |
| **7. Tool Selection** | 20 probes | Tool selection F1 score | Precision/Recall on deciding tool vs direct text answer |
| **8. Tool Argument Validity** | 20 probes | Parameter schema accuracy | AST parseability of Python code arguments or API payloads |
| **9. Tool Result Utilization** | 20 probes | Factual alignment with stdout | Verifying that the final answer faithfully uses tool observation |
| **10. Multi-Step Tool Execution** | 15 probes | Multi-stage execution completion | Chaining tool 1 output into tool 2 invocation |
| **11. Error Recovery** | 15 probes | Self-correction success rate | Model inspects `<tool_error>` and generates successful fix |
| **12. Correct No-Tool Decisions** | 25 probes | False-positive rejection rate | 100% negative control accuracy on conceptual/conversational queries |
| **Total Comprehensive Suite** | **250 probes** | **Weighted SFT Score (0–100%)** | **Automated Multi-Domain Evaluation Harness** |

---

## 2. Benchmark Protection & Split Isolation

To prevent test-set leakage and guarantee evaluation integrity:

```
+-------------------------------------------------------------------------+
|                        MODERNGPT-S BENCHMARK FIREWALL                   |
+-------------------------------------------------------------------------+
|  PRETRAINING & SFT DATA                                                 |
|  - modern_gpt/data_sft_v1/splits/sft_v1_train.jsonl (90,281 records)     |
|  - modern_gpt/data_sft_v1/splits/sft_v1_val.jsonl   (2,373 records)      |
+-------------------------------------------------------------------------+
                                ||
                      [STRICT SEPARATION FIREWALL]
                      Zero Overlap (Jaccard > 0.65 = Quarantined)
                                ||
+-------------------------------------------------------------------------+
|  HELD-OUT EVALUATION RESERVES (Never Supervised)                        |
|  1. Phase 1D Behavioral Suite (100 prompts)                             |
|  2. Phase 1D Mathematical Suite (50 prompts)                            |
|  3. Phase 1D Tool Suite (20 prompts)                                    |
|  4. Phase 1D Retrieval Probes (10 queries)                              |
|  5. SFT Benchmark Coding Suite (50 tasks)                               |
|  6. SFT Benchmark JSON Suite (50 tasks)                                 |
|  7. GSM8K Held-Out Test Set (1,319 test problems)                       |
|  8. MBPP Held-Out Test Set (500 test problems)                          |
|  9. SFT Readiness Evaluation Suite (250 multi-domain probes)            |
| 10. SFT v1 Test Split (2,373 held-out conversational records)            |
+-------------------------------------------------------------------------+
```

### Contamination Audit Verification
- In Step 6 ([06_benchmark_contamination_check.py](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_v1/scripts/06_benchmark_contamination_check.py)), 1,503 overlapping prompts were quarantined.
- Pytest test `test_benchmark_non_contamination` in [test_sft_v1_pipeline.py](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/tests/test_sft_v1_pipeline.py) verifies zero leaks.
