# ModernGPT-S 500M — Phase 1D: Evaluation Methodology & Reproducibility Protocol

**Evaluation Date:** November 13, 2026  
**Harness Script:** `modern_gpt/scripts/run_phase1d_evaluation.py`  
**Evaluation Config:** `modern_gpt/reports/phase1d_500m_evaluation/evaluation_config.json`  

---

## 1. Overview and Principles

Phase 1D establishes a rigorous, automated, fully reproducible evaluation benchmark for ModernGPT-S 500M. The evaluation methodology was designed following three inviolable scientific principles:
1. **Zero Contamination:** Ground truth test splits are strictly isolated from the 101 training binary shards.
2. **Deterministic Reproducibility:** Fixed seeds, temperature settings, and isolated sandbox subprocesses ensure that identical scores are obtained on any compatible CUDA workstation.
3. **Safety Sandboxing:** All model-generated Python code is executed in isolated subprocess sandboxes with 3.0-second timeouts, memory caps, and restricted imports (`sys`, `os`, `shutil`, `socket` blocked).

---

## 2. Benchmark Suites & Datasets

### 2.1 Validation Loss & Perplexity (Domain Evaluation)
- **Source:** 7 pre-tokenized validation shards (`val_domain_*.bin`) containing exactly 20,000,000 tokens total.
- **Slice per Domain:** Exactly 25 batches of 2,048 tokens = 51,200 tokens per domain (358,400 tokens total).
- **Domain Weighting:**
  - `web`: 40.0%
  - `edu`: 15.0%
  - `code`: 15.0%
  - `wiki`: 7.5%
  - `math`: 7.5%
  - `books`: 10.0%
  - `dialogue`: 5.0%
- **Metrics:** Cross-entropy loss ($H(p, q) = -\frac{1}{N}\sum \log q(x_i)$), Perplexity ($\text{PPL} = \exp(H)$), Early vs Late position loss (First 512 tokens vs Last 512 tokens).

### 2.2 Behavioral Evaluation (100 Cases)
- **Source:** `modern_gpt/data/eval/behavioral_eval_100.jsonl`
- **Categories:**
  - Topic Switching (20 cases)
  - Irrelevant Context Suppression (20 cases)
  - Relevant Context Retention (20 cases)
  - Returning to Previous Topics (10 cases)
  - Direct Factual QA (10 cases)
  - Simple Arithmetic (10 cases)
  - Conversational Appropriateness (10 cases)
- **Evaluation Conditions:**
  - Isolated Mode: Prompt presented without conversational history.
  - Multi-Turn Mode: Prompt presented with full multi-turn conversational history.
- **Scoring:** Keyword presence, entity match, attractor loop detection ("resignation memo").

### 2.3 Coding Benchmark (50 Tasks)
- **Source:** 50 algorithmic and function synthesis tasks (`eval_coding_tasks`).
- **Phased Verification:**
  1. **AST Validity:** Parse generated code using Python's `ast.parse()`.
  2. **Sandbox Execution:** Write code into isolated temporary test harness script and execute via `subprocess.run([sys.executable, script], capture_output=True, timeout=3.0)`.
  3. **Pass@1 Assertion:** Check exit code 0 and assertion pass across hidden test cases.

### 2.4 Mathematical Reasoning (50 Tasks)
- **Source:** 50 mathematical problems across arithmetic, algebra, fractions, sequences, percentages, and word problems.
- **Scoring:** Extract numerical answer from model output and match against ground-truth integer/float within 1% tolerance.

### 2.5 Structured JSON Generation (50 Tasks)
- **Source:** 50 JSON synthesis instructions with defined schema keys.
- **Scoring:** Strict `json.loads()` parsing followed by dictionary key schema validation.

### 2.6 Tool-Calling Benchmark (20 Tasks)
- **Source:** 20 tool-intent queries (weather, calculator, web search, file reader).
- **Scoring:** Detection of tool tags (`<tool_call>`, `[TOOL]`, `{"tool": ...}`) and semantic intent alignment.

### 2.7 4-Way Retrieval-Augmented Generation Matrix (10 Tasks)
- **Source:** 10 curated factual probes.
- **Conditions:**
  - Condition A: 100M Baseline, RAG OFF
  - Condition B: 100M Baseline, RAG ON
  - Condition C: 500M Model, RAG OFF
  - Condition D: 500M Model, RAG ON

---

## 3. Reproduction Command

To reproduce the full Phase 1D evaluation suite from scratch:

```powershell
# Activate project virtual environment
.\.venv\Scripts\Activate.ps1

# Execute the master Phase 1D evaluation runner
python modern_gpt/scripts/run_phase1d_evaluation.py
```

All metrics, logs, and outputs will be automatically generated in:
- `modern_gpt/reports/phase1d_500m_evaluation/results.json`
- `modern_gpt/reports/phase1d_500m_evaluation/results.csv`
- `modern_gpt/reports/phase1d_500m_evaluation/raw_generations.jsonl`
- `modern_gpt/reports/phase1d_500m_evaluation/per_example_scores.jsonl`
- `modern_gpt/reports/phase1d_500m_evaluation/failure_cases.jsonl`

