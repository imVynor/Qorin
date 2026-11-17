# ModernGPT-S Phase 2A — Dataset Composition Report

**Target Model:** ModernGPT-S 500M  
**Master Dataset:** [splits/train.jsonl](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release/splits/train.jsonl)  
**Total Training Records:** 226,486  
**Total Training Assistant Tokens:** 72,682,363

---

## 1. Capability Distribution Breakdown

The SFT Release dataset is structured across five core capability domains to balance conversational fluency, logical reasoning, programming, structured transformation, and agentic tool invocation:

| Capability Domain | Records | Assistant Tokens | Token Share (%) | Mean Assistant Tokens / Turn |
|---|---:|---:|---:|---:|
| **General Instructions & Tutoring** | 48,432 | 42,297,141 | 58.19% | 873.3 |
| **Reasoning & Problem Solving** | 44,282 | 14,102,574 | 19.4% | 318.5 |
| **Coding & Debugging** | 51,773 | 10,113,653 | 13.91% | 195.3 |
| **Structured Output & Extraction** | 7,631 | 959,652 | 1.32% | 125.8 |
| **Agentic Tool Use** | 74,368 | 5,209,343 | 7.17% | 70.0 |
| **Total Corpus** | **226,486** | **72,682,363** | **100.00%** | **320.9** |

---

## 2. Sequence Length Distribution

All sequences fit strictly within the model's 2,048 token context window:

| Percentile | Assistant Target Tokens | Total Conversation Tokens | Turns per Sample |
|---|---:|---:|---:|
| **Min** | 1 | 17 | 2 |
| **p25** | 67 | 295 | 2 |
| **p50 (Median)** | 137 | 496 | 3 |
| **p75** | 378 | 810 | 3 |
| **p90** | 996 | 1236 | 6 |
| **p95** | 1293 | 1500 | 6 |
| **p99** | 1636 | 1840 | 12 |
| **Max** | 1933 | 2000 | 25 |
| **Mean** | **320.91** | **605.62** | **3.3** |
