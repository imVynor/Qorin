# ModernGPT-S SFT v2 Public Dataset Audit Report

- **Total Records:** 8,600
- **Total Supervised Tokens:** 1,939,904
- **Average Sequence Length:** 225.6 tokens (Context Limit: 512)
- **P90 Sequence Length:** 399 tokens
- **Split Isolation Status:** PASSED_STRICT_ZERO_LEAKAGE

## Domain Distribution

| Category | Count | Share |
| :--- | :---: | :---: |
| **Math_reasoning** | 600 | 7.0% |
| **Python** | 3,200 | 37.2% |
| **Qa** | 1,600 | 18.6% |
| **Instruction** | 2,400 | 27.9% |
| **Json** | 800 | 9.3% |

## Source Breakdown

| Source | Inspected | Accepted | Rejections | P50 Tokens | P90 Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mbpp` | 352 | **350** | 3 | 128 | 234 |
| `code_alpaca` | 2,886 | **2,850** | 2,678 | 152 | 278 |
| `dolly_instruction` | 1,343 | **1,200** | 143 | 165 | 367 |
| `dolly_qa` | 535 | **400** | 135 | 300 | 450 |
| `ultrachat` | 2,863 | **1,200** | 1,663 | 356 | 473 |
| `squad` | 1,256 | **1,200** | 56 | 291 | 419 |
| `gsm8k` | 601 | **600** | 1 | 210 | 314 |
| `structured_json` | 996 | **800** | 196 | 162 | 179 |
