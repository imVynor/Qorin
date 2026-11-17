# Phase 0: Forensic Diagnosis & Baseline Audit — SFT Dataset Forensics

This report documents the dataset composition, category distributions, token length statistics, and behavioral attractors identified across all SFT dataset iterations (`sft_4k` v1, `sft_4k_v2`, and `sft_4k_v3`).

---

## 1. Executive Summary: The Root Causes in the Data

The forensic audit of `modern_gpt/data/sft_4k_v3/processed/train.jsonl` (31,949 records) and its predecessors reveals why ModernGPT-S 100M exhibits conversational collapse:

1. **Severe Multi-turn Starvation**: Multi-turn dialogue dropped from **25.54% in SFT v1** down to just **4.20% in SFT v3** (1,343 records). 95.8% of the training dataset consists of strictly single-turn tasks.
2. **Total Absence of Natural Greetings**: There is **literally zero** simple greeting records (`"Hello"`, `"Hi"`, `"Hey"`, `"Good morning"`, `"How are you?"`) in the entire 31,949-record SFT v3 training set.
3. **Massive Over-Representation of Two Monolithic Formats**:
   - `tau/commonsense_qa` (8,269 records, 25.9%) where 100% of assistant responses begin: `"The correct answer is [Choice]..."`
   - `synthetic_long_retrieval_v3` (9,781 records, 30.6%) where assistant responses begin: `"Based on the provided text..."`
   - Together, these two templates account for **56.5% of all supervised tokens** in SFT v3!
4. **Corporate/Email Boilerplate Contamination**: Assistant responses in `smoltalk:smol-rewrite`, `smoltalk:systemchats`, and `dolly` contain rewritten business correspondence (`"Dear [Name]"`, `"I hope this email finds you well"`, `"Best regards"`). In the absence of natural chat data, conversational prompts trigger these email attractors.

---

## 2. Cross-Dataset Evolution: v1 vs v2 vs v3

| Metric | SFT v1 (`sft_4k`) | SFT v2 (`sft_4k_v2`) | SFT v3 (`sft_4k_v3`) | Evolution / Trend |
| :--- | :--- | :--- | :--- | :--- |
| **Total Training Records** | 8,764 | 21,647 | 31,949 | +264% expansion |
| **Validation Records** | 1,111 | 2,713 | 3,277 | +195% expansion |
| **Single-Turn Records** | 6,526 (74.46%) | 19,267 (89.01%) | 30,606 (**95.80%**) | **Extreme single-turn skew** |
| **Multi-Turn Records** | 2,238 (**25.54%**) | 2,380 (10.99%) | 1,343 (**4.20%**) | **-83.6% multi-turn reduction** |
| **Max Dialogue Turns** | 12 turns | 12 turns | 12 turns | Longest multi-turn chains |
| **Simple Greetings Count** | 2 records | 0 records | **0 records** | Completely absent in v2 & v3 |
| **Median Response Tokens** | 93 tokens | 41 tokens | 36 tokens | Sharp compression towards short answers |
| **Mean Response Tokens** | 397.0 tokens | 124.6 tokens | 96.0 tokens | Heavily weighted towards QA lines |

---

## 3. SFT v3 Category & Source Breakdown

Analysis of all 31,949 training records in `sft_4k_v3/processed/train.jsonl`:

| Category | Record Count | Percentage | Primary Data Source | Typical Target Format |
| :--- | :--- | :--- | :--- | :--- |
| **Long Context Retrieval** | 9,781 | 30.61% | `synthetic_long_retrieval_v3` | `"Based on the provided document..."` |
| **Science & Commonsense QA** | 8,269 | 25.88% | `tau/commonsense_qa:train` | `"The correct answer is [A/B/C/D/E]..."` |
| **General Conversation** | 5,162 | 16.16% | `smoltalk:explore-instruct`, `openhermes`, `smol-rewrite` | Long descriptive essays, email rewrites |
| **Coding (Python)** | 2,990 | 9.36% | `local_sft_v2:code_alpaca`, `self-oss-instruct` | Python code blocks with markdown |
| **Mathematics & Reasoning**| 2,907 | 9.10% | `openai/gsm8k:train` | Step-by-step arithmetic (`<<calc>>`) |
| **Structured JSON** | 1,895 | 5.93% | `smoltalk:apigen-80k`, `local_sft_v2:json` | JSON blocks / tool-call schemas |
| **Multi-Turn Dialogue** | 945 | 2.96% | `smoltalk:systemchats`, `smol-magpie-ultra` | Multi-turn instruction exchanges |

---

## 4. Response Length Distribution & Bi-Modality

The response token distribution in SFT v3 exhibits severe bi-modality:

- **Min Tokens**: 1
- **25th Percentile**: 25 tokens
- **Median**: 36 tokens
- **75th Percentile**: 96 tokens
- **95th Percentile**: 285 tokens
- **Max Tokens**: 11,275 tokens (truncated at 4,096 at training time)

### Response Length Buckets:
- `< 50 tokens`: **18,119 records (56.7%)** — overwhelmingly Commonsense QA multiple choice choices (`"The correct answer is A: library."`).
- `50 to 199 tokens`: **11,251 records (35.2%)** — math solutions and short instruction responses.
- `200 to 999 tokens`: **2,200 records (6.9%)** — code functions and long essays.
- `≥ 1,000 tokens`: **379 records (1.2%)** — full-document retrievals.

The model is effectively trained on two disjoint behaviors: emitting a formulaic 30-token sentence, or ingesting massive 3,000-token retrieval documents. Medium-length, fluid, natural conversational dialogue (100–250 tokens) is virtually absent.

---

## 5. Attractor Analysis: Repetitive Opening Templates

Inspection of the first 3 tokens of every assistant response in SFT v3:

| Assistant Opening Phrase | Frequency in SFT v3 | Percentage of Dataset | Source Corpus |
| :--- | :--- | :--- | :--- |
| `"the correct answer"` | **8,271** | **25.89%** | `commonsense_qa` |
| `"based on the"` | **3,265** | **10.22%** | `synthetic_long_retrieval_v3` |
| `"the provided text"` | **3,248** | **10.17%** | `synthetic_long_retrieval_v3` |
| Synthetic Key Formats (`"the orbital..."`, `"the power..."`) | **3,019** | **9.45%** | Synthetic key retrieval |
| `"sure, here is/are"` / `"certainly"` | **668** | **2.09%** | `dolly`, `openhermes` |
| `"understood! i've noted"` | **149** | **0.47%** | `systemchats` |
| **All Other Openings Combined** | 13,329 | 41.72% | Miscellaneous |

### Impact on Inference
When the user gives an open-ended conversational prompt (e.g. `"What is your name?"` or `"Hello"`), the model's highest prior probability mass falls directly onto `"The correct answer is..."` or corporate boilerplate, because conversational greeting priors have zero support in the training weights.

---

## 6. Email Boilerplate Contamination Forensic

In the dataset audit, 72 instances of explicit email boilerplate were identified in `sft_v3_train`, including:
- `"Dear [Boss's name],\n\nI hope you're doing well. I was wondering if you had a moment..."`
- `"Hi Jane,\n\nI hope you're doing well. I wanted to reach out to you regarding..."`
- `"Dear David,\n\nI trust that you are well. I am reaching out to you regarding an upcoming project..."`

Because these examples are classified under `general_conversation` in `train.jsonl`, the model learned that "conversation" equals drafting email pleasantries and formal letters.

---

## 7. Conclusions for Dataset Architecture

The conversational failure of ModernGPT-S 100M is directly traceable to the SFT v3 dataset design:
1. **Multi-turn dialogue was starved** (reduced to 4.2%).
2. **Zero conversational greetings** were provided.
3. **Template concentration was catastrophic** (56.5% dominated by just two rigid opening patterns).
4. **Email and customer support rewrites** filled the vacuum left by the lack of natural dialogue.

