# Phase 0: Forensic Diagnosis & Baseline Audit — Root Cause Analysis

This report synthesizes the empirical evidence collected across all forensic investigations, classifying the observed behavioral failures of ModernGPT-S 100M into seven root-cause categories with confidence levels, severity ratings, and fixability assessments.

---

## 1. Classification & Diagnostic Matrix

| Category | Root Cause Identification | Severity | Confidence | Fixability | 100M Fixable? | 250M Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Category A: Inference Pipeline** | Non-atomic delimiters, repetition penalty corrupting `<|im_end|>`, left-truncation chopping system prompt | **High** | **100% (Empirical)** | **High** | Yes (Code fix) | Clean baseline |
| **Category B: Data Imbalance** | 95.8% single-turn skew, 0 greetings in v3, 56.5% dominated by two rigid templates (`"The correct answer is"`, `"Based on text"`) | **Critical** | **100% (Empirical)** | **High** | Yes (Data rebalance) | Essential for both |
| **Category C: Model Scale & Capacity** | 100M parameters cannot simultaneously store factual encyclopedia, math logic, code syntax, and multi-turn dialogue without interference | **High** | **95% (Empirical)** | **Low** | Partial (Specialize) | **Major (Higher capacity)** |
| **Category D: Training Dynamics** | Microbatch mean reduction weighting 20-token QA tokens 80x heavier than 1600-token documents | **Medium** | **100% (Mathematical)** | **High** | Yes (Loss weighting) | Critical for stable multi-task |
| **Category E: Tokenizer Architecture** | Delimiters `<|im_start|>` and `<|im_end|>` split into 8 and 7 subwords; heavy 64-token system prefix | **High** | **100% (Empirical)** | **High** | Yes (Add special tokens) | Clean token budget |
| **Category F: Safety Void** | Zero safety, refusal, or crisis data; model hallucinates bizarre non-sequiturs on dangerous prompts | **Medium** | **100% (Empirical)** | **High** | Yes (Add 1.5k safety pairs)| Mandatory for release |
| **Category G: Eval Benchmark Disconnect** | Behavioral eval suite penalized conversational answers while rewarding rigid template exact-matches | **Medium** | **90% (Audit)** | **High** | Yes (Refine metrics) | Unbiased metric tracking |

---

## 2. Deep Dive by Category

### Category A: Inference & Pipeline Deficiencies
- **The Evidence**:
  - `tokenizer_16k_v2` encodes `<|im_end|>` as `[28, 92, 310, 63, 490, 92, 30]`.
  - In `chat.py`, `repetition_penalty = 1.20` penalizes any token generated earlier in the sequence. If the model outputs code or punctuation (`|`, `<`, `_`), the probability of completing the 7-subword `<|im_end|>` tag drops drastically, causing runaway text until `max_new_tokens`.
  - Slicing context by raw tokens (`-(context_length - max_new_tokens)`) truncates the system prompt first.
- **Fixability**: High. Requires modifying `chat.py` to use atomic stop IDs, excluding control delimiters from repetition penalty, and preserving system messages during context eviction.

### Category B: Data Distribution & Imbalance
- **The Evidence**:
  - `sft_4k_v3` training set (31,949 samples) contains **0 simple greeting records**.
  - Multi-turn dialogue dropped from 25.54% (SFT v1) $\rightarrow$ 10.99% (SFT v2) $\rightarrow$ **4.20% (SFT v3)**.
  - Commonsense QA (`"The correct answer is [Choice]..."`) accounts for 8,271 records (25.9%), and synthetic retrieval accounts for 9,781 records (30.6%).
  - Email rewrites (`"Dear [Name]"`, `"I hope this email finds you well"`) populated the general conversation partition, causing the model to interpret conversational user queries as email writing tasks.
- **Fixability**: High. Requires data re-balancing: prune 70% of redundant Commonsense QA templates, restore 3,000 clean multi-turn chats, inject 500 greeting/casual turns, and purge email boilerplate.

### Category C: Model Scale & Capacity Constraints
- **The Evidence**:
  - On basic arithmetic (`"If I have 3 apples and eat 1, how many are left?"`), Step 8D SFT v3 memorized the GSM8K syntax tokens (`<<calc>>` and `####`), but hallucinated that $3 \times 1 = 2$, $6 - 6 = 6$, and $10 - 4 = 8$ `"in the tank"`.
  - On JSON tasks, the model consistently wrapped JSON inside Python function scripts.
  - In a 100M parameter model, attention heads and FFN layers have finite capacity. Forcing the model to memorize 9,781 synthetic 4K retrieval documents saturated weights, degrading general conversational reasoning.
- **Fixability**: Low for 100M (requires narrow domain focus); High for 250M–300M (sufficient capacity for both reasoning and chat).

### Category D: Training Dynamics & Loss Masking
- **The Evidence**:
  - Loss masking in `conversation_serializer.py` was byte-perfect: system and user tokens were correctly masked with `-100`.
  - However, `F.cross_entropy` was executed with default `reduction="mean"`. A microbatch with 20 supervised tokens produced an average loss divided by 20, whereas a microbatch with 1,600 tokens produced an average loss divided by 1,600.
  - This created an **80:1 per-token gradient weighting bias** in favor of short multiple-choice QA lines, heavily cementing `"The correct answer is..."` into early layers.
- **Fixability**: High. Use global token-level loss normalization or balance sample lengths per accumulation window.

### Category E: Tokenizer & Prompt Architecture
- **The Evidence**:
  - Vocab size is 16,000, but lacks special tokens for chat.
  - The default system prompt consumes 33 tokens, creating a 64-token static prefix for a 1-word query like `"Hello"`.
  - Small models are vulnerable to prefix anchoring: the model spends substantial attention energy attending to system prompt tokens rather than short user inputs.
- **Fixability**: High. Register `<|im_start|>` and `<|im_end|>` as special tokens (preventing subword decomposition), and use dynamic, concise system prompts.

### Category F: Safety Alignment & Guardrail Absence
- **The Evidence**:
  - On suicide / crisis prompts, Step 6B roleplayed as a depressed individual, and Step 8D gave vague advice omitting 988 lifeline details.
  - On explosives, Step 8D hallucinated that explosives are caused by sun exposure or repeated `"Fire"` 10 times.
- **Fixability**: High. Add 1,500 curated safety refusal and crisis referral training pairs.

### Category G: Evaluation Benchmark Disconnect
- **The Evidence**:
  - Previous benchmarks (e.g. strict exact-match retrieval) rewarded rigid template generation (e.g. `"The orbital frequency is 245.8 MHz"`). When models generated natural, conversational variants (e.g. `"According to the document, it is 245.8 MHz"`), strict regex evaluators scored them 0%.
  - This incentivized dataset creators in Step 8B to over-regularize the dataset toward rigid, identical opening templates.
- **Fixability**: High. Use normalized semantic evaluation alongside exact regexes.

