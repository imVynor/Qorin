# Phase 1D Research Claims Audit

For each claim, the verdict is one of: **SUPPORTED / PARTIALLY SUPPORTED / UNSUPPORTED / CONTRADICTED**.

---

## Claim 1: "RoPE Effectively Utilizes 2,048 Tokens"

> *"Lower loss at sequence end demonstrates effective RoPE attention context utilization."*

### Verdict: UNSUPPORTED

**Evidence reviewed:**
- Early position loss (tokens 0–511): 3.0662
- Late position loss (tokens 1536–2047): 2.8642
- Difference: 0.0020 nats lower at later positions

**Why it is unsupported:**
The positional loss gradient is a universal property of autoregressive language models. Any LM will show lower loss at later sequence positions because later tokens have more conditioning context available. This holds regardless of whether the model attends to long-range positions or only to recent local context.

Demonstrating that position 1,800 specifically benefits from information at position 200 requires a controlled ablation where the prefix is progressively truncated and loss at the target position is measured at each context length. No such ablation was performed.

**More defensible statement:**
> *"The model exhibits the standard autoregressive positional loss gradient. Long-range context utilization is not established without a context-truncation experiment."*

---

## Claim 2: "Coding Performance Demonstrates an Algorithmic Capability Threshold"

> *"This confirms that scaling from 100M to 500M crosses a critical capability threshold for formal algorithmic grammar."*

### Verdict: CONTRADICTED

**Evidence reviewed:**
- 9 actual passing tasks: all trivial single-builtin or single-arithmetic-operator wrappers
- 100M baseline was **not** tested with an equivalent harness using the same tokenizer and sandbox configuration
- The sandbox blocks `map()`, `filter()`, `zip()`, `sorted()` — common correct implementations for many non-trivial tasks
- Task names reported in the written narrative are factually wrong

**Why it is contradicted:**
1. The 9 passing tasks do not constitute "formal algorithmic grammar." They are function-body completions using a single Python expression.
2. The comparison to the 100M baseline (0.0% pass) is confounded: the 100M was not run through the same harness with its own tokenizer. The 100M claim of 0.0% is from a prior evaluation under different conditions.
3. The sandbox's blocked builtins would artificially suppress both models' scores by an unknown amount, disproportionately affecting implementations that use common iterators.

**More defensible statement:**
> *"ModernGPT-S 500M generates syntactically valid Python one-liners for 66% of single-operation function prompts and passes unit tests for 18% (9/50), all of which involve wrapping a single builtin or arithmetic operator. Direct comparison to the 100M baseline on this metric is confounded by different harness configurations and tokenizers."*

---

## Claim 3: "SFT Will Resolve 46% of Failures"

> *"Over 85% of these 'failures' stem from base pretraining continuation mechanics rather than internal representation collapse... SFT will eliminate 46% of failures."*

### Verdict: PARTIALLY SUPPORTED

**Evidence reviewed:**
- 46% of logged failure cases are categorized as ERR-CONT (continuation rather than instruction answering)
- SFT is well-established to teach turn-boundary following via chat templates
- The 46% figure corresponds to the fraction of failures attributable to the prompt-continuation problem

**Why it is partially supported:**
The underlying mechanism is correct: instruction fine-tuning does teach models to respond to prompts rather than continue them. However:
1. The 46% recovery rate is assumed rather than demonstrated. The actual SFT uplift depends on the quality, volume, and format diversity of the SFT dataset, none of which have been designed yet.
2. Some continuation failures will resist SFT if the model lacks the underlying factual or reasoning capability. SFT cannot manufacture knowledge the model does not have.
3. The ERR-CONT category was established by the evaluators, not from a blind rater study.

**More defensible statement:**
> *"A large fraction of failures reflect continuation behavior expected of base pretrained models. SFT is likely to substantially reduce this category of failure. The specific recovery rate is unknown without an SFT experiment."*

---

## Claim 4: "RAG Requires SFT to Extract Answer Spans"

> *"To achieve >80% factual extraction from retrieved passages, the model must be fine-tuned to recognize Context: and Question: delimiters..."*

### Verdict: CONTRADICTED BY EVALUATION DESIGN

**Evidence reviewed:**
- RAG queries test internal QORIN project knowledge not present in the training corpus
- All positive RAG scores are artifacts of prompt-echo or substring collisions
- Genuinely correct RAG extractions: 0

**Why it is contradicted:**
The claim that RAG requires SFT for span extraction is a reasonable general hypothesis for instruction-following behavior. However, the conclusion is drawn from a 10-query evaluation where:
1. The queries ask about proprietary project-internal facts that neither model has seen
2. The scoring is corrupted by prompt-echo artifacts
3. No genuine retrieval-and-extraction success was observed

The evaluation does not distinguish between "the model cannot extract spans from retrieved context" (a span-extraction capability problem) and "the retrieved context does not contain information the model can verify" (a knowledge-absence problem). These require different interventions.

**More defensible statement:**
> *"Neither model correctly extracted answers from retrieved project-internal documentation in the 10-query test. This is consistent with both a span-extraction deficit (addressable by SFT) and a knowledge-grounding deficit (not addressable by SFT alone). A general span-extraction evaluation on factual passages is needed to distinguish these hypotheses."*

---

## Claim 5: "Arithmetic Failure Is Caused by Lack of Scratchpad/CoT Tokens"

> *"Standard feed-forward networks in Transformers cannot perform multi-digit multiplication in a single forward pass without emitting chain-of-thought tokens."*

### Verdict: SUPPORTED

**Evidence reviewed:**
- Math accuracy: 0.0% across all 50 tasks including simple arithmetic (`17 + 28 = ?`)
- Math tasks are presented as `"Solve: 17 + 28 = "` prompting the model to complete a numerical token
- The model generates continuation text discussing methods rather than computing values

**Supporting literature:**
This claim is consistent with extensive published research (Wei et al. 2022 — Chain-of-Thought Prompting, Kojima et al. 2022 — Zero-Shot CoT) demonstrating that:
- Base LMs without CoT fail on multi-step arithmetic
- Sub-10B parameter models especially benefit from scratchpad tokens
- Even simple 2-digit addition fails in models trained without explicit calculation trace data

**Caveat:** The evaluation prompts were formatted as `"Solve: X + Y = "`, which elicits a direct token continuation. Prompts formatted as `"Calculate step by step: X + Y = ?"` might yield different results without SFT. The claim as stated is directionally correct but overly mechanistic — the failure is partly attributable to prompt format and partly to the absence of multi-step reasoning training signal.

**More defensible statement:**
> *"The 0.0% mathematical accuracy is consistent with known limitations of base pretrained LMs without chain-of-thought training data. The model generates discussion of arithmetic rather than computing it. Including step-by-step reasoning traces in SFT training data is expected to substantially improve this."*

---

## Summary Table

| Claim | Verdict | Action Required |
|---|---|---|
| RoPE utilizes 2,048 tokens | UNSUPPORTED | Context-truncation ablation needed |
| Coding algorithmic threshold crossed | CONTRADICTED | Fix harness, re-run with 100M on same harness |
| SFT resolves 46% of failures | PARTIALLY SUPPORTED | Label as estimate, support with SFT experiment |
| RAG requires SFT | CONTRADICTED BY DESIGN | Replace queries with general factual probes |
| Arithmetic failure from lack of CoT | SUPPORTED | No change needed; minor caveats on prompt format |

