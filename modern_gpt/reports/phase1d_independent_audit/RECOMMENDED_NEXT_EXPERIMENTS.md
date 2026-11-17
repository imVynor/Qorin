# Phase 1D Recommended Next Experiments

This document lists the minimum additional experiments required to make defensible research claims and answer the original research question.

---

## Original Research Question

> *"How does increasing the capacity of a small decoder-only Transformer affect conversational language model performance, and can external retrieval compensate for a smaller model?"*

### What Phase 1D Actually Established

| Sub-question | Phase 1D Status |
|---|---|
| Does capacity affect validation loss? | **Partially answered** — 500M has lower loss than 100M at 501M tokens, but cross-tokenizer comparison is invalid |
| Does capacity affect code synthesis? | **Inconclusive** — harness issues and no matched baseline |
| Does capacity affect instruction following? | **Not tested** — both models are base pretrained; instruction following is not measurable without SFT |
| Can RAG compensate for smaller models? | **Not answered** — scoring artifacts invalidate the results |
| Does RoPE utilize 2,048 tokens? | **Not answered** — no ablation performed |

---

## Priority 1 — Fix Existing Evaluation Bugs (No GPU Required)

These should be completed before the next GPU compute window.

### Experiment E1: Fix Coding Sandbox Builtins

**What to fix:**
Add missing builtins to `safe_globals` in `evaluate_coding_suite()`:
```python
safe_globals = {"__builtins__": {
    "range": range, "len": len, "sum": sum, "min": min, "max": max,
    "abs": abs, "str": str, "int": int, "float": float, "list": list,
    "dict": dict, "set": set, "bool": bool, "True": True, "False": False,
    "None": None,
    # Missing: add these
    "map": map, "filter": filter, "zip": zip, "sorted": sorted,
    "enumerate": enumerate, "any": any, "all": all, "type": type,
    "isinstance": isinstance, "print": print, "tuple": tuple,
    "reversed": reversed, "chr": chr, "ord": ord, "round": round,
}}
```

**Expected outcome:** Pass@1 will likely increase by 3–8 tasks. Corrects the false attribution of code failures to the model.

**Effort:** 15 minutes (code change) + ~5 minutes GPU inference.

### Experiment E2: Replace RAG Queries with General Factual Probes

**What to do:** Replace `RETRIEVAL_QUERIES` in `phase1d_benchmark_data.py` with 20–50 general factual questions (capitals, element symbols, historical dates, scientific constants) sourced from Wikipedia that both models have plausibly seen in training.

**Updated scoring:** Replace substring key_term matching with:
1. A list of acceptable answer variants (e.g. `["canberra", "australian capital territory", "act"]`)
2. No term that appears in the question text

**Expected outcome:** A clean measure of RAG vs no-RAG factual extraction on base model knowledge.

**Effort:** 1 hour (query curation) + ~10 minutes GPU inference.

---

## Priority 2 — Cross-Model Valid Comparison (Medium Effort, Requires GPU)

### Experiment E3: Bits-Per-Byte Normalized Validation Loss

**What to do:**
1. For each domain shard, decode the binary token IDs back to UTF-8 text
2. Measure character count per token for both tokenizers on the same text
3. Report BPB = CE_loss / (mean_chars_per_token × log(2))

**Why needed:** This is the only way to compare the 500M (32K vocab) and 100M (16K vocab) language models on a common scale.

**Effort:** 2–3 hours development + ~20 minutes GPU.

### Experiment E4: 100M Baseline Live Evaluation on Matching Shards

**What to do:**
Re-run `evaluate_validation_loss()` with the 100M model loaded alongside the 500M model, both evaluated on the same 32K-tokenized validation shards. Additionally, run both on 16K-tokenized shards to get the 100M's native-tokenizer baseline.

**Why needed:** The current 100M loss numbers are not from an equivalent evaluation. This is essential to validate the capacity-scaling hypothesis.

**Effort:** ~30 minutes GPU inference.

### Experiment E5: Matched Coding Evaluation for 100M Baseline

**What to do:**
Run the repaired coding suite (with fixed builtins) against the 100M model using its own 16K tokenizer. Report Pass@1 and AST validity using identical prompts, identical test cases, and identical sandbox.

**Why needed:** The current claim that 500M improved over 100M on coding is unverifiable without a matched 100M harness result.

**Effort:** ~10 minutes GPU inference after E1 is implemented.

---

## Priority 3 — Long-Context Ablation (Requires Authorized GPU Session)

### Experiment E6: Context-Truncation Ablation for RoPE Verification

**What to do:**
For 5 sequences per domain (35 total), measure cross-entropy loss on the final 256 tokens of each 2,048-token sequence while providing prefixes of lengths: 0, 128, 256, 512, 1024, and 1,536.

```python
context_lengths = [0, 128, 256, 512, 1024, 1536]
target_start = 1792  # Last 256 tokens start at position 1792 in a 2048 sequence
for ctx_len in context_lengths:
    if ctx_len == 0:
        prefix = [eos_id]  # single BOS/EOS token
    else:
        prefix = seq[target_start - ctx_len : target_start]
    target = seq[target_start : target_start + 256]
    # measure CE on target
```

**What to look for:**
- If loss at target positions decreases as `ctx_len` grows from 0 → 1,536: confirms context utilization
- If loss plateaus after `ctx_len=128` or `ctx_len=256`: model's effective attention is local
- Plot: x-axis = prefix length, y-axis = target loss per domain

**Expected GPU time:** ~15–25 minutes.

---

## Priority 4 — SFT Readiness Experiment (After 1.0B Token Checkpoint)

### Experiment E7: Post-1.0B Phase 1E Evaluation

Before starting SFT, the 1.0B-token checkpoint should undergo a Phase 1E evaluation mirroring Phase 1D but with the bugs fixed (E1–E5 applied). This will establish:
- Whether validation loss continues to decrease (expected)
- Whether coding Pass@1 improves with 2× more training (expected)
- Whether the autoregressive ramp difference widens (indicating more context utilization)

### Experiment E8: Minimal SFT Probe (100 Examples)

Before committing to a full SFT v4 campaign, run a micro-SFT with ~100 high-quality conversational examples to verify:
- That the training pipeline works for instruction fine-tuning
- That the model adopts instruction-following format (turn boundaries, role markers)
- That validation loss on SFT data decreases smoothly without catastrophic forgetting

**Effort:** Minimal data curation (~2 hours) + 15–30 minutes GPU training.

---

## Minimum Experiment Set for Defensible Research Claims

| To claim... | Requires experiment |
|---|---|
| "500M has better language modeling efficiency than 100M" | E3 (BPB normalization) + E4 (live 100M eval) |
| "500M synthesizes more Python code than 100M" | E1 (fix sandbox) + E5 (matched 100M coding eval) |
| "RoPE effectively uses 2,048-token context" | E6 (context truncation ablation) |
| "RAG provides uplift for factual retrieval" | E2 (replace RAG queries) |
| "SFT will improve conversational performance" | E8 (micro-SFT probe) |

