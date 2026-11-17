# Phase 1D Evaluation Methodology Audit

**Script audited:** `modern_gpt/scripts/run_phase1d_evaluation.py`  
**Benchmark data audited:** `modern_gpt/scripts/phase1d_benchmark_data.py` and `modern_gpt/scripts/evaluate_sft_benchmarks.py`

---

## 1. Overall Methodology Assessment

The harness is a single-process Python script that loads both models, tokenizers, and runs all 8 evaluation suites sequentially. The core logic is correctly structured: for each benchmark, it generates text, applies a scoring function, and writes records to JSONL.

**What the methodology does correctly:**
- Evaluates validation loss with `reduction='mean'` over correctly shifted targets `(x, y = seq[:-1], seq[1:])`
- Uses `torch.autocast(bfloat16)` for inference efficiency
- Applies the same tokenizer and generation function across all tasks for a given model
- Records all raw outputs to JSONL for auditability

**What the methodology does incorrectly or incompletely:**
- Validation sequence sampling is contiguous, not independent
- 100M loss figures cited in reports come from different run conditions
- Sandbox is claimed to be a subprocess but is actually `exec()` in-process
- VRAM is not instrumented
- Distinct-1/Distinct-2 metrics are computed at the word level, not token level

---

## 2. Validation Loss Implementation

```python
for i in range(num_eval_seqs):
    offset = i * seq_len
    seq = mmap[offset : offset + seq_len + 1].astype(np.int64)
    x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(device)
    y = torch.from_numpy(seq[1:]).unsqueeze(0).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, targets=y)
```

**Correctly implemented:** `y` is a one-position shift of `x`, which is standard causal LM loss. The loss returned by `model(x, targets=y)` is mean CE over all 2,048 target positions.

**Issue — Offset Bug:** The offset `i * seq_len` causes sequences to be drawn starting at token 0, 2048, 4096, etc. This is correct for non-overlapping evaluation but it means only the first 51,200 tokens of each shard are evaluated regardless of shard size. No shuffled or random offset is applied. For shards containing millions of tokens, this represents a small but non-random prefix sample.

**Issue — 100M Baseline Not Evaluated Here:** The function `evaluate_validation_loss` takes a single model and runs it on the 32K-tokenized shards. The 100M model is never passed to this function in `main()`. The 100M loss numbers in the written reports are therefore not from this evaluation run.

---

## 3. Generation Function

```python
def generate_text(model, tok, prompt, max_new_tokens=128, temperature=0.0,
                  top_p=0.9, repetition_penalty=1.15, ...):
```

**Key observation:** The default `repetition_penalty=1.15` is active during all behavioral, math, JSON, and tool evaluations. However, the generation diagnostics section explicitly sets `repetition_penalty=1.0` only for the greedy regime, then `1.15` for conservative and creative. This means the "greedy (no penalty)" distinct-1/distinct-2 scores may reflect different conditions than stated.

**Greedy decoding issue:** When `temperature <= 0.0`, the function uses `torch.argmax`. The repetition penalty is still applied before argmax. This is not standard greedy decoding; it is greedy decoding with a frequency penalty, which is a non-standard setup that makes "greedy" results non-deterministic across differently ordered context windows.

---

## 4. Behavioral Evaluation

```python
iso_pass = True
if exp and exp not in iso_clean:
    iso_pass = False
for neg in neg_triggers:
    if neg in iso_clean:
        iso_pass = False
for req in req_entities:
    if req not in iso_clean:
        iso_pass = False
```

**Scoring logic:** A case is marked correct if the expected answer substring appears in the generated text, no negative triggers appear, and all required entity strings appear. This is a substring match, which over-counts correctness when the model echoes its input prompt (the prompt is included in `iso_clean` via the full decoded output including prompt tokens if `tok.decode()` is called on all ids).

**Check:** Looking at the generation function, `new_ids = generated[len(input_ids):]` correctly isolates only new tokens. The prompt is stripped. So substring matching is applied to the continuation only. This is correct.

**Issue — multi-turn prompt format:** The multi-turn evaluation concatenates messages as `Role: Content\n`. This is a bare-text format, not an instruction-tuned prompt template. The model was never trained on this format, so multi-turn results measure raw associative recall, not instruction following.

---

## 5. Tool Calling Evaluation

```python
prompt = f"<|im_start|>user\n{task['prompt']}<|im_end|>\n<|im_start|>assistant\n"
```

The tool calling evaluation uses `<|im_start|>` / `<|im_end|>` chat delimiters. These tokens are in the 32K tokenizer but the base model was never trained on instruction-formatted data. The model is essentially being shown an unknown template and expected to respond. This is a reasonable zero-shot test of the base model but should be labelled as such; it is not a comparison point for a fine-tuned assistant.

The intent scoring logic:
```python
if expected is None:
    if not has_tool_tag:
        matched_intent = True
else:
    if expected in gen_text.lower():
        matched_intent = True
```

For `expected_tool=None` tasks (e.g. "What is the capital of France?"), the model receives credit if it does NOT emit tool tags — which it never does. This means approximately 5 of the 20 tasks are auto-credited as "correct intent" by default, inflating the 35% intent accuracy figure.

**Corrected tool intent count:** Removing the 5 auto-credit None-expected tasks, the effective intent score over the 15 tool-requiring tasks was approximately 2/15 = 13%.

---

## 6. Distinct-N Diversity Metrics

```python
words = text.split()
d1 = len(set(words)) / len(words)
bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
d2 = len(set(bigrams)) / len(bigrams)
```

**Issue:** Distinct-1 and Distinct-2 are computed at the word (whitespace-split) level, not at the BPE token level. This is a common choice but it conflates punctuation attachment. The metrics are consistent within this run; they are not comparable to papers reporting token-level distinct-N.

**Issue:** Only 5 prompts are used for the generation diagnostics suite. Variance on 5 samples is high. The reported Distinct-1 values should carry an error range; they should not be presented as precise figures.

