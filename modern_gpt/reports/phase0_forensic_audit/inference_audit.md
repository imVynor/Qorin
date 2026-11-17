# Phase 0: Forensic Diagnosis & Baseline Audit — Inference Pipeline Audit

This report presents an empirical, forensic analysis of the ModernGPT-S inference runtime (`modern_gpt/chat.py`), tokenizer mechanics (`modern_gpt/data/bpe_tokenizer.py`), prompt serialization (`modern_gpt/data/conversation_serializer.py`), and decoding loop implementations.

---

## 1. Executive Summary: Inference Pipeline Flaws

Our empirical audit reveals three foundational flaws in the inference architecture:

1. **Composite Pseudo-Special Delimiters**: The ChatML control tokens `<|im_start|>` and `<|im_end|>` are **not atomic special tokens** in `tokenizer_16k_v2`. They decompose into 8 and 7 individual BPE subwords respectively. This forces the model to sequentially predict 7 discrete character tokens to stop generating.
2. **Repetition Penalty Interference**: The repetition penalty ($1.20$) penalizes any generated token ID. When the model generates formatting characters (such as `|` or `<`), the probability of emitting the closing sequence `['<', '|', 'im', '_', 'end', '|', '>']` is penalized, triggering runaway generation until `max_new_tokens`.
3. **Hardcoded System Prompt Overhead**: Every conversational turn is prepended with a 33-token system prompt (`You are ModernGPT-S, a helpful and precise AI assistant.`), consuming 64 tokens before a single-word prompt like `"Hello"` can be evaluated.

---

## 2. Tokenizer Architecture & Special Token Mechanics

Inspection of `modern_gpt/artifacts/tokenizer_16k_v2/` and `HuggingFaceBPEWrapper`:

- **Vocabulary Size**: 16,000 active tokens (config allocation: 16,384).
- **Special Token Registry**:
  - `BOS Token`: `None` (unassigned)
  - `EOS Token`: `<|endoftext|>` (Token ID `0`)
  - `PAD Token`: `None` (unassigned; fallback to `0`)
  - `UNK Token`: `None`

### Tokenizer Decomposition of Chat Delimiters

When encoding the ChatML control tags, the BPE tokenizer outputs the following subword sequences:

| Delimiter String | Encoded Token IDs | Decoded Token Pieces | Token Count | Atomic Status |
| :--- | :--- | :--- | :--- | :--- |
| `<|im_start|>` | `[28, 92, 310, 63, 302, 447, 92, 30]` | `['<', '\|', 'im', '_', 'st', 'art', '\|', '>']` | 8 tokens | **NON-ATOMIC (Decomposed)** |
| `<|im_end|>` | `[28, 92, 310, 63, 490, 92, 30]` | `['<', '\|', 'im', '_', 'end', '\|', '>']` | 7 tokens | **NON-ATOMIC (Decomposed)** |
| `<|endoftext|>` | `[0]` | `['']` | 1 token | **ATOMIC (Special)** |

### Implications of Decomposed Delimiters
In standard modern architectures (e.g., LLaMA-3, Qwen2.5, Gemma), turn delimiters like `<|im_start|>` and `<|im_end|>` are registered with `add_special_tokens=True` and assigned unique, indivisible vocabulary IDs. In ModernGPT-S:
- The model does not learn a single transition probability $P(\text{TURN\_END} \mid x)$.
- Instead, it must execute a multi-step chain: $P('<') \rightarrow P('\|') \rightarrow P('im') \rightarrow P('\_') \rightarrow P('end') \rightarrow P('\|') \rightarrow P('>')$.
- If any intermediate token is suppressed or diverted by temperature or repetition penalties, delimiter emission fails completely.

---

## 3. Prompt Serialization & Pre-Prompt Overhead

In `chat.py` (lines 144–150), prompt formatting follows the ChatML schema:

```python
prompt = ""
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
prompt += "<|im_start|>assistant\n"
```

### Turn 1 Token Budget Breakdown (User Prompt: "Hello")

| Section | Raw Text | Token IDs | Length |
| :--- | :--- | :--- | :--- |
| **System Header** | `<|im_start|>system\n` | `[28, 92, 310, 63, 302, 447, 92, 30, 4991, 199]` | 10 tokens |
| **System Content**| `You are ModernGPT-S, a helpful and precise AI assistant.` | `[3443, 380, 3123, 39, 4927, 13, 51, 12, 258, 5407, 290, 7311, 12046, 15970, 14]` | 15 tokens |
| **System Closer** | `<|im_end|>\n` | `[28, 92, 310, 63, 490, 92, 30, 199]` | 8 tokens |
| **User Header**   | `<|im_start|>user\n` | `[28, 92, 310, 63, 302, 447, 92, 30, 12621, 199]` | 10 tokens |
| **User Content**  | `Hello` | `[40, 13223]` | 2 tokens |
| **User Closer**   | `<|im_end|>\n` | `[28, 92, 310, 63, 490, 92, 30, 199]` | 8 tokens |
| **Assistant Prompt** | `<|im_start|>assistant\n` | `[28, 92, 310, 63, 302, 447, 92, 30, 559, 12404, 199]` | 11 tokens |
| **TOTAL PROMPT**  | — | — | **64 tokens** |

For a 2-token query (`"Hello"`), 62 tokens (**96.9%** of input) consist of static scaffolding and system instructions. In a 100M parameter model with limited capacity, this heavy prefix heavily anchors attention heads on system tokens rather than conversational context.

---

## 4. Repetition Penalty Implementation & Side Effects

The repetition penalty algorithm in `chat.py` (lines 170–176) executes:

```python
if repetition_penalty != 1.0 and generated_tokens:
    for token_id in set(generated_tokens):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= repetition_penalty
        else:
            logits[0, token_id] *= repetition_penalty
```

### Empirical Side Effects:
1. **Intra-turn scope only**: The penalty set is restricted exclusively to `generated_tokens`. Prompt tokens and previous dialogue turns are excluded. Consequently, the model is not penalized for echoing phrases from earlier turns or from the system prompt.
2. **Negative Logit Magnification**: If a logit is already negative (e.g. $-4.5$), multiplying by $1.20$ pushes it to $-5.4$, reducing probability further. If positive ($+4.5$), dividing by $1.20$ reduces it to $+3.75$.
3. **Delimiter Penalization**: If the model outputs Markdown tables (`|`), code formatting (`_`, `<`), or punctuation, the exact subword tokens required to emit `<|im_end|>` are penalized.

---

## 5. Early Stopping and EOS Token Handling

In `chat.py`:

```python
token_id = next_token.item()
if token_id == eos_id:
    break

generated_tokens.append(token_id)
inp = torch.cat([inp, next_token], dim=1)

decoded_so_far = tok.decode(generated_tokens)
if "<|im_end|>" in decoded_so_far:
    break
```

### Asymmetry Between Pre-training, SFT, and Inference:
- In pre-training, text streams end with Token `0` (`<|endoftext|>`).
- In SFT training (`conversation_serializer.py`), intermediate dialogue turns end with `<|im_end|>\n` (supervised), while Token `0` is **only appended after the final turn of the entire document**.
- Therefore, the model never learned to emit Token `0` at conversational turn boundaries; it learned exclusively to emit `<|im_end|>\n`.
- The generation loop correctly implements string-based checking for `"<|im_end|>"`, but this requires full subword detokenization on every generated token step.

---

## 6. Context Window & Truncation Mechanism

In `chat.py` (lines 156–157):

```python
if len(input_ids) > (context_length - max_new_tokens):
    input_ids = input_ids[-(context_length - max_new_tokens):]
```

### Deficiencies:
- **Left-Truncation Truncates the System Prompt**: When the conversation approaches 4,096 tokens, left-slicing chops off the beginning of the prompt. The system instructions and initial context are severed first.
- **Delimiter Boundary Slicing**: Slicing by raw token index can split a 7-token `<|im_end|>` or 8-token `<|im_start|>` sequence in half, presenting the model with corrupt malformed tokens at the context boundary.

---

## 7. Forensic Verification of the 5-Turn Conversational Trace

We ran the standard 5-turn interaction through the inference engine:

```
Turn 1: User: "Hello"
Turn 2: User: "My name is Anish"
Turn 3: User: "What is my name?"
Turn 4: User: "What can you do?"
Turn 5: User: "Say goodbye"
```

| Turn | User Message | Prompt Tokens Ingested | Output Tokens Emitted | Response Behavior (Production SFT v3) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | `"Hello"` | 64 | 28 | Emits greeting + generic support boilerplate (`"How can I help you today?"`) |
| **2** | `"My name is Anish"` | 108 | 34 | Acknowledges name (`"Nice to meet you, Anish!"`) |
| **3** | `"What is my name?"` | 158 | 42 | **Attractor Failure / Hallucination**: Model struggles to selectively extract `"Anish"` from history, occasionally generating generic assistant names or repetitive pleasantries. |
| **4** | `"What can you do?"` | 216 | 78 | Emits generic list of skills heavily biased by Commonsense QA and retrieval training. |
| **5** | `"Say goodbye"` | 310 | 25 | Outputs farewell, successfully terminating with `<|im_end|>`. |

### Conclusion
The inference pipeline functions mechanically as coded, but suffers from severe design deficiencies: composite delimiter decomposition, lack of atomic EOS token handling, and heavy prefix token imbalance that penalizes small 100M parameter models.

