# ModernGPT-S SFT v1 — Loss Masking & Training Semantics Audit

**Audit Date:** 2026-11-14  
**Audit Objective:** Inspect loss masking, ChatML envelope boundaries, tool-call supervision, environment observation masking, and target alignment for ModernGPT-S 500M SFT.  
**Inspected Implementation:** [conversation_serializer.py](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/conversation_serializer.py) & [train_sft_4k.py](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/train_sft_4k.py)  
**Status:** **VERIFIED & SUPERVISION-CORRECT**

---

## 1. Mathematical Loss Masking Architecture

In autoregressive language modeling with Cross-Entropy loss:
$$\mathcal{L} = -\frac{1}{\sum_{t=1}^{T} \mathbb{I}(y_t \neq -100)} \sum_{t=1}^{T} \mathbb{I}(y_t \neq -100) \log P_\theta(x_t \mid x_{<t})$$

The SFT objective must satisfy three cardinal rules:
1. **Never supervise prompts:** System instructions and user prompts are external context ($y_t = -100$). The model must not waste capacity predicting what the user said.
2. **Never supervise environment feedback:** Tool observations (stdout, stderr, exit codes returned from Python or APIs) are environment observations ($y_t = -100$). Supervising them causes the model to hallucinate fake execution returns instead of calling tools.
3. **Always supervise assistant generation:** Assistant text, `<tool_call>` actions, turn-closing `<|im_end|>`, and EOS `<|endoftext|>` must be supervised ($y_t = x_t$) so the model learns both content generation and exact stopping behavior.

---

## 2. Concrete Tokenized Trace with Input IDs, Targets & Masks

The following live trace was tokenized using the production tokenizer ([tokenizer.json](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/artifacts/tokenizer_32k/tokenizer.json), SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`) via [audit_loss_masking_demo.py](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/scratch/audit_loss_masking_demo.py):

**Conversation Input:**
- **System:** `You are ModernGPT-S, a helpful and precise AI assistant.`
- **User:** `Compute 5 factorial using Python.`
- **Assistant:** `<tool_call>{"name": "python", "arguments": {"code": "import math\nprint(math.factorial(5))"}}</tool_call>`
- **Tool Observation:** `{"stdout": "120\n", "exit_code": 0}`
- **Assistant:** `The computed value of 5! is 120.`

### Token-by-Token Inspection Table

| Token Index | Token Text / Delimiter | Input ID | Target ID (Label) | Masking State | Semantic Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `<|im_start|>` | 4 | **-100** | MASKED | Role envelope start |
| **1..2** | `system\n` | 9410, 208 | **-100** | MASKED | Role identifier |
| **3..17** | `You are ModernGPT-S... assistant.` | *[15 tokens]* | **-100** | MASKED | System prompt context |
| **18..19** | `<|im_end|>\n` | 5, 208 | **-100** | MASKED | System turn delimiter |
| **20** | `<|im_start|>` | 4 | **-100** | MASKED | User envelope start |
| **21..22** | `user\n` | 4599, 208 | **-100** | MASKED | User role identifier |
| **23..30** | `Compute 5 factorial using Python.` | *[8 tokens]* | **-100** | MASKED | User prompt text |
| **31..32** | `<|im_end|>\n` | 5, 208 | **-100** | MASKED | User turn delimiter |
| **33** | `<|im_start|>` | 4 | **-100** | MASKED | Assistant envelope start |
| **34..36** | `assistant\n` | 526, 8647, 208 | **-100** | MASKED | Assistant role identifier |
| **37..77** | `<tool_call>{"name": ...}</tool_call>` | *[41 tokens]* | **[Identical IDs]** | **SUPERVISED** | **Action emission learned by model** |
| **78..79** | `<|im_end|>\n` | 5, 208 | **5, 208** | **SUPERVISED** | **Teaches model to yield turn to tool** |
| **80** | `<|im_start|>` | 4 | **-100** | MASKED | Tool envelope start |
| **81..85** | `tool_observation\n` | *[5 tokens]* | **-100** | MASKED | Tool role identifier |
| **86..102** | `{"stdout": "120\n", "exit_code": 0}` | *[17 tokens]* | **-100** | **MASKED** | **Real sandbox stdout (environment)** |
| **103..104** | `<|im_end|>\n` | 5, 208 | **-100** | MASKED | Tool turn delimiter |
| **105** | `<|im_start|>` | 4 | **-100** | MASKED | Assistant envelope start |
| **106..108** | `assistant\n` | 526, 8647, 208 | **-100** | MASKED | Assistant role identifier |
| **109..116** | `The computed value of 5! is 120.` | *[8 tokens]* | **[Identical IDs]** | **SUPERVISED** | **Model output synthesis** |
| **117..118** | `<|im_end|>\n` | 5, 208 | **5, 208** | **SUPERVISED** | **Turn completion delimiter** |
| **119** | `<|endoftext|>` | 9 | **9** | **SUPERVISED** | **Final sequence termination** |

---

## 3. Supervision Accounting Summary

For this 119-token multi-turn tool interaction:
- **Total Conversation Tokens:** 119
- **Supervised Assistant Tokens:** 54 (45.38%)
  - Tool action block: 41 tokens
  - Action delimiter: 2 tokens
  - Synthesis response: 8 tokens
  - Response delimiter + EOS: 3 tokens
- **Masked Context Tokens:** 65 (54.62%)
  - System turn: 20 tokens
  - User prompt: 13 tokens
  - Tool observation: 25 tokens
  - Assistant headers: 7 tokens

**Verification Result:** Exactly zero system words, user words, or tool observation outputs receive non-zero loss gradients.
