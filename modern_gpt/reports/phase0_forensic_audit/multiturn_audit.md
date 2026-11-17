# Phase 0: Forensic Diagnosis & Baseline Audit — Multi-Turn Dialogue & Context Forensics

This report details the forensic diagnosis of ModernGPT-S across multi-turn dialogue scenarios, assessing context retention, identity tracking, topic-switching insulation, and conversational history contamination.

---

## 1. Executive Summary: Multi-Turn Behavioral Breakdown

Across all tested checkpoints, multi-turn dialogue exhibits severe pathology driven by conversational history contamination and attractor lock-in:

1. **History Loop Trapping**: When an early turn produces an unstopped or repetitive response (e.g., `'Hello,\n\nHello...'` under greedy decoding), the token sequence enters the context window. On all subsequent turns (Turns 2 through 5), the model attends heavily to its own earlier repetition and duplicates the exact same loop, regardless of user input.
2. **Boilerplate Tail Echoing**: Under production sampling (`chat.py`), the model latches onto an opening boilerplate anchor (e.g. `"I'm excited about your recent research on computer programs..."`) and appends this verbatim sentence across unrelated turns (identity recall, capability queries, and farewells).
3. **Identity Inversion / Hallucination**: When the user introduces themselves (`"My name is Anish"`), the model frequently reverses speaker roles, claiming in Turn 2: `"Hello, my name is Anish."`
4. **Topic Switching Failure**: In topic-switching tests (Quantum Physics $\rightarrow$ Chocolate Chip Cookies), earlier checkpoints either continue outputting quantum mechanics or hallucinate blended concepts (`"an unpacked quantum computer that can generate a recipe"`). Step 8D SFT v3 shows partial insulation but remains anchored to rigid instructional formatting.

---

## 2. Controlled Multi-Turn Diagnostic Scenarios

We evaluated two multi-turn scenarios across Base 100M, Step 4 4K, Step 6B SFT v1, Step 7B SFT v2, and Step 8D SFT v3 under 4 decoding regimes.

### Scenario A: Identity & Context Retention (5 Turns)
```
Turn 1: "Hello"
Turn 2: "My name is Anish"
Turn 3: "What is my name?"
Turn 4: "What can you do?"
Turn 5: "Say goodbye"
```

#### Longitudinal Progression Across Checkpoints (Production Settings):

| Checkpoint | Turn 1 ("Hello") | Turn 2 ("My name is Anish") | Turn 3 ("What is my name?") | Turn 4 ("What can you do?") | Turn 5 ("Say goodbye") |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base 100M** | Loop: `user\nHello` | Forum hallucination | Loop: `Hello, !!` | Loop: `I am a software engineer` | Loop: `I am a software engineer` |
| **Step 4 4K** | Loop: `user\nHello` | Angry rant about computer | Loop: `Hello, !! I'm not a programmer` | Loop: `I am a software engineer` | Loop: `I am a software engineer` |
| **Step 6B (v1)** | `Hello, my name is Jeff.` | Repeats coding project essay | Repeats coding project essay | Repeats coding project essay | Repeats coding project essay |
| **Step 7B (v2)** | `Hello! How can I help you today?` | `Hello, Anish! How can I help you today?` | `Your name is Anish.` (Correct) | Generic bullet list | Emits farewell |
| **Step 8D (v3)** | Hallucinated essay: *"Welcome to Computer Programs!"* | Inverted identity: *"Hello, my name is Anish."* | *"Your name is Anish. I'm excited about your research on computer programs..."* | List of Python tutorials | Echos Turn 2: *"Hello, my name is Anish. I'm excited about your research..."* |

#### Key Finding on SFT v2 vs SFT v3 Regression:
In Step 7B SFT v2, simple greeting and identity turns were handled cleanly because SFT v2 retained 10.99% multi-turn dialogues and casual greeting instances.
In Step 8D SFT v3, where multi-turn data was slashed to **4.20%** and simple greetings were reduced to **0 records**, the model regressed severely: it hallucinates lengthy computer science essays and echoes the user's name as its own identity.

---

### Scenario B: Topic Switching & Context Insulation (3 Turns)
```
Turn 1: "Let's talk about quantum physics. What is entanglement?"
Turn 2: "Actually, forget that. What is a good recipe for chocolate chip cookies?"
Turn 3: "Can you list just the ingredients?"
```

#### Evaluation Matrix:

| Checkpoint | Decoding | Turn 1 Handling | Turn 2 Topic Switch | Turn 3 Follow-up | Insulation Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base 100M** | Greedy | Repetition loop | Repeats Turn 1 quantum prompt | Repeats Turn 1 quantum prompt | **Complete Failure (Unaware of ChatML)** |
| **Step 6B (v1)**| Greedy | Quantum definition loop | Tautology: *"A good recipe is a recipe for cookies."* | Lists chocolate chips in loop | **Poor (Repetitive collapse)** |
| **Step 7B (v2)**| Production | Sound quantum explanation | Clean recipe transition | Accurate bullet list | **Successful Insulation** |
| **Step 8D (v3)**| Production | Sound quantum explanation | Clean recipe transition | Accurate bullet list | **Successful Insulation** |

**Verdict on Topic Switching**: Step 8D SFT v3 successfully switches topics when the prompt aligns with structured procedural instructions (such as recipes or code). The insulation failures in Step 8D occur specifically when transitioning between **casual conversation and factual requests**, because casual dialogue is out-of-distribution.

---

## 3. The History Attractor Mechanism

Why does the model get stuck repeating boilerplate in multi-turn sessions?

$$\mathbf{h}_t = \text{SelfAttention}\left([\mathbf{x}_{\text{sys}}, \mathbf{x}_{u,1}, \mathbf{x}_{a,1}, \dots, \mathbf{x}_{u,t}]\right)$$

1. **Attention Weight Imbalance**:
   In a 100M parameter model with 12 attention heads and 14 layers, representation capacity is constrained.
2. In single-turn pretraining (95.8% of SFT v3), attention heads learn that assistant tokens primarily attend to:
   - The immediate preceding instruction
   - Standard introductory prefixes (`"The correct answer..."`, `"Based on the provided document..."`)
3. When dialogue history lengthens, attention heads in layers 8–14 exhibit high mutual attention to tokens in prior assistant turns ($\mathbf{x}_{a,1}, \mathbf{x}_{a,2}$).
4. If $\mathbf{x}_{a,1}$ contained a distinctive phrase (e.g. `"I'm excited about your recent research on computer programs"`), that phrase acts as a powerful attention attractor for subsequent generative steps, overriding the short user input $\mathbf{x}_{u,3}$ (`"What is my name?"`).

---

## 4. Multi-Turn Diagnostic Conclusions

1. The multi-turn degradation observed by users is **not caused by context length exhaustion** (the 5-turn test consumed only 310 tokens out of 4,096 available context).
2. It is caused by **data distribution collapse**: SFT v3 eliminated 83.6% of multi-turn dialogue examples relative to SFT v1, leaving the model with insufficient supervisory signal to distinguish relevant historical context from stale assistant outputs.
3. Once an unstopped or verbose assistant phrase enters the KV cache, the model's self-attention heads lock onto it as a generation prior.

