# ModernGPT-S — Phase 1B.1: SFT v3 Conversation Structure Analysis

**Dataset Version:** SFT 4K v3  
**Status:** AUDITED & VERIFIED  
**Date:** 2026-11-09  
**Directory:** `modern_gpt/data/sft_4k_v3/processed/`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report analyzes the conversational structure, dialogue turn depth, turn transition dynamics, and response length distribution of SFT v3.

The empirical findings explain why ModernGPT-S 100M exhibits conversational amnesia, responds to simple greetings with irrelevant explanations, and fails to track multi-turn context:
1. **95.80% Single-Turn Task Dominance**: Only **4.20% of records** (1,343 records) contain genuine multi-turn interaction (≥4 messages). 30,606 records consist strictly of single-turn prompts.
2. **Total Absence of Identity-Retention Chains**: There are zero examples in SFT v3 where a user introduces their identity (e.g., name or personal preference), converses over intervening turns, and subsequently prompts the assistant to recall or apply that context.
3. **Severe Response Length Polarization**: 54.73% of assistant responses are ultra-short (<50 tokens, dominated by multiple-choice answers), while 57.61% of total sequence tokens are consumed by massive retrieval documents, leaving natural mid-length dialogue (100–250 tokens) virtually unrepresented.

---

## 2. Turn Depth & Dialogue Length Distribution

In conversational datasets serialized with ChatML, message counts define the interactive turn depth:
- **2 Messages**: 1 User turn $\rightarrow$ 1 Assistant turn (Pure Single-Turn).
- **3 Messages**: 1 Context/System block $\rightarrow$ 1 User turn $\rightarrow$ 1 Assistant turn (Single-Turn Task with context header).
- **≥4 Messages**: True Multi-Turn Interaction (≥2 user-assistant exchanges).

### Turn Depth Frequency Table (Training Split)

| Message Count | Interactive Turn Count | Record Count | Percentage of Training Split | Primary Data Categories |
| :---: | :---: | :---: | :---: | :--- |
| **2 Messages** | 1 Turn | 18,097 | **56.64%** | `science_commonsense`, `mathematics`, `coding_python` |
| **3 Messages** | 1 Turn + Context | 12,509 | **39.15%** | `long_context_retrieval`, `structured_json` |
| **4 Messages** | 2 Turns | 240 | **0.75%** | `multi_turn_dialogue`, `smol-magpie-ultra` |
| **5 Messages** | 2 Turns + Context | 78 | **0.24%** | Multi-turn contextual instructions |
| **6 Messages** | 3 Turns | 709 | **2.22%** | `custom_context_insulation` (topic switching, state retention) |
| **7 Messages** | 3 Turns + Context | 113 | **0.35%** | Multi-turn tool/schema tasks |
| **8–11 Messages**| 4–5 Turns | 133 | **0.42%** | Long multi-turn instruction chains |
| **12–25 Messages**| 6–12 Turns | 70 | **0.22%** | Extended dialogues (`ultrachat`, `systemchats`) |
| **Total** | — | **31,949** | **100.00%** | — |

### Structural Aggregation:
- **Single-Turn Tasks (2–3 msgs)**: **30,606 records (95.80%)**
- **True Multi-Turn Dialogues (≥4 msgs)**: **1,343 records (4.20%)**

---

## 3. Contextual Continuity & Follow-Up Analysis

### 3.1 Context Follow-Up Frequency
Across the entire 31,949 training records:
- **Context Follow-Ups**: Only 945 records under `multi_turn_dialogue` contain contextual follow-up questions.
- **Topic Transitions**: 269 records in `custom_context_insulation:topic_switching` teach the model to switch topics when instructed.
- **State Retention**: Only 149 records in `custom_context_insulation:relevant_state_retention` teach the model to retain a factual state across turns.
- **Irrelevant Context Suppression**: Only 73 records teach the model to ignore distractors in conversation.

### 3.2 Identity Retention Forensic
In manual testing, ModernGPT-S repeatedly hallucinated user identities or failed to remember the user's name when asked.

Inspection of SFT v3 confirms the root cause:
- **Zero Intervening-Turn Identity Records**: SFT v3 has **zero** records where a user states their name at turn 1, conducts 2+ intervening conversational turns on other topics, and then asks "What is my name?".
- In `curated_conversational_diverse`, only 2 single-turn records mentioned names:
  1. `"I am Anish"` $\rightarrow$ `"Nice to meet you, Anish! What kinds of projects are you building?"`
  2. `"My name is Sarah"` $\rightarrow$ `"Hello Sarah! Great to meet you. What are you studying or working on?"`
- Because these were single-turn prompts with no subsequent recall questions, the model never saw an example where it had to extract a past name and output: `"Your name is Anish."`

---

## 4. Assistant Response Length Distribution

Assistant responses exhibit severe bi-modal polarization:

| Response Token Range | Record Count (Train) | Share of Train Split (%) | Dominant Content / Sources |
| :--- | :---: | :---: | :--- |
| **< 50 tokens** | 17,487 | **54.73%** | Commonsense QA multiple choice choices, retrieval key facts |
| **50 to 199 tokens** | 11,647 | **36.46%** | GSM8K math reasoning steps, short coding functions |
| **200 to 999 tokens** | 2,412 | **7.55%** | Detailed code files, full explanations, long essays |
| **≥ 1,000 tokens** | 403 | **1.26%** | Long document extraction and comprehensive analyses |
| **Total** | **31,949** | **100.00%** | — |

### Impact on Generation Dynamics:
- Over half the training records (54.73%) train the model to terminate generation after 30–45 tokens.
- When an open conversational query is provided, the model either halts prematurely after a single formulaic sentence or defaults to a massive explanatory essay, lacking the ability to provide natural, fluid 2–3 sentence conversational responses.

