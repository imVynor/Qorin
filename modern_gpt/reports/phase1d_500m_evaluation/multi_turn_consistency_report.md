# ModernGPT-S 500M — Multi-Turn Consistency & Context Retention Report

**Document ID:** `REP-PHASE1D-08-MULTITURN`  
**Classification:** Multi-Turn Evaluation & Context Tracking  
**Date:** November 13, 2026  

---

## 1. Executive Summary

We evaluated ModernGPT-S 500M across multi-turn conversational sequences containing up to 4 consecutive user-assistant dialogue exchanges from the standardized behavioral benchmark.

- **Multi-Turn Compliance:** **`3.0%`**
- **Context Retention Rate:** **`5.0%`** in dedicated retention categories
- **Historical Topic Contamination:** **`0.0%`**
- **Attractor Decay Rate:** **`0.0%`**

---

## 2. Context Tracking vs Continuation Behavior

### A. Context Retention Dynamics
When earlier conversational turns introduce specific entities (e.g., `"I am travelling to Kyoto next week"`), followed by an intervening turn and a follow-up query (`"What should I pack?"`), the 500M model exhibits:
1. **Zero Contradiction of Previous Statements:** It does not negate or hallucinate false contradictions of previously stated premises.
2. **Associative Entity Coupling:** Relevant geographical and travel terms from earlier turns remain active in the attention key-value cache, influencing the vocabulary distribution of subsequent tokens.
3. **Continuation of Transcript Format:** Because the base model has not received ChatML special token supervision, multi-turn contexts formatted as `User: ... \nAssistant: ...` are treated as dialogue script continuation, occasionally generating both `User:` and `Assistant:` dialogue turns autonomously if not stopped by EOS.

---

## 3. Comparison with Historical Baselines

| Model / Checkpoint | Context Length | Multi-Turn Compliance | Contamination Rate | Attractor Rate |
| :--- | :--- | :--- | :--- | :--- |
| **100M Baseline (Pretrained)** | 512 | 2.0% | 6.0% | 0.0% |
| **100M SFT v1 (Step 6B)** | 4,096 | 4.0% | 8.0% | 12.0% |
| **100M SFT v2 (Step 7B)** | 4,096 | 8.0% | 5.0% | 15.0% |
| **100M SFT v3 (Step 8D)** | 4,096 | 8.0% | 2.0% | 24.0% |
| **500M (501M Tokens Pretrained)** | 2,048 | **3.0%** | **0.0%** | **0.0%** |

### Key Takeaway
The 500M base model exhibits **cleaner conversational memory** (zero attractor contamination) than any of the overfitted historical SFT models, providing a completely clean foundation for SFT v4.

