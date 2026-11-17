# ModernGPT-S — Phase 1B.1: SFT v3 Repetition & Template Analysis

**Dataset Version:** SFT 4K v3  
**Status:** AUDITED & VERIFIED  
**Date:** 2026-11-09  
**Directory:** `modern_gpt/data/sft_4k_v3/processed/`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report documents the repetition, template dominance, near-duplication, and structural attractors discovered during the forensic audit of SFT v3.

The analysis proves that despite having zero exact byte-level duplicate records, SFT v3 is overwhelmingly dominated by a small handful of repetitive synthetic templates:
1. **Four Template Families Account for 56.50% of All Training Records**:
   - `the correct answer [is ...]`: **8,271 records (25.89%)**
   - `based on the [provided ...]`: **3,261 records (10.21%)**
   - `the provided text [does/states ...]`: **3,248 records (10.17%)**
   - Synthetic key retrieval formulas: **3,272 records (10.24%)**
2. **Dominant Attractor Formation**: When presented with open-ended or casual user inputs (e.g. `"Hello"`, `"Who are you?"`), the model's learned output distribution heavily defaults to these four dominant phrases because they appeared in over 18,000 training instances.
3. **Template Monotony**: 100% of the 8,269 `commonsense_qa` examples start with the exact phrase `"The correct answer is [Choice]..."`, teaching the model a rigid multiple-choice response habit rather than reasoning.

---

## 2. Exact & Near-Duplicate Analysis

### 2.1 Exact Duplication Audit
- **Methodology**: Canonicalization of all messages in every conversation (lowercasing, whitespace collapse, role tagging) hashed via SHA-256.
- **Result**: **0 exact duplicate records** found across the combined 35,226 records in `train.jsonl` and `val.jsonl`. Content hashing during the Step 08b build successfully eliminated exact duplicate rows.

### 2.2 Semantic & Near-Duplicate Template Redundancy
While exact duplicate rows were filtered, massive semantic near-duplication exists due to repeated prompt frameworks:
- **Commonsense Multiple-Choice Framework**: 8,269 records in train follow the identical structural skeleton:
  `[Premise Question]\n\nChoices:\nA. [Choice A]\nB. [Choice B]\nC. [Choice C]\nD. [Choice D]\nE. [Choice E]`
- **Synthetic Needle Retrieval Framework**: 9,781 records follow the identical document skeleton with synthetic distractor sentences and a single embedded key-value pair.
- **Impact**: While the individual entity names or choice letters change, the model's attention layers learn the structural invariant, heavily penalizing any conversational flexibility.

---

## 3. Assistant Response Opening Analysis (Top 15 Phrases)

An audit of the first three words of every assistant response in `train.jsonl` (31,949 records):

| Rank | Opening 3-Gram Phrase | Frequency (Train) | Share of Train Split (%) | Primary Upstream Source | Structural Attractor Impact |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | `"the correct answer"` | **8,271** | **25.89%** | `tau/commonsense_qa:train` | **Severe Primary Attractor**: Forces multiple choice choice syntax |
| **2** | `"based on the"` | **3,261** | **10.21%** | `synthetic_long_retrieval_v3` | **Severe Secondary Attractor**: Forces document hedging |
| **3** | `"the provided text"` | **3,248** | **10.17%** | `synthetic_long_retrieval_v3` | **Severe Secondary Attractor**: Forces document citation language |
| **4** | `"the orbital frequency"`| **353** | **1.10%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **5** | `"the power output"` | **352** | **1.10%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **6** | `"the coolant mixture"` | **345** | **1.08%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **7** | `"the security badge"` | **339** | **1.06%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **8** | `"the settlement value"` | **332** | **1.04%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **9** | `"the resonant frequency"`| **325** | **1.02%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **10**| `"the thermal conductivity"`| **318** | **1.00%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **11**| `"the magnetic flux"` | **315** | **0.99%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **12**| `"the pressure threshold"`| **312** | **0.98%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **13**| `"the acoustic impedance"`| **308** | **0.96%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **14**| `"the catalytic rate"` | **298** | **0.93%** | `synthetic_long_retrieval_v3` | Synthetic key retrieval answer format |
| **15**| `"sure, here is"` / `"certainly"`| **285** | **0.89%** | `openhermes`, `dolly` | Polite affirmation opening |
| — | **All Top 14 Templates Combined** | **18,347** | **57.43%** | — | **Total Template Dominance** |
| — | **All Other Openings Combined** | **13,602** | **42.57%** | Varied sources | True diverse response space |

---

## 4. Corporate & Email Template Findings

In SFT v1 and v2, assistant responses in `smoltalk:smol-rewrite`, `smoltalk:systemchats`, and `dolly` frequently generated formal business letter salutations (`"Dear [Name]"`, `"I hope this email finds you well"`, `"Best regards"`).

In SFT v3:
- Filtering regexes (`RE_EMAIL_SALUTATION` and `RE_EMAIL_BODY`) purged assistant outputs that opened with formal letters.
- **The Underlying Contamination Remained**: While the exact greeting line was cut, 460 records of email rewriting tasks from `smoltalk:smol-rewrite` remained categorized under `general_conversation`.
- When users gave open-ended conversational prompts, the model's nearest semantic neighbor in `general_conversation` remained corporate drafting and business correspondence rather than natural human conversation.

