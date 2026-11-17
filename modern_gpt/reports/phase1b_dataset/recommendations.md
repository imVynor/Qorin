# ModernGPT-S — Phase 1B.1: Technical Recommendations for SFT v4

**Dataset Target:** ModernGPT-S SFT v4  
**Status:** PROPOSED & TECHNICALLY JUSTIFIED  
**Date:** 2026-11-09  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report synthesizes the empirical measurements from the SFT v3 audit into technical recommendations for engineering the **SFT v4** dataset.

The central thesis of these recommendations is that **conversational ability cannot emerge from document retrieval and multiple-choice tests**. While SFT v3 achieved low validation perplexity on its synthetic benchmark, it collapsed in manual testing because 56.5% of its records were formulaic templates and natural conversational greetings were virtually absent.

---

## 2. Empirical Findings vs. Recommended Actions

| Measured Empirical Finding in SFT v3 | Quantitative Evidence | Root Cause Analysis | Recommended Action for SFT v4 |
| :--- | :--- | :--- | :--- |
| **Total Absence of Simple Greetings** | Only 9 records (216 tokens) in train; 0 simple "Hello/Hi/Hey" exchanges. | Builder script placed greetings at the start of pool, dumping them into val split; hash deduplication rejected repeated greetings. | Curate ~1,200 diverse, natural greeting pairs with proportionate responses (e.g. "Hey! How's it going?"). Ensure proper train/val stratification. |
| **Monolithic Multiple-Choice Attractor** | 8,269 records (25.88% of train) open with `"The correct answer is [Choice]..."`. | `tau/commonsense_qa` was ingested with a single rigid string template. | Prune from 8,269 down to ~1,500 diverse examples. Reformulate responses to explain conceptual reasons without the rigid choice formula. |
| **Massive Synthetic Retrieval Overrepresentation** | 9,781 records (30.61% of train), consuming 57.61% of all input sequence tokens. | Retrieval was generated at 5 context lengths with 14,000 synthetic candidates to satisfy a 540k token quota. | Downscale retrieval to ~5% of supervised target tokens (~180k tokens combined with structured JSON). Vary opening syntax to kill citation attractors. |
| **Multi-Turn Dialogue Starvation** | Only 945 records (2.96% of records) are true multi-turn dialogues. | SFT v3 heavily favored single-turn tasks. Multi-turn dialogues made up under 3% of interaction events. | Expand multi-turn dialogue to 20% of supervised target tokens, including 3–12 turn dialogues covering topic shifts, user corrections, and continuity. |
| **Identity Amnesia & Hallucination** | 0 records feature name introduction, intervening turns, and subsequent recall. | Assistant was never trained to persist conversational state across intervening turns. | Author 600 dedicated identity-retention dialogues covering 50+ diverse names, intervening turns, corrections, and unsupplied name probes. |
| **Corporate Email Contamination** | 460 business-email rewrite tasks categorized under `general_conversation`. | SFT v2 ingested `smoltalk:smol-rewrite` as conversation. | Re-classify legitimate text-rewriting and email-drafting tasks under `instruction_following`. Purge unhelpful business boilerplate. |
| **Zero Safety & Responsible Data** | Exactly 0 records (0.0%) and 0 tokens (0.0%) for safety. | SFT v3 had no safety category or alignment suite. | Add 5% supervised target tokens (~180k tokens) covering crisis support referral, neutral high-risk refusal, cyber education, and medical consultation advice. |

---

## 3. Recommended SFT v4 Target Composition (Supervised Tokens)

The target composition for SFT v4 is defined strictly in terms of **Supervised Assistant-Target Tokens**:

| Category | Target Share (%) | Train Target Tokens | Validation Target Tokens | Total Token Budget | Primary Data Sources & Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **General Conversation** | **25%** | 900,000 | 90,000 | 990,000 | Curated natural greetings (~1,200 items), casual everyday dialogue, filtered `smoltalk:everyday-conversations`, `dolly_qa`. |
| **Multi-Turn Dialogue** | **20%** | 720,000 | 72,000 | 792,000 | Identity-retention suite (600 items), `custom_context_insulation` (1,200 items), curated `smol-magpie-ultra` (4–12 turns). |
| **Knowledge & Explanations** | **15%** | 540,000 | 54,000 | 594,000 | Curated direct QA, `local_sft_v2:squad`, `smoltalk:explore-instruct`, pruned & reformulated `commonsense_qa` (~1,500 items). |
| **Instruction Following** | **12%** | 432,000 | 43,200 | 475,200 | `local_sft_v2:dolly_instruction`, `smoltalk:smol-rewrite` (moved from conversation), multi-constraint instruction tasks. |
| **Reasoning & Mathematics** | **10%** | 360,000 | 36,000 | 396,000 | `openai/gsm8k:train` (quarantined test), curated verified arithmetic, algebra, and logic (Python-verified solutions). |
| **Coding & Technical** | **8%** | 288,000 | 28,800 | 316,800 | `local_sft_v2:code_alpaca`, `local_sft_v2:mbpp`, `smoltalk:self-oss-instruct`. AST syntax validation (`ast.parse`) for 100% of code. |
| **Retrieval & Structured Output** | **5%** | 180,000 | 18,000 | 198,000 | Downscaled synthetic retrieval (~90k tokens, varied phrasing) + validated structured JSON (`apigen-80k`, `json`, validated via `json.loads`). |
| **Safety & Responsible Responses** | **5%** | 180,000 | 18,000 | 198,000 | Dedicated safety suite (~1,200 items): crisis helpline referral, neutral refusal of harmful tasks, cybersecurity education, medical advice boundaries. |
| **Total** | **100%** | **3,600,000** | **360,000** | **3,960,000** | Balanced, conversational, general-purpose alignment corpus. |

---

## 4. Key Engineering Safeguards for SFT v4 Build

1. **Stratified Train/Val Splitting**: To prevent the bug that starved SFT v3 of greetings, the dataset builder must shuffle all candidates within each category first, then split each category 90/10 into train and val splits, ensuring greetings and identity examples exist in both.
2. **Deterministic Random Seeds**: Fix random seed to `42` across all candidate selection, sampling, and splitting passes.
3. **Automated Syntax Gates**:
   - Every Python snippet validated via `ast.parse()`.
   - Every JSON output validated via `json.loads()`.
   - Every arithmetic example verified via programmatic execution.
4. **Attractor Elimination**:
   - Cap rigid opening phrases to $< 1\%$ of total dataset.
   - Prohibit epistolary salutations (`Dear [Name]`, `I hope this email finds you well`).
   - Prohibit rigid multiple-choice choice syntax (`"The correct answer is A"`).

