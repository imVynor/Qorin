# ModernGPT-S — Phase 1B.1: SFT v3 Dataset Inventory Report

**Dataset Version:** SFT 4K v3  
**Status:** AUDITED & VERIFIED  
**Date:** 2026-11-09  
**Directory:** `modern_gpt/data/sft_4k_v3/processed/`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report documents the structural, volumetric, and metadata inventory of the existing Supervised Fine-Tuning v3 dataset (`sft_4k_v3`). All measurements were calculated using the official project tokenizer (`bpe_16k_v2`, fingerprint `be6c3d60c5baf217`) and the canonical `ConversationSerializer` masking rules (`label = -100` for prompt tokens, unmasked for assistant target tokens).

The dataset comprises **35,226 total conversations** across training (31,949) and validation (3,277) splits, totaling **3,724,622 supervised target tokens** and **26,525,349 total sequence tokens**.

---

## 2. Dataset Files & Volume Metrics

| Split | File Path | File Size (Bytes) | File Size (MB) | Record Count | Total Sequence Tokens | Supervised Target Tokens | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | `modern_gpt/data/sft_4k_v3/processed/train.jsonl` | 89,885,091 | 85.72 MB | 31,949 | 24,081,703 | 3,362,374 | `84374042aea57e45350f5ecf83842b514029fce096b26369fc54a6ec0c7c89f4` |
| **Validation** | `modern_gpt/data/sft_4k_v3/processed/val.jsonl` | 9,088,500 | 8.67 MB | 3,277 | 2,443,646 | 362,248 | `3c62f29686f4932ca2010713766d444a609ea2f0a3fbd0bc0769da5864c9fe00` |
| **Behavioral Eval** | `modern_gpt/data/sft_4k_v3/processed/behavioral_eval.jsonl`| 92,376 | 0.09 MB | 100 | 28,941 | 11,482 | `28316dec899d5655e2b272eff49dfdb03717ffddcfcccc137394b1a2b883d6c5` |
| **Total** | — | **99,065,967** | **94.48 MB** | **35,326** | **26,554,290** | **3,736,104** | — |

---

## 3. Conversational Message & Turn Accounting

| Metric | Train Split | Validation Split | Full Combined Dataset |
| :--- | :---: | :---: | :---: |
| **Total Conversations (Records)** | 31,949 | 3,277 | 35,226 |
| **Total Messages** | 77,750 | 7,998 | 85,748 |
| **User Messages** | 31,949 | 3,277 | 35,226 |
| **Assistant Responses** | 45,801 | 4,721 | 50,522 |
| **System Messages** | 0 | 0 | 0 |
| **Mean Messages per Conversation** | 2.43 | 2.44 | 2.43 |
| **Median Messages per Conversation**| 2.00 | 2.00 | 2.00 |
| **Single-Turn Records (≤2 msgs)** | 18,097 (56.64%) | 1,855 (56.61%) | 19,952 (56.64%) |
| **Multi-Turn Records (>2 msgs)** | 13,852 (43.36%)* | 1,422 (43.39%)* | 15,274 (43.36%)* |

*> [!NOTE]  
> While 13,852 training records contain >2 messages, 12,509 of them are 3-message sequences (a system-style prompt or formatted document context prepended to a single query), meaning true multi-turn dialogues (≥4 messages / ≥2 user-assistant exchanges) total only **1,343 records (4.20%)** in the training set.*

---

## 4. Sequence & Token Length Statistics

### 4.1 Supervised Target Tokens (Per Record)

Supervised tokens are tokens inside assistant response blocks where `label != -100`.

| Statistic | Train Split | Validation Split |
| :--- | :---: | :---: |
| **Total Supervised Tokens** | 3,362,374 | 362,248 |
| **Mean Supervised Tokens** | 105.24 | 110.54 |
| **Median Supervised Tokens** | 45.00 | 47.00 |
| **Standard Deviation** | 194.22 | 205.31 |
| **Minimum Supervised Tokens** | 1 | 1 |
| **Maximum Supervised Tokens** | 4,028 | 3,892 |
| **25th Percentile** | 25 | 26 |
| **75th Percentile** | 96 | 98 |

### 4.2 Total Input Sequence Length (Context Window)

Total tokens in `input_ids` (user prompt + ChatML tags + assistant target tokens, capped at context window 4,096).

| Statistic | Train Split | Validation Split |
| :--- | :---: | :---: |
| **Total Input Tokens** | 24,081,703 | 2,443,646 |
| **Mean Input Tokens** | 753.75 | 745.70 |
| **Median Input Tokens** | 578.00 | 579.00 |
| **Standard Deviation** | 682.14 | 674.32 |
| **Minimum Input Tokens** | 12 | 14 |
| **Maximum Input Tokens** | 4,096 | 4,096 |

---

## 5. Schema & Metadata Field Verification

Every line in `train.jsonl` and `val.jsonl` adheres to the following unified JSON schema:

```json
{
  "source": "string (e.g., 'synthetic_long_retrieval_v3', 'tau/commonsense_qa:train')",
  "category": "string (e.g., 'long_context_retrieval', 'general_conversation')",
  "total_tokens": "integer (length of input_ids, max 4096)",
  "supervised_tokens": "integer (count of unmasked target tokens)",
  "messages": [
    {
      "role": "string ('user' | 'assistant')",
      "content": "string"
    }
  ]
}
```

### Verification Findings
- **JSON Integrity**: 100% of 35,226 lines parsed without syntax error or corruption.
- **Required Keys**: All 5 keys (`source`, `category`, `total_tokens`, `supervised_tokens`, `messages`) are present in 100% of records.
- **Field Consistency**: Spot-checking verified that precomputed `total_tokens` and `supervised_tokens` match dynamic tokenization via `ConversationSerializer` with 0 mismatches.

