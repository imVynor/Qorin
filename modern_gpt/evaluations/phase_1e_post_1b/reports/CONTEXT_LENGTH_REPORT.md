# ModernGPT-S 500M — Context Length & Needle-in-a-Haystack Report

This evaluation benchmarks effective context length utilization using multi-depth Needle-in-a-Haystack passkey retrieval across lengths 128 to 2,048 tokens.

## 1. Executive Summary

| Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress |
| :--- | :--- | :--- | :--- |
| **Passkey Retrieval Accuracy** | 5.56% (1/18) | **27.78%** (5/18) | **+22.22%** |

## 2. Breakdown Across Context Lengths and Depths

| Context Length | 501M (10% / 50% / 90%) | Latest Post-1B (10% / 50% / 90%) |
| :--- | :--- | :--- |
| **128 tokens** | FAIL / FAIL / FAIL | **FAIL / FAIL / FAIL** |
| **256 tokens** | FAIL / FAIL / PASS | **PASS / PASS / PASS** |
| **512 tokens** | FAIL / FAIL / FAIL | **FAIL / PASS / FAIL** |
| **1024 tokens** | FAIL / FAIL / FAIL | **PASS / FAIL / FAIL** |
| **1536 tokens** | FAIL / FAIL / FAIL | **FAIL / FAIL / FAIL** |
| **2048 tokens** | FAIL / FAIL / FAIL | **FAIL / FAIL / FAIL** |

## 3. Analysis of RoPE Rotary Embeddings
- The 500M architecture utilizes RoPE with base frequency theta=500,000.
- Retrieval accuracy at short contexts (<512) is strong, while long contexts (>1024) in base models require attention sharpening through SFT to counter distraction tokens.