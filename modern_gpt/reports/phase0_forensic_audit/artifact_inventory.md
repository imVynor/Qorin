# Phase 0: Forensic Diagnosis & Baseline Audit — Artifact Inventory

This document records the exact file paths, sizes, and cryptographic SHA-256 hashes of all foundational checkpoints, tokenizers, datasets, training logs, and benchmark suites in the ModernGPT-S repository as of Phase 0.

## 1. Protected Model Checkpoints

All five historical checkpoints are intact, read-only, and verified against recorded project manifests:

| Checkpoint Identifier | File Path | Size (Bytes) | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Base 100M Pretrained** | `modern_gpt/checkpoints/modern_gpts_100m/best_val.pt` | 1,208,690,819 | `bded509c44eabc2932dc1e293967bb0524acc31182fe9df0295b9ad2a4671e31` | Verified Intact |
| **Step 4 4K Pilot (Continued PT)** | `modern_gpt/checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt` | 617,297,215 | `3b23f37b82279c63d5913933fef22ea487ed28f28e08878cfa4d531064900817` | Verified Intact |
| **Step 6B SFT v1** | `modern_gpt/checkpoints/modern_gpts_100m_sft_4k/best_val.pt` | 617,291,183 | `97e4f8a11b7321450c0fc497b38c832080f1fc8b9b12fe7defb107d7132ae271` | Verified Intact |
| **Step 7B SFT v2** | `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v2/best_val.pt` | 617,289,583 | `4115769057d97c604f642262d462533694a52c2d27469a2ae66131c0244e0136` | Verified Intact |
| **Step 8D SFT v3** | `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt` | 617,289,583 | `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510` | Verified Intact |

## 2. Tokenizer Artifacts

The authoritative BPE tokenizer `tokenizer_16k_v2` has a vocabulary size of 16,384 tokens and canonical fingerprint `be6c3d60c5baf217`:

| Component | File Path | Size (Bytes) | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tokenizer Config** | `modern_gpt/artifacts/tokenizer_16k_v2/tokenizer_config.json` | 282 | `7e728ff718209cac30d736ff877c20d66845b1859eaec0ac6efa6a22124446c2` | Verified Intact |
| **Tokenizer Model/Vocab** | `modern_gpt/artifacts/tokenizer_16k_v2/tokenizer.json` | 1,111,929 | `6c155b6fb30153aac2f206fc14c30b78e3c3b45b16748c87faf7bf9b04ccbe69` | Verified Intact |

## 3. Supervised Fine-Tuning (SFT) Datasets

| Dataset Identifier | File Path | Records | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- |
| **SFT v3 Train** | `modern_gpt/data/sft_4k_v3/processed/train.jsonl` | 31,949 | 89,885,091 | `84374042aea57e45350f5ecf83842b514029fce096b26369fc54a6ec0c7c89f4` |
| **SFT v3 Validation** | `modern_gpt/data/sft_4k_v3/processed/val.jsonl` | 3,277 | 9,088,500 | `3c62f29686f4932ca2010713766d444a609ea2f0a3fbd0bc0769da5864c9fe00` |
| **Behavioral Evaluation** | `modern_gpt/data/sft_4k_v3/processed/behavioral_eval.jsonl` | 100 | 92,376 | `28316dec899d5655e2b272eff49dfdb03717ffddcfcccc137394b1a2b883d6c5` |
| **SFT v2 Train** | `modern_gpt/data/sft_4k_v2/processed/train.jsonl` | 26,734 | 75,187,085 | `24b30b431a14625b82c4b36e6ac4be5d81934c4b6e6e031717060ecf7ca816fc` |
| **SFT v2 Validation** | `modern_gpt/data/sft_4k_v2/processed/val.jsonl` | 2,713 | 7,610,178 | `d47ec7874ab699bb0c82b6d373eb6ed314d076eabc61c4fdcb322a7ceea0db3b` |
| **SFT v1 Train** | `modern_gpt/data/sft_4k/processed/train.jsonl` | 9,997 | 28,561,531 | `054524ba1294aaca6bc4400a2c640a22a13b9805516e674f89632474f5d19e6e` |
| **SFT v1 Validation** | `modern_gpt/data/sft_4k/processed/val.jsonl` | 1,111 | 3,254,561 | `545c440d5a70132ef196207bb0fbfcc3f6b987b90cb9d885015804e601d7f01c` |

## 4. Experiment Training Logs & Configuration Files

| Artifact | File Path | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Step 8D Training Log** | `modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d/step_08d_training_log.jsonl` | 3,087,762 | `d601b71aaadcde2ebaf89d09271cd82f5d64a6bb248291de5a1b31db5b66c30c` |
| **Step 8D Config** | `modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d/step_08d_training_config.yaml` | 1,714 | `a9062e5b0ec353cfc31e24744f69ea63a921d7bbcb17caec6252dbefdb438e88` |
| **Step 7B Training Log** | `modern_gpt/artifacts/experiments/modern_gpts_100m_step_07b/step_07b_training_log.jsonl` | 2,187,941 | `3a968da9ae4575916ee1b1caeaee081d6f1bfd8e57813a0e69ca84bc86ef3f75` |
| **Step 7B Config** | `modern_gpt/artifacts/experiments/modern_gpts_100m_step_07b/step_07b_training_config.yaml` | 1,388 | `c97eaefc8ef13636b110bc7fb16d99e5a87e5898864d262973c9fc13c32eead4` |
| **Step 6B Summary** | `modern_gpt/artifacts/experiments/modern_gpts_100m_sft_4k/training_summary.json` | 3,837 | `09363a0a38fe4eb49f4fa3ec6dcbfabeb84c98782ee91b97a2ce3907c08792c6` |

## 5. External Comparison Models

| Model | Path / Cache Location | Size | Type |
| :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-0.5B-Instruct** | HuggingFace Cache via `temp_qwen/` | ~1.0 GB | 0.49B parameter Dense Transformer |

