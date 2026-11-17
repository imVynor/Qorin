# ModernGPT-S 500M — Multi-Domain Generation Quality & Degeneration Report

This evaluation benchmarks text generation quality, lexical diversity, prompt echo, and attractor degeneration across 50 curated prompts spanning 13 functional domains under both Greedy decoding and Stochastic sampling (T=0.7, p=0.9).

## 1. Executive Summary: Degeneration & Diversity Comparison

| Evaluation Metric | Baseline 501M (Greedy) | Latest Post-1B (Greedy) | Baseline 501M (Sampling) | Latest Post-1B (Sampling) | Trend / Progress |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Distinct-1 (Unigram)** | 0.2208 | **0.303** | 0.3898 | **0.429** | Higher diversity |
| **Distinct-2 (Bigram)** | 0.2843 | **0.3733** | 0.5457 | **0.5714** | Broader vocabulary |
| **Distinct-3 (Trigram)** | 0.3179 | **0.405** | 0.6111 | **0.6356** | Phrase richness |
| **4-gram Repetition %** | 65.34% | **57.03%** | 34.92% | **31.99%** | Reduced repetition |
| **Attractor Trap Rate** | 58.0% | **48.0%** | 26.0% | **20.0%** | Loop resilience |
| **Prompt Echo Rate** | 0.0% | **2.0%** | 2.0% | **2.0%** | Completion focus |
| **Generation Speed** | 49.2 tok/s | **53.3 tok/s** | 48.4 tok/s | **60.7 tok/s** | Runtime throughput |

## 2. Per-Domain Repetition & Degeneration Breakdown (Greedy)

| Domain | Prompts | 501M 4-gram Rep % | Latest 4-gram Rep % | 501M Traps | Latest Traps | 501M Echoes | Latest Echoes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Adversarial** | 4 | 88.67% | **72.97%** | 4 | **2** | 0 | **0** |
| **Coding** | 4 | 64.91% | **61.01%** | 3 | **1** | 0 | **1** |
| **Conversational** | 4 | 79.63% | **34.73%** | 3 | **1** | 0 | **0** |
| **Creative** | 4 | 58.81% | **64.28%** | 2 | **3** | 0 | **0** |
| **Dialogue** | 4 | 36.34% | **59.38%** | 1 | **2** | 0 | **0** |
| **Factual** | 4 | 71.77% | **53.85%** | 3 | **2** | 0 | **0** |
| **Instruction_following** | 4 | 75.94% | **75.88%** | 2 | **3** | 0 | **0** |
| **Logic** | 3 | 62.54% | **55.37%** | 2 | **2** | 0 | **0** |
| **Mathematics** | 4 | 54.54% | **54.56%** | 2 | **1** | 0 | **0** |
| **Open_reasoning** | 3 | 77.45% | **86.85%** | 2 | **3** | 0 | **0** |
| **Structured_data** | 4 | 77.33% | **45.83%** | 2 | **1** | 0 | **0** |
| **Summarization** | 4 | 51.3% | **34.88%** | 1 | **1** | 0 | **0** |
| **Technical** | 4 | 52.52% | **48.88%** | 2 | **2** | 0 | **0** |

## 3. Key Qualitative Observations
- **Attractor Loop Behavior:** Pretrained base models without instruction tuning naturally encounter repetitive loops under greedy decoding on open-ended or adversarial prompts. Continued pretraining significantly dampens loop propensity.
- **Prompt Continuation vs Answering:** As an autoregressive base model, the checkpoint operates primarily by statistical continuation rather than assistant turn-taking, confirming the necessity of Supervised Fine-Tuning (SFT) for instruction alignment.
- **Lexical Diversity:** Increased pretraining exposure to 1.12B tokens elevated n-gram diversity across technical, code, and educational domains.