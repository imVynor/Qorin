# QORIN Research Showcase Readiness & Presentation Guide

**Audit Date:** November 17, 2026  
**Audited Target:** Research Showcase Website (`web/`) & Interactive Demo Harness  
**Course & Project:** ES 335 — Tool-Augmented Small Language Models  
**Status:** **READY FOR SHOWCASE DEMONSTRATION**

---

## 1. Executive Summary

The QORIN research project is fully prepared for the final ES 335 research showcase. All active frontend interfaces, metadata registries, and inference harnesses reflect the verified physical architecture of the **514M ModernGPT-S decoder-only Transformer** and its **Experiment C autonomous tool-augmented checkpoint**.

This guide outlines the showcase components, the empirical evidence base, and the strict presentation boundaries required for scientific rigor.

---

## 2. Research Showcase Components Checklist

| Component | Location / Route | Content Verified | Status |
| :--- | :--- | :--- | :--- |
| **Model Architecture Explorer** | `/model` (`Model.tsx`) | 20 layers, 24 query heads, 4 KV heads (6:1 GQA), SwiGLU, RMSNorm, 2048 context, 514,389,504 parameters | **VERIFIED** |
| **Experiment Catalog** | `/experiments` (`Experiments.tsx`) | Complete catalog of 17 experiments including Exp A, Exp B, and Exp C with exact dataset sizes and metrics | **VERIFIED** |
| **Research Methodology** | `/research` (`Research.tsx`) | Pretraining curves, SFT curves, passkey needle retrieval limits (4.0%), and tool-augmentation hypotheses | **VERIFIED** |
| **Interactive Harness** | `/harness` (`Harness.tsx`) | Live inference with model selector (all 9 checkpoints), RAG similarity slider, and independent Web/Python toggles | **VERIFIED** |
| **Checkpoint Lineage Graph** | Data: `website_data/checkpoint_lineage.json` | 4-node genealogical tree from Pretraining Base (252M) to Exp C (1.885M) with verified hashes | **VERIFIED** |
| **Telemetry & Observability** | Inspection panel (`InspectionPanel.tsx`) | Real-time TTFT, tokens/second, GPU VRAM allocation, and structured SSE event streaming | **VERIFIED** |

---

## 3. Core Scientific Distinctions for the Showcase

During the research presentation, the following distinctions must be clearly communicated:

### 1. Learned Intent vs Autonomous Agency
- **What the model learned:** QORIN learned to recognize linguistic and situational triggers where external computation or factual search is advantageous, and emit syntactically structured `<|tool_call|>` tags with valid argument schemas (F1: 0.889–1.000).
- **What the model is not:** QORIN is not a free-form autonomous agent capable of unconstrained multi-hour planning. Tool routing and safety enforcement are managed by the deterministic application harness (`two_stage_protocol.py` and `orchestrator.py`).

### 2. Execution Success vs Semantic Correctness
- **Execution Success:** The Python code parsed without errors, executed in the isolated sandbox, and exited with return code 0.
- **Semantic Correctness:** The executed algorithm actually solved the user's problem. (e.g. A naive odd-number check exits code 0, but is mathematically invalid for primality testing).

### 3. Parametric Capacity vs Non-Parametric Augmentation
- In standalone mode (both tools OFF), QORIN-500M exhibits typical sub-1B parameter limits: limited arithmetic depth, factual hallucinations on recent events, and bounded context recall.
- In augmented mode, external tools offset parametric deficits: Python provides exact arithmetic, while Web Researcher injects verifiable live citations.

---

## 4. Live Demonstration Protocol

For the ES 335 presentation demonstration, use this step-by-step walkthrough in `/harness`:

1. **Step 1: Standalone Direct Inference (Both Tools OFF)**
   - Query: *"What is the capital of France?"* or *"Explain what a binary search tree is."*
   - Observe: Model generates direct, conversational answers without emitting tool calls. Zero false positives.
2. **Step 2: Python Tool Isolation (Python ON, Web OFF)**
   - Query: *"Calculate the greatest common divisor of 123456 and 7890 using python."*
   - Observe: Model emits `<|tool_call|>{"name": "python", "arguments": {"code": "import math\nprint(math.gcd(123456, 7890))"}}`.
   - The harness displays an inline execution card with latency ($\sim 150\text{ms}$) and stdout `6`.
   - The model synthesizes: *"The greatest common divisor of 123,456 and 7,890 is 6."*
3. **Step 3: Web Researcher Isolation (Web ON, Python OFF)**
   - Query: *"Search for the latest news on NASA Artemis program."*
   - Observe: Model emits `web_research` call. Web researcher queries live sources and displays authentic URLs.
   - The model synthesizes the answer with explicit source attribution.
4. **Step 4: Two-Tool Mode (Both Tools ON)**
   - Query: *"Compute 4321 * 8765 using python."*
   - Observe: Tool selection accurately chooses Python despite both tools being active.
5. **Step 5: Safety & Timeout Demonstration**
   - Query: *"Run python code with infinite sleep: import time; time.sleep(10)"*
   - Observe: Process is cleanly terminated at 1.0s/5.0s, reporting `TimeoutError` without crashing the server.

---

## 5. Showcase Readiness Certification

The QORIN inference system and research showcase meet all institutional requirements for the ES 335 research evaluation:
- Checkpoints: Frozen and cryptographically verified.
- Architecture: 20 layers, 24 heads, 514M parameters definitively reconciled.
- Tool Harness: Safe, bounded, and closed-loop verified.
- Reporting: Transparent, honest, and reproducible.
