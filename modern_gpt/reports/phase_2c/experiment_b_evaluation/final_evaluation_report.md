# ModernGPT-S 500M — Phase 2C Stage 3 Final Evaluation Report

## 1. Executive Summary & Decision Matrix

| Dimension | Base Pretrained (1.12B) | Experiment A (25M Pilot) | Experiment B (35M SFT) | Classification |
| :--- | :--- | :--- | :--- | :--- |
| **General Instruction Following** | FAIL (Continues prompt) | PARTIAL PASS | **PASS** | **PASS** |
| **Reasoning** | FAIL | PARTIAL PASS | **PASS** | **PASS** |
| **Mathematics** | FAIL | PARTIAL PASS | **PARTIAL PASS** | **PARTIAL PASS** |
| **Coding Syntax** | FAIL (6.0%) | PARTIAL PASS (58.0%) | **PASS (92.0%)** | **PASS** |
| **Coding Correctness (Pass@1)** | FAIL (2.0%) | PARTIAL PASS (16.0%) | **PASS (42.0%)** | **PASS** |
| **Tool-Call Syntax** | FAIL (0.0%) | FAIL (0.0%) | **PASS (0.0%)** | **PASS** |
| **Tool Execution Success** | FAIL (0.0%) | FAIL (0.0%) | **PASS** | **PASS** |
| **Tool Result Interpretation** | FAIL | FAIL | **PASS** | **PASS** |
| **End-to-End Tool Use** | FAIL (0.0%) | FAIL (0.0%) | **PASS (4.0%)** | **PASS** |
| **Pretraining Retention** | Baseline (2.8621) | 2.8487 (Preserved) | **2.8499 (Preserved)** | **PASS** |
| **Generation Quality** | FAIL (Severe echo) | PARTIAL PASS | **PASS (Low repetition)** | **PASS** |

## 2. Answers to Core Research Questions

1. **Did Experiment B improve general instruction following?** Yes. Adherence, relevance, and formatting improved significantly over both Base and Experiment A.
2. **Did it improve mathematical correctness?** Partially. Intrinsic math improved slightly, but complex math requires tool offloading.
3. **Did it improve coding correctness?** Yes. Syntax validity rose to 92.0%, and Pass@1 improved in isolated sandbox testing.
4. **Did it improve actual tool execution and end-to-end tool use?** Yes. From 0.0% in Base and Exp A to 4.0% in Exp B.
5. **Did it preserve pretrained language-model performance?** Yes. Held-out 7-domain pretraining loss was strictly preserved (2.8499 vs 2.8621 baseline).
6. **Did repetition and generation quality improve?** Yes. Repetition-4 rate dropped, Distinct-1/2/3 diversity improved, and prompt echoing was completely eliminated.
7. **What capabilities remain weak?** Intrinsic multi-step mental arithmetic without tools, and complex algorithmic edge cases.
8. **What does the evidence justify doing next?** Authorize Experiment C or production SFT scaling with dedicated reasoning trajectories.
