# ModernGPT-S 500M — Phase 2C Stage 3 Failure Analysis Report

## 1. Mathematical Reasoning Failure Analysis
- **Base Pretrained**: Failed word problems and algebra completely (generated ungrounded continuation tokens).
- **Experiment A**: Correctly recognized math prompts, but exhibited arithmetic calculation mistakes (e.g. incorrect carry operations in multi-step arithmetic).
- **Experiment B**: Improved algebraic manipulation and basic calculations, but still lacks multi-step precision without external tool invocation.

## 2. Coding Correctness Failure Analysis & `is_prime` Retest
- **is_prime Failure Retest**:
  - Base Pretrained: Syntax invalid or incomplete.
  - Experiment A: Fails edge cases (0, 1, negative numbers) by not explicitly returning False for `n <= 1`.
  - Experiment B: Passes prime detection for standard numbers; edge case handling requires explicit boundary instruction.
- **Subprocess Sandbox Results**: Zero execution timeouts observed; runtime errors primarily stemmed from NameError / missing imports in complex problems.

## 3. Tool-Use Provenance & Failure Analysis
- **APIGen vs Sandbox Gap**:
  - APIGen schema demonstrations instilled strong JSON syntax compliance for `<|tool_call|>`.
  - Authentic sandbox execution was required for the model to learn argument construction and recovery.
- **Direct Direct Answering (No-Tool Controls)**:
  - Base Pretrained: Never calls tools.
  - Experiment A: Never calls tools (tool tokens were masked out of Exp A training).
  - Experiment B: Accurately distinguishes tasks requiring Python calculation vs direct factual recall.
