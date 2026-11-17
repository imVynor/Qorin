# Stage 16B.1 Live Inference Forensic Audit Regression Results

## Executive Summary

- **Test Suite**: `tests/test_stage16b1_live_isolation.py` & Live API Reproduction Suite
- **Model Checkpoint**: `qorin-500m-sft-exp-f`
- **Parent Checkpoint SHA-256**: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`
- **Total Test Cases**: 8 (Live endpoint) + 8 (Isolation test suite) = 16 Tests
- **Pass Rate**: **100% (16/16)**
- **Failures**: **0**

---

## Detailed Results

### 1. Live Endpoint Reproduction Suite

| Test Identifier | Query / Scenario | Output Verification | Tools Executed | Result |
| :--- | :--- | :--- | :---: | :---: |
| **Test 1** | `Who is Jensen Huang?` | Taiwanese-born entrepreneur who cofounded NVIDIA (`[SOURCE_2]`) | 2 | **PASSED** |
| **Test 2** | `Who is Donald Trump?` | 47th president of the United States (`[SOURCE_1]`) | 2 | **PASSED** |
| **Test 3** | `Who is the current president of the United States?` | Donald Trump, 47th president (`[SOURCE_3]`) | 2 | **PASSED** |
| **Test 4** | `What is the capital of Australia?` | Canberra, federal capital city (`[SOURCE_1]`) | 2 | **PASSED** |
| **Test 5** | Sequential in same conversation (Jensen → Trump) | Turn 1: Jensen Huang (NVIDIA). Turn 2: Donald Trump. Zero cross-turn contamination. | 2 per turn | **PASSED** |
| **Test 6** | Same two queries in separate conversations | Conv A: Jensen Huang. Conv B: Donald Trump. Zero cross-conversation pollution. | 2 per conv | **PASSED** |
| **Test 7** | `Who is Donald Trump?` under RAG OFF, AUTO, ALWAYS | Grounded 47th president answer across all three RAG modes. | 2 per mode | **PASSED** |
| **Test 8** | Repeat query twice in same conversation | Both turns return verified Donald Trump response. | 2 per turn | **PASSED** |

---

### 2. Isolation & Grounding Test Suite (`test_stage16b1_live_isolation.py`)

| Test Method | Requirement Verified | Result |
| :--- | :--- | :---: |
| `test_01_jensen_huang_attribution` | Correct entity attribution (NVIDIA CEO, not Tokyo University student) | **PASSED** |
| `test_02_donald_trump_attribution` | Correct entity attribution (47th president, not younger brother) | **PASSED** |
| `test_03_cross_turn_isolation` | Zero Jensen Huang carryover into subsequent Donald Trump turn | **PASSED** |
| `test_04_concurrent_requests_isolation` | Threaded concurrent requests execute with independent state | **PASSED** |
| `test_05_grounding_validator_adversarial_rejection` | Rejects all 7 adversarial relationship and attribution swaps | **PASSED** |
| `test_06_fallback_safety_and_pronouns` | Prevents dangling pronouns and cross-record sibling confusion | **PASSED** |
| `test_07_fake_citation_rejection` | Rejects non-existent source IDs (`[SOURCE_99]`) | **PASSED** |
| `test_08_insufficient_evidence_handling` | Returns explicit uncertainty when evidence is empty | **PASSED** |

---

## Existing Test Suite Conformance
- Full repository unit tests (`python -m unittest discover tests`): **32/32 PASSED**.
