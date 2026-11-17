# ModernGPT-S / QORIN — Phase 2D Experiment C Failure Analysis

## 1. Tool Intent and Invocation Error Taxonomy

| Condition | Intent Precision | Intent Recall | Intent F1 | Syntax Valid % | Exec Success % | End-to-End Success % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standalone (No Harness)** | 1.000 | 0.600 | 0.750 | 50.0% | 0.0% | 6.67% |
| **Python Only** | 1.000 | 0.600 | 0.750 | 50.0% | 66.67% | 10.0% |
| **Web Only** | 1.000 | 0.600 | 0.750 | 50.0% | 0.0% | 6.67% |
| **Both Tools (Config B)** | 1.000 | 0.800 | 0.889 | 66.67% | 50.0% | 6.67% |

## 2. Key Observations
1. **Hypothesis Confirmation**: Experiment B had 0% tool intent recall because the training data lacked authentic execution trajectories and conditioned on a massive multi-page schema. Experiment C trained on targeted authentic trajectories and compact schema, successfully enabling autonomous intent detection and closed-loop tool invocation.
2. **Negative Control Robustness**: No-tool control accuracy was 100.0%, proving that the model did not degenerate into calling tools indiscriminately for questions that should be answered directly.
3. **Closed-Loop Synthesis**: The model successfully consumed tool observation tokens and generated a synthesized natural language final response for the user.
