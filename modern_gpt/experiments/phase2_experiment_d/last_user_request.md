<USER_REQUEST>
# QORIN Experiment D — Stage 5: Generalization & Reliability Audit

Experiment D training is complete. The authoritative best checkpoint is:

`modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt`

Expected SHA-256:

`c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`

Experiment C remains the immutable baseline:

`6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`

Your task is to perform a rigorous evaluation-only audit. **Do not train, fine-tune, modify model weights, alter datasets, or overwrite existing artifacts.**

## 1. Integrity and reproducibility

Before evaluation:

- Verify both checkpoint hashes against the expected values.
- Verify tokenizer integrity and architecture compatibility.
- Verify that the best checkpoint corresponds to Experiment D Step 425.
- Load Experiment C and Experiment D using identical inference settings.
- Record decoding parameters, system prompts, tool configurations, context limits, and model versions.
- Confirm that both models use the same harness and tool implementations.

If any integrity check fails, stop and report it.

## 2. Build an independent held-out benchmark

Create a new benchmark of **150 tasks**, with no exact or semantic leakage from Experiment D's training and validation data.

Use these categories:

| Category | Tasks |
|---|---:|
| Colloquial Web research and entity lookup | 30 |
| Current-information and time-sensitive queries | 20 |
| Python computation and data analysis | 25 |
| Direct conceptual questions requiring no tools | 25 |
| Tool-selection discrimination | 20 |
| Multi-step Web → Python tasks | 15 |
| Tool failure and recovery | 15 |
| **Total** | **150** |

Requirements:

- Use natural, varied language rather than rigid templates.
- Include ambiguous prompts where the model must decide whether a tool is necessary.
- Include unfamiliar entities and queries absent from training.
- Include questions where Web is necessary, Python is necessary, both are necessary, and neither is appropriate.
- Include negative controls to detect unnecessary tool invocation.
- Include tasks with insufficient evidence, where the model should acknowledge uncertainty.
- For current-information tasks, record the expected answer and its source timestamp.
- Freeze the benchmark before evaluating either model.
- Audit exact and semantic overlap with Experiment D train and validation records.

Do not silently exclude difficult tasks after seeing model outputs.

## 3. Evaluate both models under identical conditions

Run Experiment C and Experiment D on exactly the same 150 tasks.

Measure the complete tool-use funnel:

1. Tool-intent recognition.
2. Correct tool selection.
3. Argument/schema validity.
4. Actual orchestrator dispatch.
5. Tool execution success.
6. Evidence utilization.
7. End-to-end semantic correctness.

Report:

- TP, FP, TN, FN.
- Precision, recall, and F1.
- Tool-selection accuracy.
- Web and Python selection recall separately.
- Schema-validity rate.
- Dispatch and execution rates.
- End-to-end semantic success.
- Unnecessary tool-call rate.
- Unsupported factual-claim rate.

Use task-level denominators consistently. Do not combine different funnel stages into a single success metric.

## 4. Investigate the five execution-to-answer failures

The earlier 30-task evaluation reported:

- 22/30 successful dispatches and executions.
- 17/30 successful final answers.

Inspect the five tasks that executed successfully but did not produce semantically correct final answers.

For each failure, identify whether the problem originated from:

- Incorrect search query.
- Poor or irrelevant retrieved sources.
- Failure to extract relevant evidence.
- Incorrect intermediate Python computation.
- Failure to transfer intermediate results.
- Hallucinated synthesis.
- Citation or source-grounding failure.
- Other identifiable causes.

Do not assume the same five failures without checking the original evaluation records.

## 5. Deep-dive into Web research

Evaluate whether Web retrieval is genuinely useful rather than merely invoked.

For every Web task, capture:

- Original user query.
- Model-selected tool.
- Generated search query.
- Actual dispatch trace.
- Retrieved URLs and source titles.
- Source relevance and credibility.
- Evidence extracted.
- Final answer.
- Citation correctness.
- Whether the final answer is supported by retrieved evidence.

Specifically investigate colloquial requests such as:

- "Grab some info on [unfamiliar public figure]."
- "What's going on with [current event]?"
- "Find out whether [recent claim] is true."
- "Give me the latest on [topic]."

Do not use only the previously successful Elon Musk query as proof of Web competence.

## 6. Generalization and memorization audit

Determine whether the unusually low Experiment D validation loss (0.0021) reflects genuine learning or narrow-pattern memorization.

Inspect:

- Assistant-target token masking.
- Validation loss normalization.
- Training/validation semantic similarity.
- Per-category validation examples.
- Output diversity across unseen prompts.
- Repeated phrasing and memorized response structures.
- Confidence and uncertainty behavior.
- Performance on paraphrases and compositional tasks.

Explain what the loss does and does not establish.

## 7. Statistical analysis

For Experiment C versus D:

- Report absolute and relative changes.
- Provide confidence intervals for important success rates.
- Use paired comparisons because both models receive identical tasks.
- Report task-level disagreements.
- Use an appropriate paired significance test where justified.
- Avoid declaring statistical significance from small subgroups without qualification.

Do not hide failures behind aggregate averages.

## 8. Artifact preservation

Store all outputs in a new directory:

`modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/`

Create:

- `benchmark.jsonl`
- `benchmark_manifest.json`
- `leakage_audit.json`
- `experiment_c_results.jsonl`
- `experiment_d_results.jsonl`
- `tool_traces.jsonl`
- `failure_analysis.json`
- `metrics.json`
- `statistical_analysis.json`
- `generalization_report.md`
- `SHA256SUMS`

Preserve all previous checkpoints, reports, datasets, and training logs.

## 9. Resource and execution safeguards

- This is evaluation-only. No optimizer updates or training.
- Follow the existing 105-minute maximum active GPU policy and minimum 15-minute cooldown.
- Save intermediate results so an interrupted audit can resume without duplicating completed evaluations.
- Avoid unnecessary concurrent model loading if it risks exhausting VRAM.
- Do not alter the live production model registry.
- Do not overwrite the existing harness configuration.

## 10. Final report

Present:

1. Integrity verification.
2. Benchmark construction and leakage audit.
3. Experiment C versus D comparison.
4. Full tool-use funnel.
5. Web research quality.
6. Python tool reliability.
7. Direct-answer preservation.
8. Multi-step tool-use performance.
9. Failure analysis.
10. Statistical confidence and limitations.
11. Whether Experiment D generalizes beyond its training distribution.
12. Recommended next research experiment.

### Final restriction

Do not start Experiment E, change the model, or initiate further fine-tuning.

The objective is to establish whether Experiment D represents a genuine improvement in general-purpose tool-augmented language-model performance, not merely better performance on its own training patterns.

If any required evaluation cannot be completed, report it explicitly. Never fabricate benchmark results, source evidence, or statistical significance.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-17T08:16:38+05:30.

The user's current state is as follows:
Other open documents:
- c:\Users\Anish\Desktop\Projects\ML-course\gpt2\gpt.ipynb (LANGUAGE_UNSPECIFIED)
- c:\Users\Anish\Desktop\Projects\ML-course\modern_gpt\scripts\phase1d_benchmark_data.py (LANGUAGE_UNSPECIFIED)
- c:\Users\Anish\Desktop\Projects\ML-course\modern_gpt\scripts\pretrain_500m.py (LANGUAGE_UNSPECIFIED)
- c:\Users\Anish\Desktop\Projects\ML-course\backend\tools\two_stage_protocol.py (LANGUAGE_UNSPECIFIED)
- c:\Users\Anish\Desktop\Projects\ML-course\modern_gpt\reports\checkpoint_architecture_audit\audit_all_9_checkpoints.py (LANGUAGE_UNSPECIFIED)
</ADDITIONAL_METADATA>