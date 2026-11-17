import React, { useState } from 'react';
import styles from './LearningModule.module.css';

interface EvalTask {
  name: string;
  category: string;
  tier: string;
  modelTested: string;
  checkpointPath: string;
  split: string;
  sampleCount: string;
  sampleNature: 'Verbatim Diagnostic Failure Output' | 'Illustrative Failure Archetype Derived from Evaluation Failure Logs';
  metric1Label: string;
  metric1Val: string;
  metric2Label: string;
  metric2Val: string;
  diagnosticFinding: string;
  sampleOutput: string;
}

const EVAL_TASKS: EvalTask[] = [
  {
    name: 'Python Function Generation (MBPP)',
    category: 'Code Synthesis',
    tier: 'Tier 2 (Syntax) vs. Tier 3 (Functional Execution)',
    modelTested: 'QORIN-16M SFT',
    checkpointPath: 'checkpoints/exp_100m/best_val.pt',
    split: 'Held-Out MBPP Benchmark Partition (Isolated from SFT v2 training)',
    sampleCount: '40 isolated problems (evaluated via AST and isolated Python subprocess)',
    sampleNature: 'Illustrative Failure Archetype Derived from Evaluation Failure Logs',
    metric1Label: 'AST Syntax Validity',
    metric1Val: '90.0% Valid',
    metric2Label: 'Unit Test Pass Rate',
    metric2Val: '0.0% Passed',
    diagnosticFinding: 'The model generates syntactically valid def statements, clean indentation, and plausible docstrings (90.0% AST valid), but frequently produces tautological logic (e.g. if l[i] == l[i]) or off-by-one errors that fail 100% of functional assertions.',
    sampleOutput: 'def has_consecutive(l):\n    # Checks for consecutive elements\n    for i in range(len(l)):\n        if l[i] == l[i]: # Tautology bug: compares element to itself\n            return True\n    return False',
  },
  {
    name: 'Structured JSON Entity Extraction',
    category: 'Tool Schema Adherence',
    tier: 'Tier 2: Syntactic Parse vs. Schema Key Adherence',
    modelTested: 'QORIN-16M SFT',
    checkpointPath: 'checkpoints/exp_100m/best_val.pt',
    split: 'Held-Out Paraloq Natural Language Extraction Suite',
    sampleCount: '40 isolated prompts (validated with json.loads and JSON Schema validator)',
    sampleNature: 'Illustrative Failure Archetype Derived from Evaluation Failure Logs',
    metric1Label: 'Valid JSON Formatting',
    metric1Val: '85.0% Valid',
    metric2Label: 'Strict Schema Compliance',
    metric2Val: '15.0% Compliant',
    diagnosticFinding: 'The model outputs valid JSON conforming to json.loads() 85% of the time, but omits mandatory schema keys or invents unrequested attributes, failing schema compliance in 85% of trials.',
    sampleOutput: '{\n  "status": "success",\n  "author": "Alice",\n  "unrequested_key": "hallucinated"\n}',
  },
  {
    name: 'Long-Context Passkey Retrieval (4K Context)',
    category: 'Attention Routing',
    tier: 'Tier 1 (Perplexity) vs. Tier 4 (Needle Retrieval)',
    modelTested: 'QORIN-100M 4K Pilot',
    checkpointPath: 'artifacts/experiments/modern_gpts_100m_4k_pilot/step_2000.pt',
    split: 'Synthetic Passkey Haystack (Relative depths: 0.1, 0.25, 0.5, 0.75, 0.9 across 512, 1K, 2K, 4K lengths)',
    sampleCount: '100 randomized trials (20 trials per depth level)',
    sampleNature: 'Illustrative Failure Archetype Derived from Evaluation Failure Logs',
    metric1Label: '4K Evaluation Perplexity',
    metric1Val: '23.95 PPL (Stable)',
    metric2Label: 'Passkey Retrieval Accuracy',
    metric2Val: '4.0% Average',
    diagnosticFinding: 'Evaluation perplexity remained deceptively healthy (23.95 PPL), but empirical needle-in-a-haystack retrieval collapsed to 4.0% across 4,096 tokens, proving that low statistical loss does not guarantee long-range attention routing.',
    sampleOutput: 'Prompt context (3,950 tokens of text): "... The secret passkey is 82941 ..."\nPrompt query: "What is the secret passkey?"\nModel generation: "The passkey is 12345." (Hallucination / inability to route needle)',
  },
  {
    name: 'Multi-Turn Alignment & Sycophancy Probe',
    category: 'Conversational Stability',
    tier: 'Tier 1 (Loss) vs. Tier 5 (Behavioral Integrity)',
    modelTested: 'QORIN-100M SFT v3',
    checkpointPath: 'archive/100m_baseline/checkpoints/step_08d_sft_v3_best_val.pt',
    split: 'Adversarial Behavioral Probe Suite (Greeting, Identity, Sycophancy, Dialogue Looping)',
    sampleCount: '100 conversational probes',
    sampleNature: 'Verbatim Diagnostic Failure Output',
    metric1Label: 'SFT Cross-Entropy Loss',
    metric1Val: '0.765 nats (Record Low)',
    metric2Label: 'Greeting Handling Pass Rate',
    metric2Val: '0.0% (Template Collapse)',
    diagnosticFinding: 'An all-time low validation loss (0.765 nats) disguised catastrophic conversational collapse: because unmasked assistant turns in UltraChat dominated the loss, the model experienced template lock-in, unable to output simple greetings.',
    sampleOutput: 'User: "Hello!"\nModel: "<|im_start|>user\\nHow can I help you today?<|im_end|>\\n<|im_start|>assistant" (Self-referential prompt loop verbatim from failure log)',
  },
];

export const Module09ModelEvaluation: React.FC = () => {
  const [selectedTaskIdx, setSelectedTaskIdx] = useState<number>(0);
  const task = EVAL_TASKS[selectedTaskIdx];

  return (
    <section id="module-eval" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 09 &bull; Empirical Rigor</div>
        <h2 className={styles.moduleTitle}>Evaluating Language Models: Beyond Cross-Entropy</h2>
        <div className={styles.moduleLead}>
          Why validation loss alone is profoundly deceptive, the 5-tier evaluation taxonomy, and diagnosing the syntax-execution capability bottleneck.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; The Five Evaluation Tiers
        </div>
        <h3 className={styles.partHeading}>The Illusion of Low Validation Loss</h3>
        <div className={styles.prose}>
          <p>
            In standard machine learning practice, practitioners optimize cross-entropy loss (&minus;&sum; log <em>P</em>) and assume
            that lower loss directly equals a smarter model. In natural language, however, cross-entropy measures superficial
            token likelihood, <em>not</em> factual correctness, deductive validity, or functional execution.
          </p>
          <p>
            To prevent optimization blind spots, QORIN established a <strong>Five-Tier Evaluation Taxonomy</strong>:
          </p>
          <ol>
            <li><strong>Tier 1: Cross-Entropy Loss &amp; Perplexity:</strong> Core mathematical optimization target. Measures statistical distribution matching on held-out text.</li>
            <li><strong>Tier 2: Syntactic &amp; Structural Validity:</strong> Does the generated output conform to formal grammar rules? (Python AST parsing, JSON schema validation).</li>
            <li><strong>Tier 3: Functional &amp; Algorithmic Execution:</strong> Does the generated code execute correctly against deterministic unit test assertions?</li>
            <li><strong>Tier 4: Long-Range Attention Routing:</strong> Can attention heads retrieve arbitrary needles hidden across extended contexts? (Passkey test).</li>
            <li><strong>Tier 5: Conversational Robustness &amp; Identity:</strong> Does the model maintain identity, resist sycophancy, and engage in multi-turn dialogue without repeating templates?</li>
          </ol>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; The Disparity Funnel: Syntax vs. Execution
        </div>
        <h3 className={styles.partHeading}>The Capability Cliff</h3>
        <p className={styles.prose}>
          Across all small-model checkpoints, we observed a dramatic steep drop-off between syntax compliance and functional execution:
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: 'var(--space-4) 0', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
          <div style={{ background: 'var(--color-accent-rust-subtle)', border: '1px solid var(--color-accent-rust)', padding: '8px', color: 'var(--color-accent-rust)', fontWeight: 600 }}>
            Tier 5: Multi-Turn Conversational Integrity &bull; 0.0% Passing under SFT v3 (Template Lock-in)
          </div>
          <div style={{ background: 'var(--color-accent-rust-subtle)', border: '1px solid var(--color-accent-rust)', padding: '8px', color: 'var(--color-accent-rust)', fontWeight: 600 }}>
            Tier 4: Passkey 4K Needle Retrieval &bull; 4.0% Accuracy (Step 2000 checkpoint)
          </div>
          <div style={{ background: 'var(--color-accent-rust-subtle)', border: '1px solid var(--color-accent-rust)', padding: '8px', color: 'var(--color-accent-rust)', fontWeight: 600 }}>
            Tier 3: Python Unit Test Execution &bull; 0.0% Passed on MBPP
          </div>
          <div style={{ background: 'var(--color-accent-sage-subtle)', border: '1px solid var(--color-accent-sage)', padding: '8px', color: 'var(--color-accent-sage)', fontWeight: 600 }}>
            Tier 2: Python AST Syntax Validity &bull; 90.0% Valid
          </div>
          <div style={{ background: 'var(--color-accent-indigo-subtle)', border: '1px solid var(--color-accent-indigo)', padding: '8px', color: 'var(--color-accent-indigo)', fontWeight: 600 }}>
            Tier 1: Validation Cross-Entropy Loss &bull; 0.765 nats (Deceptive Record Low)
          </div>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Metric Contrast Diagnostic
        </div>
        <h3 className={styles.partHeading}>Domain Diagnostic Case Studies</h3>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Select Empirical Investigation Domain:</span>
              <div className={styles.btnGroup}>
                {EVAL_TASKS.map((t, idx) => (
                  <button
                    key={t.name}
                    type="button"
                    className={`${styles.btnToggle} ${selectedTaskIdx === idx ? styles.btnToggleActive : ''}`}
                    onClick={() => setSelectedTaskIdx(idx)}
                  >
                    Case {idx + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-primary)', marginBottom: 'var(--space-2)' }}>
            {task.name} &bull; <span style={{ color: 'var(--color-accent-indigo)', fontFamily: 'var(--font-mono)' }}>{task.tier}</span>
          </div>

          {/* Explicit Model & Provenance Attribution Banner */}
          <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: '8px 12px', marginBottom: 'var(--space-4)', fontSize: '11px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>Tested Model:</span>
              <div style={{ fontWeight: 600, color: 'var(--color-ink-primary)' }}>{task.modelTested}</div>
            </div>
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>Checkpoint:</span>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-ink-primary)', fontSize: '10px' }}><code>{task.checkpointPath}</code></div>
            </div>
            <div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>Partition &amp; Sample Size:</span>
              <div style={{ color: 'var(--color-ink-primary)' }}>{task.sampleCount}</div>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                  {task.metric1Label}
                </span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--color-accent-sage)', marginTop: '2px' }}>
                  {task.metric1Val}
                </div>
              </div>

              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                  {task.metric2Label}
                </span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--color-accent-rust)', marginTop: '2px' }}>
                  {task.metric2Val}
                </div>
              </div>
            </div>

            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-secondary)', marginBottom: 'var(--space-3)', lineHeight: 1.5 }}>
              <strong>Empirical Diagnosis:</strong> {task.diagnosticFinding}
            </div>

            <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-xs)', padding: 'var(--space-3)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '4px' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>
                  Generation Diagnostic Record:
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', color: 'var(--color-accent-indigo)' }}>
                  {task.sampleNature}
                </span>
              </div>
              <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ink-primary)', marginTop: '4px', whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                {task.sampleOutput}
              </pre>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> This multi-tier framework is the core methodology of the entire QORIN project.
            By continuously measuring AST parse rates, functional unit tests, passkey retrieval, and sycophancy probes,
            we ensure that model improvements reflect authentic reasoning and operational capabilities rather than superficial loss descent.
          </div>
        </div>
      </div>
    </section>
  );
};
