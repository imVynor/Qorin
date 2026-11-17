import React, { useState } from 'react';
import styles from './LearningModule.module.css';

export const Module06HowModelsLearn: React.FC = () => {
  const [probTarget, setProbTarget] = useState<number>(0.35); // 0.01 to 0.99
  const [learningStep, setLearningStep] = useState<number>(2000);

  // Target loss and perplexity formulas
  const lossNats = -Math.log(Math.max(probTarget, 0.001));
  const perplexity = Math.exp(lossNats);

  // Simulated learning curve data point
  const trainLossSim = (2.2 + 4.5 * Math.exp(-learningStep / 1500)).toFixed(2);
  const valLossSim = (2.4 + 4.6 * Math.exp(-learningStep / 1500) + (learningStep > 3500 ? (learningStep - 3500) * 0.0003 : 0)).toFixed(2);

  return (
    <section id="module-learning" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 06 &bull; Optimization &amp; Convergence</div>
        <h2 className={styles.moduleTitle}>How Language Models Learn</h2>
        <div className={styles.moduleLead}>
          Next-token prediction, cross-entropy loss, perplexity, AdamW gradient descent, and identifying the boundary between healthy generalization and overfitting.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>The Next-Token Prediction Objective</h3>
        <div className={styles.prose}>
          <p>
            Language models are trained on a deceptively simple objective: <strong>given all preceding tokens, predict the probability distribution over the next token</strong>.
            For a vocabulary of size <em>V</em>, the final Transformer layer outputs unnormalized scores (logits <em>z</em><sub><em>i</em></sub>),
            which are turned into probabilities via softmax:
          </p>
          <div className={styles.formulaBox}>
            P(w<sub>t</sub> = v | w<sub>&lt;t</sub>) = exp(z<sub>v</sub>) / &sum;<sub>j=1..V</sub> exp(z<sub>j</sub>)
          </div>
          <p>
            The training objective minimizes <strong>Cross-Entropy Loss (<em>L</em><sub>CE</sub>)</strong> on the ground-truth target token <em>y</em>:
          </p>
          <div className={styles.formulaBox}>
            L<sub>CE</sub> = &minus; log P(y) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Perplexity (PPL) = exp( L<sub>CE</sub> ) = 1 / P(y)
          </div>
          <p>
            <strong>What is Perplexity?</strong> Intuitively, perplexity represents the effective number of tokens the model is equally uncertain between.
            A random model on a 32,768-token vocabulary has an initial perplexity of 32,768 (loss = 10.39). As training progresses and the model
            learns syntax and facts, perplexity drops toward 20&ndash;30.
          </p>
          <p>
            <strong>The AdamW Optimizer:</strong> Parameters are updated via gradient descent with decoupled weight decay:
            &theta;<sub><em>t</em>+1</sub> = &theta;<sub><em>t</em></sub> &minus; &eta; (<em>m</em><sub><em>t</em></sub> / (&radic;<em>v</em><sub><em>t</em></sub> + &epsilon;)) &minus; &eta;&lambda;&theta;<sub><em>t</em></sub>.
            Learning rate &eta; follows a cosine decay schedule, warming up from 0 to 6 &times; 10<sup>&minus;4</sup> and annealing to 10% of maximum.
          </p>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Predicted Next-Token Probability Distribution
        </div>
        <h3 className={styles.partHeading}>Predicting &ldquo;The capital of France is [ ? ]&rdquo;</h3>
        <div className={styles.prose}>
          <p>
            Observe how probability is divided among candidates. If the model assigns 35% probability to the correct target (&ldquo;Paris&rdquo;),
            it suffers a cross-entropy loss of 1.05 nats:
          </p>
        </div>

        <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: 'var(--space-4)', margin: 'var(--space-4) 0' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', marginBottom: '2px' }}>
                <span style={{ fontWeight: 600, color: 'var(--color-accent-sage)' }}>Paris (Ground Truth Target)</span>
                <span>{(probTarget * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: '8px', background: 'var(--color-rule-subtle)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${probTarget * 100}%`, background: 'var(--color-accent-sage)' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', marginBottom: '2px' }}>
                <span style={{ color: 'var(--color-ink-muted)' }}>London (Distractor 1)</span>
                <span>{(((1 - probTarget) * 0.45) * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: '8px', background: 'var(--color-rule-subtle)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${(1 - probTarget) * 45}%`, background: 'var(--color-rule-strong)' }} />
              </div>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', marginBottom: '2px' }}>
                <span style={{ color: 'var(--color-ink-muted)' }}>located (Distractor 2)</span>
                <span>{(((1 - probTarget) * 0.35) * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: '8px', background: 'var(--color-rule-subtle)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${(1 - probTarget) * 35}%`, background: 'var(--color-rule-strong)' }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Loss &amp; Perplexity Playground
        </div>
        <h3 className={styles.partHeading}>Probability &amp; Gradient Penalty Explorer</h3>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem} style={{ flex: 1 }}>
              <label className={styles.controlLabel}>
                Predicted Probability for Ground Truth Target: {(probTarget * 100).toFixed(1)}%
              </label>
              <input
                type="range"
                min={0.01}
                max={0.99}
                step={0.01}
                value={probTarget}
                onChange={(e) => setProbTarget(Number(e.target.value))}
                className={styles.sliderInput}
                style={{ width: '100%' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                <span>1% (Severe Penalty)</span>
                <span>50% (Moderate)</span>
                <span>99% (Near Zero Loss)</span>
              </div>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-4)', textAlign: 'center' }}>
              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                  Cross-Entropy Loss (-log P)
                </span>
                <div style={{ fontSize: 'var(--text-xl)', fontWeight: 700, color: lossNats > 2.5 ? 'var(--color-accent-rust)' : 'var(--color-ink-primary)', marginTop: '4px' }}>
                  {lossNats.toFixed(4)} nats
                </div>
              </div>

              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                  Perplexity (exp(Loss))
                </span>
                <div style={{ fontSize: 'var(--text-xl)', fontWeight: 700, color: 'var(--color-accent-indigo)', marginTop: '4px' }}>
                  {perplexity.toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          {/* Training vs Validation divergence simulator */}
          <div style={{ marginTop: 'var(--space-5)' }}>
            <div className={styles.outputTitle}>Learning Curve &bull; Training vs. Validation Loss (Step {learningStep})</div>
            <div className={styles.controlsRow}>
              <div className={styles.controlItem} style={{ width: '100%' }}>
                <input
                  type="range"
                  min={100}
                  max={5000}
                  step={100}
                  value={learningStep}
                  onChange={(e) => setLearningStep(Number(e.target.value))}
                  className={styles.sliderInput}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: '11px', padding: '6px 0', borderTop: '1px solid var(--color-rule-subtle)' }}>
              <span>Train Loss: <strong style={{ color: 'var(--color-accent-indigo)' }}>{trainLossSim} nats</strong></span>
              <span>Validation Loss: <strong style={{ color: learningStep > 3500 ? 'var(--color-accent-rust)' : 'var(--color-accent-sage)' }}>{valLossSim} nats</strong></span>
              <span>Status: {learningStep > 3500 ? <span style={{ color: 'var(--color-accent-rust)' }}>Overfitting Divergence</span> : <span style={{ color: 'var(--color-accent-sage)' }}>Healthy Generalization</span>}</span>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> QORIN-100M pretraining scaled from an initial loss of 6.03 down to
            2.9152 (perplexity 18.45) across 1.5 billion tokens. Our experiments demonstrated that while training loss descends smoothly,
            validation loss begins diverging if dataset diversity is exhausted.
          </div>
        </div>
      </div>
    </section>
  );
};

