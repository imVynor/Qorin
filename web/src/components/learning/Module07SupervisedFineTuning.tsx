import React, { useState } from 'react';
import styles from './LearningModule.module.css';

export const Module07SupervisedFineTuning: React.FC = () => {
  const [targetMaskingActive, setTargetMaskingActive] = useState<boolean>(true);

  const tokens = [
    { text: '<|im_start|>system', role: 'system', isTarget: false },
    { text: 'You are an AI assistant.<|im_end|>', role: 'system', isTarget: false },
    { text: '<|im_start|>user', role: 'user', isTarget: false },
    { text: 'Explain gravity in one sentence.<|im_end|>', role: 'user', isTarget: false },
    { text: '<|im_start|>assistant', role: 'assistant_prompt', isTarget: false },
    { text: 'Gravity is the mutual curvature', role: 'assistant', isTarget: true },
    { text: 'of spacetime caused by mass.', role: 'assistant', isTarget: true },
    { text: '<|im_end|>', role: 'assistant', isTarget: true },
  ];

  const totalTokens = tokens.length;
  const targetTokens = tokens.filter((t) => t.isTarget).length;

  return (
    <section id="module-sft" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 07 &bull; Instruction Alignment</div>
        <h2 className={styles.moduleTitle}>Supervised Fine-Tuning (SFT) &amp; Target Masking</h2>
        <div className={styles.moduleLead}>
          Transforming a raw text predictor into an interactive assistant: instruction formatting, supervised target masking, and avoiding the trap of conversational collapse.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>From Document Completer to Instruction Follower</h3>
        <div className={styles.prose}>
          <p>
            A pretrained base model simply continues whatever text it receives. If you prompt it with &ldquo;Write a function to sort a list&rdquo;,
            it might complete it with &ldquo;in C++ written by Alice on StackOverflow&rdquo;. It does not understand that it is expected to answer as an assistant.
          </p>
          <p>
            <strong>Supervised Fine-Tuning (SFT)</strong> trains the model on curated (User Prompt &rarr; Assistant Response) pairs using
            structured markup tokens such as <strong>ChatML</strong>:
          </p>
          <div className={styles.formulaBox}>
            &lt;|im_start|&gt;user\nHello!&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\nHello! How can I help you today?&lt;|im_end|&gt;
          </div>
          <p>
            <strong>The Target Masking Rule:</strong> If the model calculates loss on the user prompt tokens, it wastes parameter capacity
            memorizing the user&rsquo;s questions instead of learning how to formulate responses. We apply a target mask <em>m</em><sub><em>t</em></sub> &isin; &#123;0, 1&#125;:
          </p>
          <div className={styles.formulaBox}>
            L<sub>SFT</sub> = &minus; (1 / N<sub>supervised</sub>) &times; &sum;<sub>t=1..T</sub> [ m<sub>t</sub> &times; log P(y<sub>t</sub> | x<sub>&lt;t</sub>) ]<br />
            where m<sub>t</sub> = 0 for System and User tokens (Target ID = &minus;100), and m<sub>t</sub> = 1 strictly for Assistant tokens.
          </div>
          <p>
            <strong>The SFT Deception Trap:</strong> Lower SFT training loss does <em>not</em> guarantee better conversational capability!
            In QORIN&rsquo;s SFT v3 experiment, validation loss dropped to an all-time low of <strong>0.765 nats</strong>, yet the model
            suffered catastrophic conversational collapse: it became completely sycophantic, hallucinated identities, and failed trivial greetings.
            Forensic analysis revealed this was caused by an 80.3% dialogue imbalance and unweighted padding loss.
          </p>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Token Sequence Gradient Masking
        </div>
        <h3 className={styles.partHeading}>Active Gradient Flow vs. Masked Prompts</h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: 'var(--space-4) 0' }}>
          {tokens.map((tok, idx) => {
            const isActive = targetMaskingActive ? tok.isTarget : true;
            return (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '6px 12px',
                  background: isActive ? 'var(--color-accent-sage-subtle)' : 'var(--color-paper-subtle)',
                  border: `1px solid ${isActive ? 'var(--color-accent-sage)' : 'var(--color-rule-subtle)'}`,
                  borderRadius: 'var(--radius-xs)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  transition: 'all 0.15s ease'
                }}
              >
                <div>
                  <span style={{ color: 'var(--color-ink-muted)', marginRight: '8px' }}>[{tok.role}]</span>
                  <span style={{ color: isActive ? 'var(--color-ink-primary)' : 'var(--color-ink-muted)', fontWeight: isActive ? 600 : 400 }}>
                    {tok.text}
                  </span>
                </div>
                <span style={{ fontSize: '10px', color: isActive ? 'var(--color-accent-sage)' : 'var(--color-ink-muted)', fontWeight: 600 }}>
                  {isActive ? 'Target (m_t = 1) \u2022 Active Gradient' : 'Masked (m_t = 0) \u2022 Loss Ignored'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Target Masking Sandbox
        </div>
        <h3 className={styles.partHeading}>Masking Policy Comparison</h3>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Target Masking Mode:</span>
              <button
                type="button"
                className={`${styles.btnToggle} ${targetMaskingActive ? styles.btnToggleActive : ''}`}
                style={{ border: '1px solid var(--color-rule)', padding: '5px 12px' }}
                onClick={() => setTargetMaskingActive(!targetMaskingActive)}
              >
                {targetMaskingActive ? 'Target Masking ENABLED (Standard SFT)' : 'Target Masking DISABLED (Flawed SFT)'}
              </button>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div className={styles.outputTitle}>Gradient Attribution Analysis</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-4)' }}>
              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                  Active Tokens in Gradient Update:
                </span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--color-ink-primary)', marginTop: '2px' }}>
                  {targetMaskingActive ? `${targetTokens} Tokens (Assistant Only)` : `${totalTokens} Tokens (Prompt + Assistant)`}
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                  Training Consequence:
                </span>
                <div style={{ fontSize: 'var(--text-xs)', color: targetMaskingActive ? 'var(--color-accent-sage)' : 'var(--color-accent-rust)', fontWeight: 600, marginTop: '2px', lineHeight: 1.4 }}>
                  {targetMaskingActive
                    ? 'Model learns conditional response generation; prompt representations serve purely as attention context.'
                    : 'Model wastes gradients memorizing user prompts; triggers repetition loops and template lock-in!'}
                </div>
              </div>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> In our Forensic Audit, we identified that early fine-tuning scripts
            normalized gradients per-sequence rather than per-active-token, creating an 80:1 gradient distortion favoring short sequences.
            Implementing <strong>Token-Weighted Cross-Entropy</strong> eliminated this gradient bias and stabilized multi-turn reasoning.
          </div>
        </div>
      </div>
    </section>
  );
};
