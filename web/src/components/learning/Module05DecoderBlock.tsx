import React, { useState } from 'react';
import styles from './LearningModule.module.css';

type ComponentKey = 'emb' | 'rmsnorm1' | 'rope_gqa' | 'residual1' | 'rmsnorm2' | 'swiglu' | 'residual2' | 'head';

interface BlockLayer {
  key: ComponentKey;
  name: string;
  stage: string;
  formula: string;
  shape: string;
  explanation: string;
}

const LAYERS: BlockLayer[] = [
  {
    key: 'emb',
    name: 'Token Embedding Matrix (W_emb)',
    stage: 'Input Vectorization',
    formula: 'X_0 = W_emb[t] \u2208 \u211D^(B \u00D7 S \u00D7 d)',
    shape: 'Input: [B, S] \u2192 Output: [B, S, 1536]',
    explanation: 'Converts integer token IDs into dense vector representations. Vocabulary size is 32,768, hidden dimension is 1,536.',
  },
  {
    key: 'rmsnorm1',
    name: 'Pre-Attention RMSNorm',
    stage: 'Activation Scaling',
    formula: 'x\u0302 = (x / RMS(x)) \u2299 \u03B3,  RMS(x) = \u221A((1/d)\u2211x_i\u00B2 + \u03B5)',
    shape: 'Input: [B, S, 1536] \u2192 Output: [B, S, 1536]',
    explanation: 'Normalizes input activations by root-mean-square without mean-centering. Eliminates activation drift and accelerates training.',
  },
  {
    key: 'rope_gqa',
    name: 'RoPE + Causal Grouped-Query Attention',
    stage: 'Contextual Routing',
    formula: 'Attn(X) = softmax( (R_m Q)(R_n K)^T / \u221A64 + M ) V',
    shape: 'Q: [B, S, 24, 64], K,V: [B, S, 4, 64] \u2192 Out: [B, S, 1536]',
    explanation: 'Computes contextualized token interactions across 24 query heads and 4 key-value heads with relative rotary position encoding.',
  },
  {
    key: 'residual1',
    name: 'Residual Highway 1',
    stage: 'Gradient Preservation',
    formula: 'X_attn = X_in + Dropout(Attn_out)',
    shape: 'Shape: [B, S, 1536]',
    explanation: 'Adds the original input back to the attention output, allowing gradients to flow unattenuated across all 20 layers during backpropagation.',
  },
  {
    key: 'rmsnorm2',
    name: 'Pre-FFN RMSNorm',
    stage: 'Activation Scaling',
    formula: 'x\u0302_attn = (X_attn / RMS(X_attn)) \u2299 \u03B3',
    shape: 'Shape: [B, S, 1536]',
    explanation: 'Normalizes the post-attention residual representation before passing it into non-linear expansion.',
  },
  {
    key: 'swiglu',
    name: 'SwiGLU Gated Feed-Forward Network',
    stage: 'Non-Linear Representation',
    formula: 'FFN(x) = ( Swish(x W_gate) \u2299 (x W_up) ) W_down',
    shape: 'Inner Dim: 3,840 \u2192 Output: [B, S, 1536]',
    explanation: 'Bilinear gating mechanism: computes an element-wise product of Swish-activated gate projection and linear up-projection before down-projecting.',
  },
  {
    key: 'residual2',
    name: 'Residual Highway 2',
    stage: 'Gradient Preservation',
    formula: 'X_block = X_attn + Dropout(FFN_out)',
    shape: 'Shape: [B, S, 1536]',
    explanation: 'Second residual summation, delivering the finalized block output ready for the next layer or terminal output projection.',
  },
  {
    key: 'head',
    name: 'Final RMSNorm & Output Logits Head',
    stage: 'Prediction Softmax',
    formula: 'P(w_{t+1}) = softmax( (RMSNorm(X_L)) W_emb^T )',
    shape: 'Output: [B, S, 32768] vocabulary logits',
    explanation: 'Transposes the tied input embedding matrix to project 1,536-dimensional hidden representations into 32,768 vocabulary logits.',
  },
];

export const Module05DecoderBlock: React.FC = () => {
  const [selectedKey, setSelectedKey] = useState<ComponentKey>('swiglu');
  const selectedLayer = LAYERS.find((l) => l.key === selectedKey)!;

  return (
    <section id="module-decoder" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 05 &bull; Architectural Blueprint</div>
        <h2 className={styles.moduleTitle}>Anatomy of a Transformer Decoder Block</h2>
        <div className={styles.moduleLead}>
          The complete end-to-end computational graph of modern autoregressive Transformers: Pre-LN RMSNorm, RoPE GQA, residual highways, and SwiGLU gating.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>The Modern Pre-LN Transformer Invariants</h3>
        <div className={styles.prose}>
          <p>
            Early Transformers (Vaswani et al., 2017) utilized <em>Post-LayerNorm</em>, placing normalization after the residual
            addition (<em>x</em> + Sublayer(<em>x</em>)). Post-LN suffered from severe gradient instability near the input layers,
            requiring meticulous learning rate warmups. Modern models—including LLaMA, Mistral, and QORIN—adopt four foundational invariants:
          </p>
          <ol>
            <li><strong>Pre-Layer Normalization (Pre-LN):</strong> Normalization is applied strictly <em>before</em> the sublayers, preserving an unimpeded identity mapping (<em>x</em> + <em>f</em>(Norm(<em>x</em>))) across the residual highway.</li>
            <li><strong>RMSNorm over LayerNorm:</strong> Root Mean Square Normalization replaces standard LayerNorm, discarding the mean-centering step (&mu; = 0) to save 7% computational overhead without degrading accuracy.</li>
            <li><strong>SwiGLU Feed-Forward Networks:</strong> Replacing ReLU or GELU with SwiGLU (Shazeer, 2020) provides multiplicative gating, substantially increasing model parameter efficiency.</li>
            <li><strong>Bias-Free Linear Layers:</strong> All linear projections set additive bias to zero (<em>b</em> = 0), reducing memory footprint and preventing numerical drift on consumer GPUs.</li>
          </ol>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Data Flow &amp; Tensor Projections
        </div>
        <h3 className={styles.partHeading}>The Decoder Signal Pipeline</h3>
        <div className={styles.prose}>
          <p>
            Notice the two parallel residual bypasses. Information flows along the straight residual backbone, while the attention
            and SwiGLU blocks compute incremental additive deltas (&Delta;<em>x</em>):
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', maxWidth: '480px', margin: 'var(--space-4) auto' }}>
          {LAYERS.map((l) => {
            const isSelected = selectedKey === l.key;
            return (
              <div
                key={l.key}
                onClick={() => setSelectedKey(l.key)}
                style={{
                  background: isSelected ? 'var(--color-surface)' : 'var(--color-paper-subtle)',
                  border: isSelected ? '2px solid var(--color-accent-indigo)' : '1px solid var(--color-rule)',
                  borderRadius: 'var(--radius-xs)',
                  padding: '8px 12px',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  transition: 'all 0.15s ease'
                }}
              >
                <div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>
                    {l.stage}
                  </span>
                  <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-primary)' }}>
                    {l.name}
                  </div>
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: isSelected ? 'var(--color-accent-indigo)' : 'var(--color-ink-muted)' }}>
                  {isSelected ? '\u25C0 Active' : 'Select'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Component Inspector
        </div>
        <h3 className={styles.partHeading}>Inspecting: {selectedLayer.name}</h3>

        <div className={styles.labCard}>
          <div className={styles.formulaBox}>
            {selectedLayer.formula}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-4)', margin: 'var(--space-4) 0' }}>
            <div style={{ background: 'var(--color-paper-subtle)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)' }}>
              <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                Tensor Dimension
              </span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-primary)', marginTop: '2px' }}>
                {selectedLayer.shape}
              </div>
            </div>

            <div style={{ background: 'var(--color-paper-subtle)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)' }}>
              <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                Architectural Function
              </span>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-secondary)', marginTop: '2px', lineHeight: 1.5 }}>
                {selectedLayer.explanation}
              </div>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> In QORIN-500M, this entire block is repeated across <strong>20 stacked layers</strong>.
            Each layer contains 5,505,024 attention parameters and 17,694,720 SwiGLU parameters, totaling 23,202,816 parameters per block
            and 464,057,856 non-embedding parameters across the full network.
          </div>
        </div>
      </div>
    </section>
  );
};

