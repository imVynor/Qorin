import React from 'react';
import { modelsData } from '../data/modelsData';
import { TransformerDataFlow } from '../components/visualizations/TransformerDataFlow';
import { MathView } from '../components/common/MathView';
import styles from './Model.module.css';

export const Model: React.FC = () => {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Architectural Blueprint &bull; Parameter Accounting</div>
          <h1 className={styles.title}>Model Architecture &amp; Specifications</h1>
          <div className={styles.lead}>
            A rigorous engineering ledger documenting parameter allocations, attention mechanics,
            non-linear activations, and modern architectural invariants across all QORIN variants.
          </div>
        </div>
      </header>

      {/* 1. Comparative Ledger */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>Comparative Architectural Ledger</h2>
          <div className={styles.proseMeasure}>
            <p>
              Every parameter count below has been independently calculated from layer dimensions and verified
              against physical PyTorch checkpoint tensors. In accordance with open research principles, we explicitly
              separate total parameters from non-embedding parameters:
            </p>
            <div style={{ margin: 'var(--space-3) 0' }}>
              <MathView math="N_{\text{non-emb}} = N_{\text{total}} - (V \times d)" block />
            </div>
          </div>

          <div className={styles.matrixTableWrap}>
            <table className={styles.matrixTable}>
              <thead>
                <tr>
                  <th>Model Identifier</th>
                  <th>Total Params</th>
                  <th>Non-Embedding</th>
                  <th>Layers</th>
                  <th>Dim (d)</th>
                  <th>Heads (Q/KV)</th>
                  <th>FFN Type</th>
                  <th>FFN Dim</th>
                  <th>Context</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.map((m) => (
                  <tr key={m.model_id}>
                    <td>
                      <strong style={{ color: 'var(--color-ink-primary)' }}>{m.name}</strong>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-ink-muted)' }}>
                        {m.model_id}
                      </div>
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{m.total_parameters.toLocaleString()}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.non_embedding_parameters.toLocaleString()}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.layers}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.hidden_dim}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.query_heads} / {m.kv_heads} ({m.gqa_ratio}:1)</td>
                    <td>{m.ffn_type.toUpperCase()}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.ffn_hidden_dim}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{m.context_length}</td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: m.status === 'production_sft' ? 'var(--color-accent-sage)' : m.status === 'active_pretraining' ? 'var(--color-accent-indigo)' : 'inherit', fontWeight: m.status === 'production_sft' ? 600 : 'normal' }}>
                        {m.status.replace(/_/g, ' ')}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className={styles.proseMeasure}>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-muted)' }}>
              <em>Parameter Accounting Note:</em> In early documentation, QORIN-16M was estimated at 16,252,800 parameters.
              Independent physical tensor inspection confirms exactly <strong>16,768,128 trainable parameters</strong>{' '}
              (<MathView math="L=6, d=384, H=6, d_{\text{ffn}}=1024, V=16000" />).
              Both figures are recorded for historical provenance.
            </p>
          </div>
        </div>
      </section>

      {/* 2. Interactive Data Flow */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>Computational Data Flow</h2>
          <div className={styles.proseMeasure}>
            <p>
              Select any model scale below to inspect tensor projections, memory consumption, activation functions,
              and mathematical operations inside each layer.
            </p>
          </div>
          <TransformerDataFlow />
        </div>
      </section>

      {/* 3. Flagship Blueprint: QORIN-500M */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>The Anatomy of QORIN-500M</h2>
          <div className={styles.proseMeasure}>
            <p>
              QORIN-500M represents the scaled synthesis of all experimental findings from our 16M, 50M, and 100M
              investigations. It contains exactly <strong>514,389,504 total parameters</strong>, with{' '}
              <strong>464,057,856 non-embedding parameters</strong> distributed across 20 transformer blocks.
            </p>
            <p>
              The exact parameter allocation per block is mathematically structured as follows:
            </p>
            <ul style={{ paddingLeft: '1.2rem', marginBottom: 'var(--space-4)', lineHeight: 1.7, fontSize: 'var(--text-sm)', color: 'var(--color-ink-secondary)' }}>
              <li><strong>Embedding Matrix:</strong> 32,768 vocabulary tokens &times; 1,536 hidden dimensions = 50,331,648 parameters (tied).</li>
              <li><strong>Grouped-Query Attention (24Q, 4KV):</strong>
                <br />Query projection: 1,536 &times; 1,536 = 2,359,296 weights.
                <br />Key projection: 1,536 &times; 256 = 393,216 weights.
                <br />Value projection: 1,536 &times; 256 = 393,216 weights.
                <br />Output projection: 1,536 &times; 1,536 = 2,359,296 weights.
                <br />Total attention per layer = 5,505,024 weights.
              </li>
              <li><strong>SwiGLU Feed-Forward Network:</strong> Hidden dimension expanded to 3,840 dimensions.
                <br />Gate projection (<MathView math="W_{\text{gate}}" />): 1,536 &times; 3,840 = 5,898,240 weights.
                <br />Up projection (<MathView math="W_{\text{up}}" />): 1,536 &times; 3,840 = 5,898,240 weights.
                <br />Down projection (<MathView math="W_{\text{down}}" />): 3,840 &times; 1,536 = 5,898,240 weights.
                <br />Total FFN per layer = 17,694,720 weights.
              </li>
              <li><strong>RMSNorm Scale Weights:</strong> Pre-attention norm (1,536) + Pre-FFN norm (1,536) = 3,072 weights per layer.</li>
              <li><strong>Final Output Norm:</strong> 1,536 scale weights. Tied input/output embeddings.</li>
            </ul>
          </div>

          {/* Architectural Invariants */}
          <div className={styles.decisionGrid}>
            <div className={styles.decisionCard}>
              <div className={styles.decisionTag}>Attention Invariant</div>
              <div className={styles.decisionHeading}>Grouped-Query Attention (6:1 GQA)</div>
              <div style={{ margin: 'var(--space-2) 0' }}>
                <MathView
                  math="\text{Attention}(Q_i, K_{\lfloor i/g \rfloor}, V_{\lfloor i/g \rfloor}) = \text{softmax}\left( \frac{Q_i K_{\lfloor i/g \rfloor}^T}{\sqrt{d_k}} \right) V_{\lfloor i/g \rfloor}"
                  block
                />
              </div>
              <p className={styles.decisionBody}>
                Standard multi-head attention (MHA) creates an unsustainable KV-cache memory footprint as sequence lengths
                scale to 2048 and 4096 tokens. By grouping 24 query heads into 4 key-value heads (<MathView math="g=6" />), GQA achieves an 83.3% reduction in
                KV-cache memory bandwidth while preserving full multi-head query expressivity.
              </p>
            </div>

            <div className={styles.decisionCard}>
              <div className={styles.decisionTag}>Activation Invariant</div>
              <div className={styles.decisionHeading}>SwiGLU Gated Activation</div>
              <div style={{ margin: 'var(--space-2) 0' }}>
                <MathView
                  math="\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}} \right) W_{\text{down}}"
                  block
                />
              </div>
              <p className={styles.decisionBody}>
                Replacing standard GELU with SwiGLU introduces multiplicative gating. Following Shazeer (2020),
                this significantly enhances representation capacity per parameter without increasing layer depth.
              </p>
            </div>

            <div className={styles.decisionCard}>
              <div className={styles.decisionTag}>Positional Invariant</div>
              <div className={styles.decisionHeading}>Rotary Position Embeddings (RoPE)</div>
              <div style={{ margin: 'var(--space-2) 0' }}>
                <MathView
                  math="R_{\theta_i, m} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2(i-1)/d}"
                  block
                />
              </div>
              <p className={styles.decisionBody}>
                Absolute position embeddings fail to generalize beyond their training length. RoPE encodes relative token
                distances directly into the query-key dot product via 2D rotation matrices (<MathView math="\theta = 10,000" />), enabling
                linear frequency interpolation (<MathView math="s = 8.0" />) for context window scaling.
              </p>
            </div>

            <div className={styles.decisionCard}>
              <div className={styles.decisionTag}>Optimization Invariant</div>
              <div className={styles.decisionHeading}>Bias-Free Linear Layers &amp; RMSNorm</div>
              <div style={{ margin: 'var(--space-2) 0' }}>
                <MathView
                  math="\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma"
                  block
                />
              </div>
              <p className={styles.decisionBody}>
                All linear layers set bias to zero (<MathView math="b = 0" />), eliminating redundant gradient buffers and improving numerical
                stability on consumer GPUs. Pre-layer RMSNorm (<MathView math="\epsilon = 10^{-6}" />) enforces scale invariance without
                mean-centering computational overhead.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Architectural Evolution Narrative */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>Four Generations of Architectural Evolution</h2>
          <div className={styles.proseMeasure}>
            <p>
              Each model scale was not arbitrarily selected; each served as an empirical stepping stone to diagnose
              a specific mathematical bottleneck in transformer training under consumer hardware constraints.
            </p>

            <h3 style={{ fontSize: 'var(--text-lg)', fontWeight: 600, marginTop: 'var(--space-6)', marginBottom: 'var(--space-2)', color: 'var(--color-ink-primary)' }}>
              Generation I: QORIN-16M (The Calibration Baseline)
            </h3>
            <p>
              <strong>Architecture:</strong> 6 layers, 384 hidden dimensions, 6 standard attention heads, <MathView math="d_{\text{ffn}} = 1024" />, 16,000 vocabulary.
              Total verified parameters: <strong>16,768,128</strong>.
            </p>
            <p>
              <strong>Training Recipe:</strong> Pretrained from scratch on 102M tokens using AdamW (<MathView math="\beta_1=0.9, \beta_2=0.95" />, weight decay 0.1),
              peak learning rate <MathView math="\eta_{\max} = 6.0 \times 10^{-4}" />, cosine decay to <MathView math="6.0 \times 10^{-5}" /> with 500 warmup updates.
              Achieved validation loss of <strong>3.4862</strong> (perplexity 32.66).
            </p>
            <p>
              <strong>Key Empirical Lesson:</strong> The model demonstrated that modern PyTorch implementations could train stably on a consumer GPU.
              However, capability testing exposed the <em>syntax-execution divergence</em>: while achieving 90.0% formatting compliance, it scored <strong>0.0% on functional code execution</strong>,
              revealing that minimal parameter scale cannot sustain algorithmic state tracking.
            </p>

            <h3 style={{ fontSize: 'var(--text-lg)', fontWeight: 600, marginTop: 'var(--space-6)', marginBottom: 'var(--space-2)', color: 'var(--color-ink-primary)' }}>
              Generation II: QORIN-50M (The Architecture Ablation Testbed)
            </h3>
            <p>
              <strong>Architecture:</strong> 12 layers, 512 hidden dimensions, 8 query heads, 4 KV heads (2:1 GQA), <MathView math="d_{\text{ffn}} = 1365" /> SwiGLU, 16,384 vocabulary.
              Total verified parameters: <strong>49,105,408</strong>.
            </p>
            <p>
              <strong>Training Recipe:</strong> Trained across 250M tokens at <MathView math="\eta_{\max} = 4.0 \times 10^{-4}" />.
              Achieved validation loss of <strong>3.1152</strong> (perplexity 22.54).
            </p>
            <p>
              <strong>Key Empirical Lesson:</strong> Rigorous architectural ablations comparing Multi-Head Attention vs. Grouped-Query Attention vs. Deep-Thin (16 layers, 448 dim)
              proved that deep-thin topologies suffered a <strong>56.6% latency penalty</strong> in time-to-first-token (14.58ms vs 9.82ms) and 28% slower decoding due to sequential GPU kernel launch overhead.
              Meanwhile, GQA-4 matched MHA perplexity exactly while cutting KV-cache memory in half (192MB &rarr; 96MB per batch of 16), confirming GQA as an essential architectural primitive.
            </p>

            <h3 style={{ fontSize: 'var(--text-lg)', fontWeight: 600, marginTop: 'var(--space-6)', marginBottom: 'var(--space-2)', color: 'var(--color-ink-primary)' }}>
              Generation III: QORIN-100M (Context Extension &amp; The Starvation Paradox)
            </h3>
            <p>
              <strong>Architecture:</strong> 14 layers, 768 hidden dimensions, 12 query heads, 4 KV heads (3:1 GQA), <MathView math="d_{\text{ffn}} = 2048" /> SwiGLU, 32,768 vocabulary.
              Total verified parameters: <strong>100,401,408</strong>.
            </p>
            <p>
              <strong>Training Recipe:</strong> Pretrained across 1.5B tokens at <MathView math="\eta_{\max} = 3.5 \times 10^{-4}" />.
              Achieved validation loss of <strong>2.7135</strong> (perplexity 15.08, a 68.3% perplexity reduction relative to 16M).
            </p>
            <p>
              <strong>Key Empirical Lesson:</strong> Context extension from 1K to 4K tokens was validated via RoPE frequency interpolation (<MathView math="s=8.0" />).
              However, during supervised fine-tuning (SFT v3), validation loss reached a record low of <strong>0.765</strong>, yet the model suffered conversational collapse on simple greetings (emitting synthetic Wikipedia headers).
              Forensic diagnosis discovered the <em>Greeting Starvation Paradox</em>: greetings comprised only 0.026% of training records (9 out of 34,700),
              proving that loss metrics can mask severe behavioral blindspots without balanced conversational curriculum design.
            </p>

            <h3 style={{ fontSize: 'var(--text-lg)', fontWeight: 600, marginTop: 'var(--space-6)', marginBottom: 'var(--space-2)', color: 'var(--color-ink-primary)' }}>
              Generation IV: QORIN-514M (Autonomous Tool Chaining &amp; Pretraining)
            </h3>
            <p>
              <strong>Architecture:</strong> 20 layers, 1536 hidden dimensions, 24 query heads, 4 KV heads (6:1 GQA), <MathView math="d_{\text{ffn}} = 3840" /> SwiGLU (2.5&times; expansion),
              32,768 vocabulary, 2048 context length. Verified total parameters: <strong>514,389,504</strong> (464,057,856 non-embedding).
            </p>
            <p>
              <strong>Pretraining Infrastructure:</strong> Target 5-billion-token curated corpus across 101 SHA-256 verified binary shards.
              Configured with AdamW (<MathView math="\beta_1=0.9, \beta_2=0.95, \lambda=0.1" />), <MathView math="\eta_{\max}=3.0\times 10^{-4}" />, <MathView math="\eta_{\min}=3.0\times 10^{-5}" />,
              2,000 warmup updates, micro-batch 1 with gradient accumulation 16 (32,768 tokens per parameter update), BF16 mixed precision,
              gradient checkpointing, and a thermal duty cycle watchdog (30m active / 5m cooldown).
            </p>
            <p>
              <strong>Autonomous Tool-Chaining SFT:</strong> Trained for 100 updates on multi-turn sequential tool chaining
              (Web Research &rarr; Observation &rarr; Python Execution &rarr; Grounded Synthesis).
              Hyperparameters: <code>bitsandbytes.optim.AdamW8bit</code> (<MathView math="\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}, \lambda=0.1" />),
              peak learning rate <MathView math="\eta_{\max} = 1.5 \times 10^{-5}" /> decaying to <MathView math="\eta_{\min} = 1.5 \times 10^{-6}" />,
              20 warmup updates, micro-batch 1 with gradient accumulation 4 (effective batch 4),
              400 presentations interleaved 4:3:3 across 3 curricula (Corpus A: Web Search, Corpus B: Python Sandbox, Corpus C: Grounded Synthesis),
              and Token-Weighted Cross-Entropy on assistant response tokens (<MathView math="m_{b,t}=1" />).
              Validation loss converged from 3.0854 to <strong>0.8947</strong>.
            </p>
            <p>
              <strong>Blind Benchmark Verification:</strong> In held-out capability testing (100 untouched records), this model achieved:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li><strong>First-Turn Web Selection:</strong> 70.0% (28/40) vs 17.5% in parent baseline.</li>
              <li><strong>Second-Turn Python Chaining:</strong> 50.0% (15/30) vs 0.0% in parent baseline.</li>
              <li><strong>AST Code Syntax Validity:</strong> 43.3% (13/30) vs 0.0% in parent baseline.</li>
              <li><strong>Evidence-Grounded Synthesis:</strong> 100.0% (15/15) full-support precision and 0.0% unsupported claims when verified by the runtime evidence validator.</li>
            </ul>
          </div>
        </div>
      </section>
    </main>
  );
};
