import React, { useState } from 'react';
import styles from './LearningModule.module.css';

export const Module03RotaryEmbeddings: React.FC = () => {
  const [posM, setPosM] = useState<number>(4);
  const [posN, setPosN] = useState<number>(1);
  const [channelIdx, setChannelIdx] = useState<number>(1); // 0..31
  const [scalingFactor, setScalingFactor] = useState<number>(1.0); // 1.0, 4.0, or 8.0

  const headDim = 64;
  const thetaBase = 10000;
  // Linear RoPE frequency scaling: theta_i = (thetaBase^(-2*i/d)) / s
  const theta_i = Math.pow(thetaBase, -(2 * channelIdx) / headDim) / scalingFactor;

  const angleM = (posM * theta_i);
  const angleN = (posN * theta_i);
  const deltaAngle = angleM - angleN;
  const innerProduct = Math.cos(deltaAngle);

  // Canvas coordinates for 2D compass
  const radius = 55;
  const xM = radius * Math.cos(angleM);
  const yM = -radius * Math.sin(angleM);
  const xN = radius * Math.cos(angleN);
  const yN = -radius * Math.sin(angleN);

  return (
    <section id="module-rope" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 03 &bull; Relative Positional Geometry</div>
        <h2 className={styles.moduleTitle}>Positional Encoding &amp; RoPE</h2>
        <div className={styles.moduleLead}>
          Why Transformers require positional signals, how 2D rotation matrices naturally encode relative distances, and how frequency scaling enables context extension.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>The Geometry of Relative Position</h3>
        <div className={styles.prose}>
          <p>
            Self-attention is mathematically permutation-invariant: if you shuffle the order of input words, the dot products
            remain identical. The model cannot tell &ldquo;dog bites man&rdquo; from &ldquo;man bites dog&rdquo; without positional coordinates.
          </p>
          <p>
            Traditional models added absolute position vectors to the token embeddings: <em>x</em> + <em>p</em><sub><em>pos</em></sub>.
            However, absolute vectors do not generalize to unseen sequence lengths, and the model must awkwardly memorize every absolute index.
          </p>
          <p>
            <strong>Rotary Position Embeddings (RoPE)</strong> (Su et al., 2024) solves this by rotating the Query and Key vectors
            in 2D coordinate planes:
          </p>
          <div className={styles.formulaBox}>
            q<sub>m</sub> = R<sub>&Theta;,m</sub> &times; (W<sub>Q</sub> x<sub>m</sub>)<br />
            k<sub>n</sub> = R<sub>&Theta;,n</sub> &times; (W<sub>K</sub> x<sub>n</sub>)<br />
            &lang; q<sub>m</sub>, k<sub>n</sub> &rang; = q<sup>T</sup> R<sub>&Theta;,m</sub><sup>T</sup> R<sub>&Theta;,n</sub> k = q<sup>T</sup> R<sub>&Theta;,n&minus;m</sub> k
          </div>
          <p>
            Because rotating vector <em>A</em> by angle &theta;<sub><em>m</em></sub> and vector <em>B</em> by angle &theta;<sub><em>n</em></sub>
            makes their inner product depend strictly on the difference angle (&theta;<sub><em>m</em></sub> &minus; &theta;<sub><em>n</em></sub>),
            the attention mechanism naturally captures <strong>relative distance (<em>m</em> &minus; <em>n</em>)</strong> regardless of absolute location in the document!
          </p>
          <p>
            The rotation angle decays across channel pairs via the geometric progression:
          </p>
          <div className={styles.formulaBox}>
            &theta;<sub>i</sub> = 10,000<sup>&minus;2(i&minus;1) / d</sup> &times; (1 / s) &nbsp;&nbsp;(for channel index i &isin; [0, 1, ..., d/2 &minus; 1])
          </div>
          <p>
            Low dimension channels rotate rapidly (capturing local syntax), while high dimension channels rotate very slowly (capturing document-level relationships).
          </p>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Internal State &amp; The 2D Rotation Compass
        </div>
        <h3 className={styles.partHeading}>2D Orthogonal Rotation Plane</h3>
        <div className={styles.prose}>
          <p>
            Each 64-dimensional attention head is paired into 32 two-dimensional coordinate slices (<em>x</em><sub>0</sub>, <em>x</em><sub>1</sub>),
            (<em>x</em><sub>2</sub>, <em>x</em><sub>3</sub>), etc. Each slice rotates independently according to its channel frequency:
          </p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'center', margin: 'var(--space-6) 0' }}>
          <svg width="220" height="220" viewBox="-80 -80 160 160" style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: '50%' }}>
            {/* Compass axes */}
            <line x1="-70" y1="0" x2="70" y2="0" stroke="var(--color-rule-subtle)" strokeWidth="1" />
            <line x1="0" y1="-70" x2="0" y2="70" stroke="var(--color-rule-subtle)" strokeWidth="1" />
            <circle cx="0" cy="0" r={radius} fill="none" stroke="var(--color-rule)" strokeDasharray="2 3" />

            {/* Key Vector at posN */}
            <line x1="0" y1="0" x2={xN} y2={yN} stroke="#3B82F6" strokeWidth="2.5" />
            <circle cx={xN} cy={yN} r="4" fill="#3B82F6" />
            <text x={xN + 8} y={yN} fill="#3B82F6" fontSize="10" fontFamily="var(--font-mono)">k (pos {posN})</text>

            {/* Query Vector at posM */}
            <line x1="0" y1="0" x2={xM} y2={yM} stroke="var(--color-accent-rust)" strokeWidth="2.5" />
            <circle cx={xM} cy={yM} r="4" fill="var(--color-accent-rust)" />
            <text x={xM + 8} y={yM} fill="var(--color-accent-rust)" fontSize="10" fontFamily="var(--font-mono)">q (pos {posM})</text>

            {/* Relative arc indicator */}
            <circle cx="0" cy="0" r="18" fill="none" stroke="var(--color-accent-indigo)" strokeWidth="1.5" />
          </svg>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive RoPE Laboratory
        </div>
        <h3 className={styles.partHeading}>Live RoPE Rotation &amp; Frequency Scaling Explorer</h3>
        <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', marginBottom: 'var(--space-3)' }}>
          <strong>Educational Controls Notice:</strong> The interactive sliders below let you inspect relative rotation angles across synthetic token positions and channel indices.
        </p>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem}>
              <label className={styles.controlLabel}>Query Position (m): {posM}</label>
              <input
                type="range"
                min={0}
                max={24}
                value={posM}
                onChange={(e) => setPosM(Number(e.target.value))}
                className={styles.sliderInput}
              />
            </div>

            <div className={styles.controlItem}>
              <label className={styles.controlLabel}>Key Position (n): {posN}</label>
              <input
                type="range"
                min={0}
                max={24}
                value={posN}
                onChange={(e) => setPosN(Number(e.target.value))}
                className={styles.sliderInput}
              />
            </div>

            <div className={styles.controlItem}>
              <label className={styles.controlLabel}>Channel Pair Index (i &isin; 0..31): {channelIdx}</label>
              <input
                type="range"
                min={0}
                max={31}
                value={channelIdx}
                onChange={(e) => setChannelIdx(Number(e.target.value))}
                className={styles.sliderInput}
              />
            </div>

            <div className={styles.controlItem}>
              <label className={styles.controlLabel}>RoPE Linear Scaling Factor (s):</label>
              <div className={styles.btnGroup}>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${scalingFactor === 1.0 ? styles.btnToggleActive : ''}`}
                  onClick={() => setScalingFactor(1.0)}
                >
                  1.0x (1K Base)
                </button>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${scalingFactor === 4.0 ? styles.btnToggleActive : ''}`}
                  onClick={() => setScalingFactor(4.0)}
                >
                  4.0x (Pedagogical)
                </button>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${scalingFactor === 8.0 ? styles.btnToggleActive : ''}`}
                  onClick={() => setScalingFactor(8.0)}
                >
                  8.0x (QORIN 4K Pilot Actual)
                </button>
              </div>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div className={styles.outputTitle}>Calculated Relative Positional Metrics</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 'var(--space-3)' }}>
              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>Relative Distance (m - n):</span>
                <div style={{ fontSize: 'var(--text-base)', fontWeight: 600, color: 'var(--color-ink-primary)' }}>
                  {posM - posN} tokens
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>Effective Angular Frequency (&theta;_i):</span>
                <div style={{ fontSize: 'var(--text-base)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {theta_i.toFixed(5)} rad/tok
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>Angular Shift (&Delta;&theta; = (m - n)&theta;_i):</span>
                <div style={{ fontSize: 'var(--text-base)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {((deltaAngle * 180 / Math.PI) % 360).toFixed(1)}&deg;
                </div>
              </div>

              <div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>Projected Inner Product &lang;q, k&rang;:</span>
                <div style={{ fontSize: 'var(--text-base)', fontWeight: 600, fontFamily: 'var(--font-mono)', color: innerProduct > 0 ? 'var(--color-accent-sage)' : 'var(--color-accent-rust)' }}>
                  {innerProduct.toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> In the QORIN-100M 4K context extension pilot (<code>modern_gpts_100m_4k_pilot</code>),
            the model was extended from 512 base sequence length to 4,096 tokens using linear RoPE frequency scaling with <strong>s = 8.0</strong>.
            This reduced the angular velocity of all channel frequencies by a factor of 8, keeping query-key dot products
            within stable numerical distributions across 4,096 tokens without retraining from scratch.
          </div>
        </div>
      </div>
    </section>
  );
};
