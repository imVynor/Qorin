import React, { useState } from 'react';
import styles from './LearningModule.module.css';

interface HeadProfile {
  name: string;
  role: string;
  matrix: number[][]; // 4x4 attention matrix
}

export const Module02SelfAttention: React.FC = () => {
  const [causalMask, setCausalMask] = useState(true);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  const sampleTokens = ['The', 'model', 'predicts', 'tokens'];

  // 3 distinct attention heads demonstrating different routing behaviors
  const heads: HeadProfile[] = [
    {
      name: 'Head 1: Syntactic Predecessor Head',
      role: 'Attends primarily to the immediately preceding token to track local grammar and verb-noun agreements.',
      matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.82, 0.18, 0.0, 0.0],
        [0.12, 0.74, 0.14, 0.0],
        [0.08, 0.15, 0.68, 0.09],
      ],
    },
    {
      name: 'Head 2: Semantic Association Head',
      role: 'Routes attention between semantically coupled concepts ("model" <-> "predicts", "predicts" <-> "tokens").',
      matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.35, 0.65, 0.0, 0.0],
        [0.10, 0.60, 0.30, 0.0],
        [0.05, 0.25, 0.55, 0.15],
      ],
    },
    {
      name: 'Head 3: Identity / Self-Focus Head',
      role: 'Preserves the direct token identity along the diagonal while dispersing diffuse background attention.',
      matrix: [
        [1.0, 0.0, 0.0, 0.0],
        [0.15, 0.85, 0.0, 0.0],
        [0.10, 0.12, 0.78, 0.0],
        [0.05, 0.08, 0.14, 0.73],
      ],
    },
  ];

  const currentHead = heads[selectedHead];

  // Modify matrix if causal mask is toggled off (for bidirection demonstration)
  const displayMatrix = currentHead.matrix.map((row, rIdx) => {
    if (causalMask) return row;
    // Normalized non-causal mock distribution
    const unmasked = [...row];
    if (rIdx === 0) return [0.45, 0.25, 0.15, 0.15];
    if (rIdx === 1) return [0.30, 0.40, 0.18, 0.12];
    if (rIdx === 2) return [0.10, 0.35, 0.40, 0.15];
    return unmasked;
  });

  return (
    <section id="module-attention" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 02 &bull; Core Computational Engine</div>
        <h2 className={styles.moduleTitle}>How Self-Attention Works</h2>
        <div className={styles.moduleLead}>
          The mathematical mechanics of Queries, Keys, and Values: calculating token relevance, causal masking, and weighted information routing.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>The Query-Key-Value Information Retrieval Triad</h3>
        <div className={styles.prose}>
          <p>
            Unlike recurrent networks (RNNs) that process tokens one-by-one in temporal lockstep, the Transformer compares
            every token against every other token simultaneously. Each token vector <em>x</em><sub><em>i</em></sub> is linearly projected
            into three separate representations:
          </p>
          <ul>
            <li><strong>Query (<em>Q</em> = <em>XW</em><sub><em>Q</em></sub>):</strong> What information this token is actively seeking.</li>
            <li><strong>Key (<em>K</em> = <em>XW</em><sub><em>K</em></sub>):</strong> What information this token offers to others.</li>
            <li><strong>Value (<em>V</em> = <em>XW</em><sub><em>V</em></sub>):</strong> The actual semantic payload transferred if a Query matches this Key.</li>
          </ul>

          <div className={styles.formulaBox}>
            Attention(Q, K, V) = softmax( (Q K<sup>T</sup> / &radic;d<sub>k</sub>) + M ) V
          </div>

          <p>
            <strong>Why scale by &radic;<em>d</em><sub><em>k</em></sub>?</strong><br />
            Assume the components of <em>q</em> and <em>k</em> are independent random variables with zero mean and unit variance.
            Their dot product &sum; <em>q</em><sub><em>i</em></sub> <em>k</em><sub><em>i</em></sub> has a mean of 0 and a variance
            of <em>d</em><sub><em>k</em></sub>. For large head dimensions (e.g. <em>d</em><sub><em>k</em></sub> = 64), the dot product magnitudes grow
            very large. Large inputs push the <code>softmax</code> function into regions where the gradient is near zero
            (vanishing gradient). Dividing by &radic;64 = 8 preserves unit variance and ensures clean gradient backpropagation.
          </p>

          <p>
            <strong>Causal Masking (<em>M</em>):</strong> In autoregressive language modeling, token <em>t</em> must never peek into the future
            tokens (<em>t</em>+1, <em>t</em>+2). We add an upper-triangular mask matrix <em>M</em> where future positions are set to &minus;&infin;,
            ensuring exp(&minus;&infin;) = 0 in the softmax.
          </p>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Internal State &amp; Worked Numerical Step
        </div>
        <h3 className={styles.partHeading}>Worked Numerical Example</h3>
        <div className={styles.prose}>
          <p>
            Consider token &ldquo;predicts&rdquo; (row 2) attending to prior tokens. The raw query-key dot products are computed, scaled,
            masked, and passed through softmax:
          </p>
          <div className={styles.formulaBox}>
            Raw Dot Products QK<sup>T</sup>: [ 0.96, 4.80, 1.12, &minus;&infin; ]<br />
            Scaled by &radic;64 (8.0): [ 0.12, 0.60, 0.14, &minus;&infin; ]<br />
            Softmax (Row Sum = 1.0): [ 0.12, 0.74, 0.14, 0.00 ]<br />
            &rarr; Output Vector = 0.12 &times; V<sub>The</sub> + 0.74 &times; V<sub>model</sub> + 0.14 &times; V<sub>predicts</sub> + 0.00 &times; V<sub>tokens</sub>
          </div>
          <p>
            The resulting contextualized representation of &ldquo;predicts&rdquo; now contains 74% of the semantic properties of &ldquo;model&rdquo;!
          </p>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Attention Heatmap
        </div>
        <h3 className={styles.partHeading}>Conceptual Multi-Head Attention Inspector</h3>
        <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: '8px 12px', marginBottom: 'var(--space-4)', fontSize: '11px', color: 'var(--color-ink-secondary)' }}>
          <strong>Pedagogical Notice:</strong> The attention matrices below represent stylized conceptual archetypes (syntactic routing, semantic coupling, and identity preservation) constructed for teaching purposes. They are illustrative pedagogical representations, not empirical activation weights dumped from a specific trained QORIN model checkpoint.
        </div>
        <p className={styles.prose}>
          Switch between attention heads and toggle causal masking to observe how attention routing shifts between syntactic
          predecessors, semantic associations, and self-focus:
        </p>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Select Simulated Attention Head:</span>
              <div className={styles.btnGroup}>
                {heads.map((h, idx) => (
                  <button
                    key={h.name}
                    type="button"
                    className={`${styles.btnToggle} ${selectedHead === idx ? styles.btnToggleActive : ''}`}
                    onClick={() => setSelectedHead(idx)}
                  >
                    Head {idx + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Causal Masking (Autoregressive):</span>
              <button
                type="button"
                className={`${styles.btnToggle} ${causalMask ? styles.btnToggleActive : ''}`}
                style={{ border: '1px solid var(--color-rule)', padding: '4px 10px' }}
                onClick={() => setCausalMask(!causalMask)}
              >
                {causalMask ? 'Causal Mask ON (Lower Triangle)' : 'Causal Mask OFF (Full Bidirectional)'}
              </button>
            </div>
          </div>

          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-ink-secondary)', marginBottom: 'var(--space-3)' }}>
            <strong>{currentHead.name}:</strong> {currentHead.role}
          </div>

          <div className={styles.outputArea} style={{ overflowX: 'auto' }}>
            <div className={styles.outputTitle}>Attention Weight Heatmap &bull; Softmax(QK^T / &radic;d_k)</div>

            <table style={{ borderCollapse: 'collapse', margin: '8px auto', fontFamily: 'var(--font-mono)', fontSize: '11px' }}>
              <thead>
                <tr>
                  <th style={{ padding: '6px 10px', color: 'var(--color-ink-muted)' }}>Query \ Key</th>
                  {sampleTokens.map((tok) => (
                    <th key={tok} style={{ padding: '6px 12px', textAlign: 'center', color: 'var(--color-ink-primary)' }}>
                      {tok}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sampleTokens.map((qTok, rIdx) => (
                  <tr key={qTok}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: 'var(--color-ink-primary)' }}>{qTok}</td>
                    {sampleTokens.map((kTok, cIdx) => {
                      const weight = displayMatrix[rIdx][cIdx];
                      const isMasked = causalMask && cIdx > rIdx;
                      const opacity = isMasked ? 0.05 : Math.max(weight, 0.08);
                      const isHovered = hoveredCell?.row === rIdx && hoveredCell?.col === cIdx;

                      return (
                        <td
                          key={kTok}
                          onMouseEnter={() => setHoveredCell({ row: rIdx, col: cIdx })}
                          onMouseLeave={() => setHoveredCell(null)}
                          style={{
                            padding: '10px 14px',
                            textAlign: 'center',
                            backgroundColor: isMasked
                              ? 'transparent'
                              : `rgba(30, 58, 138, ${opacity})`,
                            color: isMasked
                              ? 'var(--color-ink-muted)'
                              : weight > 0.45
                              ? '#ffffff'
                              : 'var(--color-ink-primary)',
                            border: isHovered
                              ? '1px solid var(--color-ink-primary)'
                              : '1px solid var(--color-rule-subtle)',
                            cursor: 'pointer',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          {isMasked ? '0.00 (M)' : weight.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>

            {hoveredCell && (
              <div style={{ marginTop: '8px', fontSize: '11px', textAlign: 'center', color: 'var(--color-ink-secondary)' }}>
                Query <strong>&ldquo;{sampleTokens[hoveredCell.row]}&rdquo;</strong> assigns{' '}
                <strong>{(displayMatrix[hoveredCell.row][hoveredCell.col] * 100).toFixed(0)}%</strong> attention weight to Key{' '}
                <strong>&ldquo;{sampleTokens[hoveredCell.col]}&rdquo;</strong>.
              </div>
            )}
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> QORIN-16M employs 6 attention heads (head dimension <em>d</em><sub><em>k</em></sub> = 64).
            QORIN-500M scales to 24 query heads across 20 transformer layers (totaling 480 independent attention heads!),
            each learning specialized syntactic, semantic, and factual routing functions.
          </div>
        </div>
      </div>
    </section>
  );
};

