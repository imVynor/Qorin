import React, { useState } from 'react';
import styles from './LearningModule.module.css';

export const Module04GroupedQueryAttention: React.FC = () => {
  const [seqLen, setSeqLen] = useState<number>(2048);
  const [batchSize, setBatchSize] = useState<number>(1);
  const [precision, setPrecision] = useState<'bf16' | 'int8' | 'fp8'>('bf16');
  const [modelType, setModelType] = useState<'500M' | '100M' | '50M'>('500M');

  // Ground truth model specifications strictly aligned with modelsData.ts and training checkpoints
  const modelSpecs = {
    '500M': { layers: 20, qHeads: 24, kvHeads: 4, headDim: 64, dModel: 1536, ratio: '6:1 GQA', status: 'Active 5B Pretraining Configuration' },
    '100M': { layers: 14, qHeads: 12, kvHeads: 4, headDim: 64, dModel: 768, ratio: '3:1 GQA', status: 'Empirical Baseline Configuration' },
    '50M': { layers: 12, qHeads: 8, kvHeads: 4, headDim: 64, dModel: 512, ratio: '2:1 GQA', status: 'Empirical Iso-Data Ablation Champion' },
  };

  const spec = modelSpecs[modelType];
  const bytesPerVal = precision === 'bf16' ? 2 : 1; // 2 bytes for BF16/FP16, 1 byte for INT8/FP8

  // KV cache memory formula: 2 (K + V) * layers * batchSize * seqLen * kvHeads * headDim * bytes
  const calcKVCacheMB = (heads: number) => {
    const bytes = 2 * spec.layers * batchSize * seqLen * heads * spec.headDim * bytesPerVal;
    return bytes / (1024 * 1024);
  };

  const mhaMB = calcKVCacheMB(spec.qHeads);
  const gqaMB = calcKVCacheMB(spec.kvHeads);
  const mqaMB = calcKVCacheMB(1);

  const savingsPct = (((mhaMB - gqaMB) / mhaMB) * 100).toFixed(1);

  return (
    <section id="module-gqa" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 04 &bull; Memory Bandwidth Optimization</div>
        <h2 className={styles.moduleTitle}>Multi-Head Attention vs. Grouped-Query Attention (GQA)</h2>
        <div className={styles.moduleLead}>
          Overcoming the autoregressive memory bandwidth wall: why loading past Keys and Values bottlenecks generation, and how GQA slashes KV-cache VRAM by up to 83.3%.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>The Autoregressive Memory Wall</h3>
        <div className={styles.prose}>
          <p>
            During training, all tokens are processed in parallel (compute-bound). But during interactive text generation,
            the model generates <strong>one token at a time</strong>. At every generation step, the model must read all
            previously cached Key and Value vectors from GPU high-bandwidth memory (VRAM) into fast on-chip SRAM:
          </p>
          <div className={styles.formulaBox}>
            KV-Cache Memory (Bytes) = 2 &times; L &times; B &times; S &times; H<sub>KV</sub> &times; d<sub>k</sub> &times; P<br />
            where: L = layers, B = batch size, S = sequence length, H<sub>KV</sub> = key/value heads, d<sub>k</sub> = head dimension, P = bytes/param (2 for BF16)
          </div>
          <p>
            Under standard <strong>Multi-Head Attention (MHA)</strong>, every single query head has its own private key and value head
            (<em>H</em><sub><em>Q</em></sub> = <em>H</em><sub><em>KV</em></sub>). As sequence length grows to 4K or 8K tokens, the KV-cache
            swallows gigabytes of memory, choking GPU memory buses and slowing decoding to a crawl.
          </p>
          <p>
            <strong>Grouped-Query Attention (GQA)</strong> (Ainslie et al., 2023) groups multiple query heads into clusters that share
            a single pooled key-value head:
          </p>
          <ul>
            <li><strong>Multi-Head Attention (MHA):</strong> 1 Query Head per 1 KV Head (1:1 ratio). Highest memory, full expressivity.</li>
            <li><strong>Grouped-Query Attention (GQA):</strong> <em>G</em> Query Heads per 1 KV Head (e.g. 6:1 in QORIN-500M, 3:1 in QORIN-100M). Preserves 99.4% quality with a fraction of the memory.</li>
            <li><strong>Multi-Query Attention (MQA):</strong> All Query Heads share a single global KV Head. Maximum memory savings, but causes noticeable degradation in complex reasoning.</li>
          </ul>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Architectural Head Sharing Topologies
        </div>
        <h3 className={styles.partHeading}>Head Routing Topologies</h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--space-4)', margin: 'var(--space-4) 0' }}>
          <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', padding: 'var(--space-4)', borderRadius: 'var(--radius-xs)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>MHA Baseline (1:1)</span>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink-primary)', margin: '4px 0' }}>8 Q-Heads &bull; 8 KV-Heads</div>
            <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', lineHeight: 1.5 }}>
              Each query head reads its own private KV-cache stream. Extreme memory footprint at long contexts.
            </p>
          </div>

          <div style={{ background: 'var(--color-paper-subtle)', border: '2px solid var(--color-accent-indigo)', padding: 'var(--space-4)', borderRadius: 'var(--radius-xs)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-accent-indigo)' }}>QORIN GQA Standard ({spec.ratio})</span>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink-primary)', margin: '4px 0' }}>{spec.qHeads} Q-Heads &bull; {spec.kvHeads} KV-Heads</div>
            <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', lineHeight: 1.5 }}>
              Query heads form groups of {spec.qHeads / spec.kvHeads}, sharing 1 pooled KV head. Cuts cache size by {savingsPct}% with near-identical perplexity.
            </p>
          </div>

          <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', padding: 'var(--space-4)', borderRadius: 'var(--radius-xs)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>MQA Extreme ({spec.qHeads}:1)</span>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink-primary)', margin: '4px 0' }}>{spec.qHeads} Q-Heads &bull; 1 KV-Head</div>
            <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', lineHeight: 1.5 }}>
              Single KV stream for all queries. Highest memory efficiency, but limits multi-head factual attention diversity.
            </p>
          </div>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Memory &amp; Bandwidth Calculator
        </div>
        <h3 className={styles.partHeading}>Live KV-Cache VRAM Calculator</h3>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Target Architecture (Empirical Ground Truth):</span>
              <div className={styles.btnGroup}>
                {(['500M', '100M', '50M'] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    className={`${styles.btnToggle} ${modelType === m ? styles.btnToggleActive : ''}`}
                    onClick={() => setModelType(m)}
                  >
                    QORIN-{m}
                  </button>
                ))}
              </div>
              <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
                {spec.layers} layers &bull; {spec.qHeads}Q / {spec.kvHeads}KV heads ({spec.ratio}) &bull; {spec.status}
              </div>
            </div>

            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Sequence Length: {seqLen} tokens</span>
              <input
                type="range"
                min={512}
                max={8192}
                step={512}
                value={seqLen}
                onChange={(e) => setSeqLen(Number(e.target.value))}
                className={styles.sliderInput}
              />
            </div>

            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Concurrent Batch Size: {batchSize}</span>
              <input
                type="range"
                min={1}
                max={32}
                step={1}
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className={styles.sliderInput}
              />
            </div>

            <div className={styles.controlItem}>
              <span className={styles.controlLabel}>Precision Mode:</span>
              <div className={styles.btnGroup}>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${precision === 'bf16' ? styles.btnToggleActive : ''}`}
                  onClick={() => setPrecision('bf16')}
                >
                  BF16 [Tested]
                </button>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${precision === 'int8' ? styles.btnToggleActive : ''}`}
                  onClick={() => setPrecision('int8')}
                >
                  INT8 [Simulated]
                </button>
                <button
                  type="button"
                  className={`${styles.btnToggle} ${precision === 'fp8' ? styles.btnToggleActive : ''}`}
                  onClick={() => setPrecision('fp8')}
                >
                  FP8 [Simulated]
                </button>
              </div>
              <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: precision === 'bf16' ? 'var(--color-accent-sage)' : 'var(--color-ink-muted)', marginTop: '4px' }}>
                {precision === 'bf16' ? '✓ Tested in QORIN checkpoints (2 bytes/val)' : 'Hypothetical simulated quantization (1 byte/val)'}
              </div>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div className={styles.outputTitle}>KV-Cache Memory Footprint Comparison</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-4)', textAlign: 'center' }}>
              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>MHA Cache (1:1 Uncompressed)</span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600, color: 'var(--color-accent-rust)', marginTop: '4px' }}>
                  {mhaMB >= 1024 ? `${(mhaMB / 1024).toFixed(2)} GB` : `${mhaMB.toFixed(1)} MB`}
                </div>
              </div>

              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '2px solid var(--color-accent-indigo)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-indigo)' }}>QORIN GQA Cache ({spec.ratio})</span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--color-ink-primary)', marginTop: '4px' }}>
                  {gqaMB >= 1024 ? `${(gqaMB / 1024).toFixed(2)} GB` : `${gqaMB.toFixed(1)} MB`}
                </div>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-sage)', fontWeight: 600 }}>
                  &minus;{savingsPct}% Memory Footprint
                </span>
              </div>

              <div style={{ background: 'var(--color-surface)', padding: 'var(--space-3)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>MQA Extreme ({spec.qHeads}:1)</span>
                <div style={{ fontSize: 'var(--text-lg)', fontWeight: 600, color: 'var(--color-ink-secondary)', marginTop: '4px' }}>
                  {mqaMB >= 1024 ? `${(mqaMB / 1024).toFixed(2)} GB` : `${mqaMB.toFixed(1)} MB`}
                </div>
              </div>
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> In our 50M architecture ablation at iso-data (250M tokens), QORIN GQA-4
            achieved 101.8 tok/s decoding speed while consuming only 96 MB KV-cache for batch 16, exactly halving memory
            compared to MHA (192 MB) with virtually identical validation perplexity (22.67 vs 22.54). In QORIN-100M (14 layers, 12 Q-heads, 4 KV-heads)
            and QORIN-500M (20 layers, 24 Q-heads, 4 KV-heads), GQA delivers 66.7% and 83.3% KV-cache reductions, respectively.
          </div>
        </div>
      </div>
    </section>
  );
};
