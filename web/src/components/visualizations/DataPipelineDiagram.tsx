import React from 'react';
import styles from './DataPipelineDiagram.module.css';

interface Props {
  figureNumber?: number;
}

const PIPELINE_STEPS = [
  {
    step: 'Step 01',
    title: 'Multi-Source Acquisition',
    desc: '7 balanced domains: FineWeb-Edu, Cosmopedia, StarCoder, Books, Wikipedia, OpenWebMath, and UltraChat.',
    stats: '4.14M docs \u2022 22.8 GB raw'
  },
  {
    step: 'Step 02',
    title: 'Hygiene & Deduplication',
    desc: 'Min-hash deduplication (Jaccard 0.8), PII & secret stripping, language identification, and 5% dialogue ceiling.',
    stats: 'Zero hash collisions'
  },
  {
    step: 'Step 03',
    title: 'Byte-Pair Encoding',
    desc: 'Custom BPE-32k tokenizer (fingerprint: 3559c743). Sequence packing with causal end-of-text delimiters.',
    stats: '32,768 vocabulary'
  },
  {
    step: 'Step 04',
    title: 'Binary Sharding',
    desc: '101 contiguous uint16 binary shards with 64-byte headers. Enables zero-copy memory mapping at >100k tok/s.',
    stats: '101 shards (5.00B tok)'
  },
  {
    step: 'Step 05',
    title: 'Isolated Validation Split',
    desc: '20M dedicated validation tokens sampled uniformly across domains. Cryptographically quarantined with 0.00% leakage.',
    stats: '20.02M reserved tok'
  },
  {
    step: 'Step 06',
    title: 'Multi-Tier Evaluation',
    desc: 'Continuous checkpoint logging against 5 capability tiers: PPL, Python AST, JSON Schema, Passkey 4K, and Alignment.',
    stats: '5 capability tiers'
  }
];

export const DataPipelineDiagram: React.FC<Props> = ({ figureNumber = 9 }) => {
  return (
    <figure className={styles.diagramWrap}>
      <header className={styles.diagramHeader}>
        <div className={styles.figureIndex}>Figure {figureNumber} &bull; Data Lifecycle Architecture</div>
        <h3 className={styles.figureTitle}>From Raw Internet Corpora to Verified Binary Shards &amp; Evaluations</h3>
      </header>

      <div className={styles.pipelineGrid}>
        {PIPELINE_STEPS.map((step) => (
          <div key={step.step} className={styles.stepCard}>
            <span className={styles.stepIndex}>{step.step}</span>
            <h4 className={styles.stepTitle}>{step.title}</h4>
            <p className={styles.stepDesc}>{step.desc}</p>
            <div className={styles.stepStats}>{step.stats}</div>
          </div>
        ))}
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure {figureNumber} Source:</strong> Documented in <code>reports/phase1b_1_tokenizer_corpus_audit/corpus_pipeline_design.md</code> and <code>reports/phase1b_7_final_sharding/training_shard_manifest.json</code>. Raw text passes through deduplication, tokenization, and memory-mapped binary compilation into 101 production shards with a cryptographically isolated 20M validation split.
      </figcaption>
    </figure>
  );
};

