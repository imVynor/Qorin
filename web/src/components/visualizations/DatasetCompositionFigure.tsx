import React from 'react';
import styles from './DatasetCompositionFigure.module.css';

interface Props {
  figureNumber?: number;
}

interface Segment {
  label: string;
  share: number;
  color: string;
}

interface CorpusData {
  title: string;
  total: string;
  segments: Segment[];
}

const DATASETS: CorpusData[] = [
  {
    title: '5B Token Pretraining Corpus (Verified Shards 000-100)',
    total: '5.0 Billion Tokens / 101 Shards',
    segments: [
      { label: 'FineWeb-Edu (60%)', share: 60, color: 'var(--color-accent-indigo)' },
      { label: 'DCLM-Baseline (25%)', share: 25, color: '#3B82F6' },
      { label: 'Math & Code (10%)', share: 10, color: 'var(--color-accent-sage)' },
      { label: 'Dialogue Cap (5%)', share: 5, color: 'var(--color-accent-rust)' },
    ],
  },
  {
    title: 'Flawed Initial SFT Corpus (Triggered Conversational Collapse)',
    total: '34,700 Instruction Records',
    segments: [
      { label: 'UltraChat Conversational (80.3%)', share: 80.3, color: 'var(--color-accent-rust)' },
      { label: 'LMSYS Arena (16.5%)', share: 16.5, color: '#D97706' },
      { label: 'Synthetic Retrieval (3.2%)', share: 3.2, color: 'var(--color-ink-muted)' },
    ],
  },
  {
    title: 'Remediated Multi-turn Diagnostic Curriculum',
    total: '12,500 High-Entropy Records',
    segments: [
      { label: 'Synthetic Multi-turn Reasoning (56.5%)', share: 56.5, color: 'var(--color-accent-sage)' },
      { label: 'Code & Algorithmic Logic (25.0%)', share: 25.0, color: 'var(--color-accent-indigo)' },
      { label: 'Hermes Filtered Instruction (18.5%)', share: 18.5, color: '#059669' },
    ],
  },
];

export const DatasetCompositionFigure: React.FC<Props> = ({ figureNumber = 7 }) => {
  return (
    <figure className={styles.figureWrap}>
      <header className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>Corpus Token Composition: 5B Pretraining vs. SFT Collapse vs. Remediated Curriculum</span>
        </div>
      </header>

      <div className={styles.barsGrid}>
        {DATASETS.map((dataset) => (
          <div key={dataset.title} className={styles.corpusRow}>
            <div className={styles.corpusLabelRow}>
              <span className={styles.corpusName}>{dataset.title}</span>
              <span className={styles.corpusTotal}>{dataset.total}</span>
            </div>
            <div className={styles.stackedBar}>
              {dataset.segments.map((seg) => (
                <div
                  key={seg.label}
                  className={styles.segment}
                  style={{ width: `${seg.share}%`, backgroundColor: seg.color }}
                  title={`${seg.label}: ${seg.share}%`}
                >
                  {seg.share >= 8 ? seg.label : `${seg.share}%`}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className={styles.legendGrid}>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--color-accent-indigo)' }} />
          <span>Educational Web (FineWeb-Edu)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: '#3B82F6' }} />
          <span>Curated Web (DCLM)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--color-accent-sage)' }} />
          <span>Reasoning, Math & Code</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--color-accent-rust)' }} />
          <span>Uncapped Dialogue (Collapse Risk)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: '#D97706' }} />
          <span>General Instruction / Arena</span>
        </div>
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure {figureNumber} Source:</strong> Dataset manifest logs in <code>reports/corpus_manifest_5b.json</code>, <code>data_preparer.py</code>, and SFT audit logs in <code>reports/sft_forensic_audit.json</code>. Uncapped dialogue density (80.3%) in early fine-tuning caused sycophantic degeneration, leading to a mandatory 5% strict ceiling on conversational data in the 5B pretraining corpus.
      </figcaption>
    </figure>
  );
};

