import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { sftAnomalyData } from '../../data/chartData';
import styles from './SFTLossBehavioralChart.module.css';

interface Props {
  figureNumber?: number;
}

export const SFTLossBehavioralChart: React.FC<Props> = ({ figureNumber = 3 }) => {
  const [selectedIndex, setSelectedIndex] = useState<number>(2); // Default to SFT v3
  const current = sftAnomalyData[selectedIndex];

  return (
    <figure className={styles.figureWrap}>
      <div className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>Cross-Entropy Loss vs. Multi-Turn Behavioral Degradation</span>
        </div>
        <div className={styles.toggleGroup}>
          {sftAnomalyData.map((d, idx) => (
            <button
              key={d.version}
              type="button"
              className={`${styles.toggleBtn} ${selectedIndex === idx ? styles.toggleBtnActive : ''}`}
              onClick={() => setSelectedIndex(idx)}
            >
              {d.version}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.splitGrid}>
        {/* Left: Loss Descent */}
        <div className={styles.panel}>
          <div>
            <div className={styles.panelTitle}>Supervised Cross-Entropy</div>
            <div className={styles.metricBig}>{current.valLoss.toFixed(3)}</div>
            <div className={styles.metricSubtext}>
              {selectedIndex === 2 ? 'Project-record low loss' : 'Intermediate checkpoint loss'} across {current.records.toLocaleString()} sequences
            </div>
          </div>

          <div className={styles.lossBarRow}>
            {sftAnomalyData.map((d, i) => {
              const heightFrac = (d.valLoss / 1.6) * 100;
              const isSelected = i === selectedIndex;
              return (
                <div key={d.version} className={styles.lossCol} onClick={() => setSelectedIndex(i)}>
                  <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)' }}>{d.valLoss.toFixed(3)}</span>
                  <div
                    className={styles.lossColBar}
                    style={{
                      height: `${heightFrac}%`,
                      background: isSelected ? 'var(--color-ink-primary)' : 'var(--color-rule-strong)',
                    }}
                  />
                  <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)' }}>{d.version}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Behavioral divergence */}
        <div className={styles.panel}>
          <div className={styles.panelTitle}>100-Case Behavioral Challenge Metrics</div>
          <div className={styles.behaviorRows}>
            <div className={styles.behaviorRow}>
              <div className={styles.behaviorMeta}>
                <span>Isolated Question Accuracy (0-turn)</span>
                <strong>{current.isolatedAccuracy.toFixed(1)}%</strong>
              </div>
              <div className={styles.barTrack}>
                <div className={styles.barFillIndigo} style={{ width: `${current.isolatedAccuracy * 3}%` }} />
              </div>
            </div>

            <div className={styles.behaviorRow}>
              <div className={styles.behaviorMeta}>
                <span>Multi-Turn Context Accuracy (3–5 turn)</span>
                <strong>{current.multiTurnAccuracy.toFixed(1)}%</strong>
              </div>
              <div className={styles.barTrack}>
                <div className={styles.barFillIndigo} style={{ width: `${current.multiTurnAccuracy * 3}%` }} />
              </div>
            </div>

            <div className={styles.behaviorRow}>
              <div className={styles.behaviorMeta}>
                <span style={{ color: 'var(--color-accent-rust)' }}>History Delta Penalty (&Delta;<sub>hist</sub>)</span>
                <strong style={{ color: 'var(--color-accent-rust)' }}>{current.historyDeltaPenalty.toFixed(1)}%</strong>
              </div>
              <div className={styles.barTrack}>
                <div className={styles.barFillRust} style={{ width: `${Math.abs(current.historyDeltaPenalty) * 5}%` }} />
              </div>
            </div>

            <div className={styles.behaviorRow}>
              <div className={styles.behaviorMeta}>
                <span>Attractor Token Bias Rate</span>
                <strong>{current.attractorTokenBias.toFixed(1)}%</strong>
              </div>
              <div className={styles.barTrack}>
                <div className={styles.barFillRust} style={{ width: `${current.attractorTokenBias * 6}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure 3.</strong> Divergence between optimization loss and conversational stability across SFT iterations.
        In SFT v3, cross-entropy reached a record low of 0.765, yet multi-turn evaluation revealed that history delta penalty plunged to &minus;13.0% and the model failed basic greetings.
        Audit revealed greetings represented only 0.026% of candidate records (9 of 34,700) and 0.003% of supervised tokens, while UltraChat in SFT v1 contributed 80.3% of supervised tokens, pulling generation into an assistance attractor basin.
        Full forensic audit report in <Link to="/experiments" style={{ textDecoration: 'underline' }}>Experiments Archive &rarr;</Link>
      </figcaption>
    </figure>
  );
};
