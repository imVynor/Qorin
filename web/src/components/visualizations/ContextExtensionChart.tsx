import React from 'react';
import { Link } from 'react-router-dom';
import {
  contextScalingLossData,
  passkeyAccuracyMatrix,
  passkeySummary,
} from '../../data/chartData';
import styles from './ContextExtensionChart.module.css';

interface Props {
  figureNumber?: number;
}

export const ContextExtensionChart: React.FC<Props> = ({ figureNumber = 4 }) => {
  const depths = [0.1, 0.25, 0.5, 0.75, 0.9];
  const lengths = [512, 1024, 2048, 4096];

  return (
    <figure className={styles.figureWrap}>
      <div className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>4K Context Scaling and Passkey Retrieval</span>
        </div>
      </div>

      <div className={styles.splitGrid}>
        <div>
          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Validation Loss Progression
          </div>
          <div className={styles.lossList}>
            {contextScalingLossData.map((item) => (
              <div key={item.contextLength} className={styles.lossItem}>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{item.contextLength} tokens</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{item.valLoss.toFixed(4)} NLL</span>
                <span style={{ color: 'var(--color-ink-muted)' }}>{item.valPerplexity.toFixed(2)} ppl</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
            Passkey Depth Matrix ({passkeySummary.overallAccuracy}% Overall)
          </div>
          <div className={styles.matrixGrid}>
            <div className={styles.colHeader}>Window</div>
            {depths.map((d) => (
              <div key={d} className={styles.colHeader}>
                {(d * 100).toFixed(0)}%
              </div>
            ))}

            {lengths.map((len) => (
              <React.Fragment key={len}>
                <div className={styles.rowHeader}>{len}</div>
                {depths.map((d) => {
                  const cell = passkeyAccuracyMatrix.find(
                    (c) => c.contextLength === len && c.depth === d
                  );
                  const isSuccess = cell && cell.accuracy > 0;
                  return (
                    <div
                      key={`${len}-${d}`}
                      className={`${styles.cell} ${isSuccess ? styles.cellPass : styles.cellFail}`}
                      title={`${len} tokens, ${(d * 100).toFixed(0)}% depth: ${cell?.retrieved || '0/5'}`}
                    >
                      {cell ? `${cell.accuracy}%` : '0%'}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure 4.</strong> RoPE linear scaling (s = 8.0) during 4K continued pretraining.
        Validation loss improved monotonically out to 4,096 tokens (2.8153 &rarr; 2.6997), yet synthetic passkey needle retrieval scored only 4.0% accuracy across 100 trials.
        This demonstrates that continued pretraining expands the receptive field, but deliberate needle extraction requires supervised instruction tuning.
        Full report in <Link to="/experiments" style={{ textDecoration: 'underline' }}>Experiments Archive &rarr;</Link>
      </figcaption>
    </figure>
  );
};
