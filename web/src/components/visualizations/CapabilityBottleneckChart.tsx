import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { capabilityBottleneckData } from '../../data/chartData';
import styles from './CapabilityBottleneckChart.module.css';

interface Props {
  figureNumber?: number;
}

export const CapabilityBottleneckChart: React.FC<Props> = ({ figureNumber = 2 }) => {
  const [model, setModel] = useState<'50M' | '16M'>('50M');

  return (
    <figure className={styles.figureWrap}>
      <div className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>Syntax Memorization vs. Functional Execution</span>
        </div>
        <div className={styles.toggleGroup}>
          <button
            type="button"
            className={`${styles.toggleBtn} ${model === '50M' ? styles.toggleBtnActive : ''}`}
            onClick={() => setModel('50M')}
          >
            QORIN-50M SFT
          </button>
          <button
            type="button"
            className={`${styles.toggleBtn} ${model === '16M' ? styles.toggleBtnActive : ''}`}
            onClick={() => setModel('16M')}
          >
            QORIN-16M SFT
          </button>
        </div>
      </div>

      <div className={styles.tasksList}>
        {capabilityBottleneckData.map((item) => {
          const train = model === '50M' ? item.m50Train : item.m16Train;
          const val = model === '50M' ? item.m50Val : item.m16Val;
          const isExecutionFailure = val <= 5.0;

          return (
            <div key={item.task} className={styles.taskItem}>
              <div className={styles.taskHeader}>
                <span className={styles.taskName}>{item.task}</span>
                <span className={styles.taskDesc}>{item.description}</span>
              </div>

              {/* Train bar */}
              <div className={styles.barRow}>
                <span className={styles.barLabel}>Train memorized (400)</span>
                <div className={styles.track}>
                  <div className={styles.fillTrain} style={{ width: `${train}%` }} />
                </div>
                <span className={styles.barValue}>{train.toFixed(1)}%</span>
              </div>

              {/* Held-out generalization bar */}
              <div className={styles.barRow}>
                <span className={styles.barLabel}>Held-out test (100)</span>
                <div className={styles.track}>
                  <div
                    className={isExecutionFailure ? styles.fillValFail : styles.fillVal}
                    style={{ width: `${Math.max(val, 2)}%` }}
                  />
                </div>
                <span
                  className={styles.barValue}
                  style={{ color: isExecutionFailure ? 'var(--color-accent-rust)' : 'inherit' }}
                >
                  {val.toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure 2.</strong> Diagnostic capability evaluation across 400 training tasks and 100 held-out tasks.
        In QORIN-50M, fine-tuning achieved 100% training syntax and schema compliance with 48.8% training execution.
        On held-out tasks, syntactic validity remained strong (87.5% AST, 95.0% JSON), yet functional execution collapsed to 2.5% (1/40 passed), and 0.0% on 16M.
        This demonstrates that structural formatting is a statistical surface regularity that models acquire easily, whereas functional execution requires deeper parametric capacity.
        Explore full diagnostic logs in <Link to="/experiments" style={{ textDecoration: 'underline' }}>Experiments Archive &rarr;</Link>
      </figcaption>
    </figure>
  );
};
