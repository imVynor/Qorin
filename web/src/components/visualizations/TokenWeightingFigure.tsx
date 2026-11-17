import React from 'react';
import styles from './TokenWeightingFigure.module.css';

interface Props {
  figureNumber?: number;
}

export const TokenWeightingFigure: React.FC<Props> = ({ figureNumber = 6 }) => {
  return (
    <figure className={styles.figureWrap}>
      <header className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>Loss Weighting Formulation: Unweighted Micro-batch Mean vs. Token-Weighted Invariant</span>
        </div>
      </header>

      <div className={styles.splitComparison}>
        <div className={`${styles.box} ${styles.boxFlawed}`}>
          <div className={styles.boxTitle}>Flawed: Sample-Averaged Batch Loss</div>
          <div className={styles.formula}>
            L<sub>batch</sub> = (1 / B) &Sigma;<sub>i=1..B</sub> [ (1 / N<sub>i</sub>) &Sigma;<sub>t=1..N<sub>i</sub></sub> &ell;<sub>i,t</sub> ]
          </div>
          <p className={styles.boxExplanation}>
            Each sequence in the micro-batch is normalized by its active token count <em>N<sub>i</sub></em> before averaging across sequences. Under variable-length padding, short sequences receive artificially inflated gradients per token, while long sequences are heavily downweighted. Micro-batch gradients vary with batch composition even when identical tokens are processed.
          </p>
        </div>

        <div className={`${styles.box} ${styles.boxFixed}`}>
          <div className={styles.boxTitle}>Corrected: Token-Weighted Invariant Loss</div>
          <div className={styles.formula}>
            L<sub>batch</sub> = (1 / &Sigma;<sub>i=1..B</sub> N<sub>i</sub>) &Sigma;<sub>i=1..B</sub> &Sigma;<sub>t=1..N<sub>i</sub></sub> &ell;<sub>i,t</sub>
          </div>
          <p className={styles.boxExplanation}>
            Every supervised target token contributes exactly equal gradient weight (1.0) regardless of sequence packaging. Loss sums across all active tokens and normalizes by total active tokens across the micro-batch. Verified by unit tests across 1, 2, and 4 gradient accumulation steps to guarantee identical analytical gradients.
          </p>
        </div>
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure {figureNumber} Source:</strong> Mathematical derivation and verification suite in <code>tests/test_token_weighted_loss.py</code> and <code>modern_gpt/training/trainer.py</code>. This correction eliminated gradient variance across heterogeneous prompt lengths during instruction tuning.
      </figcaption>
    </figure>
  );
};

