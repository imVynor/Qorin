import React from 'react';
import { Link } from 'react-router-dom';
import styles from './Footer.module.css';

export const Footer: React.FC = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.container}>
        <div className={styles.colophonGrid}>
          <div className={styles.brandCol}>
            <div className={styles.colophonTitle}>
              <img
                src={`${import.meta.env.BASE_URL}qorin.svg`}
                alt="QORIN Logo"
                className={styles.footerLogo}
                width="20"
                height="20"
              />
              <span>QORIN Research Project</span>
            </div>
            <p className={styles.colophonText}>
              An independent investigation into small language models, token-weighted optimization,
              and external retrieval, trained from scratch on consumer hardware.
            </p>
            <div className={styles.authorCredit}>
              <div>Conducted by <strong>Tumarada Sai Anish Raj</strong> &amp; <strong>Depuru Jnan Deep</strong></div>
              <div>Originally proposed as <em>&ldquo;Tool-Augmented Small Language Models&rdquo;</em></div>
            </div>
          </div>

          <div>
            <div className={styles.colHeader}>Research</div>
            <ul className={styles.linkList}>
              <li><Link to="/research" className={styles.footerLink}>Narrative &amp; Origins</Link></li>
              <li><Link to="/research#token-weighting" className={styles.footerLink}>Token-Weighted Loss</Link></li>
              <li><Link to="/research#pretraining-corpus" className={styles.footerLink}>5B Pretraining Corpus</Link></li>
              <li><Link to="/research#methodology" className={styles.footerLink}>Methodology</Link></li>
            </ul>
          </div>

          <div>
            <div className={styles.colHeader}>Evidence</div>
            <ul className={styles.linkList}>
              <li><Link to="/experiments" className={styles.footerLink}>All 14 Experiments</Link></li>
              <li><Link to="/experiments?category=pretraining" className={styles.footerLink}>Pretraining Trajectory</Link></li>
              <li><Link to="/experiments?category=supervised_fine_tuning" className={styles.footerLink}>SFT Forensic Audit</Link></li>
              <li><Link to="/experiments?category=diagnostic_audit" className={styles.footerLink}>Syntax vs. Execution</Link></li>
            </ul>
          </div>

          <div>
            <div className={styles.colHeader}>Model &amp; Lab</div>
            <ul className={styles.linkList}>
              <li><Link to="/model" className={styles.footerLink}>Architecture Ledger</Link></li>
              <li><Link to="/learn" className={styles.footerLink}>Transformer Learning Lab</Link></li>
              <li><Link to="/about" className={styles.footerLink}>About &amp; Environment</Link></li>
              <li><Link to="/about#references" className={styles.footerLink}>Academic Citations</Link></li>
            </ul>
          </div>
        </div>

        <div className={styles.bottomBar}>
          <div>
            &copy; 2026 QORIN. Empirical evaluation on deterministic benchmarks and local checkpoints.
          </div>
          <div>
            Single NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) &bull; Verified Provenance
          </div>
        </div>
      </div>
    </footer>
  );
};
