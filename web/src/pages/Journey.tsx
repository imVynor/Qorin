import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { journeyChaptersData } from '../data/journeyData';
import { Terminal, Cpu } from 'lucide-react';
import styles from './Journey.module.css';

export const Journey: React.FC = () => {
  const [activeChapterSlug, setActiveChapterSlug] = useState<string>('the-question');

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 200;
      for (const chap of journeyChaptersData) {
        const el = document.getElementById(chap.slug);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPosition >= top && scrollPosition < top + height) {
            setActiveChapterSlug(chap.slug);
            break;
          }
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <main className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Chronological Scientific Monograph</div>
          <h1 className={styles.title}>The QORIN Research Journey</h1>
          <div className={styles.lead}>
            An honest, 9-chapter narrative documenting the evolution of our thinking: why we formulated
            the research question, how we designed and scaled the Transformer models, why retrieval became mandatory,
            how the harness evolved, the controlled tool ablation verdict, and the next research frontier at five billion tokens.
          </div>
        </div>
      </header>

      <div className={styles.container}>
        <div className={styles.journeyLayout}>
          {/* Sticky Navigation Timeline */}
          <aside className={styles.timelineNav}>
            <div className={styles.navHeading}>{journeyChaptersData.length} Research Chapters</div>
            {journeyChaptersData.map((chap) => (
              <a
                key={chap.id}
                href={`#${chap.slug}`}
                className={`${styles.navItem} ${activeChapterSlug === chap.slug ? styles.navItemActive : ''}`}
                onClick={() => setActiveChapterSlug(chap.slug)}
              >
                <span className={styles.chapterNumber}>0{chap.id}</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {chap.title.split(': ')[1]}
                </span>
              </a>
            ))}

            <div style={{ marginTop: 'var(--space-4)', borderTop: '1px solid var(--color-rule-subtle)', paddingTop: 'var(--space-4)' }}>
              <Link
                to="/harness"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-accent-indigo)',
                  textDecoration: 'none',
                  fontWeight: 600,
                }}
              >
                <Terminal size={12} />
                <span>Test in Harness &rarr;</span>
              </Link>
            </div>
          </aside>

          {/* Chapters Feed */}
          <section className={styles.chaptersFeed}>
            {journeyChaptersData.map((chap) => (
              <article key={chap.id} id={chap.slug} className={styles.chapterArticle}>
                <div className={styles.chapterMetaRow}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '12px', fontWeight: 700, color: 'var(--color-accent-indigo)' }}>
                      CHAPTER 0{chap.id}
                    </span>
                    <span className={styles.chapterTag}>{chap.tag}</span>
                  </div>
                </div>

                <h2 className={styles.chapterTitle}>{chap.title}</h2>
                <div className={styles.chapterSubtitle}>{chap.subtitle}</div>

                {/* Central Inquiry Callout */}
                <div className={styles.questionCallout}>
                  <strong>The Underlying Question</strong>
                  &ldquo;{chap.question}&rdquo;
                </div>

                {/* Hypothesis Callout */}
                <div className={styles.hypothesisCallout}>
                  <strong>Working Hypothesis:</strong> {chap.keyHypothesis}
                </div>

                {/* Empirical Evidence */}
                <div className={styles.evidenceSection}>
                  <div className={styles.evidenceSectionTitle}>Empirical Evidence &amp; Observations</div>
                  <ul className={styles.evidenceList}>
                    {chap.evidence.map((ev, evIdx) => (
                      <li key={evIdx}>{ev}</li>
                    ))}
                  </ul>
                </div>

                {/* Architectural Decisions & Findings */}
                <div className={styles.decisionsGrid}>
                  <div className={styles.decisionBox}>
                    <strong>Key Findings</strong>
                    <ul style={{ margin: 0, paddingLeft: '14px', lineHeight: 1.5 }}>
                      {chap.findings.map((f, fIdx) => (
                        <li key={fIdx} style={{ marginTop: '4px' }}>{f}</li>
                      ))}
                    </ul>
                  </div>

                  <div className={styles.decisionBox}>
                    <strong>Architectural Decisions</strong>
                    <ul style={{ margin: 0, paddingLeft: '14px', lineHeight: 1.5 }}>
                      {chap.architecturalDecisions.map((d, dIdx) => (
                        <li key={dIdx} style={{ marginTop: '4px' }}>{d}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Scientific Insight Box */}
                <div className={styles.es335TakeawayBox}>
                  <div className={styles.takeawayLabel}>Core Scientific Insight</div>
                  <div>{chap.es335Takeaway}</div>
                </div>
              </article>
            ))}

            {/* Bottom Actions */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 'var(--space-8)', background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
              <div>
                <h4 style={{ margin: 0, fontFamily: 'var(--font-serif)', fontSize: 'var(--text-lg)', color: 'var(--color-ink-primary)' }}>
                  Ready to inspect the experimental data?
                </h4>
                <p style={{ margin: '4px 0 0', fontSize: '13px', color: 'var(--color-ink-secondary)' }}>
                  Examine all 22 experiments, the held-out blind capability evaluation, and interactive failure autopsies.
                </p>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
                <Link to="/experiments" className={styles.navItemActive} style={{ padding: '8px 16px', borderRadius: '4px', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                  <Cpu size={14} /> Experiment Archive
                </Link>
                <Link to="/harness" style={{ padding: '8px 16px', borderRadius: '4px', background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule)', color: 'var(--color-ink-primary)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600 }}>
                  <Terminal size={14} /> Inference Harness
                </Link>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
};
