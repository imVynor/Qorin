import React from 'react';
import { Link } from 'react-router-dom';
import { PretrainLossChart } from '../components/visualizations/PretrainLossChart';
import { CapabilityBottleneckChart } from '../components/visualizations/CapabilityBottleneckChart';
import { SFTLossBehavioralChart } from '../components/visualizations/SFTLossBehavioralChart';
import { ControlledAblationFigure } from '../components/visualizations/ControlledAblationFigure';
import { journeyChaptersData } from '../data/journeyData';
import { MathView } from '../components/common/MathView';
import { BookOpen, Cpu, Terminal, ShieldAlert, Award } from 'lucide-react';
import styles from './Home.module.css';

export const Home: React.FC = () => {
  return (
    <main className={styles.page}>
      {/* =========================================================
          1. HERO / OPENING & RESEARCH QUESTION
          ========================================================= */}
      <section className={styles.hero}>
        <div className={styles.container}>
          <div className={styles.heroOverline}>
            <span>Machine Learning Research Monograph</span>
            <span>&bull;</span>
            <span>Foundational Model Architecture &amp; Tool Augmentation</span>
            <span>&bull;</span>
            <span>November 2026</span>
          </div>

          <h1 className={styles.heroTitle}>QORIN</h1>

          <div className={styles.heroSubtitle}>
            The Quiet Origin of Reasoning and Intelligence Networks
          </div>

          <div style={{ fontSize: 'var(--text-xl)', fontWeight: 600, color: 'var(--color-ink-primary)', marginBottom: 'var(--space-4)' }}>
            Tool-Augmented Small Language Models: Building and Evaluating a Lightweight Transformer
          </div>

          <div className={styles.heroLead}>
            An empirical investigation into decoder-only Transformer scaling, supervised instruction tuning,
            and external tool augmentation trained entirely from scratch on accessible consumer hardware.
          </div>

          {/* Central Research Question Callout Block */}
          <div className={styles.researchQuestionBlock}>
            <div className={styles.rqLabel}>
              <Award size={14} />
              <span>Central Research Question</span>
            </div>
            <blockquote className={styles.rqText}>
              &ldquo;How does increasing the capacity of a small decoder-only Transformer affect conversational LM performance, and can external retrieval compensate for smaller model capacity?&rdquo;
            </blockquote>
            <div className={styles.rqMeta}>
              <div>
                <strong>Independent Variables:</strong> Model Capacity (<MathView math="N \in [1.68 \times 10^7, 5.14 \times 10^8]" /> parameters) &bull; Tool Augmentation (BM25 Retrieval, Web Search, Python Sandbox Execution)
              </div>
              <div>
                <strong>Dependent Variables:</strong> Validation Perplexity, AST Syntax Validity, Functional Code Execution, Proposition Evidence Grounding, Citation Precision
              </div>
              <div>
                <strong>Hardware Boundary:</strong> Single NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) &bull; No cloud compute clusters or third-party pretrained weights used
              </div>
            </div>
          </div>

          {/* Key Metrics Bar */}
          <div className={styles.metricsGrid}>
            <div className={styles.metricCard}>
              <div className={styles.metricVal}>514,389,504</div>
              <div className={styles.metricLabel}>Verified Parameters</div>
              <div className={styles.metricSub}>20L &bull; 24Q/4KV &bull; GQA 6:1 &bull; 2048 ctx</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricVal}>0.8947</div>
              <div className={styles.metricLabel}>Autonomous Chaining Loss</div>
              <div className={styles.metricSub}>100 updates (3.0854 &rarr; 0.8947)</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricVal}>100.0%</div>
              <div className={styles.metricLabel}>Grounded Precision</div>
              <div className={styles.metricSub}>Held-out benchmark (15/15 cases)</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricVal}>0.0%</div>
              <div className={styles.metricLabel}>Unsupported Claims</div>
              <div className={styles.metricSub}>With Runtime Evidence Grounding</div>
            </div>
          </div>

          <div className={styles.heroLinks}>
            <Link to="/research" className={styles.heroPrimaryLink}>
              Read Research Monograph &rarr;
            </Link>
            <Link to="/experiments" className={styles.heroSecondaryLink}>
              Explore Experiment Archive &rarr;
            </Link>
            <Link to="/harness" className={styles.heroSecondaryLink}>
              Interactive Inference Harness &rarr;
            </Link>
            <Link to="/model" className={styles.heroSecondaryLink}>
              Architecture Specifications &rarr;
            </Link>
          </div>
        </div>
      </section>

      {/* =========================================================
          2. CAPACITY VS. TOOL AUGMENTATION MATRIX
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <div className={styles.kicker}>The Central Duality</div>
            <h2 className={styles.storyHeading}>Model Capacity vs. External Tool Augmentation</h2>
          </div>

          <p className={styles.storyParagraph} style={{ maxWidth: 'var(--measure-prose)' }}>
            Our research tests two orthogonal approaches to language modeling on consumer hardware:
            expanding the parametric capacity of the neural network versus augmenting a smaller network with
            deterministic execution environments and dynamic retrieval.
          </p>

          <div className={styles.capacityVsToolGrid}>
            <div className={styles.dimensionCard}>
              <div className={styles.dimensionHeader}>
                <h3 className={styles.dimensionTitle}>Dimension A: Parametric Capacity Scaling</h3>
                <span className={styles.dimensionBadge}>16.8M &rarr; 514.4M Params</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--color-ink-secondary)', margin: 0 }}>
                Investigating what language abilities emerge purely from expanding Transformer layers, attention heads, and parameter counts under causal next-token pretraining.
              </p>
              <ul className={styles.dimensionList}>
                <li><strong>Validation Perplexity Drops Monotonically:</strong> 47.53 (16M) &rarr; 22.54 (50M) &rarr; 15.08 (100M).</li>
                <li><strong>Syntax Memorization Emerges:</strong> Python AST validity increases from 77.5% (16M) to 87.5% (50M) and 96.5% (500M).</li>
                <li><strong>Capacity Ceiling on Execution:</strong> Functional code execution remains near zero (0.0% in 16M, 2.5% in 50M) regardless of parameter scaling.</li>
                <li><strong>Factual Hallucination Persists:</strong> Parametric weights suffer lossy recall, inventing false historical dates and facts.</li>
              </ul>
            </div>

            <div className={styles.dimensionCard}>
              <div className={styles.dimensionHeader}>
                <h3 className={styles.dimensionTitle}>Dimension B: External Tool Augmentation</h3>
                <span className={styles.dimensionBadge}>Web + Python + Harness</span>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--color-ink-secondary)', margin: 0 }}>
                Equipping a 514M parameter model with non-parametric BM25/Web retrieval and an isolated Python sandbox to delegate computation and dynamic fact retrieval.
              </p>
              <ul className={styles.dimensionList}>
                <li><strong>Deterministic Computation:</strong> Python sandbox execution yields 100% mathematical precision, bypassing next-token arithmetic limits.</li>
                <li><strong>Zero-Hallucination Factual Grounding:</strong> Web research + Runtime Evidence Grounding eliminates unsupported claims (0/15 vs 10/15 in baseline).</li>
                <li><strong>Autonomous Sequential Chaining:</strong> Experiment F Step 100 pilot achieved 50% second-turn Python invocation following Web observation.</li>
                <li><strong>Economic &amp; Hardware Efficiency:</strong> Enables a 514M model on an 8GB GPU to outperform multi-billion parameter models on factual precision.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* =========================================================
          2b. CONTROLLED TOOL-AUGMENTATION ABLATION (THE EMPIRICAL ANSWER)
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <div className={styles.kicker}>Empirical Resolution</div>
            <h2 className={styles.storyHeading}>Do External Tools Actually Improve Small LMs?</h2>
          </div>

          <p className={styles.storyParagraph} style={{ maxWidth: 'var(--measure-prose)' }}>
            We directly answered our central research question through a rigorous controlled ablation, holding checkpoints constant across the 100-record Stage 13 held-out benchmark:
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)', margin: 'var(--space-4) 0 var(--space-6) 0' }}>
            <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', padding: 'var(--space-4)' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>Primary 500M Finding</div>
              <div style={{ fontSize: 'var(--text-xl)', fontWeight: 700, color: 'var(--color-accent-sage, #15803d)', margin: '4px 0' }}>+21.0% Absolute Gain</div>
              <p style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', margin: 0 }}>
                Tools Disabled: <strong>43.0%</strong> &rarr; Tools Enabled: <strong>64.0%</strong> (relative +48.8%). McNemar's paired test yields χ² = 12.90, p = 0.000328 (statistically significant).
              </p>
            </div>

            <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', padding: 'var(--space-4)' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>100M Scale Finding</div>
              <div style={{ fontSize: 'var(--text-xl)', fontWeight: 700, color: 'var(--color-accent-coral, #b91c1c)', margin: '4px 0' }}>-7.0% Net Regression</div>
              <p style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', margin: 0 }}>
                Tools Disabled: <strong>36.0%</strong> &rarr; Tools Enabled: <strong>29.0%</strong>. Tools do not compensate for inadequate capacity; sub-threshold models suffer schema parsing breakdown and context distraction.
              </p>
            </div>
          </div>

          <ControlledAblationFigure figureNumber={1} />
        </div>
      </section>

      {/* =========================================================
          3. THE 9-CHAPTER RESEARCH JOURNEY OVERVIEW
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <div className={styles.kicker}>The Scientific Narrative</div>
            <h2 className={styles.storyHeading}>The 9-Chapter Research Journey</h2>
          </div>

          <p className={styles.storyParagraph} style={{ maxWidth: 'var(--measure-prose)' }}>
            Follow the chronological evolution of QORIN: from the initial motivation and baseline architectural ablations
            to model capacity scaling, retrieval augmentation, runtime harness engineering, and forensic failure analysis.
          </p>

          <div className={styles.journeyGrid}>
            {journeyChaptersData.map((chap) => (
              <Link to={`/research#${chap.slug}`} key={chap.id} className={styles.journeyCard}>
                <div className={styles.journeyCardHeader}>
                  <span className={styles.journeyCardNumber}>CH. 0{chap.id}</span>
                  <span className={styles.journeyCardTag}>{chap.tag}</span>
                </div>
                <h4 className={styles.journeyCardTitle}>{chap.title.split(': ')[1]}</h4>
                <div className={styles.journeyCardSubtitle}>{chap.subtitle}</div>
                <div className={styles.journeyCardQuestion}>&ldquo;{chap.question}&rdquo;</div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* =========================================================
          4. THE STARTING POINT (16M) & SYNTAX VS EXECUTION
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.splitLayout}>
            <div className={styles.splitNarrative}>
              <div className={styles.sectionHeader}>
                <div className={styles.kicker}>Chapter 2 &amp; 3 Finding</div>
                <h2 className={styles.storyHeading}>The disparity between syntax and execution</h2>
              </div>

              <p className={styles.storyParagraph}>
                We began with <strong>QORIN-16M</strong> (16.8M parameters, 6 layers, 384 dim), confirming stable convergence
                to 3.8614 validation loss on 100M tokens. Scaling to <strong>QORIN-50M</strong> (49.1M parameters) revealed an essential
                machine learning lesson:
              </p>

              <p className={styles.storyParagraph}>
                On held-out coding tasks, the 50M model achieved <strong>87.5% AST syntax validity</strong> and{' '}
                <strong>95.0% JSON syntax compliance</strong>. Yet actual functional execution collapsed to just{' '}
                <strong>2.5% (1 out of 40 passed)</strong>, while 16M passed 0.0%.
              </p>

              <blockquote>
                &ldquo;Syntax is a low-rank statistical regularity that small Transformers memorize easily.
                Functional execution requires state composition that parameter scaling alone fails to deliver.&rdquo;
              </blockquote>
            </div>

            <div className={styles.splitAside}>
              <div className={styles.marginalia}>
                <strong>Hardware Constraints:</strong>
                All models were trained locally on a single <strong>NVIDIA RTX 5060 Laptop GPU (8GB VRAM)</strong>.
                Zero cloud compute, zero external APIs used for model weights.
              </div>

              <div className={styles.marginalia}>
                <strong>Verified Checkpoints:</strong>
                <ul style={{ paddingLeft: '1.2rem', marginTop: '6px', lineHeight: 1.6 }}>
                  <li>QORIN-16M: 16,768,128 params (ckpt_100m)</li>
                  <li>QORIN-50M: 49,105,408 params (gqa4_250m)</li>
                  <li>QORIN-100M: 100,401,408 params (base_100m)</li>
                  <li>QORIN-500M: 514,389,504 params (ckpt_pilot_best.pt)</li>
                </ul>
              </div>
            </div>
          </div>

          <CapabilityBottleneckChart />
        </div>
      </section>

      {/* =========================================================
          5. PRETRAIN LOSS SCALING & SFT STARVATION PARADOX
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <div className={styles.kicker}>Empirical Scaling &amp; SFT Forensic</div>
            <h2 className={styles.storyHeading}>Pretraining scaling and the Starvation Paradox</h2>
          </div>

          <div className={styles.proseMeasure}>
            <p className={styles.storyParagraph}>
              Scaling from 16M to 100M across 1.5 billion tokens reduced validation perplexity from 47.53 to 15.08 (a 68.3% reduction).
              In controlled 100M-token iso-data tests, increasing parameters from 16.8M to 49.1M dropped validation loss by 0.4047 (-33.3% ppl)
              from capacity alone.
            </p>
          </div>

          <PretrainLossChart />

          <div className={styles.proseMeasure} style={{ marginTop: 'var(--space-10)' }}>
            <p className={styles.storyParagraph}>
              However, in early SFT fine-tuning (SFT v3), validation loss reached a record low of <strong>0.765</strong>, yet the model suffered
              from complete conversational collapse on simple greetings (generating synthetic Wikipedia headers).
              Forensic analysis revealed <strong>greeting starvation</strong>: greetings represented only 0.026% of training records
              (9 out of 34,700), proving that validation loss alone is an incomplete metric.
            </p>
          </div>

          <SFTLossBehavioralChart />
        </div>
      </section>

      {/* =========================================================
          6. AUTONOMOUS TOOL CHAINING & RUNTIME EVIDENCE GROUNDING
          ========================================================= */}
      <section className={styles.storySection}>
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <div className={styles.kicker}>Tool Augmentation Milestone</div>
            <h2 className={styles.storyHeading}>Autonomous Sequential Tool Chaining &amp; Grounding</h2>
          </div>

          <div className={styles.proseMeasure}>
            <p className={styles.storyParagraph}>
              In the <strong>Autonomous Sequential Tool Chaining</strong> investigation, QORIN-500M was trained for 100 updates on multi-turn sequential
              tool chaining (Web Research &rarr; Observation &rarr; Python Execution &rarr; Grounded Synthesis).
              Validation loss converged from 3.0854 to <strong>0.8947</strong>.
            </p>

            <p className={styles.storyParagraph}>
              In the <strong>blind held-out capability evaluation (100-record test set)</strong>, the autonomous chaining model demonstrated:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.8, fontSize: 'var(--text-sm)' }}>
              <li><strong>First-turn Web selection:</strong> 28/40 (70.0%) vs. 7/40 (17.5%) in parent baseline.</li>
              <li><strong>Second-turn Python invocation:</strong> 15/30 (50.0%) vs. 0/30 (0.0%) in parent baseline.</li>
              <li><strong>AST code syntax validity:</strong> 13/30 (43.3%) vs. 0/30 in parent baseline.</li>
              <li><strong>Evidence-grounded synthesis:</strong> 15/15 (100.0%) full support precision with 0.0% unsupported claims.</li>
            </ul>

            <p className={styles.storyParagraph} style={{ marginTop: 'var(--space-4)' }}>
              Subsequent <strong>runtime harness engineering</strong> instituted structured evidence representations ([E1], [E2]),
              an <code>EvidenceGroundingValidator</code> with proposition-level matching, ephemeral execution-ID isolation, and turn-scoped dialogue context.
            </p>
          </div>

          {/* Honest Active Limitations Callout */}
          <div className={styles.limitationBlock}>
            <div className={styles.limitationTitle}>
              <ShieldAlert size={14} />
              <span>Current Live System Limitations (Documented Transparently)</span>
            </div>
            <ul className={styles.limitationList}>
              <li><strong>Contradictory &amp; Insufficient Evidence:</strong> In held-out evaluation, cases with contradictory sources or missing data scored 0% success without harness fallback.</li>
              <li><strong>Affirmative Tool Confirmation:</strong> When prompted with &ldquo;Yes, execute that calculation&rdquo;, the live model currently produces conversational text rather than emitting a tool call token block.</li>
              <li><strong>Repetitive Follow-Up &amp; Reference Resolution:</strong> Complex multi-hop pronouns across turns remain brittle under low temperature settings.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* =========================================================
          7. INTERACTIVE RESEARCH CALL TO ACTION
          ========================================================= */}
      <section className={styles.storySectionNoBorder}>
        <div className={styles.container}>
          <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', padding: 'var(--space-8)' }}>
            <div className={styles.kicker}>Interactive Research Instrument</div>
            <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: 'var(--text-2xl)', margin: 'var(--space-2) 0 var(--space-4) 0' }}>
              Experience QORIN in the Research Inference Harness
            </h3>
            <p style={{ maxWidth: '640px', color: 'var(--color-ink-secondary)', fontSize: 'var(--text-sm)', lineHeight: 1.6, marginBottom: 'var(--space-6)' }}>
              Test native checkpoints, inspect live token streaming telemetry (TTFT, tok/s), observe BM25 and Web retrieval traces,
              and examine isolated Python execution outputs with fixed internal scrolling and evidence verification.
            </p>
            <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
              <Link to="/harness" className={styles.heroPrimaryLink} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <Terminal size={14} /> Launch Inference Harness &rarr;
              </Link>
              <Link to="/experiments" className={styles.heroSecondaryLink} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={14} /> Inspect Experiment Archive &rarr;
              </Link>
              <Link to="/research" className={styles.heroSecondaryLink} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                <BookOpen size={14} /> Read Research Monograph &rarr;
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
};
