import React, { useState, useMemo, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Search, ChevronDown, ChevronUp, Bug, Award } from 'lucide-react';
import { experimentsData } from '../data/experimentsData';
import type { ExperimentRecord } from '../types/experiment';
import { PretrainLossChart } from '../components/visualizations/PretrainLossChart';
import { CapabilityBottleneckChart } from '../components/visualizations/CapabilityBottleneckChart';
import { SFTLossBehavioralChart } from '../components/visualizations/SFTLossBehavioralChart';
import { ContextExtensionChart } from '../components/visualizations/ContextExtensionChart';
import { ArchitectureAblationFigure } from '../components/visualizations/ArchitectureAblationFigure';
import { TokenWeightingFigure } from '../components/visualizations/TokenWeightingFigure';
import { DatasetCompositionFigure } from '../components/visualizations/DatasetCompositionFigure';
import { stage15BlindEvalData, stage14PilotCurve } from '../data/chartData';
import { failureCasesData } from '../data/failuresData';
import styles from './Experiments.module.css';

interface ClusterDef {
  id: string;
  number: string;
  title: string;
  lead: string;
  prose: string;
  figure: React.ReactNode;
  experimentIds: string[];
}

export const Experiments: React.FC = () => {
  const [viewMode, setViewMode] = useState<'clusters' | 'archive' | 'failures'>('clusters');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedFailureCategory, setSelectedFailureCategory] = useState<string>('all');
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>({});
  const location = useLocation();

  useEffect(() => {
    if (location.hash) {
      const id = location.hash.replace('#', '');
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, [location.hash]);

  const toggleExpand = (id: string) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const categories: { label: string; value: string }[] = [
    { label: `All (${experimentsData.length})`, value: 'all' },
    { label: 'Pretraining', value: 'pretraining' },
    { label: 'Architecture Ablation', value: 'architecture_ablation' },
    { label: 'Context Extension', value: 'context_extension' },
    { label: 'Supervised Fine-Tuning', value: 'supervised_fine_tuning' },
    { label: 'Evaluation', value: 'evaluation' },
    { label: 'Harness Engineering', value: 'harness_engineering' },
    { label: 'Diagnostic Audit', value: 'diagnostic_audit' },
    { label: 'Algorithm Verification', value: 'algorithm_verification' },
  ];

  const experimentsMap = useMemo(() => {
    const map = new Map<string, ExperimentRecord>();
    experimentsData.forEach((exp) => map.set(exp.experiment_id, exp));
    return map;
  }, []);

  const filteredExperiments = useMemo(() => {
    return experimentsData.filter((exp) => {
      const matchesCategory =
        selectedCategory === 'all' || exp.category === selectedCategory;
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        !q ||
        exp.name.toLowerCase().includes(q) ||
        exp.codename.toLowerCase().includes(q) ||
        exp.objective.toLowerCase().includes(q) ||
        exp.outcome.toLowerCase().includes(q) ||
        (exp.consequent_decision && exp.consequent_decision.toLowerCase().includes(q)) ||
        (exp.dataset && exp.dataset.toLowerCase().includes(q));

      return matchesCategory && matchesSearch;
    });
  }, [searchQuery, selectedCategory]);

  const filteredFailures = useMemo(() => {
    return failureCasesData.filter((f) => {
      if (selectedFailureCategory === 'all') return true;
      return f.category === selectedFailureCategory;
    });
  }, [selectedFailureCategory]);

  const clusters: ClusterDef[] = [
    {
      id: 'cluster-pretraining',
      number: 'Cluster 01',
      title: 'Pretraining Trajectory & Capacity Scaling',
      lead: 'Empirical scaling trajectories across four model generations, mapping power-law loss curves and the transition from syntactic babbling to coherent prose.',
      prose:
        'Pretraining across the QORIN series demonstrated clean Chinchilla-aligned convergence. While early 5M and 25M pilots exhibited word-level repetition, scaling to 100M tokens on the 16.8M model reduced validation loss to 3.8614. The 100M parameter model consumed 1.5B tokens to reach 2.7137 loss. QORIN-500M represents the ongoing target campaign across a verified 5B-token corpus.',
      figure: <PretrainLossChart figureNumber={1} />,
      experimentIds: [
        'EXP-PRETRAIN-5M',
        'EXP-PRETRAIN-25M',
        'EXP-PRETRAIN-16M-100M',
        'EXP-PRETRAIN-100M-1.5B',
        'EXP-PRETRAIN-500M-5B',
      ],
    },
    {
      id: 'cluster-ablation',
      number: 'Cluster 02',
      title: 'Architecture Ablation at Iso-Data (50M)',
      lead: 'Direct empirical comparison between Multi-Head Attention, Grouped-Query Attention, and Deep-Thin layer allocations under fixed token budgets.',
      prose:
        'To establish the optimal architectural blueprint before large-scale compute commitments, we trained three 50M parameter topologies on identical 250M-token datasets. Deep-thin configurations suffered a catastrophic 56.6% latency penalty due to sequential kernel dispatching. Grouped-Query Attention (GQA-4) matched standard MHA validation perplexity (22.67 vs 22.54) while cutting inference KV-cache memory exactly in half.',
      figure: <ArchitectureAblationFigure figureNumber={5} />,
      experimentIds: ['EXP-PRETRAIN-50M-GQA4-250M', 'EXP-PRETRAIN-50M-DEEP-THIN-250M'],
    },
    {
      id: 'cluster-capability',
      number: 'Cluster 03',
      title: 'Capability Emergence & The Syntax-Execution Disparity',
      lead: 'Probing the divergence between grammatical fluency and functional algorithmic execution in small language models.',
      prose:
        'A central discovery in the QORIN project is the deceptive nature of cross-entropy validation loss. Models below 100M parameters readily learn surface syntax, achieving 90.0% formatting compliance on programming prompts. However, functional unit-test execution remained at 0.0%, demonstrating that formal reasoning circuits do not spontaneously emerge from next-token cross-entropy without critical parameter density or targeted execution traces.',
      figure: <CapabilityBottleneckChart figureNumber={2} />,
      experimentIds: ['EXP-SFT-16M-MIDSCALE-500'],
    },
    {
      id: 'cluster-sft',
      number: 'Cluster 04',
      title: 'Supervised Fine-Tuning & Forensic Diagnostic Audits',
      lead: 'Tracking conversational collapse, sycophancy degeneration, and the forensic reconstruction of instruction curricula.',
      prose:
        'Early fine-tuning trials on 100M checkpoints showed steady cross-entropy loss descent, yet manual inspection revealed severe behavioral collapse: sycophantic repetition, persistent self-referential conversational looping, and hallucinated identities. Forensic token auditing traced the root cause to conversational dataset imbalance (80.3% UltraChat density) and unweighted padding loss. Remediating with high-entropy synthetic multi-turn reasoning restored coherent multi-turn dialogue.',
      figure: (
        <>
          <SFTLossBehavioralChart figureNumber={3} />
          <DatasetCompositionFigure figureNumber={7} />
        </>
      ),
      experimentIds: [
        'EXP-SFT-100M-V1',
        'EXP-SFT-100M-V2',
        'EXP-SFT-100M-V3',
        'EXP-PHASE0-FORENSICS',
      ],
    },
    {
      id: 'cluster-context',
      number: 'Cluster 05',
      title: 'Context Extension & Needle-in-a-Haystack Retrieval',
      lead: 'Empirical diagnostics of Rotary Position Embedding (RoPE) frequency scaling from 1,024 to 4,096 tokens.',
      prose:
        'Context extension is frequently claimed on the basis of low evaluation perplexity alone. We extended QORIN-100M to 4,096 tokens via RoPE linear frequency scaling. While perplexity remained stable at 23.95, needle-in-a-haystack passkey retrieval revealed an actual retrieval accuracy of 4.0%. Without explicit synthetic long-range dependency training, sub-1B attention heads fail to route long-range attention signals despite mathematically valid position encodings.',
      figure: <ContextExtensionChart figureNumber={4} />,
      experimentIds: ['EXP-PILOT-100M-4K-CONTEXT'],
    },
    {
      id: 'cluster-optimization',
      number: 'Cluster 06',
      title: 'Mathematical Optimization & Token-Weighted Invariant Loss',
      lead: 'Eliminating micro-batch gradient variance through exact token-level loss normalization.',
      prose:
        'During instruction tuning with variable-length sequences, standard micro-batch averaging introduces gradient skew: short sequences receive disproportionate gradient magnitude per token. We reformulated the training objective to normalize across active, non-padded supervised tokens strictly across micro-batches, proving mathematical equivalence across gradient accumulation steps.',
      figure: <TokenWeightingFigure figureNumber={6} />,
      experimentIds: ['EXP-PHASE1A-LOSS-CORRECTION'],
    },
    {
      id: 'cluster-autonomous-tools',
      number: 'Cluster 07',
      title: 'Autonomous Tool Invocation & Closed-Loop Execution',
      lead: 'Resolving the tool-use bottleneck in small language models: progressing from 0% intent recall to 80% recall and closed-loop error recovery.',
      prose:
        'To establish autonomous tool capabilities, we fine-tuned the frozen 514M parameter foundation across controlled curricular iterations. Initial general instruction alignment eliminated prompt echoing, followed by multi-task fine-tuning that elevated coding syntax to 92.0%. A dedicated tool-calling curriculum solved the decision bottleneck (0.889 F1, 96% coding syntax), followed by negative-control calibration to prevent tool overtriggering.',
      figure: <CapabilityBottleneckChart figureNumber={7} />,
      experimentIds: [
        'EXP-PHASE2B-EXP-A',
        'EXP-PHASE2C-EXP-B',
        'EXP-PHASE2D-EXP-C',
        'EXP-PHASE2E-EXP-D',
        'EXP-PHASE2F-EXP-E',
      ],
    },
    {
      id: 'cluster-chaining-grounding',
      number: 'Cluster 08',
      title: 'Sequential Tool Chaining & Runtime Evidence Grounding',
      lead: 'Autonomous multi-step tool chaining (Web Search to Python calculation), blind capability evaluation across 100 held-out tasks, and runtime grounding contracts.',
      prose:
        'We achieved autonomous sequential tool chaining in QORIN (Web Search -> Observation -> Python Calculation -> Grounded Synthesis), with validation loss converging from 3.0854 to 0.8947 across 100 updates. In a blind capability evaluation on 100 held-out records, the model reached 70.0% first-turn web selection, 50.0% second-turn Python invocation, and 100% full-support precision. Subsequent runtime harness engineering resolved live hallucination, cross-request state bleed, and observation leakage through deterministic proposition validation.',
      figure: (
        <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', padding: 'var(--space-6)', margin: 'var(--space-6) 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--color-accent-indigo)', marginBottom: 'var(--space-3)' }}>
            <Award size={14} />
            <span>Blind Capability Evaluation &bull; Held-Out 100-Record Benchmark</span>
          </div>

          <div className={styles.evalTableWrap}>
            <table className={styles.evalTable}>
              <thead>
                <tr>
                  <th>Evaluation Task &amp; Metric</th>
                  <th>Parent Baseline</th>
                  <th>Tool Chaining Model</th>
                  <th>Absolute Delta</th>
                  <th>Empirical Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {stage15BlindEvalData.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: 600, color: 'var(--color-ink-primary)' }}>
                      {row.metric}
                    </td>
                    <td>{row.parentD} ({row.parentDValue.toFixed(1)}%)</td>
                    <td><strong>{row.pilotF100} ({row.pilotF100Value.toFixed(1)}%)</strong></td>
                    <td style={{ color: row.pilotF100Value >= row.parentDValue ? 'var(--color-accent-sage)' : 'var(--color-accent-coral)' }}>
                      {row.pilotF100Value >= row.parentDValue ? `+${(row.pilotF100Value - row.parentDValue).toFixed(1)}%` : `${(row.pilotF100Value - row.parentDValue).toFixed(1)}%`}
                    </td>
                    <td style={{ fontSize: '11px' }}>{row.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pilot Loss Table */}
          <div style={{ marginTop: 'var(--space-6)', borderTop: '1px solid var(--color-rule-subtle)', paddingTop: 'var(--space-4)' }}>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ink-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
              Sequential Tool Chaining Optimization Curve (100 Updates)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
              {stage14PilotCurve.map((pt) => (
                <div key={pt.step} style={{ padding: '6px 10px', background: 'var(--color-paper-subtle)', borderRadius: '4px', border: '1px solid var(--color-rule-subtle)' }}>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-ink-muted)' }}>Step {pt.step}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 700, color: 'var(--color-ink-primary)' }}>Loss: {pt.valLoss.toFixed(4)}</div>
                  <div style={{ fontSize: '10px', color: 'var(--color-ink-secondary)', marginTop: '2px' }}>{pt.note}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ),
      experimentIds: [
        'EXP-PHASE2G-EXP-F',
        'EXP-PHASE2G-STAGE15-EVAL',
        'EXP-PHASE2G-STAGE16-HARNESS',
      ],
    },
  ];

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <span className={`${styles.statusIndicator} ${styles.statusCompleted}`}>Completed</span>;
      case 'in_progress':
        return <span className={`${styles.statusIndicator} ${styles.statusInProgress}`}>In Progress</span>;
      default:
        return <span className={`${styles.statusIndicator} ${styles.statusArchived}`}>Archived</span>;
    }
  };

  const renderExperimentCard = (exp: ExperimentRecord) => {
    const isExpanded = !!expandedCards[exp.experiment_id];

    return (
      <article key={exp.experiment_id} id={exp.experiment_id} className={styles.entryCard}>
        <div className={styles.entryTop}>
          <div>
            <span className={styles.codename}>{exp.codename}</span>
            <span className={styles.date}>&bull; {exp.date}</span>
          </div>
          <div>{getStatusBadge(exp.status)}</div>
        </div>

        <h3 className={styles.entryTitle}>{exp.name}</h3>

        <div className={styles.specGrid}>
          <div className={styles.specItem}>
            <span className={styles.specLabel}>Model ID</span>
            <span className={styles.specVal}>{exp.model_id}</span>
          </div>
          <div className={styles.specItem}>
            <span className={styles.specLabel}>Parameters</span>
            <span className={styles.specVal}>
              {exp.parameter_count ? exp.parameter_count.toLocaleString() : 'N/A'}
            </span>
          </div>
          <div className={styles.specItem}>
            <span className={styles.specLabel}>Tokens</span>
            <span className={styles.specVal}>
              {exp.training_tokens ? `${(exp.training_tokens / 1_000_000).toFixed(1)}M` : '0M'}
            </span>
          </div>
          <div className={styles.specItem}>
            <span className={styles.specLabel}>Steps</span>
            <span className={styles.specVal}>{exp.steps || 0}</span>
          </div>
        </div>

        <p className={styles.narrativeSnippet}>
          <strong>Objective:</strong> {exp.objective}
        </p>

        <p className={styles.narrativeSnippet}>
          <strong>Outcome:</strong> {exp.outcome}
        </p>

        {/* Metrics Ticker */}
        {exp.metrics && Object.keys(exp.metrics).length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: 'var(--space-3)' }}>
            {Object.entries(exp.metrics).map(([key, val]) => (
              <span
                key={key}
                style={{
                  padding: '2px 8px',
                  background: 'var(--color-paper-subtle)',
                  borderRadius: '3px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-ink-secondary)',
                  border: '1px solid var(--color-rule-subtle)',
                }}
              >
                {key.replace(/_/g, ' ')}: <strong>{typeof val === 'number' ? (val < 1 && val > 0 ? val.toFixed(4) : val) : val}</strong>
              </span>
            ))}
          </div>
        )}

        <button
          type="button"
          className={styles.detailsToggle}
          onClick={() => toggleExpand(exp.experiment_id)}
        >
          {isExpanded ? (
            <>
              Hide Technical Details <ChevronUp size={12} />
            </>
          ) : (
            <>
              View Technical Details <ChevronDown size={12} />
            </>
          )}
        </button>

        {isExpanded && (
          <div className={styles.collapsibleDetails}>
            <div className={styles.specGrid}>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Context Length</span>
                <span className={styles.specVal}>{exp.context_length || 'N/A'}</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Tokenizer</span>
                <span className={styles.specVal}>{exp.tokenizer} ({exp.tokenizer_fingerprint.slice(0, 8)})</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Dataset Path</span>
                <span className={styles.specVal}>{exp.dataset || 'N/A'}</span>
              </div>
              <div className={styles.specItem}>
                <span className={styles.specLabel}>Checkpoint Path</span>
                <span className={styles.specVal} style={{ fontSize: '10px' }}>{exp.checkpoint || 'None'}</span>
              </div>
            </div>

            {exp.parameter_reconciliation_note && (
              <div style={{ marginTop: 'var(--space-3)', fontSize: '11px', color: 'var(--color-ink-muted)', fontStyle: 'italic' }}>
                Note: {exp.parameter_reconciliation_note}
              </div>
            )}
          </div>
        )}
      </article>
    );
  };

  return (
    <main className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Empirical Archive &bull; 22 Verified Investigations</div>
          <h1 className={styles.title}>The QORIN Experiment Archive</h1>
          <div className={styles.lead}>
            A complete chronological registry of all 22 experimental milestones, architecture ablations,
            diagnostic audits, fine-tuning curricula, blind capability evaluations, and failure autopsies.
          </div>
        </div>
      </header>

      {/* Controls Bar */}
      <section className={styles.controlsBar}>
        <div className={styles.container}>
          <div className={styles.controlsInner}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <div className={styles.segmentedBtns}>
                <button
                  type="button"
                  className={`${styles.segBtn} ${viewMode === 'clusters' ? styles.segBtnActive : ''}`}
                  onClick={() => setViewMode('clusters')}
                >
                  8 Thematic Clusters
                </button>
                <button
                  type="button"
                  className={`${styles.segBtn} ${viewMode === 'archive' ? styles.segBtnActive : ''}`}
                  onClick={() => setViewMode('archive')}
                >
                  Chronological Archive ({experimentsData.length})
                </button>
                <button
                  type="button"
                  className={`${styles.segBtn} ${viewMode === 'failures' ? styles.segBtnActive : ''}`}
                  onClick={() => setViewMode('failures')}
                  style={{ color: viewMode === 'failures' ? '#ffffff' : '#dc2626' }}
                >
                  Failure Analysis Explorer ({failureCasesData.length})
                </button>
              </div>
            </div>

            {viewMode !== 'failures' ? (
              <div className={styles.searchWrap}>
                <Search size={14} className={styles.searchIcon} />
                <input
                  type="text"
                  placeholder="Filter by name, codename, dataset..."
                  className={styles.searchInput}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            ) : null}
          </div>

          {/* Filter Categories Pill Bar (Archive Mode) */}
          {viewMode === 'archive' && (
            <div className={styles.filterGroup} style={{ marginTop: 'var(--space-3)' }}>
              {categories.map((c) => (
                <button
                  key={c.value}
                  type="button"
                  className={`${styles.filterPill} ${selectedCategory === c.value ? styles.filterPillActive : ''}`}
                  onClick={() => setSelectedCategory(c.value)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          )}

          {/* Failure Categories Pill Bar (Failures Mode) */}
          {viewMode === 'failures' && (
            <div className={styles.filterGroup} style={{ marginTop: 'var(--space-3)' }}>
              {[
                { label: `All Failures (${failureCasesData.length})`, value: 'all' },
                { label: 'Tool Execution State', value: 'tool_isolation' },
                { label: 'Evidence Grounding', value: 'model_hallucination' },
                { label: 'Intent & Routing', value: 'harness_dispatch' },
                { label: 'Cross-Turn Contamination', value: 'cross_turn_leakage' },
                { label: 'Dataset Starvation', value: 'data_starvation' },
                { label: 'Active Limitations', value: 'live_limitation' },
              ].map((c) => (
                <button
                  key={c.value}
                  type="button"
                  className={`${styles.filterPill} ${selectedFailureCategory === c.value ? styles.filterPillActive : ''}`}
                  onClick={() => setSelectedFailureCategory(c.value)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* Main Content Area */}
      <section className={styles.container}>
        {viewMode === 'clusters' && (
          <div className={styles.clusterList}>
            {clusters.map((cluster) => (
              <section key={cluster.id} id={cluster.id} className={styles.clusterSection}>
                <div className={styles.clusterKicker}>{cluster.number}</div>
                <h2 className={styles.clusterTitle}>{cluster.title}</h2>
                <div className={styles.clusterLead}>{cluster.lead}</div>
                <p className={styles.clusterProse}>{cluster.prose}</p>

                {/* Embedded Scientific Visualizations */}
                {cluster.figure}

                {/* Cluster's Experiments */}
                <div className={styles.clusterExperiments}>
                  <h4 style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', textTransform: 'uppercase', color: 'var(--color-ink-muted)' }}>
                    Associated Empirical Investigations ({cluster.experimentIds.length})
                  </h4>
                  {cluster.experimentIds
                    .map((id) => experimentsMap.get(id))
                    .filter((exp): exp is ExperimentRecord => exp !== undefined)
                    .map((exp) => renderExperimentCard(exp))}
                </div>
              </section>
            ))}
          </div>
        )}

        {viewMode === 'archive' && (
          <div className={styles.archiveList}>
            {filteredExperiments.length === 0 ? (
              <div style={{ textAlign: 'center', padding: 'var(--space-16) 0', color: 'var(--color-ink-muted)' }}>
                No experiments match your search criteria.
              </div>
            ) : (
              filteredExperiments.map((exp) => renderExperimentCard(exp))
            )}
          </div>
        )}

        {viewMode === 'failures' && (
          <div className={styles.failureExplorerSection}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#dc2626', marginBottom: 'var(--space-2)' }}>
              <Bug size={18} />
              <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: 'var(--text-2xl)', margin: 0, color: 'var(--color-ink-primary)' }}>
                Interactive Failure Analysis Explorer
              </h2>
            </div>
            <p style={{ maxWidth: 'var(--measure-prose)', color: 'var(--color-ink-secondary)', fontSize: 'var(--text-sm)', lineHeight: 1.6 }}>
              A hallmark of genuine scientific research is the rigorous forensic autopsy of reproducible failures.
              Below are 8 concrete anomalies discovered during the development of QORIN, tracing their exact root causes,
              the engineering or algorithmic remedies applied, and the core machine learning lessons learned.
            </p>

            <div className={styles.failureGrid}>
              {filteredFailures.map((f) => (
                <div key={f.id} className={styles.failureCard}>
                  <div className={styles.failureHeader}>
                    <h3 className={styles.failureTitle}>{f.title}</h3>
                    <span className={styles.failureCategoryBadge}>{f.categoryLabel}</span>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-ink-muted)' }}>
                      {f.investigationContext}
                    </span>
                    <span
                      className={`${styles.failureStatusBadge} ${
                        f.status === 'active_limitation' ? styles.statusLimitation : styles.statusResolved
                      }`}
                    >
                      {f.status === 'active_limitation' ? 'Active Limitation' : 'Resolved'}
                    </span>
                  </div>

                  <div className={styles.failureBox}>
                    <div className={styles.failureBoxLabel}>Reproduction Prompt:</div>
                    <div className={styles.failureBoxContent} style={{ fontStyle: 'italic' }}>&ldquo;{f.query}&rdquo;</div>
                  </div>

                  <div className={styles.failureBox} style={{ background: 'rgba(239, 68, 68, 0.04)', borderColor: 'rgba(239, 68, 68, 0.2)' }}>
                    <div className={styles.failureBoxLabel} style={{ color: '#dc2626' }}>Observed Failure Output:</div>
                    <div className={styles.failureBoxContent} style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{f.observedOutput}</div>
                  </div>

                  <div className={styles.failureBox}>
                    <div className={styles.failureBoxLabel}>Forensic Root Cause:</div>
                    <div className={styles.failureBoxContent}>{f.rootCause}</div>
                  </div>

                  <div className={styles.failureBox} style={{ background: 'rgba(34, 197, 94, 0.04)', borderColor: 'rgba(34, 197, 94, 0.2)' }}>
                    <div className={styles.failureBoxLabel} style={{ color: '#16a34a' }}>Harness / Algorithmic Remedy:</div>
                    <div className={styles.failureBoxContent}>{f.remedy}</div>
                  </div>

                  <div className={styles.takeawayBox}>
                    <strong>Scientific Takeaway:</strong> {f.scientificTakeaway}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
};
