import React, { useState, useMemo } from 'react';
import {
  controlledAblationSummary,
  controlledAblationCategories,
} from '../../data/chartData';
import {
  ablationTasksData,
} from '../../data/ablationTasksData';
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Search,
  ArrowRight,
} from 'lucide-react';
import styles from './ControlledAblationFigure.module.css';

interface Props {
  figureNumber?: number;
}

export const ControlledAblationFigure: React.FC<Props> = ({ figureNumber = 9 }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'inspector' | 'capacity' | 'categories' | 'efficiency'>('overview');
  const [verdictFilter, setVerdictFilter] = useState<string>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedTaskId, setSelectedTaskId] = useState<string>(ablationTasksData[3]?.id || ablationTasksData[0].id);

  // Filter tasks
  const filteredTasks = useMemo(() => {
    return ablationTasksData.filter((task) => {
      const matchVerdict =
        verdictFilter === 'all' || task.verdict === verdictFilter;
      const matchCategory =
        categoryFilter === 'all' || task.category === categoryFilter;
      const q = searchQuery.toLowerCase().trim();
      const matchQuery =
        !q ||
        task.id.toLowerCase().includes(q) ||
        task.prompt.toLowerCase().includes(q) ||
        task.target.toLowerCase().includes(q) ||
        task.toolsDisabled.response.toLowerCase().includes(q) ||
        task.toolsEnabled.response.toLowerCase().includes(q);

      return matchVerdict && matchCategory && matchQuery;
    });
  }, [verdictFilter, categoryFilter, searchQuery]);

  // Selected task
  const activeTask = useMemo(() => {
    return (
      ablationTasksData.find((t) => t.id === selectedTaskId) ||
      filteredTasks[0] ||
      ablationTasksData[0]
    );
  }, [selectedTaskId, filteredTasks]);

  const openInspectorWithFilter = (verdict: string) => {
    setVerdictFilter(verdict);
    setCategoryFilter('all');
    setSearchQuery('');
    setActiveTab('inspector');
    const firstMatching = ablationTasksData.find(
      (t) => verdict === 'all' || t.verdict === verdict
    );
    if (firstMatching) {
      setSelectedTaskId(firstMatching.id);
    }
  };

  const openInspectorForCategory = (cat: string) => {
    const categoryMapping: Record<string, string[]> = {
      'Parametric knowledge': ['CAL_DIRECT_CONCEPT'],
      'Current information': ['CAL_DIRECT_FACTUAL_WEB'],
      'Numerical computation': ['CAL_DIRECT_PROVIDED_CALC'],
      'Multi-step reasoning': ['MULTITURN_STANDARD_WEB_PYTHON'],
      'Evidence synthesis': ['SYN_FULL_SUPPORT'],
      'Partial evidence': ['SYN_PARTIAL_SUPPORT'],
      'Contradictory evidence': ['SYN_CONTRADICTORY_EVIDENCE', 'MULTITURN_CONTRADICTORY_SOURCES'],
      'Insufficient evidence': ['SYN_INSUFFICIENT_EVIDENCE'],
      'Tool discrimination': ['CAL_CONTRASTIVE_PAIRS', 'MULTITURN_WEB_SUFFICIENT_DIRECT'],
      'Conversational follow-ups': ['MULTITURN_WEB_FAILURE_RECOVERY'],
    };

    const allowed = categoryMapping[cat] || [];
    setVerdictFilter('all');
    setSearchQuery('');
    setActiveTab('inspector');
    const firstMatching = ablationTasksData.find((t) =>
      allowed.includes(t.category)
    );
    if (firstMatching) {
      setSelectedTaskId(firstMatching.id);
    }
  };

  return (
    <figure className={styles.figureWrap} aria-label="Controlled Tool-Augmentation Ablation Visualization">
      {/* Header */}
      <div className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <h3 className={styles.figureHeading}>Controlled Tool-Augmentation Ablation (Central Question)</h3>
          <span className={styles.figureSubtitle}>
            Direct empirical comparison of identical checkpoints on 100 held-out tasks: Click any metric to inspect live answers
          </span>
        </div>

        <div className={styles.toggleGroup} role="tablist" aria-label="Ablation Views">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'overview'}
            className={`${styles.toggleBtn} ${activeTab === 'overview' ? styles.toggleBtnActive : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Primary 500M Ablation
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'inspector'}
            className={`${styles.toggleBtn} ${activeTab === 'inspector' ? styles.toggleBtnActive : ''}`}
            onClick={() => setActiveTab('inspector')}
            style={{ color: activeTab === 'inspector' ? 'var(--color-accent-indigo)' : undefined }}
          >
            Realtime Answers ({ablationTasksData.length})
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'capacity'}
            className={`${styles.toggleBtn} ${activeTab === 'capacity' ? styles.toggleBtnActive : ''}`}
            onClick={() => setActiveTab('capacity')}
          >
            2×2 Capacity Matrix
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'categories'}
            className={`${styles.toggleBtn} ${activeTab === 'categories' ? styles.toggleBtnActive : ''}`}
            onClick={() => setActiveTab('categories')}
          >
            Category Breakdown
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'efficiency'}
            className={`${styles.toggleBtn} ${activeTab === 'efficiency' ? styles.toggleBtnActive : ''}`}
            onClick={() => setActiveTab('efficiency')}
          >
            Latency &amp; Efficiency
          </button>
        </div>
      </div>

      {/* TAB 1: OVERVIEW & MCNEMAR TEST */}
      {activeTab === 'overview' && (
        <div className={styles.overviewGrid}>
          {/* Success Comparison Card */}
          <div className={styles.statCard}>
            <div className={styles.statHeader}>
              <span>Primary Outcome (QORIN-500M)</span>
              <span className={styles.statBadge}>N = 100 Tasks</span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginTop: '4px' }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>Tools Disabled</div>
                <div className={styles.statNumber} style={{ color: '#64748b' }}>
                  {controlledAblationSummary.toolsDisabledPct.toFixed(1)}%
                </div>
                <div className={styles.statSub}>43/100 correct [95% CI: 34–53%]</div>
              </div>

              <div style={{ textAlign: 'center', fontSize: '18px', color: 'var(--color-ink-muted)' }}>&rarr;</div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '11px', color: 'var(--color-accent-indigo)' }}>Tools Enabled</div>
                <div className={styles.statNumber} style={{ color: 'var(--color-accent-indigo)' }}>
                  {controlledAblationSummary.toolsEnabledPct.toFixed(1)}%
                </div>
                <div className={styles.statSub}>64/100 correct [95% CI: 55–74%]</div>
              </div>
            </div>

            <div style={{ marginTop: '8px', padding: '8px', background: 'var(--color-surface)', borderRadius: '4px', border: '1px solid var(--color-rule-subtle)' }}>
              <div style={{ fontSize: '12px' }}>
                Net Gain:{' '}
                <span className={styles.statDeltaPositive}>
                  +{controlledAblationSummary.absoluteGainPct.toFixed(1)}% pts
                </span>{' '}
                <span style={{ color: 'var(--color-ink-secondary)', fontSize: '11px' }}>
                  ({controlledAblationSummary.relativeGainPct.toFixed(1)}% relative gain)
                </span>
              </div>
            </div>

            <button
              type="button"
              className={styles.toggleBtn}
              onClick={() => openInspectorWithFilter('all')}
              style={{
                marginTop: '10px',
                background: 'var(--color-accent-indigo)',
                color: '#fff',
                padding: '8px 12px',
                borderRadius: '4px',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '6px',
                fontWeight: 600,
              }}
            >
              <span>Inspect Realtime Answers (All 100)</span>
              <ArrowRight size={13} />
            </button>
          </div>

          {/* McNemar's Paired Test & Clickable Discordant Matrix */}
          <div className={styles.statCard}>
            <div className={styles.statHeader}>
              <span>Statistical Significance</span>
              <span className={styles.statBadge}>McNemar's Paired Test</span>
            </div>

            <div style={{ display: 'flex', gap: '16px', alignItems: 'baseline' }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>Chi-Square (χ²)</div>
                <div className={styles.statNumber}>{controlledAblationSummary.mcnemarChi2.toFixed(4)}</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>p-value</div>
                <div className={styles.statNumber} style={{ color: 'var(--color-accent-sage, #15803d)' }}>
                  0.000328
                </div>
              </div>
            </div>

            <div className={styles.statSub} style={{ marginTop: '4px' }}>
              <strong style={{ color: 'var(--color-accent-sage, #15803d)' }}>Statistically Significant (p &lt; 0.001)</strong>. Click any cell below to view the real answers:
            </div>

            {/* Clickable Discordant Matrix */}
            <div className={styles.matrixWrap}>
              <button
                type="button"
                className={`${styles.matrixCell} ${styles.matrixCellSuccess} ${styles.matrixCellInteractive}`}
                onClick={() => openInspectorWithFilter('TOOLS_IMPROVED')}
                title="Click to view tasks where Tools Enabled was Right"
              >
                <div className={styles.matrixLabel}>Tools Enabled Only (Improved)</div>
                <div className={styles.matrixValue} style={{ color: '#15803d' }}>
                  {controlledAblationSummary.contingency.toolsEnabledOnly} tasks &rarr;
                </div>
              </button>

              <button
                type="button"
                className={`${styles.matrixCell} ${styles.matrixCellFailure} ${styles.matrixCellInteractive}`}
                onClick={() => openInspectorWithFilter('TOOLS_REGRESSED')}
                title="Click to view tasks where Tools Disabled was Right"
              >
                <div className={styles.matrixLabel}>Tools Disabled Only (Regressed)</div>
                <div className={styles.matrixValue} style={{ color: '#b91c1c' }}>
                  {controlledAblationSummary.contingency.toolsDisabledOnly} tasks &rarr;
                </div>
              </button>

              <button
                type="button"
                className={`${styles.matrixCell} ${styles.matrixCellNeutral} ${styles.matrixCellInteractive}`}
                onClick={() => openInspectorWithFilter('BOTH_SUCCEEDED')}
                title="Click to view tasks where both were Right"
              >
                <div className={styles.matrixLabel}>Both Succeeded</div>
                <div className={styles.matrixValue}>
                  {controlledAblationSummary.contingency.bothSucceeded} tasks &rarr;
                </div>
              </button>

              <button
                type="button"
                className={`${styles.matrixCell} ${styles.matrixCellNeutral} ${styles.matrixCellInteractive}`}
                onClick={() => openInspectorWithFilter('BOTH_FAILED')}
                title="Click to view tasks where both Failed"
              >
                <div className={styles.matrixLabel}>Both Failed</div>
                <div className={styles.matrixValue}>
                  {controlledAblationSummary.contingency.bothFailed} tasks &rarr;
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: REALTIME ANSWER INSPECTOR (SIDE-BY-SIDE ANSWERS & WHICH ONE IS RIGHT) */}
      {activeTab === 'inspector' && (
        <div className={styles.inspectorContainer}>
          {/* Controls Bar */}
          <div className={styles.inspectorFilterBar}>
            <div className={styles.filterPills}>
              <button
                type="button"
                className={`${styles.filterPill} ${verdictFilter === 'all' ? styles.filterPillActive : ''}`}
                onClick={() => setVerdictFilter('all')}
              >
                All Tasks ({ablationTasksData.length})
              </button>
              <button
                type="button"
                className={`${styles.filterPill} ${verdictFilter === 'TOOLS_IMPROVED' ? styles.filterPillActive : ''}`}
                onClick={() => setVerdictFilter('TOOLS_IMPROVED')}
                style={{ borderColor: verdictFilter === 'TOOLS_IMPROVED' ? undefined : '#15803d' }}
              >
                Tools Right (+26)
              </button>
              <button
                type="button"
                className={`${styles.filterPill} ${verdictFilter === 'TOOLS_REGRESSED' ? styles.filterPillActive : ''}`}
                onClick={() => setVerdictFilter('TOOLS_REGRESSED')}
                style={{ borderColor: verdictFilter === 'TOOLS_REGRESSED' ? undefined : '#b91c1c' }}
              >
                Base Right (5)
              </button>
              <button
                type="button"
                className={`${styles.filterPill} ${verdictFilter === 'BOTH_SUCCEEDED' ? styles.filterPillActive : ''}`}
                onClick={() => setVerdictFilter('BOTH_SUCCEEDED')}
              >
                Both Right (38)
              </button>
              <button
                type="button"
                className={`${styles.filterPill} ${verdictFilter === 'BOTH_FAILED' ? styles.filterPillActive : ''}`}
                onClick={() => setVerdictFilter('BOTH_FAILED')}
              >
                Both Failed (31)
              </button>
            </div>

            <div className={styles.searchBox}>
              <Search size={13} style={{ color: 'var(--color-ink-muted)' }} />
              <input
                type="text"
                placeholder="Search prompt, answer, or ID..."
                className={styles.searchInput}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          {/* Split View: Task List on Left, Realtime Comparison on Right */}
          <div className={styles.explorerSplit}>
            {/* Task List Pane */}
            <div className={styles.taskListPane} role="listbox" aria-label="Tasks List">
              {filteredTasks.length === 0 ? (
                <div style={{ padding: 'var(--space-6)', textAlign: 'center', color: 'var(--color-ink-muted)', fontSize: '12px' }}>
                  No tasks matched your filter criteria.
                </div>
              ) : (
                filteredTasks.map((t) => {
                  const isSelected = t.id === activeTask.id;
                  const isImproved = t.verdict === 'TOOLS_IMPROVED';
                  const isRegressed = t.verdict === 'TOOLS_REGRESSED';
                  const isBothPass = t.verdict === 'BOTH_SUCCEEDED';

                  return (
                    <button
                      key={t.id}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      className={`${styles.taskItemCard} ${isSelected ? styles.taskItemCardActive : ''}`}
                      onClick={() => setSelectedTaskId(t.id)}
                    >
                      <div className={styles.taskItemHeader}>
                        <span>{t.id}</span>
                        <span>{t.category.replace(/^(CAL_|SYN_|MULTITURN_)/, '')}</span>
                      </div>
                      <div className={styles.taskItemPrompt}>{t.prompt}</div>
                      <div className={styles.taskBadgeRow}>
                        {isImproved && <span className={styles.badgeRight}>Tools Right (+1)</span>}
                        {isRegressed && <span className={styles.badgeWrong}>Base Right (-1)</span>}
                        {isBothPass && <span className={styles.badgeRight}>Both Right</span>}
                        {!isImproved && !isRegressed && !isBothPass && (
                          <span className={styles.badgeWrong}>Both Failed</span>
                        )}
                        <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginLeft: 'auto' }}>
                          {t.toolsEnabled.latencySec.toFixed(2)}s
                        </span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>

            {/* Task Realtime Comparison Detail */}
            {activeTask && (
              <div className={styles.detailPane}>
                {/* 1. WHICH ONE IS RIGHT? BANNER */}
                {activeTask.verdict === 'TOOLS_IMPROVED' && (
                  <div className={`${styles.verdictBanner} ${styles.verdictBannerImproved}`}>
                    <CheckCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <div className={styles.verdictTitle}>Tools Enabled is RIGHT (+1 Gain)</div>
                      <div className={styles.verdictText}>
                        Without tools, the model generated an incorrect or hallucinated response. With tools enabled, the model retrieved evidence and produced the verified answer.
                      </div>
                    </div>
                  </div>
                )}

                {activeTask.verdict === 'TOOLS_REGRESSED' && (
                  <div className={`${styles.verdictBanner} ${styles.verdictBannerRegressed}`}>
                    <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <div className={styles.verdictTitle}>Tools Disabled is RIGHT (Base Model Was Correct)</div>
                      <div className={styles.verdictText}>
                        The parametric baseline answered correctly without tools. Tool invocation introduced observation noise or token distraction that degraded the answer.
                      </div>
                    </div>
                  </div>
                )}

                {activeTask.verdict === 'BOTH_SUCCEEDED' && (
                  <div className={`${styles.verdictBanner} ${styles.verdictBannerBothPass}`}>
                    <CheckCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <div className={styles.verdictTitle}>Both Conditions are RIGHT</div>
                      <div className={styles.verdictText}>
                        Both the standalone parametric weights and the tool-augmented pipeline produced verified correct completions.
                      </div>
                    </div>
                  </div>
                )}

                {activeTask.verdict === 'BOTH_FAILED' && (
                  <div className={`${styles.verdictBanner} ${styles.verdictBannerBothFail}`}>
                    <XCircle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <div className={styles.verdictTitle}>Both Conditions FAILED</div>
                      <div className={styles.verdictText}>
                        Neither condition satisfied the target verification (e.g. contradictory evidence or unsupported numerical calculation).
                      </div>
                    </div>
                  </div>
                )}

                {/* 2. THE QUESTION / PROMPT */}
                <div className={styles.sectionPrompt}>
                  <div className={styles.promptLabel}>Task Prompt / Inquiry [{activeTask.id}] &bull; {activeTask.category}</div>
                  <div className={styles.promptText}>{activeTask.prompt}</div>

                  <div style={{ marginTop: '8px' }}>
                    <div className={styles.promptLabel}>Ground Truth Verification Target</div>
                    <div className={styles.targetText}>{activeTask.target}</div>
                  </div>
                </div>

                {/* 3. SIDE-BY-SIDE REALTIME ANSWERS */}
                <div className={styles.comparisonColumns}>
                  {/* Left Column: Tools Disabled */}
                  <div
                    className={styles.answerColumn}
                    style={{
                      borderColor: activeTask.toolsDisabled.success
                        ? 'rgba(21, 128, 61, 0.4)'
                        : 'rgba(185, 28, 28, 0.3)',
                    }}
                  >
                    <div className={styles.columnHeader}>
                      <span className={styles.columnTitle}>Condition A: Tools Disabled</span>
                      {activeTask.toolsDisabled.success ? (
                        <span className={styles.badgeRight}>RIGHT (Passed)</span>
                      ) : (
                        <span className={styles.badgeWrong}>WRONG (Failed)</span>
                      )}
                    </div>

                    <div className={styles.columnBody}>
                      <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                        Parametric Autonomous Response:
                      </div>
                      <div className={styles.answerOutput}>
                        {activeTask.toolsDisabled.response || '<No output generated>'}
                      </div>
                    </div>

                    <div className={styles.columnFooter}>
                      <span>Latency: {activeTask.toolsDisabled.latencySec.toFixed(3)}s</span>
                      <span>Tokens: {activeTask.toolsDisabled.tokensOut}</span>
                    </div>
                  </div>

                  {/* Right Column: Tools Enabled */}
                  <div
                    className={styles.answerColumn}
                    style={{
                      borderColor: activeTask.toolsEnabled.success
                        ? 'rgba(21, 128, 61, 0.4)'
                        : 'rgba(185, 28, 28, 0.3)',
                    }}
                  >
                    <div className={styles.columnHeader}>
                      <span className={styles.columnTitle}>Condition B: Tools Enabled</span>
                      {activeTask.toolsEnabled.success ? (
                        <span className={styles.badgeRight}>RIGHT (Passed)</span>
                      ) : (
                        <span className={styles.badgeWrong}>WRONG (Failed)</span>
                      )}
                    </div>

                    <div className={styles.columnBody}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                          Tool-Augmented Response:
                        </span>
                        {activeTask.toolsEnabled.citationsPresent && (
                          <span style={{ fontSize: '10px', color: 'var(--color-accent-indigo)', fontFamily: 'var(--font-mono)' }}>
                            [Citations Present]
                          </span>
                        )}
                      </div>
                      <div className={styles.answerOutput}>
                        {activeTask.toolsEnabled.response || '<No output generated>'}
                      </div>
                    </div>

                    <div className={styles.columnFooter}>
                      <span>Latency: {activeTask.toolsEnabled.latencySec.toFixed(3)}s</span>
                      <span>Tokens: {activeTask.toolsEnabled.tokensOut}</span>
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', background: 'var(--color-paper-subtle)', padding: '6px 10px', borderRadius: '4px', border: '1px solid var(--color-rule-subtle)' }}>
                  <strong>Evaluator Diagnostic:</strong> {activeTask.explanation}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: 2x2 CAPACITY MATRIX */}
      {activeTab === 'capacity' && (
        <div className={styles.capacityGrid}>
          {controlledAblationSummary.twoFactorCapacity.map((item) => (
            <div key={item.model} className={styles.capacityCard}>
              <div className={styles.capacityCardHeader}>
                <span className={styles.modelTitle}>{item.model}</span>
                <span className={`${styles.verdictBadge} ${item.deltaPct > 0 ? styles.verdictGain : styles.verdictDrop}`}>
                  {item.verdict}
                </span>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', margin: '4px 0' }}>
                <div className={styles.barRow}>
                  <span className={styles.barLabel}>Tools Disabled</span>
                  <div className={styles.barTrack}>
                    <div className={styles.barFillDisabled} style={{ width: `${item.toolsDisabledPct}%` }} />
                  </div>
                  <span className={styles.barValue}>{item.toolsDisabledPct.toFixed(1)}%</span>
                </div>

                <div className={styles.barRow}>
                  <span className={styles.barLabel}>Tools Enabled</span>
                  <div className={styles.barTrack}>
                    <div className={styles.barFillEnabled} style={{ width: `${item.toolsEnabledPct}%` }} />
                  </div>
                  <span className={styles.barValue}>{item.toolsEnabledPct.toFixed(1)}%</span>
                </div>
              </div>

              <div style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', background: 'var(--color-paper-subtle)', padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-rule-subtle)' }}>
                <strong>Finding:</strong> {item.explanation}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* TAB 4: CATEGORY BREAKDOWN */}
      {activeTab === 'categories' && (
        <div className={styles.categoryTableWrap}>
          <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)', marginBottom: '8px', fontFamily: 'var(--font-mono)' }}>
            Tip: Click any category row to inspect the real answers for those tasks.
          </div>
          <table className={styles.categoryTable}>
            <thead>
              <tr>
                <th>Category</th>
                <th>Sample</th>
                <th>Tools Disabled</th>
                <th>Tools Enabled</th>
                <th>Absolute Delta</th>
                <th>Relative Gain</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {controlledAblationCategories.map((cat) => (
                <tr
                  key={cat.category}
                  className={styles.categoryRowInteractive}
                  onClick={() => openInspectorForCategory(cat.category)}
                  title="Click to view real answers for this category"
                >
                  <td style={{ fontWeight: 600 }}>{cat.category}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>N={cat.sampleSize}</td>
                  <td>
                    {cat.toolsDisabledSuccess}/{cat.sampleSize} ({cat.toolsDisabledPct.toFixed(1)}%)
                  </td>
                  <td>
                    <strong>
                      {cat.toolsEnabledSuccess}/{cat.sampleSize} ({cat.toolsEnabledPct.toFixed(1)}%)
                    </strong>
                  </td>
                  <td style={{ color: cat.deltaPct > 0 ? 'var(--color-accent-sage, #15803d)' : cat.deltaPct < 0 ? 'var(--color-accent-coral, #b91c1c)' : 'var(--color-ink-muted)' }}>
                    {cat.deltaPct > 0 ? `+${cat.deltaPct.toFixed(1)}%` : `${cat.deltaPct.toFixed(1)}%`}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {cat.relativeImprovementPct > 0 ? `+${cat.relativeImprovementPct.toFixed(1)}%` : '0.0%'}
                  </td>
                  <td>
                    <span style={{ fontSize: '11px', color: 'var(--color-accent-indigo)', fontWeight: 600 }}>
                      Inspect &rarr;
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* TAB 5: EFFICIENCY & LATENCY */}
      {activeTab === 'efficiency' && (
        <div className={styles.overviewGrid}>
          <div className={styles.statCard}>
            <div className={styles.statHeader}>
              <span>Inference Latency Trade-Off</span>
              <span className={styles.statBadge}>Consumer GPU (RTX 5060)</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '6px' }}>
              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>Tools Disabled (Parametric Inference)</div>
                <div className={styles.statNumber} style={{ fontSize: '22px' }}>
                  {controlledAblationSummary.efficiency.toolsDisabledMeanLatencySec.toFixed(3)}s
                </div>
                <div className={styles.statSub}>Pure autoregressive next-token forward passes</div>
              </div>

              <div>
                <div style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>Tools Enabled (Dispatch + Execution + Synthesis)</div>
                <div className={styles.statNumber} style={{ fontSize: '22px', color: 'var(--color-accent-indigo)' }}>
                  {controlledAblationSummary.efficiency.toolsEnabledMeanLatencySec.toFixed(3)}s
                </div>
                <div className={styles.statSub}>
                  Overhead Ratio: <strong>{controlledAblationSummary.efficiency.overheadRatio.toFixed(2)}×</strong> latency cost
                </div>
              </div>
            </div>
          </div>

          <div className={styles.statCard}>
            <div className={styles.statHeader}>
              <span>Time Decomposition</span>
              <span className={styles.statBadge}>Execution Profile</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
              <div style={{ fontSize: '12px' }}>
                <strong>Model Forward Generation:</strong> {controlledAblationSummary.efficiency.meanGenerationSec.toFixed(2)}s (99.5% of turn time)
              </div>
              <div style={{ fontSize: '12px' }}>
                <strong>Tool Execution Runtime:</strong> {controlledAblationSummary.efficiency.meanToolExecutionSec.toFixed(3)}s (0.5% of turn time)
              </div>
              <div className={styles.statSub} style={{ marginTop: '8px' }}>
                Subprocess sandbox and BM25 local retrieval execute in under 15ms. The primary latency driver is multi-turn autoregressive token generation across tool observation loops.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className={styles.calloutFooter}>
        <strong>Scientific Verdict:</strong> External tools do not compensate for small model capacity below the critical parametric threshold (at 100M, performance degraded by 7.0% pts due to schema parsing errors). Rather, external tools act as a <em>capability multiplier</em> that requires sufficient base representational density (500M+ parameters) to unlock, yielding a statistically verified +21.0% absolute improvement (McNemar χ² = 12.90, p = 0.000328).
      </div>
    </figure>
  );
};
