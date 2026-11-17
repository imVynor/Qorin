import React, { useState, useEffect } from 'react';
import type { TrainingRunSummary, TrainingRunDetail, MetricSeriesPoint } from '../../types/harness';
import { getTrainingRuns, getTrainingRunDetail } from '../../services/api';
import { Flame, AlertCircle, RefreshCw } from 'lucide-react';
import styles from './Harness.module.css';

export const TrainingDashboardView: React.FC = () => {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('qorin-500m-pretraining');
  const [runDetail, setRunDetail] = useState<TrainingRunDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadRuns();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      loadRunDetail(selectedRunId);
    }
  }, [selectedRunId]);

  const loadRuns = async () => {
    try {
      const data = await getTrainingRuns();
      setRuns(data);
      if (data.length > 0 && !selectedRunId) {
        setSelectedRunId(data[0].id);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch runs');
    }
  };

  const loadRunDetail = async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const detail = await getTrainingRunDetail(id);
      setRunDetail(detail);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch run details');
    } finally {
      setLoading(false);
    }
  };

  // Helper to render an SVG line chart
  const renderLineChart = (
    series: MetricSeriesPoint[],
    xKey: 'step' | 'tokens',
    yKey: 'loss' | 'lr' | 'gpu_temp_c' | 'gpu_power_w',
    _yLabel: string,
    strokeColor: string,
    formatY?: (v: number) => string
  ) => {
    const validPoints = series.filter(p => p[yKey] !== undefined && p[yKey] !== null);
    if (validPoints.length < 2) {
      return (
        <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-ink-muted)', fontSize: '12px' }}>
          Insufficient telemetry series data points.
        </div>
      );
    }

    const minX = validPoints[0][xKey];
    const maxX = validPoints[validPoints.length - 1][xKey] || 1;
    const yValues = validPoints.map(p => (p[yKey] as number));
    let minY = Math.min(...yValues);
    let maxY = Math.max(...yValues);
    if (minY === maxY) {
      minY = minY * 0.9;
      maxY = maxY * 1.1;
    }
    const yPad = (maxY - minY) * 0.08;
    minY = Math.max(0, minY - yPad);
    maxY = maxY + yPad;

    const width = 500;
    const height = 200;
    const padL = 50;
    const padR = 20;
    const padT = 20;
    const padB = 30;

    const scaleX = (x: number) => padL + ((x - minX) / (maxX - minX || 1)) * (width - padL - padR);
    const scaleY = (y: number) => height - padB - ((y - minY) / (maxY - minY || 1)) * (height - padT - padB);

    const pathData = validPoints
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p[xKey]).toFixed(1)} ${scaleY(p[yKey] as number).toFixed(1)}`)
      .join(' ');

    const valPoints = series.filter(p => p.val_loss !== undefined && yKey === 'loss');

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className={styles.chartSvg} preserveAspectRatio="none">
        {/* Horizontal gridlines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
          const yVal = minY + (maxY - minY) * ratio;
          const yPos = scaleY(yVal);
          return (
            <g key={ratio}>
              <line
                x1={padL}
                y1={yPos}
                x2={width - padR}
                y2={yPos}
                stroke="var(--color-rule-subtle)"
                strokeDasharray="2,2"
                strokeWidth="1"
              />
              <text
                x={padL - 6}
                y={yPos + 3}
                fill="var(--color-ink-muted)"
                fontSize="9"
                fontFamily="var(--font-mono)"
                textAnchor="end"
              >
                {formatY ? formatY(yVal) : yVal.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* X axis endpoints */}
        <text
          x={padL}
          y={height - 8}
          fill="var(--color-ink-muted)"
          fontSize="9"
          fontFamily="var(--font-mono)"
          textAnchor="start"
        >
          {xKey === 'tokens' ? `${(minX / 1e6).toFixed(0)}M tok` : `Step ${minX}`}
        </text>
        <text
          x={width - padR}
          y={height - 8}
          fill="var(--color-ink-muted)"
          fontSize="9"
          fontFamily="var(--font-mono)"
          textAnchor="end"
        >
          {xKey === 'tokens' ? `${(maxX / 1e6).toFixed(1)}M tok` : `Step ${maxX}`}
        </text>

        {/* Train series path */}
        <path
          d={pathData}
          fill="none"
          stroke={strokeColor}
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Validation loss markers if applicable */}
        {valPoints.map((vp, idx) => (
          <circle
            key={idx}
            cx={scaleX(vp[xKey])}
            cy={scaleY(vp.val_loss as number)}
            r="3.5"
            fill="var(--color-accent-rust)"
            stroke="#fff"
            strokeWidth="1"
          />
        ))}
      </svg>
    );
  };

  return (
    <div style={{ marginTop: 'var(--space-6)' }}>
      {/* Run Selector Navigation */}
      <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', marginBottom: 'var(--space-6)' }}>
        {runs.map((r) => (
          <button
            key={r.id}
            type="button"
            onClick={() => setSelectedRunId(r.id)}
            className={`${styles.tabBtn} ${selectedRunId === r.id ? styles.tabBtnActive : ''}`}
            style={{ borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule)' }}
          >
            <strong>{r.name}</strong>
            <span style={{ fontSize: '10px', color: 'var(--color-ink-muted)' }}>({r.generation})</span>
          </button>
        ))}
      </div>

      {loading && (
        <div style={{ padding: 'var(--space-8)', textAlign: 'center', color: 'var(--color-ink-muted)' }}>
          <RefreshCw size={20} className="animate-spin" style={{ margin: '0 auto var(--space-2)' }} />
          Loading verified telemetry artifacts...
        </div>
      )}

      {error && (
        <div style={{ padding: 'var(--space-6)', background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-xs)', marginBottom: 'var(--space-6)', fontSize: '13px', color: 'var(--color-ink-secondary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, color: 'var(--color-ink-primary)', marginBottom: '6px' }}>
            <AlertCircle size={16} style={{ color: 'var(--color-accent-amber)' }} />
            <span>Telemetry Backend Service Offline</span>
          </div>
          <p style={{ margin: 0, lineHeight: 1.6 }}>
            Live GPU hardware telemetry and training run streams require the local API service. Verified pretraining scaling curves, loss trajectories, and benchmark evaluations are statically documented in the <a href="#/experiments" style={{ color: 'var(--color-accent-indigo)', textDecoration: 'underline' }}>Experiments Archive</a> and <a href="#/research" style={{ color: 'var(--color-accent-indigo)', textDecoration: 'underline' }}>Research Monograph</a>.
          </p>
        </div>
      )}

      {runDetail && !loading && (
        <div>
          {/* Run Header Card */}
          <div className={styles.topBar}>
            <div>
              <div className={styles.kicker}>{runDetail.generation} Telemetry Record</div>
              <h2 style={{ fontSize: '20px', fontFamily: 'var(--font-serif)', margin: '2px 0 6px' }}>
                {runDetail.name}
              </h2>
              <div style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', fontFamily: 'var(--font-mono)' }}>
                {runDetail.architecture} • {runDetail.context_length} Context Window • {(runDetail.parameters / 1e6).toFixed(1)}M Parameters
              </div>
            </div>

            <div style={{ display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
              <div className={styles.metricCard}>
                <div className={styles.metricCardTitle}>Tokens Processed</div>
                <div className={styles.metricCardVal}>
                  {(runDetail.tokens_processed / 1e6).toFixed(1)}M
                  <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)' }}> / {(runDetail.total_tokens_target / 1e6).toFixed(0)}M</span>
                </div>
              </div>

              <div className={styles.metricCard}>
                <div className={styles.metricCardTitle}>Best Validation</div>
                <div className={styles.metricCardVal}>
                  {runDetail.best_val_loss ? runDetail.best_val_loss.toFixed(4) : '—'}
                  {runDetail.best_val_perplexity && (
                    <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)' }}> (PPL {runDetail.best_val_perplexity.toFixed(1)})</span>
                  )}
                </div>
              </div>

              <div className={styles.metricCard}>
                <div className={styles.metricCardTitle}>Current Step</div>
                <div className={styles.metricCardVal}>
                  {runDetail.current_step.toLocaleString()}
                  <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)' }}> / {runDetail.target_steps.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Duty Cycle Card if present (500M Campaign) */}
          {runDetail.duty_cycle_telemetry && (
            <div className={styles.dutyCycleCard}>
              <div style={{ maxWidth: '340px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-accent-indigo)', fontWeight: 600, fontSize: '12px', fontFamily: 'var(--font-mono)' }}>
                  <Flame size={14} /> Thermal Management & Duty Cycle
                </div>
                <div style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', marginTop: '4px' }}>
                  Hardware protection schedule: <strong>105m active training / 15m cooldown</strong>.
                  Completed {runDetail.duty_cycle_telemetry.completed_cycles} thermal safety cycles on RTX 5060 Laptop GPU.
                </div>
              </div>

              <div className={styles.dutyProgressWrap}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>
                  <span>5B Token Campaign Progress</span>
                  <span>{runDetail.progress_pct.toFixed(2)}%</span>
                </div>
                <div className={styles.progressBarBg}>
                  <div className={styles.progressBarFill} style={{ width: `${Math.max(1, runDetail.progress_pct)}%` }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--color-ink-muted)' }}>
                  <span>Step {runDetail.current_step} ({(runDetail.tokens_processed / 1e6).toFixed(1)}M tokens)</span>
                  <span>Target: 5,000,000,000 tokens</span>
                </div>
              </div>

              <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
                <div style={{ background: 'var(--color-surface)', padding: '6px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                  <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>LATEST GPU TEMP</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: runDetail.duty_cycle_telemetry.latest_temp_c > 75 ? 'var(--color-accent-rust)' : 'var(--color-accent-sage)' }}>
                    {runDetail.duty_cycle_telemetry.latest_temp_c.toFixed(1)} °C
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface)', padding: '6px 12px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
                  <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>LATEST POWER</div>
                  <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--color-ink-primary)' }}>
                    {runDetail.duty_cycle_telemetry.latest_power_w.toFixed(1)} W
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SVG Charts Grid */}
          <div className={styles.chartsGrid}>
            <div className={styles.chartCard}>
              <div className={styles.chartCardTitle}>Cross-Entropy Loss vs Steps</div>
              <div className={styles.chartCardSubtitle}>
                Empirical training loss trajectory (solid line) with validation evaluation markers.
              </div>
              {renderLineChart(runDetail.series, 'step', 'loss', 'Loss', 'var(--color-accent-indigo)')}
            </div>

            <div className={styles.chartCard}>
              <div className={styles.chartCardTitle}>Learning Rate Trajectory</div>
              <div className={styles.chartCardSubtitle}>
                Warmup phase followed by Cosine Annealing decay down to 10% minimum.
              </div>
              {renderLineChart(runDetail.series, 'step', 'lr', 'Learning Rate', 'var(--color-accent-sage)', (v) => v.toExponential(1))}
            </div>

            {runDetail.duty_cycle_telemetry && (
              <>
                <div className={styles.chartCard}>
                  <div className={styles.chartCardTitle}>GPU Core Temperature (°C)</div>
                  <div className={styles.chartCardSubtitle}>
                    Thermal telemetry sampled during active gradient descent windows.
                  </div>
                  {renderLineChart(runDetail.series, 'step', 'gpu_temp_c', 'Temp (°C)', 'var(--color-accent-rust)', (v) => `${v.toFixed(0)}°C`)}
                </div>

                <div className={styles.chartCard}>
                  <div className={styles.chartCardTitle}>GPU Total Board Power (Watts)</div>
                  <div className={styles.chartCardSubtitle}>
                    Real-time power dissipation tracked under 115W TGP ceiling.
                  </div>
                  {renderLineChart(runDetail.series, 'step', 'gpu_power_w', 'Power (W)', 'var(--color-ink-secondary)', (v) => `${v.toFixed(0)}W`)}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
