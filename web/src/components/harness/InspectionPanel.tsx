import React, { useState } from 'react';
import type { TelemetryInfo, ModelInfo } from '../../types/harness';
import { Activity, Eye, ChevronDown, ChevronUp } from 'lucide-react';
import styles from './Harness.module.css';

interface InspectionPanelProps {
  telemetry: TelemetryInfo | null;
  activeModel?: ModelInfo | null;
  lastStats?: {
    ttft_ms?: number;
    tok_per_sec?: number;
    latency_ms?: number;
    token_count?: number;
    injected_prompt?: string;
  };
}

export const InspectionPanel: React.FC<InspectionPanelProps> = ({
  telemetry,
  lastStats,
}) => {
  const [promptOpen, setPromptOpen] = useState(false);

  const qorin = telemetry?.qorin;
  const vramAllocated = qorin?.vram_allocated_mb || 0;
  const vramTotal = qorin?.vram_total_mb || 8188;
  const vramPct = Math.min(100, Math.round((vramAllocated / (vramTotal || 1)) * 100));

  return (
    <div className={styles.inspectionDrawer}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Activity size={16} color="var(--color-accent-indigo)" />
          <h3 style={{ fontSize: '12px', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.05em', margin: 0 }}>
            Runtime Telemetry & Hardware Inspector
          </h3>
        </div>
        <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
          Device: <strong style={{ color: 'var(--color-ink-primary)' }}>{qorin?.device_name || (qorin?.cuda_available ? 'CUDA (Active)' : 'CPU')}</strong>
        </div>
      </div>

      <div className={styles.metricsGrid}>
        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>VRAM Allocation</div>
          <div className={styles.metricCardVal}>
            {vramAllocated.toFixed(1)} <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)' }}>/ {vramTotal.toFixed(0)} MB</span>
          </div>
          <div className={styles.progressBarBg} style={{ height: '4px', marginTop: '6px' }}>
            <div
              className={styles.progressBarFill}
              style={{
                width: `${vramPct}%`,
                background: vramPct > 80 ? 'var(--color-accent-rust)' : 'var(--color-accent-indigo)',
              }}
            />
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Time to First Token (TTFT)</div>
          <div className={styles.metricCardVal}>
            {lastStats?.ttft_ms !== undefined ? `${lastStats.ttft_ms.toFixed(0)} ms` : '—'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Prompt processing latency
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Generation Speed</div>
          <div className={styles.metricCardVal}>
            {lastStats?.tok_per_sec !== undefined ? `${lastStats.tok_per_sec.toFixed(1)} tok/s` : '—'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            Autoregressive decode throughput
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Total Execution</div>
          <div className={styles.metricCardVal}>
            {lastStats?.latency_ms !== undefined ? `${(lastStats.latency_ms / 1000).toFixed(2)} s` : '—'}
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '4px' }}>
            {lastStats?.token_count !== undefined ? `${lastStats.token_count} tokens generated` : 'Awaiting prompt'}
          </div>
        </div>
      </div>

      {lastStats?.injected_prompt && (
        <div style={{ marginTop: 'var(--space-3)' }}>
          <button
            type="button"
            onClick={() => setPromptOpen(!promptOpen)}
            style={{
              background: 'transparent',
              border: 'none',
              padding: '4px 0',
              fontFamily: 'var(--font-mono)',
              fontSize: '11px',
              color: 'var(--color-accent-indigo)',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <Eye size={13} />
            {promptOpen ? 'Hide Injected Model Prompt' : 'Inspect Exact Injected Model Prompt'}
            {promptOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>

          {promptOpen && (
            <div className={styles.promptInspectionBox} style={{ marginTop: 'var(--space-2)' }}>
              {lastStats.injected_prompt}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
