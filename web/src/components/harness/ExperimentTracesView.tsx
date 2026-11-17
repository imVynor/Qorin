import React, { useState, useEffect } from 'react';
import type { ExperimentTraceRecord } from '../../types/harness';
import { getExperiments, getExportUrl } from '../../services/api';
import { Download, Search, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react';
import styles from './Harness.module.css';

export const ExperimentTracesView: React.FC = () => {
  const [traces, setTraces] = useState<ExperimentTraceRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);
  const [filterQuery, setFilterQuery] = useState<string>('');

  useEffect(() => {
    loadTraces();
  }, []);

  const loadTraces = async () => {
    setLoading(true);
    try {
      const data = await getExperiments(100);
      setTraces(data);
    } catch (err) {
      console.error('Failed to load experiment traces', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredTraces = traces.filter((t) => {
    if (!filterQuery) return true;
    const q = filterQuery.toLowerCase();
    return (
      t.model_id.toLowerCase().includes(q) ||
      t.prompt.toLowerCase().includes(q) ||
      t.response_text.toLowerCase().includes(q)
    );
  });

  // Calculate statistics
  const totalTraces = traces.length;
  const avgTTFT = totalTraces > 0
    ? traces.reduce((acc, t) => acc + (t.ttft_ms || 0), 0) / totalTraces
    : 0;
  const avgSpeed = totalTraces > 0
    ? traces.reduce((acc, t) => acc + (t.tokens_per_second || 0), 0) / totalTraces
    : 0;
  const retrievalCount = traces.filter((t) => t.retrieval_trace?.triggered).length;
  const toolsCount = traces.reduce((acc, t) => acc + (t.tool_traces?.length || 0), 0);

  return (
    <div style={{ marginTop: 'var(--space-6)' }}>
      {/* Metrics Banner */}
      <div className={styles.metricsGrid}>
        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Recorded Traces</div>
          <div className={styles.metricCardVal}>{totalTraces}</div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Persistent inference runs logged
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Mean TTFT</div>
          <div className={styles.metricCardVal}>{avgTTFT.toFixed(1)} ms</div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Prompt processing & prefill
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>Mean Throughput</div>
          <div className={styles.metricCardVal}>{avgSpeed.toFixed(1)} tok/s</div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            RTX 5060 autoregressive decode
          </div>
        </div>

        <div className={styles.metricCard}>
          <div className={styles.metricCardTitle}>RAG & Tool Calls</div>
          <div className={styles.metricCardVal}>
            {retrievalCount} <span style={{ fontSize: '11px', fontWeight: 400, color: 'var(--color-ink-muted)' }}>RAG / {toolsCount} tools</span>
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
            Non-parametric actions executed
          </div>
        </div>
      </div>

      {/* Traces Table Container */}
      <div className={styles.tracesWrap}>
        <div className={styles.tracesToolbar}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flex: 1, minWidth: '220px' }}>
            <Search size={14} color="var(--color-ink-muted)" />
            <input
              type="text"
              placeholder="Search traces by model, prompt, or output..."
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              style={{
                background: 'transparent',
                border: 'none',
                outline: 'none',
                fontFamily: 'var(--font-sans)',
                fontSize: '12px',
                width: '100%',
                color: 'var(--color-ink-primary)',
              }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <button
              type="button"
              onClick={loadTraces}
              className={styles.clearBtn}
              title="Refresh trace records"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
            </button>
            <a
              href={getExportUrl('json')}
              download="qorin_experiment_traces.json"
              className={styles.presetBtn}
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
            >
              <Download size={12} /> Export JSON
            </a>
            <a
              href={getExportUrl('jsonl')}
              download="qorin_experiment_traces.jsonl"
              className={styles.presetBtn}
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px' }}
            >
              <Download size={12} /> Export JSONL
            </a>
          </div>
        </div>

        <div className={styles.tableScroll}>
          <table className={styles.tracesTable}>
            <thead>
              <tr>
                <th style={{ width: '130px' }}>Timestamp</th>
                <th style={{ width: '150px' }}>Model</th>
                <th>Prompt</th>
                <th style={{ width: '60px' }}>Tokens</th>
                <th style={{ width: '80px' }}>TTFT</th>
                <th style={{ width: '80px' }}>Speed</th>
                <th style={{ width: '80px' }}>RAG</th>
                <th style={{ width: '80px' }}>Tools</th>
                <th style={{ width: '40px' }}></th>
              </tr>
            </thead>
            <tbody>
              {filteredTraces.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: 'var(--space-8)', color: 'var(--color-ink-muted)' }}>
                    {loading
                      ? 'Loading traces...'
                      : 'No live inference traces recorded in this session. Traces are captured dynamically when requests are processed by the active local backend.'}
                  </td>
                </tr>
              ) : (
                filteredTraces.map((trace) => {
                  const isExpanded = expandedTraceId === trace.id;
                  const dateStr = new Date(trace.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                  return (
                    <React.Fragment key={trace.id}>
                      <tr
                        className={styles.tracesTableRow}
                        onClick={() => setExpandedTraceId(isExpanded ? null : trace.id)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--color-ink-secondary)' }}>
                          {dateStr}
                        </td>
                        <td>
                          <strong style={{ fontSize: '11px', fontFamily: 'var(--font-mono)' }}>{trace.model_id}</strong>
                        </td>
                        <td style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {trace.prompt}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{trace.token_count}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{trace.ttft_ms ? `${trace.ttft_ms.toFixed(0)} ms` : '—'}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{trace.tokens_per_second ? `${trace.tokens_per_second.toFixed(1)}/s` : '—'}</td>
                        <td>
                          {trace.retrieval_trace?.triggered ? (
                            <span className={`${styles.pillBadge} ${styles.pillBadgeTrue}`}>
                              {trace.retrieval_trace.retrieved_chunks?.length || 0} chunks
                            </span>
                          ) : (
                            <span className={`${styles.pillBadge} ${styles.pillBadgeFalse}`}>Off</span>
                          )}
                        </td>
                        <td>
                          {trace.tool_traces && trace.tool_traces.length > 0 ? (
                            <span className={`${styles.pillBadge} ${styles.pillBadgeTrue}`}>
                              {trace.tool_traces.length} calls
                            </span>
                          ) : (
                            <span className={`${styles.pillBadge} ${styles.pillBadgeFalse}`}>0</span>
                          )}
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                        </td>
                      </tr>

                      {/* Expanded Trace Detail Drawer */}
                      {isExpanded && (
                        <tr>
                          <td colSpan={9} style={{ background: 'var(--color-paper)', padding: 'var(--space-4)' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
                              <div>
                                <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                                  Full Prompt:
                                </div>
                                <div style={{ fontSize: '12px', background: 'var(--color-paper-subtle)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', marginTop: '4px', whiteSpace: 'pre-wrap' }}>
                                  {trace.prompt}
                                </div>
                              </div>

                              {trace.retrieval_trace?.triggered && (
                                <div>
                                  <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-indigo)', textTransform: 'uppercase' }}>
                                    Injected Retrieval Context ({trace.retrieval_trace.retrieved_chunks?.length} chunks, {trace.retrieval_trace.retrieval_latency_ms.toFixed(1)} ms):
                                  </div>
                                  <div className={styles.promptInspectionBox} style={{ marginTop: '4px' }}>
                                    {trace.retrieval_trace.injected_context}
                                  </div>
                                </div>
                              )}

                              {trace.tool_traces && trace.tool_traces.length > 0 && (
                                <div>
                                  <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-sage)', textTransform: 'uppercase' }}>
                                    Tool Invocations ({trace.tool_traces.length}):
                                  </div>
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '4px' }}>
                                    {trace.tool_traces.map((tt, i) => (
                                      <div key={i} style={{ fontSize: '11px', background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', padding: '8px', borderRadius: 'var(--radius-xs)' }}>
                                        <strong>{tt.tool_name}</strong> ({tt.latency_ms.toFixed(0)} ms) — Status: <em>{tt.status}</em>
                                        <pre style={{ margin: '4px 0 0', fontSize: '10px', background: 'var(--color-surface)', padding: '4px 6px', overflowX: 'auto' }}>
                                          {JSON.stringify(tt.output_payload, null, 2)}
                                        </pre>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              <div>
                                <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', textTransform: 'uppercase' }}>
                                  Generated Output ({trace.token_count} tokens):
                                </div>
                                <div style={{ fontSize: '12px', background: 'var(--color-paper-subtle)', padding: '8px 12px', borderRadius: 'var(--radius-xs)', marginTop: '4px', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                                  {trace.response_text}
                                </div>
                              </div>

                              <div style={{ display: 'flex', gap: 'var(--space-4)', fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                                <span>Trace ID: {trace.id}</span>
                                <span>Total Latency: {trace.total_latency_ms?.toFixed(0)} ms</span>
                                <span>Config: {JSON.stringify(trace.generation_config)}</span>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
