import React, { useState } from 'react';
import type { ModelInfo } from '../../types/harness';
import { compareRetrieval } from '../../services/api';
import { ArrowRight, AlertTriangle, RefreshCw } from 'lucide-react';
import styles from './Harness.module.css';

interface RetrievalComparisonViewProps {
  models: ModelInfo[];
  selectedModelId: string;
}

const BENCHMARK_PRESETS = [
  {
    title: 'GQA Head Architecture',
    query: 'What is QORIN\'s GQA ratio, query heads, and KV heads across the 16M, 50M, and 100M variants?',
  },
  {
    title: '4K Passkey Retrieval Accuracy',
    query: 'What was the exact passkey retrieval accuracy measured in the QORIN 100M 4K context pilot?',
  },
  {
    title: 'Token-Weighted Loss Formulation',
    query: 'What is the token-weighted loss correction implemented to eliminate batch size bias?',
  },
  {
    title: 'Greeting Density Arithmetic',
    query: 'Explain the greeting density arithmetic in the SFT v2 dataset and the revised percentage.',
  },
];

export const RetrievalComparisonView: React.FC<RetrievalComparisonViewProps> = ({
  models,
  selectedModelId,
}) => {
  const [modelId, setModelId] = useState<string>(selectedModelId || 'qorin-100m-sft-v3');
  const [prompt, setPrompt] = useState<string>(BENCHMARK_PRESETS[0].query);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunComparison = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await compareRetrieval(modelId, prompt);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 'var(--space-6)' }}>
      {/* Configuration & Input Bar */}
      <div className={styles.sidebarCard} style={{ marginBottom: 'var(--space-6)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
          <div>
            <div className={styles.kicker}>Empirical Evaluation Testbed</div>
            <h3 style={{ fontSize: '18px', fontFamily: 'var(--font-serif)', margin: '2px 0' }}>
              Condition A vs Condition B: Grounding vs Parametric Memory
            </h3>
            <p style={{ fontSize: '12px', color: 'var(--color-ink-secondary)', margin: 0 }}>
              Directly contrast closed-book parametric generation against non-parametric TF-IDF context injection.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>TEST MODEL:</span>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className={styles.modelSelect}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.generation})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Benchmark Presets */}
        <div style={{ marginTop: 'var(--space-2)' }}>
          <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', marginBottom: '6px' }}>
            SELECT EMPIRICAL BENCHMARK QUERY:
          </div>
          <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            {BENCHMARK_PRESETS.map((p, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setPrompt(p.query)}
                className={styles.presetBtn}
                style={{
                  borderColor: prompt === p.query ? 'var(--color-accent-indigo)' : undefined,
                  color: prompt === p.query ? 'var(--color-accent-indigo)' : undefined,
                }}
              >
                {p.title}
              </button>
            ))}
          </div>
        </div>

        {/* Prompt Input */}
        <div style={{ marginTop: 'var(--space-3)' }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className={styles.inputTextarea}
            placeholder="Type a research question to evaluate parametric vs retrieval performance..."
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 'var(--space-2)' }}>
          <button
            type="button"
            onClick={handleRunComparison}
            disabled={loading || !prompt.trim()}
            className={styles.sendBtn}
          >
            {loading ? (
              <>
                <RefreshCw size={14} className="animate-spin" /> Running Dual Inference...
              </>
            ) : (
              <>
                <ArrowRight size={14} /> Execute Side-by-Side Comparison
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ padding: 'var(--space-4)', background: 'var(--color-accent-rust-subtle)', color: 'var(--color-accent-rust)', borderRadius: 'var(--radius-xs)', marginBottom: 'var(--space-6)' }}>
          <AlertTriangle size={16} style={{ display: 'inline', marginRight: '6px' }} />
          {error}
        </div>
      )}

      {/* Side-by-side Result Layout */}
      {result && (() => {
        const condA = result.without_retrieval || result.condition_a_no_retrieval || {};
        const condB = result.with_retrieval || result.condition_b_retrieval || {};
        const retrievedSources = condB.retrieved_sources || condB.retrieval_trace?.retrieved_chunks || [];
        const injectedContext = condB.injected_context || condB.retrieval_trace?.injected_context;

        return (
          <div>
            <div className={styles.compareLayout}>
              {/* Condition A: No Retrieval */}
              <div className={`${styles.compareCard} ${styles.compareCardConditionA}`}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className={styles.compareBadgeA}>CONDITION A: CLOSED-BOOK</span>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                    Parametric Weights Only
                  </span>
                </div>

                <div style={{ fontSize: '12px', color: 'var(--color-ink-secondary)' }}>
                  Generation relies purely on pre-trained weights without external context injection.
                </div>

                <div className={styles.compareOutputText}>
                  {condA.response || 'No response generated.'}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', borderTop: '1px solid var(--color-rule-subtle)', paddingTop: 'var(--space-2)' }}>
                  <span>Latency: {condA.latency_ms !== undefined ? `${condA.latency_ms.toFixed(0)} ms` : '—'}</span>
                  <span>Citations: {condA.citations_present ? 'Yes' : 'None'}</span>
                </div>

                <div style={{ padding: '8px', background: 'var(--color-paper-subtle)', borderRadius: 'var(--radius-xs)', fontSize: '11px', color: 'var(--color-ink-muted)' }}>
                  <strong>Behavioral Note:</strong> For fine-grained architectural numbers or specific percentages, small models (&lt; 1B) frequently hallucinate or confuse historical parameters without retrieval.
                </div>
              </div>

              {/* Condition B: Retrieval Injected */}
              <div className={`${styles.compareCard} ${styles.compareCardConditionB}`}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className={styles.compareBadgeB}>CONDITION B: NON-PARAMETRIC RETRIEVAL</span>
                  <span style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                    TF-IDF + Cosine Grounded
                  </span>
                </div>

                <div style={{ fontSize: '12px', color: 'var(--color-ink-secondary)' }}>
                  Injected authoritative chunks from the QORIN specification corpus.
                </div>

                {/* Retrieved Context Preview */}
                {injectedContext && (
                  <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: 'var(--space-2)', maxHeight: '120px', overflowY: 'auto' }}>
                    <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: 'var(--color-accent-indigo)', marginBottom: '4px', textTransform: 'uppercase' }}>
                      Injected Knowledge Chunks ({retrievedSources.length} sources):
                    </div>
                    <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-secondary)', whiteSpace: 'pre-wrap' }}>
                      {injectedContext}
                    </div>
                  </div>
                )}

                <div className={styles.compareOutputText}>
                  {condB.response || 'No response generated.'}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)', borderTop: '1px solid var(--color-rule-subtle)', paddingTop: 'var(--space-2)' }}>
                  <span>Latency: {condB.latency_ms !== undefined ? `${condB.latency_ms.toFixed(0)} ms` : '—'}</span>
                  <span>Citations: {condB.citations_present ? 'Verified' : 'None'}</span>
                </div>

                <div style={{ padding: '8px', background: 'var(--color-accent-sage-subtle)', borderRadius: 'var(--radius-xs)', fontSize: '11px', color: 'var(--color-accent-sage)' }}>
                  <strong>Behavioral Note:</strong> Grounded context anchors small-model generation to verified facts, eliminating parameter confusion and producing verifiable scientific citations.
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
