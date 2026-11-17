import React, { useState } from 'react';
import styles from './LearningModule.module.css';

interface DocChunk {
  id: string;
  title: string;
  snippet: string;
  keywords: string[];
}

const KNOWLEDGE_BASE: DocChunk[] = [
  {
    id: 'chunk-1',
    title: 'QORIN-500M Architecture Specification',
    snippet: 'QORIN-500M contains 514,389,504 total parameters and 464,057,856 non-embedding parameters across 20 layers with 24 query heads and 4 KV heads (6:1 GQA).',
    keywords: ['500m', 'parameters', 'layers', 'gqa', 'architecture', 'heads']
  },
  {
    id: 'chunk-2',
    title: 'Rotary Position Embeddings & 4K Context',
    snippet: 'Context extension from 512 to 4,096 tokens was achieved via linear RoPE frequency scaling (s = 8.0) with base frequency theta = 10,000.',
    keywords: ['rope', 'context', '4k', 'position', 'rotary', 'frequency', 'scaling']
  },
  {
    id: 'chunk-3',
    title: '5B Token Pretraining Mixture Breakdown',
    snippet: 'The 5B corpus spans 7 domains: FineWeb-Edu (40%), Cosmopedia (15%), StarCoder (15%), Books (10%), Wikipedia (7.5%), Math (7.5%), and dialogue capped at 5%.',
    keywords: ['dataset', 'corpus', '5b', 'fineweb', 'tokens', 'shards', 'ultrachat', 'mixture']
  },
  {
    id: 'chunk-4',
    title: 'Hardware & Thermal Duty Cycle',
    snippet: 'Pretraining is sustained on an NVIDIA RTX 5060 Laptop GPU (8GB VRAM) at ~5,054 tokens/sec under an automated 105m active / 15m idle cooling cycle.',
    keywords: ['gpu', 'hardware', 'rtx', '5060', 'throughput', 'cooling', 'vram', 'duty']
  }
];

export const Module08RetrievalAugmentation: React.FC = () => {
  const [query, setQuery] = useState('how many parameters in 500m');

  // Client-side token overlap heuristic for in-browser educational demonstration
  const queryWords = query.toLowerCase().split(/\s+/).filter(Boolean);

  const scoredDocs = KNOWLEDGE_BASE.map((doc) => {
    let matches = 0;
    queryWords.forEach((q) => {
      if (doc.keywords.some((k) => k.includes(q) || q.includes(k))) matches += 1;
    });
    const score = queryWords.length > 0 ? Math.min(0.98, matches / Math.sqrt(queryWords.length * doc.keywords.length) * 2.2) : 0;
    return { ...doc, score };
  }).sort((a, b) => b.score - a.score);

  const topMatch = scoredDocs[0];

  return (
    <section id="module-rag" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 08 &bull; Non-Parametric Memory</div>
        <h2 className={styles.moduleTitle}>Retrieval-Augmented Generation (RAG)</h2>
        <div className={styles.moduleLead}>
          Bridging the parameter gap: how small language models leverage external knowledge retrieval to access vast factual corpora without catastrophic parameter bloat.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>Parametric vs. Non-Parametric Knowledge</h3>
        <div className={styles.prose}>
          <p>
            Language models possess two fundamentally different types of memory:
          </p>
          <ul>
            <li><strong>Parametric Memory:</strong> Factual associations stored inside Transformer weight matrices (<em>W</em><sub><em>Q</em></sub>, <em>W</em><sub><em>FFN</em></sub>). Learned during pretraining, static, and bounded by total parameter count.</li>
            <li><strong>Non-Parametric Memory:</strong> External documents stored in an indexed vector database. Dynamic, easily updated in real time, and practically unbounded in volume.</li>
          </ul>
          <p>
            A sub-1B parameter model simply cannot memorize the entire web—its parametric capacity is too constrained.
            <strong>Retrieval-Augmented Generation (RAG)</strong> offloads the burden of encyclopedic memorization to an external retrieval engine,
            freeing the small model to specialize in reasoning, instruction following, and synthesis:
          </p>
          <div className={styles.formulaBox}>
            Dense Vector Cosine Similarity = (u &bull; v) / ( ||u|| &times; ||v|| )<br />
            Prompt<sub>Augmented</sub> = &ldquo;&lt;|im_start|&gt;system\nContext: [Retrieved Passages]\n&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n[User Query]\n&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n&rdquo;
          </div>
          <p>
            <strong>The Critical Limitation:</strong> RAG only provides external factual context; it does <em>not</em> fix
            underlying algorithmic reasoning deficits. If a model lacks the capacity to follow logic or verify code syntax,
            injecting retrieved documents into the context window will not prevent hallucinations.
          </p>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Production Neural Pipeline vs. Educational Demonstration
        </div>
        <h3 className={styles.partHeading}>The Two Retrieval Paradigms</h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)', margin: 'var(--space-4) 0' }}>
          <div style={{ background: 'var(--color-paper-subtle)', padding: 'var(--space-4)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-accent-indigo)', fontWeight: 600 }}>Production Neural Vector RAG</span>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink-primary)', margin: '4px 0' }}>Dense Vector Embeddings (MiniLM / BERT)</div>
            <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', lineHeight: 1.5 }}>
              Passages and queries are projected into dense continuous embedding spaces (e.g. 384 dimensions) via a separate neural encoder.
              Similarity is computed via Approximate Nearest Neighbors (FAISS/HNSW). Captures conceptual synonyms even with zero keyword overlap.
            </p>
          </div>

          <div style={{ background: 'var(--color-paper-subtle)', padding: 'var(--space-4)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-accent-sage)', fontWeight: 600 }}>In-Browser Pedagogical Simulator</span>
            <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--color-ink-primary)', margin: '4px 0' }}>Client-Side Token Overlap Heuristic</div>
            <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', lineHeight: 1.5 }}>
              The interactive lab below runs entirely in your browser using deterministic token and keyword overlap matching against QORIN technical specifications,
              illustrating the retrieval-prompt-generation workflow without downloading heavy neural model weights.
            </p>
          </div>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Retrieval &amp; Prompt Augmentation Sandbox
        </div>
        <h3 className={styles.partHeading}>Interactive Retrieval Simulator</h3>

        <div className={styles.labCard}>
          <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: '8px 12px', marginBottom: 'var(--space-4)', fontSize: '11px', color: 'var(--color-ink-secondary)' }}>
            <strong>Demonstration Notice:</strong> This live simulator evaluates queries against a mini-corpus of QORIN technical specifications using client-side keyword and token overlap scoring to demonstrate in-context prompt assembly.
          </div>

          <div className={styles.controlsRow}>
            <div className={styles.controlItem} style={{ flex: 1 }}>
              <label className={styles.controlLabel} htmlFor="rag-query-input">Search Query against QORIN Technical Specs:</label>
              <input
                id="rag-query-input"
                type="text"
                className={styles.textInput}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., parameters in 500m, rope frequency, rtx 5060..."
              />
            </div>
          </div>

          <div className={styles.outputArea}>
            <div className={styles.outputTitle}>Ranked Knowledge Chunks (Token Overlap Simulation Score)</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {scoredDocs.map((doc, idx) => (
                <div
                  key={doc.id}
                  style={{
                    background: idx === 0 && doc.score > 0.2 ? 'var(--color-accent-indigo-subtle)' : 'var(--color-surface)',
                    border: `1px solid ${idx === 0 && doc.score > 0.2 ? 'var(--color-accent-indigo)' : 'var(--color-rule-subtle)'}`,
                    borderRadius: 'var(--radius-xs)',
                    padding: '8px 12px'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '2px' }}>
                    <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--color-ink-primary)' }}>
                      {doc.title}
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: doc.score > 0.3 ? 'var(--color-accent-indigo)' : 'var(--color-ink-muted)' }}>
                      Overlap Score: {doc.score.toFixed(3)} {idx === 0 && doc.score > 0.2 ? '(Top Ranked Match)' : ''}
                    </span>
                  </div>
                  <p style={{ fontSize: '11px', color: 'var(--color-ink-secondary)', margin: 0, lineHeight: 1.45 }}>
                    {doc.snippet}
                  </p>
                </div>
              ))}
            </div>

            {topMatch && topMatch.score > 0.2 && (
              <div style={{ marginTop: 'var(--space-4)', padding: 'var(--space-3)', background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-xs)' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', textTransform: 'uppercase', color: 'var(--color-accent-sage)', fontWeight: 600 }}>
                  Assembled In-Context Augmented Prompt:
                </span>
                <pre style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--color-ink-primary)', marginTop: '4px', whiteSpace: 'pre-wrap' }}>
{`<|im_start|>system
Use the following verified context to answer the user question:
${topMatch.snippet}<|im_end|>
<|im_start|>user
${query}<|im_end|>
<|im_start|>assistant`}
                </pre>
              </div>
            )}
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> This reflects the original foundation of our research proposal:
            <em>&ldquo;Tool-Augmented Small Language Models&rdquo;</em>. Our experiments demonstrated that while retrieval
            supplies factual context, small models require a critical architectural capacity threshold (&ge; 100M parameters)
            before they can consistently extract, adhere to, and synthesize injected evidence without hallucinating.
          </div>
        </div>
      </div>
    </section>
  );
};
