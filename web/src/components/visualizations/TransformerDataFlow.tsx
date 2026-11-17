import React, { useState } from 'react';
import styles from './TransformerDataFlow.module.css';

type ModelKey = '16M' | '50M' | '100M' | '500M';
type NodeKey = 'embedding' | 'norm1' | 'gqa' | 'norm2' | 'swiglu' | 'head';

interface NodeDetail {
  title: string;
  tag: string;
  formula: string;
  stats: (model: ModelKey) => { label: string; value: string }[];
  description: (model: ModelKey) => string;
}

const DETAILS: Record<NodeKey, NodeDetail> = {
  embedding: {
    title: 'Token Embedding Matrix',
    tag: 'Input Projection',
    formula: 'X_0 = W_emb[tokens]  where  W_emb \u2208 \u211D^(V \u00D7 d)',
    stats: (m) => {
      switch (m) {
        case '16M':
          return [
            { label: 'Vocabulary (V)', value: '16,000' },
            { label: 'Hidden Dim (d)', value: '384' },
            { label: 'Embedding Weights', value: '6,144,000' },
            { label: 'Tied Weights', value: 'Yes (Input / Output)' },
          ];
        case '50M':
          return [
            { label: 'Vocabulary (V)', value: '16,000' },
            { label: 'Hidden Dim (d)', value: '512' },
            { label: 'Embedding Weights', value: '8,192,000' },
            { label: 'Tied Weights', value: 'Yes' },
          ];
        case '100M':
          return [
            { label: 'Vocabulary (V)', value: '32,768' },
            { label: 'Hidden Dim (d)', value: '768' },
            { label: 'Embedding Weights', value: '25,165,824' },
            { label: 'Tied Weights', value: 'Yes' },
          ];
        case '500M':
          return [
            { label: 'Vocabulary (V)', value: '32,768' },
            { label: 'Hidden Dim (d)', value: '1,536' },
            { label: 'Embedding Weights', value: '50,331,648' },
            { label: 'Tied Weights', value: 'Yes' },
          ];
      }
    },
    description: () =>
      'Discrete input token IDs are mapped to dense vector representations. Vocabulary weights are tied between the initial input embedding matrix and the final pre-softmax output projection, eliminating redundant storage while regularizing semantic alignment.',
  },
  norm1: {
    title: 'Pre-Attention RMSNorm',
    tag: 'Normalization Layer',
    formula: 'x\u0302 = x / \u221A( (1/d) \u2211 x_i\u00B2 + \u03B5 ) \u2299 \u03B3',
    stats: (m) => {
      const dim = m === '16M' ? '384' : m === '50M' ? '512' : m === '100M' ? '768' : '1,536';
      return [
        { label: 'Norm Dimension', value: dim },
        { label: 'Epsilon (\u03B5)', value: '1e-5' },
        { label: 'Trainable Weights', value: dim },
        { label: 'Bias Vector', value: 'Zero (b = 0)' },
      ];
    },
    description: () =>
      'Root Mean Square Layer Normalization stabilizes activation magnitudes without the computational cost of mean-centering required by standard LayerNorm. Applied strictly before the attention block (Pre-LN) to preserve uninhibited gradient flow through the residual highway.',
  },
  gqa: {
    title: 'RoPE + Grouped-Query Attention (GQA)',
    tag: 'Attention Block',
    formula: 'Attn(Q, K, V) = softmax( (R_\u0398 Q)(R_\u0398 K)\u1D40 / \u221Ad_k + M ) V',
    stats: (m) => {
      switch (m) {
        case '16M':
          return [
            { label: 'Heads (Q / KV)', value: '6 / 6 (MHA)' },
            { label: 'Head Dim (d_k)', value: '64' },
            { label: 'Attention Weights / Layer', value: '589,824' },
            { label: 'KV-Cache / Token', value: '0.75 KB' },
          ];
        case '50M':
          return [
            { label: 'Heads (Q / KV)', value: '8 / 4 (2:1 GQA)' },
            { label: 'Head Dim (d_k)', value: '64' },
            { label: 'Attention Weights / Layer', value: '786,432' },
            { label: 'KV-Cache Reduction', value: '50.0% vs MHA' },
          ];
        case '100M':
          return [
            { label: 'Heads (Q / KV)', value: '12 / 4 (3:1 GQA)' },
            { label: 'Head Dim (d_k)', value: '64' },
            { label: 'Attention Weights / Layer', value: '1,572,864' },
            { label: 'KV-Cache Reduction', value: '66.7% vs MHA' },
          ];
        case '500M':
          return [
            { label: 'Heads (Q / KV)', value: '24 / 4 (6:1 GQA)' },
            { label: 'Head Dim (d_k)', value: '64' },
            { label: 'Attention Weights / Layer', value: '5,505,024' },
            { label: 'KV-Cache Reduction', value: '83.3% vs MHA' },
          ];
      }
    },
    description: (m) =>
      m === '16M'
        ? 'Initial 16M pilot utilized standard Multi-Head Attention (6 heads). Positional encoding is managed via Rotary Position Embeddings (RoPE), rotating query and key representations in 2D coordinate slices.'
        : `Grouped-Query Attention pools multiple query heads per key-value head (${m === '50M' ? '2:1' : m === '500M' ? '6:1' : '3:1'} ratio). This compresses memory footprint and memory-bandwidth saturation during autoregressive generation without degrading validation perplexity.`,
  },
  norm2: {
    title: 'Pre-FFN RMSNorm',
    tag: 'Normalization Layer',
    formula: 'x\u0302_attn = x_attn / \u221A( (1/d) \u2211 x_i\u00B2 + \u03B5 ) \u2299 \u03B3',
    stats: (m) => {
      const dim = m === '16M' ? '384' : m === '50M' ? '512' : m === '100M' ? '768' : '1,536';
      return [
        { label: 'Norm Dimension', value: dim },
        { label: 'Epsilon (\u03B5)', value: '1e-5' },
        { label: 'Trainable Weights', value: dim },
        { label: 'Placement', value: 'Pre-FeedForward' },
      ];
    },
    description: () =>
      'Independent RMSNorm scale vector applied to the residual output of the attention block before passing into the non-linear feed-forward expansion.',
  },
  swiglu: {
    title: 'SwiGLU Non-Linear Feed-Forward Network',
    tag: 'Feed-Forward Block',
    formula: 'FFN(x) = ( Swish(x W_gate) \u2299 (x W_up) ) W_down',
    stats: (m) => {
      switch (m) {
        case '16M':
          return [
            { label: 'Hidden FFN Dim', value: '1,024 (Standard GELU)' },
            { label: 'FFN Projections', value: 'Up, Down (2 matrices)' },
            { label: 'FFN Weights / Layer', value: '786,432' },
            { label: 'Non-Linearity', value: 'GELU' },
          ];
        case '50M':
          return [
            { label: 'Hidden FFN Dim', value: '1,376 (SwiGLU)' },
            { label: 'FFN Projections', value: 'Gate, Up, Down (3 matrices)' },
            { label: 'FFN Weights / Layer', value: '2,113,536' },
            { label: 'Non-Linearity', value: 'SwiGLU (Shazeer 2020)' },
          ];
        case '100M':
          return [
            { label: 'Hidden FFN Dim', value: '2,048 (SwiGLU)' },
            { label: 'FFN Projections', value: 'Gate, Up, Down (3 matrices)' },
            { label: 'FFN Weights / Layer', value: '4,718,592' },
            { label: 'Non-Linearity', value: 'SwiGLU' },
          ];
        case '500M':
          return [
            { label: 'Hidden FFN Dim', value: '3,840 (SwiGLU)' },
            { label: 'FFN Projections', value: 'Gate, Up, Down (3 matrices)' },
            { label: 'FFN Weights / Layer', value: '17,694,720' },
            { label: 'Non-Linearity', value: 'SwiGLU' },
          ];
      }
    },
    description: (m) =>
      m === '16M'
        ? 'QORIN-16M used traditional two-matrix GELU feed-forward projections. Later generations transitioned permanently to SwiGLU.'
        : 'Gated linear unit with SiLU activation function. Rather than a single non-linear projection, SwiGLU computes an element-wise product of a gated linear projection and an up-projection, significantly boosting expressive capacity per parameter.',
  },
  head: {
    title: 'Final RMSNorm & Output Logits',
    tag: 'Output Projection',
    formula: 'P(w_{t+1}) = softmax( (RMSNorm(x_L)) W_emb\u1D40 )',
    stats: (m) => {
      const dim = m === '16M' ? '384' : m === '50M' ? '512' : m === '100M' ? '768' : '1,536';
      const vocab = m === '16M' || m === '50M' ? '16,000' : '32,768';
      return [
        { label: 'Final Norm Dimension', value: dim },
        { label: 'Logit Projections', value: vocab },
        { label: 'Independent Parameters', value: '0 (Tied to W_emb)' },
        { label: 'Softmax Temperature', value: 'Configurable' },
      ];
    },
    description: () =>
      'The output representations from the final Transformer block pass through a terminal RMSNorm layer and are multiplied by the transpose of the input embedding matrix to produce vocabulary logits.',
  },
};

const ArrowDown: React.FC = () => (
  <div className={styles.arrowDownWrap}>
    <svg width="12" height="14" viewBox="0 0 12 14" fill="none" className={styles.arrowSvg} aria-hidden="true">
      <path d="M6 0V11M6 11L2.5 7.5M6 11L9.5 7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </div>
);

export const TransformerDataFlow: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<ModelKey>('500M');
  const [selectedNode, setSelectedNode] = useState<NodeKey>('gqa');

  const detail = DETAILS[selectedNode];
  const stats = detail.stats(selectedModel);

  return (
    <figure className={styles.diagramWrap}>
      <div className={styles.diagramHeader}>
        <div className={styles.titleArea}>
          <span className={styles.figureIndex}>Figure 8 &bull; Interactive Architecture Graph</span>
          <span className={styles.figureTitle}>QORIN Computational Data Flow &amp; Tensor Dimensions</span>
        </div>

        <div className={styles.modelSelector}>
          {(['16M', '50M', '100M', '500M'] as ModelKey[]).map((m) => (
            <button
              key={m}
              type="button"
              className={`${styles.modelBtn} ${selectedModel === m ? styles.modelBtnActive : ''}`}
              onClick={() => setSelectedModel(m)}
            >
              QORIN-{m}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.flowLayout}>
        <div className={styles.flowColumn}>
          <div
            className={`${styles.flowNode} ${selectedNode === 'embedding' ? styles.flowNodeActive : ''}`}
            onClick={() => setSelectedNode('embedding')}
          >
            <div className={styles.flowNodeTag}>Input Embedding</div>
            <div className={styles.flowNodeLabel}>Tokens &rarr; Vectors (W_emb)</div>
          </div>

          <ArrowDown />

          <div className={styles.residualWrap}>
            <div className={styles.residualLabel}>Transformer Block (Repeated \u00D7 L)</div>

            <div
              className={`${styles.flowNode} ${selectedNode === 'norm1' ? styles.flowNodeActive : ''}`}
              onClick={() => setSelectedNode('norm1')}
            >
              <div className={styles.flowNodeTag}>Pre-LN</div>
              <div className={styles.flowNodeLabel}>RMSNorm (Attention)</div>
            </div>

            <ArrowDown />

            <div
              className={`${styles.flowNode} ${selectedNode === 'gqa' ? styles.flowNodeActive : ''}`}
              onClick={() => setSelectedNode('gqa')}
            >
              <div className={styles.flowNodeTag}>Attention Core</div>
              <div className={styles.flowNodeLabel}>RoPE + GQA (Q, K, V Projections)</div>
            </div>

            <ArrowDown />

            <div
              className={`${styles.flowNode} ${selectedNode === 'norm2' ? styles.flowNodeActive : ''}`}
              onClick={() => setSelectedNode('norm2')}
            >
              <div className={styles.flowNodeTag}>Pre-LN</div>
              <div className={styles.flowNodeLabel}>RMSNorm (Feed-Forward)</div>
            </div>

            <ArrowDown />

            <div
              className={`${styles.flowNode} ${selectedNode === 'swiglu' ? styles.flowNodeActive : ''}`}
              onClick={() => setSelectedNode('swiglu')}
            >
              <div className={styles.flowNodeTag}>Non-Linearity</div>
              <div className={styles.flowNodeLabel}>SwiGLU (Gate \u2299 Up &rarr; Down)</div>
            </div>
          </div>

          <ArrowDown />

          <div
            className={`${styles.flowNode} ${selectedNode === 'head' ? styles.flowNodeActive : ''}`}
            onClick={() => setSelectedNode('head')}
          >
            <div className={styles.flowNodeTag}>Terminal Projection</div>
            <div className={styles.flowNodeLabel}>Final RMSNorm &rarr; Softmax</div>
          </div>
        </div>

        <div className={styles.detailsPanel}>
          <div className={styles.detailKicker}>{detail.tag} &bull; QORIN-{selectedModel} Specification</div>
          <h3 className={styles.detailTitle}>{detail.title}</h3>

          <div className={styles.detailFormula}>{detail.formula}</div>

          <div className={styles.detailStats}>
            {stats.map((s) => (
              <div key={s.label} className={styles.statItem}>
                <div className={styles.statLabel}>{s.label}</div>
                <div className={styles.statValue}>{s.value}</div>
              </div>
            ))}
          </div>

          <p className={styles.detailDesc}>{detail.description(selectedModel)}</p>
        </div>
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure 8 Source:</strong> Implementation specifications in <code>modern_gpt/model/transformer.py</code>. Click on any component node in the computational flow to inspect its exact mathematical formulation, parameter footprint, and architectural rationale across QORIN-16M, 50M, 100M, and 500M configurations.
      </figcaption>
    </figure>
  );
};

