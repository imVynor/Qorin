import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { pretrainScalingData } from '../../data/chartData';
import type { PretrainPoint } from '../../data/chartData';
import styles from './PretrainLossChart.module.css';

interface Props {
  figureNumber?: number;
}

export const PretrainLossChart: React.FC<Props> = ({ figureNumber = 1 }) => {
  const [metric, setMetric] = useState<'loss' | 'perplexity'>('loss');
  const [hoveredPoint, setHoveredPoint] = useState<PretrainPoint | null>(null);

  const width = 720;
  const height = 310;
  const padLeft = 55;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 45;

  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;

  const logMinTokens = 7.0; // 10^7
  const logMaxTokens = 9.301; // 2 * 10^9

  const getX = (tokens: number) => {
    const logVal = Math.log10(tokens);
    const frac = (logVal - logMinTokens) / (logMaxTokens - logMinTokens);
    return padLeft + frac * plotWidth;
  };

  const minLoss = 2.4;
  const maxLoss = 4.1;
  const getYLoss = (val: number) => {
    const frac = (maxLoss - val) / (maxLoss - minLoss);
    return padTop + frac * plotHeight;
  };

  const minPpl = 10;
  const maxPpl = 52;
  const getYPpl = (val: number) => {
    const logMin = Math.log10(minPpl);
    const logMax = Math.log10(maxPpl);
    const frac = (logMax - Math.log10(val)) / (logMax - logMin);
    return padTop + frac * plotHeight;
  };

  const getY = (pt: PretrainPoint) =>
    metric === 'loss' ? getYLoss(pt.valLoss) : getYPpl(pt.valPerplexity);

  const xGridTokens = [
    { label: '10M', val: 10_000_000 },
    { label: '100M', val: 100_000_000 },
    { label: '500M', val: 500_000_000 },
    { label: '1.0B', val: 1_000_000_000 },
    { label: '1.5B', val: 1_500_000_000 },
  ];

  const yGridLoss = [2.5, 3.0, 3.5, 4.0];
  const yGridPpl = [15, 20, 30, 40, 50];

  const isoX = getX(100_000_000);

  return (
    <figure className={styles.figureWrap}>
      <div className={styles.figureHeader}>
        <div className={styles.figureTitle}>
          <span className={styles.figureIndex}>Figure {figureNumber}</span>
          <span>Empirical Pretraining Scaling</span>
        </div>
        <div className={styles.toggleGroup}>
          <button
            type="button"
            className={`${styles.toggleBtn} ${metric === 'loss' ? styles.toggleBtnActive : ''}`}
            onClick={() => setMetric('loss')}
          >
            Validation Loss (nats)
          </button>
          <button
            type="button"
            className={`${styles.toggleBtn} ${metric === 'perplexity' ? styles.toggleBtnActive : ''}`}
            onClick={() => setMetric('perplexity')}
          >
            Perplexity (exp)
          </button>
        </div>
      </div>

      <div className={styles.svgWrapper}>
        <svg viewBox={`0 0 ${width} ${height}`} className={styles.svg} role="img" aria-label="Pretraining scaling trajectory">
          {/* X Grid Lines */}
          {xGridTokens.map((g) => {
            const x = getX(g.val);
            return (
              <g key={g.label}>
                <line x1={x} y1={padTop} x2={x} y2={padTop + plotHeight} className={styles.grid} />
                <text x={x} y={padTop + plotHeight + 16} textAnchor="middle" className={styles.axisLabel}>
                  {g.label}
                </text>
              </g>
            );
          })}

          {/* Y Grid Lines */}
          {(metric === 'loss' ? yGridLoss : yGridPpl).map((val) => {
            const y = metric === 'loss' ? getYLoss(val) : getYPpl(val);
            return (
              <g key={val}>
                <line x1={padLeft} y1={y} x2={padLeft + plotWidth} y2={y} className={styles.grid} />
                <text x={padLeft - 8} y={y + 3} textAnchor="end" className={styles.axisLabel}>
                  {val.toFixed(metric === 'loss' ? 1 : 0)}
                </text>
              </g>
            );
          })}

          {/* Iso-Data Vertical Guideline */}
          <line x1={isoX} y1={padTop} x2={isoX} y2={padTop + plotHeight} className={styles.isoLine} />
          <text x={isoX + 6} y={padTop + 12} className={styles.isoLabel}>
            Iso-data snapshot (~100M tokens)
          </text>

          {/* Trajectory line connecting measured checkpoints */}
          <path
            d={`M ${getX(pretrainScalingData[0].tokens)} ${getY(pretrainScalingData[0])} 
               L ${getX(pretrainScalingData[1].tokens)} ${getY(pretrainScalingData[1])} 
               L ${getX(pretrainScalingData[2].tokens)} ${getY(pretrainScalingData[2])} 
               L ${getX(pretrainScalingData[3].tokens)} ${getY(pretrainScalingData[3])}`}
            fill="none"
            stroke="var(--color-ink-muted)"
            strokeWidth="1.25"
            strokeDasharray="3 3"
          />

          {/* Axis borders */}
          <line x1={padLeft} y1={padTop} x2={padLeft} y2={padTop + plotHeight} className={styles.axis} />
          <line x1={padLeft} y1={padTop + plotHeight} x2={padLeft + plotWidth} y2={padTop + plotHeight} className={styles.axis} />

          <text x={padLeft + plotWidth / 2} y={height - 6} textAnchor="middle" className={styles.axisTitle}>
            Tokens ingested (logarithmic scale)
          </text>
          <text x={-height / 2} y={16} transform="rotate(-90)" textAnchor="middle" className={styles.axisTitle}>
            {metric === 'loss' ? 'Cross-Entropy Loss (nats)' : 'Validation Perplexity'}
          </text>

          {/* Data Points */}
          {pretrainScalingData.map((pt) => {
            const cx = getX(pt.tokens);
            const cy = getY(pt);
            const isHovered = hoveredPoint?.model === pt.model;

            return (
              <g
                key={pt.model}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 6 : 4.5}
                  fill={pt.isIsoData ? 'var(--color-accent-rust)' : 'var(--color-ink-primary)'}
                  stroke="var(--color-surface)"
                  strokeWidth="1.5"
                />
                <text
                  x={cx}
                  y={cy - 10}
                  textAnchor="middle"
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '10px',
                    fontWeight: 600,
                    fill: 'var(--color-ink-primary)',
                  }}
                >
                  {metric === 'loss' ? pt.valLoss.toFixed(3) : pt.valPerplexity.toFixed(1)}
                </text>
              </g>
            );
          })}
        </svg>

        {hoveredPoint && (
          <div
            className={styles.tooltip}
            style={{
              left: `${(getX(hoveredPoint.tokens) / width) * 100}%`,
              top: `${(getY(hoveredPoint) / height) * 100}%`,
            }}
          >
            <div style={{ fontWeight: 600 }}>{hoveredPoint.model}</div>
            <div>{hoveredPoint.paramsLabel} &bull; {hoveredPoint.tokensLabel}</div>
            <div>Loss: <strong>{hoveredPoint.valLoss.toFixed(4)}</strong> | PPL: <strong>{hoveredPoint.valPerplexity.toFixed(2)}</strong></div>
          </div>
        )}
      </div>

      <figcaption className={styles.caption}>
        <strong>Figure 1.</strong> Cross-entropy validation loss evaluated on identical validation tokens across model scales.
        At the 100M token iso-data budget, expanding capacity from 16.8M to 49.1M parameters produced a &Delta; = &minus;0.4047 loss reduction (33.3% perplexity improvement).
        Scaling to 100M parameters across 1.5B tokens yielded a final validation loss of 2.7137 (15.08 perplexity; best checkpoint 2.621).
        See full checkpoints and logs in <Link to="/experiments" style={{ textDecoration: 'underline' }}>Experiments Archive &rarr;</Link>
      </figcaption>
    </figure>
  );
};
