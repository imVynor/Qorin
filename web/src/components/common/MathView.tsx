import React, { useMemo } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathViewProps {
  math: string;
  block?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

export const MathView: React.FC<MathViewProps> = ({ math, block = false, className, style }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
      });
    } catch {
      return math;
    }
  }, [math, block]);

  if (block) {
    return (
      <div
        className={className}
        dangerouslySetInnerHTML={{ __html: html }}
        style={{ margin: '1rem 0', overflowX: 'auto', textAlign: 'center', ...style }}
      />
    );
  }

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
      style={style}
    />
  );
};

export default MathView;
