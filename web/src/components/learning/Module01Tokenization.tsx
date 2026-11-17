import React, { useState } from 'react';
import styles from './LearningModule.module.css';

export const Module01Tokenization: React.FC = () => {
  const [inputText, setInputText] = useState('small language models compute reasoning');
  const [mergeSteps, setMergeSteps] = useState(3);

  // Deterministic subword BPE simulation for teaching
  const simulateBpe = (text: string, steps: number) => {
    if (!text.trim()) return [];
    const words = text.trim().split(/\s+/);
    const tokens: { text: string; id: number; merged: boolean }[] = [];
    let id = 1000;

    words.forEach((word) => {
      if (steps === 0) {
        // Character level
        for (let i = 0; i < word.length; i++) {
          tokens.push({ text: i === 0 ? `Ġ${word[i]}` : word[i], id: id++, merged: false });
        }
      } else if (steps === 1) {
        // Bigram merges
        let i = 0;
        while (i < word.length) {
          if (i + 1 < word.length && (i === 0 || i % 2 === 0)) {
            tokens.push({ text: i === 0 ? `Ġ${word.slice(i, i + 2)}` : word.slice(i, i + 2), id: id++, merged: true });
            i += 2;
          } else {
            tokens.push({ text: i === 0 ? `Ġ${word[i]}` : word[i], id: id++, merged: false });
            i += 1;
          }
        }
      } else if (steps === 2) {
        // Intermediate subwords (stem + affix)
        if (word.length > 5) {
          tokens.push({ text: `Ġ${word.slice(0, 4)}`, id: id++, merged: true });
          tokens.push({ text: word.slice(4), id: id++, merged: true });
        } else {
          tokens.push({ text: `Ġ${word}`, id: id++, merged: true });
        }
      } else {
        // Full subword vocabulary
        const commonSubwords = ['Ġsmall', 'Ġlanguage', 'Ġmodels', 'Ġcompute', 'Ġreason', 'ing', 'Ġthe', 'Ġtransformer'];
        let matched = false;
        for (const sub of commonSubwords) {
          const raw = sub.replace('Ġ', '');
          if (word.startsWith(raw) && word.length > raw.length) {
            tokens.push({ text: `Ġ${raw}`, id: id++, merged: true });
            tokens.push({ text: word.slice(raw.length), id: id++, merged: true });
            matched = true;
            break;
          }
        }
        if (!matched) {
          tokens.push({ text: `Ġ${word}`, id: id++, merged: true });
        }
      }
    });

    return tokens;
  };

  const tokens = simulateBpe(inputText, mergeSteps);
  const charCount = inputText.length;
  const tokenCount = Math.max(tokens.length, 1);
  const compressionRatio = (charCount / tokenCount).toFixed(2);

  return (
    <section id="module-tokenization" className={styles.moduleSection}>
      <header className={styles.moduleHeader}>
        <div className={styles.moduleKicker}>Module 01 &bull; Discrete Input Representation</div>
        <h2 className={styles.moduleTitle}>Tokenization &amp; Byte-Pair Encoding (BPE)</h2>
        <div className={styles.moduleLead}>
          How raw UTF-8 text is converted into discrete numerical sequences, balancing sequence length against embedding table memory.
        </div>
      </header>

      {/* Part A: Understand */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part A</span> &bull; First Principles &amp; Mathematical Formulation
        </div>
        <h3 className={styles.partHeading}>Why Transformers Cannot Read Raw Text</h3>
        <div className={styles.prose}>
          <p>
            Computers store text as byte streams (ASCII/UTF-8), but neural networks require dense vector inputs.
            One could theoretically pass raw characters, but an English paragraph of 200 words contains over 1,000 characters.
            Because self-attention scales quadratically with sequence length (<em>O</em>(<em>N</em><sup>2</sup>)), character-level models
            exhaust GPU memory rapidly and struggle to capture long-range semantic dependencies.
          </p>
          <p>
            Conversely, full word-level vocabularies fail because human language has an unbounded lexicon (morphology, compound words,
            slang, misspellings). A dictionary with 500,000 words creates an astronomical embedding matrix:
          </p>
          <div className={styles.formulaBox}>
            Embedding Matrix Footprint = V &times; d &times; 2 bytes (FP16)<br />
            For V = 500,000, d = 1,536: 500,000 &times; 1,536 &times; 2 = 1.536 GB (just for input weights!)
          </div>
          <p>
            <strong>Byte-Pair Encoding (BPE)</strong> resolves this dilemma by learning statistical subwords from corpus frequency:
          </p>
          <ol>
            <li>Initialize the vocabulary with single bytes or base characters (256 bytes + basic alphabet).</li>
            <li>Count the co-occurrence frequency of all adjacent token bigrams (<em>c</em><sub><em>i</em></sub>, <em>c</em><sub><em>i</em>+1</sub>) across the entire dataset.</li>
            <li>Identify the most frequent pair (e.g., &lsquo;e&rsquo; + &lsquo;r&rsquo; &rarr; &lsquo;er&rsquo;) and merge it into a single atomic token.</li>
            <li>Repeat this merge operation for <em>K</em> predetermined steps until the vocabulary reaches target size <em>V</em>.</li>
          </ol>
        </div>
      </div>

      {/* Part B: Visualise */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part B</span> &bull; Internal State &amp; Successive Merge Hierarchy
        </div>
        <h3 className={styles.partHeading}>The BPE Merge Tree</h3>
        <div className={styles.prose}>
          <p>
            Frequent letter combinations merge into suffixes, prefixes, and common roots, while rare or misspelled words
            gracefully fall back to shorter subword constituents without ever triggering an &ldquo;Out of Vocabulary&rdquo; ([UNK]) crash:
          </p>
        </div>

        <div style={{ background: 'var(--color-paper-subtle)', border: '1px solid var(--color-rule-subtle)', borderRadius: 'var(--radius-xs)', padding: 'var(--space-4)', margin: 'var(--space-4) 0', fontFamily: 'var(--font-mono)', fontSize: '11px', lineHeight: 1.8, color: 'var(--color-ink-primary)' }}>
          <div>[Step 0 &bull; Base Bytes] : &lsquo;r&rsquo; &lsquo;e&rsquo; &lsquo;a&rsquo; &lsquo;s&rsquo; &lsquo;o&rsquo; &lsquo;n&rsquo; &lsquo;i&rsquo; &lsquo;n&rsquo; &lsquo;g&rsquo;</div>
          <div>[Step 1 &bull; Merge 1] &nbsp; &nbsp;: &lsquo;r&rsquo; &lsquo;e&rsquo; &lsquo;a&rsquo; &lsquo;s&rsquo; &lsquo;o&rsquo; &lsquo;n&rsquo; + (&lsquo;i&rsquo; + &lsquo;n&rsquo; &rarr; &lsquo;in&rsquo;) &lsquo;g&rsquo;</div>
          <div>[Step 2 &bull; Merge 2] &nbsp; &nbsp;: &lsquo;r&rsquo; &lsquo;e&rsquo; &lsquo;a&rsquo; &lsquo;s&rsquo; &lsquo;o&rsquo; &lsquo;n&rsquo; + (&lsquo;in&rsquo; + &lsquo;g&rsquo; &rarr; &lsquo;ing&rsquo;)</div>
          <div>[Step 3 &bull; Merge 3] &nbsp; &nbsp;: (&lsquo;r&rsquo; &lsquo;e&rsquo; &lsquo;a&rsquo; &lsquo;s&rsquo; &lsquo;o&rsquo; &lsquo;n&rsquo; &rarr; &lsquo;reason&rsquo;) + &lsquo;ing&rsquo;</div>
          <div style={{ color: 'var(--color-accent-sage)', fontWeight: 600 }}>&rarr; Final Token Sequence: [&ldquo;Ġreason&rdquo; (#1004), &ldquo;ing&rdquo; (#1005)]</div>
        </div>
      </div>

      {/* Part C: Experiment */}
      <div className={styles.teachingBlock}>
        <div className={styles.partKicker}>
          <span className={styles.partTag}>Part C</span> &bull; Interactive Tokenizer Laboratory
        </div>
        <h3 className={styles.partHeading}>Live Subword Merge Simulator</h3>
        <p className={styles.prose}>
          Type any custom phrase and use the merge slider to watch how increasing merge iterations compresses text from
          raw characters into dense semantic subwords:
        </p>

        <div className={styles.labCard}>
          <div className={styles.controlsRow}>
            <div className={styles.controlItem} style={{ flex: 1 }}>
              <label className={styles.controlLabel} htmlFor="bpe-text-input">Input Phrase</label>
              <input
                id="bpe-text-input"
                type="text"
                className={styles.textInput}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type a sentence..."
              />
            </div>

            <div className={styles.controlItem}>
              <label className={styles.controlLabel}>Merge Iteration Stage: {mergeSteps}</label>
              <input
                type="range"
                min={0}
                max={3}
                step={1}
                value={mergeSteps}
                onChange={(e) => setMergeSteps(Number(e.target.value))}
                className={styles.sliderInput}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', fontFamily: 'var(--font-mono)', color: 'var(--color-ink-muted)' }}>
                <span>0 (Bytes)</span>
                <span>1 (Pairs)</span>
                <span>2 (Subwords)</span>
                <span>3 (Vocab)</span>
              </div>
            </div>
          </div>

          <div className={styles.outputArea}>
            <div className={styles.outputTitle}>
              Resulting Tokens ({tokens.length} tokens &bull; {compressionRatio} chars/token)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {tokens.map((t, idx) => (
                <span
                  key={idx}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'baseline',
                    gap: '4px',
                    background: t.merged ? 'var(--color-accent-indigo-subtle)' : 'var(--color-surface)',
                    border: `1px solid ${t.merged ? 'var(--color-accent-indigo)' : 'var(--color-rule)'}`,
                    borderRadius: 'var(--radius-xs)',
                    padding: '3px 7px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    color: 'var(--color-ink-primary)'
                  }}
                >
                  <span>{t.text}</span>
                  <span style={{ fontSize: '9px', color: 'var(--color-ink-muted)' }}>#{t.id}</span>
                </span>
              ))}
            </div>
          </div>

          <div className={styles.qorinConnection}>
            <strong>Connection to QORIN:</strong> QORIN-16M used <code>bpe_16k_v2</code> (16,000 vocabulary), allocating
            6.14M parameters (36.6% of the entire model!) to embeddings. QORIN-500M upgraded to <code>bpe_32k</code> (32,768 vocabulary),
            reducing sequence length by 18.4% on source code and technical math, while consuming 50.33M parameters (9.78% of the model).
          </div>
        </div>
      </div>
    </section>
  );
};

