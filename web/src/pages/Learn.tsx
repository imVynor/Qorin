import React from 'react';
import { Module01Tokenization } from '../components/learning/Module01Tokenization';
import { Module02SelfAttention } from '../components/learning/Module02SelfAttention';
import { Module03RotaryEmbeddings } from '../components/learning/Module03RotaryEmbeddings';
import { Module04GroupedQueryAttention } from '../components/learning/Module04GroupedQueryAttention';
import { Module05DecoderBlock } from '../components/learning/Module05DecoderBlock';
import { Module06HowModelsLearn } from '../components/learning/Module06HowModelsLearn';
import { Module07SupervisedFineTuning } from '../components/learning/Module07SupervisedFineTuning';
import { Module08RetrievalAugmentation } from '../components/learning/Module08RetrievalAugmentation';
import { Module09ModelEvaluation } from '../components/learning/Module09ModelEvaluation';
import styles from './Learn.module.css';

export const Learn: React.FC = () => {
  const modules = [
    { id: 'module-tokenization', number: '01', name: 'Tokenization & BPE' },
    { id: 'module-attention', number: '02', name: 'Self-Attention' },
    { id: 'module-rope', number: '03', name: 'RoPE Embeddings' },
    { id: 'module-gqa', number: '04', name: 'GQA & KV-Cache' },
    { id: 'module-decoder', number: '05', name: 'Decoder Block' },
    { id: 'module-learning', number: '06', name: 'How Models Learn' },
    { id: 'module-sft', number: '07', name: 'Supervised Fine-Tuning' },
    { id: 'module-rag', number: '08', name: 'Retrieval (RAG)' },
    { id: 'module-eval', number: '09', name: 'Evaluation Tiers' },
  ];

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Interactive Laboratory &bull; 9 Foundation Modules</div>
          <h1 className={styles.title}>The Transformer Learning Lab</h1>
          <div className={styles.lead}>
            A rigorous interactive educational curriculum explaining the mathematics, internal representations,
            and engineering mechanics behind modern language models: from discrete tokenization to rotary geometry and multi-tier evaluation.
          </div>
        </div>
      </header>

      {/* Sticky Module Navigator */}
      <nav className={styles.moduleNavSticky} aria-label="Learning Modules Quick Navigation">
        <div className={styles.container}>
          <div className={styles.moduleNavList}>
            {modules.map((m) => (
              <a key={m.id} href={`#${m.id}`} className={styles.moduleNavBtn}>
                <span style={{ color: 'var(--color-accent-indigo)' }}>M{m.number}</span>
                <span>{m.name}</span>
              </a>
            ))}
          </div>
        </div>
      </nav>

      {/* 9 Educational Modules */}
      <div className={styles.container}>
        <Module01Tokenization />
        <Module02SelfAttention />
        <Module03RotaryEmbeddings />
        <Module04GroupedQueryAttention />
        <Module05DecoderBlock />
        <Module06HowModelsLearn />
        <Module07SupervisedFineTuning />
        <Module08RetrievalAugmentation />
        <Module09ModelEvaluation />
      </div>
    </main>
  );
};
