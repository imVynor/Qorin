import React from 'react';
import { DataPipelineDiagram } from '../components/visualizations/DataPipelineDiagram';
import { DatasetInventoryTable } from '../components/visualizations/DatasetInventoryTable';
import { datasetsData } from '../data/datasetsData';
import styles from './About.module.css';

export const About: React.FC = () => {
  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Identity &bull; Provenance &bull; Dataset Catalogue</div>
          <h1 className={styles.title}>About the QORIN Project</h1>
          <div className={styles.lead}>
            The academic origins, engineering philosophy, local hardware environment, researchers,
            and comprehensive dataset inventory powering the QORIN investigation.
          </div>
        </div>
      </header>

      {/* 1. Project Identity & Central Inquiry */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>1. Project Identity &amp; The Research Question</h2>
          <div className={styles.proseMeasure}>
            <p>
              <strong>QORIN</strong> (<em>The Quiet Origin of Reasoning and Intelligence Networks</em>) is an independent
              language model research program built entirely from scratch. Unlike projects that fine-tune existing foundation
              models or wrap proprietary commercial APIs, QORIN trains decoder-only Transformers from <strong>randomly initialized
              weights</strong> on consumer-grade hardware.
            </p>
            <p>
              The central research question driving QORIN is:
            </p>
            <blockquote style={{ borderLeft: '3px solid var(--color-accent-indigo)', paddingLeft: 'var(--space-4)', margin: 'var(--space-4) 0', fontStyle: 'italic', color: 'var(--color-ink-primary)' }}>
              &ldquo;How do modern architectural invariants (GQA, RoPE, SwiGLU, RMSNorm), token-weighted optimization objectives,
              and balanced data curricula interact under strict consumer hardware budgets, and at what parameter density do
              verifiable reasoning circuits emerge?&rdquo;
            </blockquote>
            <p>
              By training models ourselves, we gain complete visibility into every gradient update, optimizer state,
              token distribution, and behavioral failure mode—insights that are impossible to obtain through closed-source APIs.
            </p>
          </div>
        </div>
      </section>

      {/* 2. Project History & Evolution */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>2. Project History &amp; Academic Evolution</h2>
          <div className={styles.proseMeasure}>
            <p>
              The project originated in October 2026 under the academic proposal title:
            </p>
            <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', background: 'var(--color-paper-subtle)', padding: '10px 14px', borderRadius: 'var(--radius-xs)', border: '1px solid var(--color-rule-subtle)' }}>
              &ldquo;Tool-Augmented Small Language Models: Building and Evaluating a Lightweight Transformer&rdquo;
            </p>
            <p>
              The initial hypothesis posited that system-level external tools (lexical retrieval, calculator modules, and external memory)
              could compensate for limited internal parameter capacity in sub-100M language models.
            </p>
            <p>
              However, our early empirical capability evaluations revealed a fundamental roadblock: <strong>the syntax-execution disparity</strong>.
              A model that produces syntactically flawless Python or JSON (90% formatting accuracy) but fails functional unit-test execution (0.0% accuracy)
              cannot reliably invoke external tools or interpret retrieved context.
            </p>
            <p>
              Consequently, the research program expanded its scope to systematically investigate core architectural scaling:
            </p>
            <ul style={{ paddingLeft: '1.2rem', margin: 'var(--space-4) 0', lineHeight: 1.7, fontSize: 'var(--text-sm)', color: 'var(--color-ink-secondary)' }}>
              <li><strong>Generation I (QORIN-16M):</strong> Established training pipeline stability, BPE tokenization, and confirmed the syntax-execution bottleneck.</li>
              <li><strong>Generation II (QORIN-50M):</strong> Conducted iso-data architectural ablations comparing Multi-Head Attention, Grouped-Query Attention (GQA-4), and Deep-Thin topologies.</li>
              <li><strong>Generation III (QORIN-100M):</strong> Extended context from 1K to 4K tokens via RoPE frequency scaling and diagnosed passkey needle-in-a-haystack retrieval limits (4.0%).</li>
              <li><strong>Generation IV (QORIN-500M):</strong> Formulated token-weighted invariant cross-entropy, curated a verified 5B-token pretraining corpus across 101 shards, and launched the 500M pretraining campaign.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 3. The Research Team */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>3. Research Authors &amp; Core Contributions</h2>
          <div className={styles.proseMeasure}>
            <p>
              QORIN is conceived and executed by an independent two-person research collaboration:
            </p>
          </div>

          <div className={styles.twoCol}>
            <div className={styles.card}>
              <div className={styles.cardTag}>Co-Lead Researcher &amp; Architect</div>
              <div className={styles.cardHeading}>Tumarada Sai Anish Raj</div>
              <p className={styles.cardText}>
                Primary responsibility for model architecture specification (GQA, RoPE, SwiGLU implementations), custom Byte-Pair
                Encoding tokenizer training (16k and 32k vocabularies), mathematical derivation and unit-testing of token-weighted
                invariant cross-entropy loss, and distributed pretraining engine orchestration.
              </p>
            </div>

            <div className={styles.card}>
              <div className={styles.cardTag}>Co-Lead Researcher &amp; Systems Engineer</div>
              <div className={styles.cardHeading}>Depuru Jnan Deep</div>
              <p className={styles.cardText}>
                Primary responsibility for consumer GPU performance profiling, automated thermal duty-cycle management (105m active /
                15m cool), dataset acquisition and cryptographic shard verification, passkey needle-in-a-haystack long-context
                benchmarks, and multi-tier capability evaluation infrastructure.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Hardware & Engineering Environment */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>4. Hardware &amp; Engineering Infrastructure</h2>
          <div className={styles.proseMeasure}>
            <p>
              A cornerstone of the QORIN investigation is physical transparency regarding hardware constraints.
              All pretraining, fine-tuning, and diagnostic evaluations are executed on a single consumer laptop machine:
            </p>

            <ul style={{ paddingLeft: '1.2rem', margin: 'var(--space-4) 0', lineHeight: 1.7, fontSize: 'var(--text-sm)', color: 'var(--color-ink-secondary)' }}>
              <li><strong>Physical GPU:</strong> Single NVIDIA GeForce RTX 5060 Laptop GPU (8,192 MB GDDR7 dedicated VRAM).</li>
              <li><strong>Compute Platform:</strong> PyTorch 2.5.1 + CUDA 12.4 on x86_64 host system.</li>
              <li><strong>Memory Optimizations:</strong> BF16 mixed-precision training, gradient accumulation, and activation gradient checkpointing.</li>
              <li><strong>Sustained Throughput:</strong> ~5,054 tokens/second during 500M pretraining (micro-batch size 2, gradient accumulation 16, sequence length 2,048).</li>
              <li><strong>Thermal Management:</strong> Automated hardware watchdog enforcing a 105-minute active pretraining window followed by a 15-minute idle cooling period to maintain GPU junction temperatures below 78&deg;C.</li>
              <li><strong>Hardware Scope Note:</strong> Multi-node cluster setups or enterprise H100 clusters were merely analyzed for comparative theoretical runtime scaling; all actual weights and runs documented on this site originated on the physical RTX 5060 setup.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. Research Philosophy */}
      <section className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>5. Research Philosophy: Open Science &amp; Negative Results</h2>
          <div className={styles.proseMeasure}>
            <p>
              In contemporary machine learning literature, negative results and training failures are frequently omitted,
              creating a distorted perception of seamless convergence. QORIN adheres strictly to four principles:
            </p>
            <ol style={{ paddingLeft: '1.2rem', margin: 'var(--space-4) 0', lineHeight: 1.7, fontSize: 'var(--text-sm)', color: 'var(--color-ink-secondary)' }}>
              <li><strong>Transparent Failure Analysis:</strong> We explicitly publish our catastrophic conversational collapse in SFT v3 and our 4.0% passkey retrieval failure rather than obscuring them behind low cross-entropy scores.</li>
              <li><strong>Cryptographic Reproducibility:</strong> Every dataset, tokenizer vocabulary, and model checkpoint is fingerprinted with SHA-256 hashes and logged in immutable manifests.</li>
              <li><strong>Parameter Honesty:</strong> We report exact parameter counts directly calculated from physical checkpoint tensors (reconciling 16,768,128 parameters against historical estimates) and explicitly separate non-embedding parameters.</li>
              <li><strong>Distinguishing Intrinsic Capacity from Prompt Scaffolding:</strong> We never conflate system-level prompt engineering or external retrieval with intrinsic model reasoning capacity.</li>
            </ol>
          </div>
        </div>
      </section>

      {/* 6. Comprehensive Dataset Catalogue */}
      <section id="datasets" className={styles.bodySection}>
        <div className={styles.container}>
          <div className={styles.kicker}>Data Governance &bull; {datasetsData.length} Documented Datasets &amp; Benchmarks</div>
          <h2 className={styles.sectionTitle}>6. Comprehensive Dataset Catalogue &amp; Provenance</h2>
          <div className={styles.proseMeasure}>
            <p>
              Complete transparency in data acquisition, domain interleaving, filtering heuristics, and licensing
              is essential for ethical open science and scientific reproducibility. Below is the complete inventory of
              every dataset utilized across QORIN pretraining, instruction tuning, capability evaluation, and retrieval.
            </p>
          </div>

          {/* Data Pipeline Diagram */}
          <DataPipelineDiagram figureNumber={9} />

          {/* Dataset Inventory Table */}
          <DatasetInventoryTable />

          <div className={styles.proseMeasure} style={{ marginTop: 'var(--space-6)' }}>
            <h3 style={{ fontSize: 'var(--text-base)', fontWeight: 600, color: 'var(--color-ink-primary)', marginBottom: 'var(--space-2)' }}>
              Licensing Compliance &amp; Contamination Controls
            </h3>
            <p style={{ fontSize: 'var(--text-xs)', lineHeight: 1.6, color: 'var(--color-ink-secondary)' }}>
              <strong>Pretraining Openness:</strong> All 7 domains in the 5B-token pretraining corpus utilize strictly permissive open-source licenses
              (ODC-By, Apache-2.0, MIT, CC0, and CC-BY-SA 4.0). High-quality candidate datasets with non-commercial restrictions—such as
              Evol-Instruct-Code-80K (CC-BY-NC-SA)—were formally audited and strictly rejected from the production corpus.
            </p>
            <p style={{ fontSize: 'var(--text-xs)', lineHeight: 1.6, color: 'var(--color-ink-secondary)' }}>
              <strong>SFT Pilot Disclosure:</strong> The early 5,000-example SFT pilot incorporated 2,778 examples from Stanford Alpaca, which carries
              a CC-BY-NC 4.0 license and is preserved strictly for non-commercial academic research calibration. Subsequent SFT v2 and v4
              curricula transitioned exclusively to commercially permissible open-source sources.
            </p>
            <p style={{ fontSize: 'var(--text-xs)', lineHeight: 1.6, color: 'var(--color-ink-secondary)' }}>
              <strong>Contamination Isolation:</strong> Pretraining training shards (000&ndash;100) were cryptographically quarantined against
              the 20M validation reserve using 25,137 document SHA-256 hashes, confirming exactly 0.00% validation leakage.
            </p>
          </div>
        </div>
      </section>

      {/* 7. Seminal Academic References */}
      <section id="references" className={styles.bodySection}>
        <div className={styles.container}>
          <h2 className={styles.sectionTitle}>7. Seminal Academic Literature</h2>
          <div className={styles.proseMeasure}>
            <p>
              The architecture, token weighting, and scaling laws implemented in QORIN build upon foundational open literature:
            </p>
          </div>

          <ul className={styles.citationList}>
            <li className={styles.citationItem}>
              <strong>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &amp; Polosukhin, I. (2017).</strong>{' '}
              <em>Attention Is All You Need.</em> Advances in Neural Information Processing Systems (NeurIPS 2017).
            </li>
            <li className={styles.citationItem}>
              <strong>Touvron, H., Lavril, T., Izacard, G., et al. (2023).</strong>{' '}
              <em>LLaMA: Open and Efficient Foundation Language Models.</em> arXiv:2302.13971.
            </li>
            <li className={styles.citationItem}>
              <strong>Shazeer, N. (2020).</strong>{' '}
              <em>GLU Variants Improve Transformer.</em> arXiv:2002.05202.
            </li>
            <li className={styles.citationItem}>
              <strong>Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., &amp; Liu, Y. (2024).</strong>{' '}
              <em>RoFormer: Enhanced Transformer with Rotary Position Embedding.</em> Neurocomputing, 568, 127063.
            </li>
            <li className={styles.citationItem}>
              <strong>Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022).</strong>{' '}
              <em>Training Compute-Optimal Large Language Models.</em> arXiv:2203.15556 (Chinchilla).
            </li>
            <li className={styles.citationItem}>
              <strong>Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., &amp; Sanghai, S. (2023).</strong>{' '}
              <em>GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.</em> EMNLP 2023.
            </li>
            <li className={styles.citationItem}>
              <strong>Zhang, B., &amp; Sennrich, R. (2019).</strong>{' '}
              <em>Root Mean Square Layer Normalization.</em> Advances in Neural Information Processing Systems (NeurIPS 2019).
            </li>
          </ul>
        </div>
      </section>
    </main>
  );
};
