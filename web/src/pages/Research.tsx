import React from 'react';
import { Link } from 'react-router-dom';
import { Award, AlertTriangle } from 'lucide-react';
import { stage15BlindEvalData } from '../data/chartData';
import { ControlledAblationFigure } from '../components/visualizations/ControlledAblationFigure';
import { MathView } from '../components/common/MathView';
import styles from './Research.module.css';

export const Research: React.FC = () => {
  return (
    <main className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.container}>
          <div className={styles.kicker}>Machine Learning Research Monograph</div>
          <h1 className={styles.title}>Research Foundations &amp; Empirical Methodology</h1>
          <div className={styles.lead}>
            A rigorous technical specification of the problem formulation, machine learning principles,
            optimization dynamics, baseline methodology, multi-tier evaluation taxonomy, and empirical findings
            of the QORIN investigation.
          </div>
        </div>
      </header>

      {/* Navigation Quicklinks for Monograph Sections */}
      <nav className={styles.section} style={{ padding: 'var(--space-6) 0', background: 'var(--color-surface)' }}>
        <div className={styles.container}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--color-ink-muted)', marginBottom: 'var(--space-3)' }}>
            Monograph Contents
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 16px', fontSize: '12px' }}>
            <a href="#sec-1" style={{ color: 'var(--color-accent-indigo)' }}>1. Problem Formulation</a>
            <a href="#sec-2" style={{ color: 'var(--color-accent-indigo)' }}>2. Motivation</a>
            <a href="#sec-3" style={{ color: 'var(--color-accent-indigo)' }}>3. ML vs DL Foundations</a>
            <a href="#sec-4" style={{ color: 'var(--color-accent-indigo)' }}>4. Data Composition</a>
            <a href="#sec-5" style={{ color: 'var(--color-accent-indigo)' }}>5. Preprocessing &amp; Loss</a>
            <a href="#sec-6" style={{ color: 'var(--color-accent-indigo)' }}>6. Baseline Selection</a>
            <a href="#sec-7" style={{ color: 'var(--color-accent-indigo)' }}>7. Candidate Architectures</a>
            <a href="#sec-8" style={{ color: 'var(--color-accent-indigo)' }}>8. Experimental Design</a>
            <a href="#sec-9" style={{ color: 'var(--color-accent-indigo)' }}>9. Training &amp; Optimization</a>
            <a href="#sec-10" style={{ color: 'var(--color-accent-indigo)' }}>10. Evaluation Taxonomy</a>
            <a href="#sec-11" style={{ color: 'var(--color-accent-indigo)' }}>11. Empirical Findings</a>
            <a href="#sec-12" style={{ color: 'var(--color-accent-indigo)' }}>12. Threats to Validity</a>
            <a href="#sec-13" style={{ color: 'var(--color-accent-indigo)' }}>13. Limitations &amp; Frontiers</a>
            <a href="#sec-14" style={{ color: 'var(--color-accent-indigo)' }}>14. References</a>
          </div>
        </div>
      </nav>

      {/* 1. Problem Formulation and Scope */}
      <section id="sec-1" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 01 &bull; Problem Formulation &amp; Scope</div>
          <h2 className={styles.sectionTitle}>1. Problem Formulation and Operational Scope</h2>
          <div className={styles.proseMeasure}>
            <p>
              This investigation examines whether external retrieval and computational tools can compensate for
              constrained neural parameter capacity in small language models.
            </p>

            <div className={styles.calloutBox}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, color: 'var(--color-accent-indigo)', marginBottom: '6px' }}>
                <Award size={13} />
                <span>CENTRAL RESEARCH QUESTION</span>
              </div>
              <blockquote style={{ margin: 0, fontStyle: 'italic', fontSize: '15px', color: 'var(--color-ink-primary)' }}>
                &ldquo;How does increasing the capacity of a small decoder-only Transformer affect conversational LM performance,
                and can external retrieval compensate for smaller model capacity?&rdquo;
              </blockquote>
            </div>

            <p>
              To give this question empirical precision, we operationally define each construct:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)', color: 'var(--color-ink-secondary)' }}>
              <li>
                <strong>Conversational Language-Model Performance:</strong> Evaluated across four joint criteria:
                (a) next-token cross-entropy perplexity on held-out prose,
                (b) structural syntax validity (AST code syntax and RFC 8259 JSON parsing),
                (c) factual proposition correctness against verified evidence without hallucination, and
                (d) multi-turn conversational stability without degenerative repetition or contextual amnesia.
              </li>
              <li>
                <strong>Model Capacity:</strong> The expressive volume of the neural network, formalized as total parameter count{' '}
                <MathView math="N \in [1.68 \times 10^7, 5.14 \times 10^8]" />, hidden state dimensionality{' '}
                <MathView math="d_{\text{model}} \in [384, 1536]" />, layer depth <MathView math="L \in [6, 20]" />, and attention head topology.
              </li>
              <li>
                <strong>Experimental Unit:</strong> A multi-turn dialogue session comprising an initial user prompt,
                intermediate model decisions (direct answer vs. tool invocation), external environment feedback
                (retrieval snippets or execution stdout), and final grounded synthesis.
              </li>
              <li>
                <strong>Information Available at Prediction Time:</strong> In standalone mode, only preceding tokens in the context window{' '}
                <MathView math="x_{<t}" />. In tool-augmented mode, dynamic BM25 knowledge chunks, live search excerpts, or Python subprocess execution output.
              </li>
              <li>
                <strong>Scope of Claims:</strong> Our experiments establish what can and cannot be achieved by small models trained from scratch
                under single-GPU consumer hardware constraints (NVIDIA RTX 5060 8GB VRAM). We do not claim universal artificial intelligence;
                we isolate the trade-off boundary between parametric memory and external tool augmentation.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 2. Research Motivation */}
      <section id="sec-2" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 02 &bull; Motivation</div>
          <h2 className={styles.sectionTitle}>2. Research Motivation: The Lower Bound of Intelligence</h2>
          <div className={styles.proseMeasure}>
            <p>
              Contemporary deep learning disproportionately emphasizes scaling up: training 70B+ parameter models on massive
              clusters costing millions of dollars. While effective, this creates severe environmental, economic, and accessibility
              barriers, leaving edge devices and privacy-preserving local environments unable to run capable AI systems.
            </p>
            <p>
              More critically, brute-force scaling masks fundamental algorithmic questions:
              <em> Must all knowledge be memorized into static neural weights?</em>
              A model storing all human facts in weights is fundamentally inefficient; real-time facts change constantly,
              and arithmetic calculations require exact symbolic manipulation rather than stochastic next-token probability guessing.
            </p>
            <p>
              By training small language models completely from scratch and augmenting them with runtime tools (retrieval engines
              and Python execution sandboxes), we explore a compute-optimal alternative: a lightweight 514M neural reasoning engine
              orchestrating deterministic external tools.
            </p>
          </div>
        </div>
      </section>

      {/* 3. Machine Learning Framing vs Deep Learning */}
      <section id="sec-3" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 03 &bull; Theoretical Framing</div>
          <h2 className={styles.sectionTitle}>3. Machine Learning Foundations vs. Deep Learning Architecture</h2>
          <div className={styles.proseMeasure}>
            <p>
              We draw a strict distinction between deep learning as an architectural function class and machine learning
              as the empirical science of generalization:
            </p>

            <div className={styles.frameworkGrid}>
              <div className={styles.frameworkCard}>
                <div className={styles.cardTag}>Deep Learning (The Architecture)</div>
                <div className={styles.cardHeading}>Nonlinear Function Approximation</div>
                <ul className={styles.featureList}>
                  <li>Transformer blocks, multi-head attention, and SwiGLU feed-forward networks.</li>
                  <li>Rotary Position Embeddings (RoPE) and RMSNorm layer stabilization.</li>
                  <li>Grouped Query Attention (GQA 6:1) reducing KV-cache VRAM by 83.3%.</li>
                  <li>Auto-regressive next-token conditional probability modeling <MathView math="P_\theta(x_t \mid x_{<t})" />.</li>
                </ul>
              </div>

              <div className={styles.frameworkCard}>
                <div className={styles.cardTag}>Machine Learning (The Science)</div>
                <div className={styles.cardHeading}>Generalization, Bias, and Risk</div>
                <ul className={styles.featureList}>
                  <li>Empirical Risk Minimization (ERM) vs. true out-of-distribution generalization.</li>
                  <li>Capacity vs. sample complexity: the Chinchilla scaling relationship.</li>
                  <li>Inductive bias: causal attention masking and loss masking on assistant tokens.</li>
                  <li>Data contamination prevention: strict cryptographic SHA-256 train/val/test isolation.</li>
                </ul>
              </div>
            </div>

            <p>
              The research focuses squarely on machine learning questions: Does increasing parameter capacity reduce inductive bias?
              Does supervised fine-tuning lead to distribution shift or catastrophic forgetting?
              And how do non-parametric tools alter the effective sample complexity required for factual reasoning?
            </p>
          </div>
        </div>
      </section>

      {/* 4. Data Composition and Provenance */}
      <section id="sec-4" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 04 &bull; Corpus Architecture</div>
          <h2 className={styles.sectionTitle}>4. Data Understanding, Composition, and Provenance</h2>
          <div className={styles.proseMeasure}>
            <p>
              A language model is strictly bounded by the distribution, diversity, and filtering rigor of its training data.
              Our project maintains a multi-phase data regimen with exact token accounting and cryptographic verification:
            </p>

            <h3 className={styles.subheading}>Pretraining Corpus (5,000,000,000 Token Target)</h3>
            <p>
              Assembled across 7 balanced domains into 101 binary shards, verified with SHA-256 checksums:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li><strong>Quality Web (FineWeb-Edu):</strong> 40.0% (2.0B tokens, 40 shards) &bull; Filtered for educational prose and conceptual clarity.</li>
              <li><strong>Educational Text (Cosmopedia v2):</strong> 15.0% (750M tokens, 15 shards) &bull; High-density synthetic textbooks and course notes.</li>
              <li><strong>Source Code (StarCoder Python):</strong> 15.0% (750M tokens, 15 shards) &bull; Cleaned, syntax-verified Python code and documentation.</li>
              <li><strong>Curated Literature (Books):</strong> 10.0% (500M tokens, 10 shards) &bull; Long-form narrative structure and discourse coherence.</li>
              <li><strong>Wikipedia Clean Text:</strong> 7.5% (375M tokens, 8 shards) &bull; Encyclopedic factual reference and entity definitions.</li>
              <li><strong>Synthetic Mathematics (OpenMathInstruct):</strong> 7.5% (375M tokens, 8 shards) &bull; Step-by-step arithmetic derivations and proofs.</li>
              <li><strong>Curated Dialogue (UltraChat 200k):</strong> 5.0% (250M tokens, 5 shards) &bull; Strictly capped to prevent conversational attractor bias.</li>
            </ul>

            <h3 className={styles.subheading}>Held-Out Capability Benchmark Split (100 Untouched Records)</h3>
            <p>
              The held-out evaluation dataset establishes a strictly isolated, untouched 100-record benchmark with zero overlap against any pretraining or fine-tuning split:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li><strong>Corpus A (Multi-Turn Tool Chaining):</strong> 40 records requiring initial Web research followed by observational comprehension.</li>
              <li><strong>Corpus B (Numerical &amp; Algorithmic Tasks):</strong> 30 records requiring Python sandbox code synthesis and live calculation.</li>
              <li><strong>Corpus C (Evidence-Grounded Synthesis):</strong> 30 records evaluating factual citation fidelity, partial support, and contradictory source handling.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 5. Preprocessing and Loss Formulation */}
      <section id="sec-5" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 05 &bull; Statistical Preprocessing</div>
          <h2 className={styles.sectionTitle}>5. Preprocessing, Masking, and Loss Formulation</h2>
          <div className={styles.proseMeasure}>
            <p>
              Naïve token-level autoregression during supervised fine-tuning introduces severe statistical distortions.
              We instituted three core mathematical and data engineering protocols:
            </p>

            <h3 className={styles.subheading}>1. Custom Byte-Pair Encoding (BPE) Tokenization</h3>
            <p>
              Trained custom tokenizers using the HuggingFace tokenizers library: BPE-16K (16,384 tokens) for early 16M/50M/100M models,
              and BPE-32K (32,768 tokens, fingerprint <code>3559c7435e1d9330</code>) for QORIN-500M.
              The vocabulary incorporates dedicated special tokens:
              <code>&lt;|im_start|&gt;</code>, <code>&lt;|im_end|&gt;</code>, <code>&lt;tool_call&gt;</code>,
              <code>&lt;/tool_call&gt;</code>, <code>&lt;tool_response&gt;</code>, and <code>&lt;/tool_response&gt;</code>.
            </p>

            <h3 className={styles.subheading}>2. Supervised Target Loss Masking</h3>
            <p>
              During SFT, backpropagating gradients on user instructions teaches the model to predict user prompts rather than
              learn to answer. We apply a binary mask <MathView math="m_{b,t} \in \{0, 1\}" />:
            </p>
            <div className={styles.mathBlock}>
              <MathView
                math="\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{\sum_{b=1}^B \sum_{t=1}^T m_{b,t}} \sum_{b=1}^{B} \sum_{t=1}^{T} m_{b,t} \log P_\theta(y_{b,t} \mid x_{b,<t})"
                block
              />
            </div>
            <p>
              where <MathView math="m_{b,t} = 1" /> strictly for assistant response tokens and <MathView math="m_{b,t} = 0" /> for system prompts, user turns, and padding tokens.
            </p>

            <h3 className={styles.subheading}>3. Token-Weighted Loss Normalization</h3>
            <p>
              Standard PyTorch micro-batch loss averaging divides by batch sequence count <MathView math="B" />. When sequence lengths vary
              (e.g., 20 tokens vs. 800 tokens), short sequences exert an 80:1 per-token gradient distortion:
            </p>
            <div className={styles.mathBlock}>
              <MathView
                math="\mathcal{L}_{\text{TW}} = \frac{\sum_{k=1}^K L_k}{\sum_{k=1}^K N_k} \quad \text{vs.} \quad \mathcal{L}_{\text{naive}} = \frac{1}{K} \sum_{k=1}^K \frac{L_k}{N_k}"
                block
              />
            </div>
            <p>
              Our Token-Weighted Cross-Entropy normalizes gradients globally by total active tokens across micro-batches,
              guaranteeing exact mathematical equivalence to single large-batch training.
            </p>
          </div>
        </div>
      </section>

      {/* 6. Baseline Selection */}
      <section id="sec-6" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 06 &bull; Baselines</div>
          <h2 className={styles.sectionTitle}>6. Baseline Methodology and Reference Configurations</h2>
          <div className={styles.proseMeasure}>
            <p>
              To isolate capacity and tool augmentation cleanly, we established clear reference baselines:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li>
                <strong>Architectural Baseline:</strong> Decoder-only causal Transformer. We chose decoder-only over encoder-decoder (T5)
                because unified auto-regressive generation naturally supports conversational multi-turn dialogue, scratchpad reasoning,
                and inline tool call generation within a single uniform sequence.
              </li>
              <li>
                <strong>Parametric Baseline Model:</strong> QORIN-16M (16.8M parameters, 102M pretraining tokens) established the baseline for basic syntax.
              </li>
              <li>
                <strong>Single-Turn Dispatch Baseline:</strong> Frozen Experiment D Step 425 checkpoint (514,389,504 parameters).
                This baseline possessed balanced single-turn tool dispatch and high direct-answer accuracy (96.5%), but had zero multi-turn sequential chaining capability (0/30 in held-out tests).
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 7. Candidate Architectures and Attention Ablations */}
      <section id="sec-7" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 07 &bull; Architectural Exploration</div>
          <h2 className={styles.sectionTitle}>7. Candidate Architectures and Attention Ablations</h2>
          <div className={styles.proseMeasure}>
            <p>
              Before settling on QORIN-500M's final architecture, we conducted rigorous ablation experiments on 50M parameter variants:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li>
                <strong>Multi-Head Attention (8Q/8KV) vs. Grouped-Query Attention (8Q/4KV):</strong>
                GQA achieved identical validation perplexity (22.67 vs 22.54) while cutting KV-cache memory from 192MB to 96MB per batch of 16,
                enabling twice the batch size on our 8GB GPU.
                <div style={{ margin: 'var(--space-3) 0' }}>
                  <MathView
                    math="\text{Attention}(Q_i, K_{\lfloor i/g \rfloor}, V_{\lfloor i/g \rfloor}) = \text{softmax}\left( \frac{Q_i K_{\lfloor i/g \rfloor}^T}{\sqrt{d_k}} \right) V_{\lfloor i/g \rfloor}"
                    block
                  />
                </div>
                where <MathView math="g = H_Q / H_{KV} = 24 / 4 = 6" /> in QORIN-500M.
              </li>
              <li>
                <strong>Deep-Thin (16 layers, 448 dim, 7Q/7KV):</strong>
                Suffered 56% higher time-to-first-token latency (14.58ms vs 9.82ms) and 28% slower token decoding with zero perplexity gain.
              </li>
              <li>
                <strong>SwiGLU Gated Feed-Forward Activation:</strong>
                <div style={{ margin: 'var(--space-3) 0' }}>
                  <MathView
                    math="\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}} \right) W_{\text{down}}"
                    block
                  />
                </div>
                Hidden dimension expanded to <MathView math="d_{\text{ffn}} = 3840" /> (2.5&times; expansion ratio).
              </li>
              <li>
                <strong>Final QORIN-500M Configuration:</strong>
                20 layers, 1536 hidden dimension, 24 Query heads, 4 KV heads (6:1 GQA ratio), SwiGLU hidden dim 3840,
                RoPE rotary embeddings (<MathView math="\theta = 10000" />), Pre-RMSNorm (<MathView math="\epsilon = 10^{-6}" />), and tied token embeddings.
                Verified total parameters: <strong>514,389,504</strong>.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 8. Experimental Design */}
      <section id="sec-8" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 08 &bull; Factorial Matrix</div>
          <h2 className={styles.sectionTitle}>8. Experimental Design: The 2&times;2 Factorial Framework</h2>
          <div className={styles.proseMeasure}>
            <p>
              To resolve our central research question, we evaluated models across a controlled 2&times;2 factorial design:
            </p>
            <div className={styles.taxonomyTableWrap}>
              <table className={styles.taxonomyTable}>
                <thead>
                  <tr>
                    <th>Factorial Cell</th>
                    <th>Model Configuration</th>
                    <th>Runtime Tooling</th>
                    <th>Primary Capability Tested</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className={styles.metricName}>Cell 1 (Low Cap / Standalone)</td>
                    <td>QORIN-16M &amp; QORIN-50M</td>
                    <td>None (Raw parametric generation)</td>
                    <td>Baseline internal syntax and memorization ceiling</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Cell 2 (High Cap / Standalone)</td>
                    <td>QORIN-500M Base Checkpoint</td>
                    <td>None (Raw parametric generation)</td>
                    <td>Parametric scaling impact on syntax, reasoning, and hallucination</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Cell 3 (Low Cap / Augmented)</td>
                    <td>QORIN-16M / 50M + Harness</td>
                    <td>BM25 Retrieval + Python Sandbox</td>
                    <td>Can external tools compensate for small parameter count?</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Cell 4 (High Cap / Augmented)</td>
                    <td>QORIN-500M Tool-Chaining + Harness</td>
                    <td>Web Research + Python + Evidence Validator</td>
                    <td>Autonomous sequential chaining &amp; zero-hallucination synthesis</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p>
              In all comparative evaluations, we held learning rate schedules, optimizer hyperparameters (AdamW, <MathView math="\beta_1=0.9, \beta_2=0.95" />, weight decay <MathView math="\lambda=0.1" />),
              and evaluation test records constant.
            </p>
          </div>
        </div>
      </section>

      {/* 9. Comprehensive Training & Optimization Methodology (Detailed Section) */}
      <section id="sec-9" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 09 &bull; Optimization Mechanics</div>
          <h2 className={styles.sectionTitle}>9. Training and Optimization Methodology</h2>
          <div className={styles.proseMeasure}>
            <p>
              Training a 514M parameter model from scratch on a single consumer GPU (NVIDIA RTX 5060, 8GB VRAM)
              demands complete mathematical and algorithmic discipline. We document the exact training recipes,
              loss objectives, optimizer formulations, and operational watchdogs that governed our experiments.
            </p>

            <h3 className={styles.subheading}>Optimizer Formulation &amp; Decoupled Weight Decay</h3>
            <p>
              Pretraining utilizes standard AdamW, while supervised fine-tuning employs 8-bit AdamW
              (<code>bitsandbytes.optim.AdamW8bit</code>) to conserve 1.03 GB of GPU memory for optimizer states.
              The parameter update update rule decouples weight decay from adaptive moment estimates:
            </p>
            <div className={styles.mathBlock}>
              <MathView
                math="\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2"
                block
              />
              <MathView
                math="\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}"
                block
              />
              <MathView
                math="\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1} \right)"
                block
              />
            </div>
            <p>
              We enforce <MathView math="\beta_1 = 0.9" />, <MathView math="\beta_2 = 0.95" />, <MathView math="\epsilon = 10^{-8}" />,
              and decoupled weight decay <MathView math="\lambda = 0.1" />. Global gradient norms are strictly clipped to{' '}
              <MathView math="\|\mathbf{g}\|_2 \le 1.0" /> before each optimizer step to prevent catastrophic gradient spikes.
            </p>

            <h3 className={styles.subheading}>Cosine Learning Rate Schedule with Linear Warmup</h3>
            <p>
              The learning rate schedule follows a piecewise trajectory: linear warmup from zero to <MathView math="\eta_{\max}" /> over{' '}
              <MathView math="T_{\text{warmup}}" /> updates, followed by half-period cosine decay down to <MathView math="\eta_{\min} = 0.1 \times \eta_{\max}" />:
            </p>
            <div className={styles.mathBlock}>
              <MathView
                math="\eta_t = \begin{cases} \frac{t}{T_{\text{warmup}}} \eta_{\max}, & t < T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}} \pi\right)\right), & t \ge T_{\text{warmup}} \end{cases}"
                block
              />
            </div>
            <p>
              <strong>Pretraining Hyperparameters:</strong> <MathView math="\eta_{\max} = 3.0 \times 10^{-4}" />, <MathView math="\eta_{\min} = 3.0 \times 10^{-5}" />,{' '}
              <MathView math="T_{\text{warmup}} = 2000" /> updates. This peak was selected following Chinchilla scaling recommendations for models of width <MathView math="d=1536" />.
            </p>
            <p>
              <strong>Supervised Tool Fine-Tuning:</strong> <MathView math="\eta_{\max} = 1.5 \times 10^{-5}" />, <MathView math="\eta_{\min} = 1.5 \times 10^{-6}" />,{' '}
              <MathView math="T_{\text{warmup}} = 20" /> updates over 100 total updates. A conservative learning rate is critical during SFT: higher rates (<MathView math="\ge 5.0 \times 10^{-5}" />)
              cause catastrophic forgetting of pretrained linguistic representations, destroying general reasoning.
            </p>

            <h3 className={styles.subheading}>Batching, Gradient Accumulation, and Precision</h3>
            <p>
              With sequence length <MathView math="T = 2048" /> tokens and an 8GB VRAM envelope:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li><strong>Pretraining:</strong> Micro-batch size 1, gradient accumulation steps 16 &rarr; Effective batch size 16 sequences (<MathView math="32,768" /> tokens per parameter update).</li>
              <li><strong>Tool-Chaining SFT:</strong> Micro-batch size 1, gradient accumulation steps 4 &rarr; Effective batch size 4 sequences (<MathView math="8,192" /> tokens per parameter update).</li>
              <li><strong>Precision:</strong> Bfloat16 (<code>torch.bfloat16</code>) mixed precision via PyTorch AMP. BF16 retains the full 8-bit dynamic range of FP32, eliminating the loss-scaling instability inherent to FP16.</li>
              <li><strong>Activation Checkpointing:</strong> Gradient checkpointing enabled across all 20 Transformer blocks, recalculating attention activations during backpropagation to reduce peak activation memory by 68%.</li>
            </ul>

            <h3 className={styles.subheading}>Thermal Management and Atomic Checkpointing</h3>
            <p>
              To guarantee hardware longevity and numerical reproducibility on a mobile workstation:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li><strong>Thermal Watchdog Cycle:</strong> Mandatory duty cycle of 30 minutes active training followed by a 5-minute cooling cooldown guard, keeping GPU die temperature below 74&deg;C and eliminating thermal clock throttling.</li>
              <li><strong>Validation Cadence:</strong> Validation loss evaluated every 10 updates during SFT and every 250 updates during pretraining on untouched validation shards.</li>
              <li><strong>Atomic Checkpointing:</strong> Checkpoints are written to a temporary file (<code>ckpt.tmp</code>) and committed via atomic filesystem rename (<code>os.replace</code>), preventing corrupted or partial checkpoint states in case of process termination.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 10. Evaluation Design and Multi-Tier Taxonomy */}
      <section id="sec-10" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 10 &bull; Measurement Framework</div>
          <h2 className={styles.sectionTitle}>10. Evaluation Design and Multi-Tier Taxonomy</h2>
          <div className={styles.proseMeasure}>
            <p>
              A central thesis of QORIN is that single-number metrics (such as validation loss or BLEU) are fundamentally deceptive.
              We measure system performance across a 5-tier evaluation taxonomy:
            </p>
            <div className={styles.taxonomyTableWrap}>
              <table className={styles.taxonomyTable}>
                <thead>
                  <tr>
                    <th>Tier</th>
                    <th>Metric Name</th>
                    <th>Formal Definition &amp; Target</th>
                    <th>Experimental Significance</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className={styles.metricName}>Tier 1: Statistical</td>
                    <td>Cross-Entropy Loss &amp; Perplexity</td>
                    <td>
                      <MathView math="\text{PPL} = \exp\left( -\frac{1}{N} \sum_{t=1}^N \log P(x_t) \right)" />
                    </td>
                    <td>Evaluates general distribution fit; drops monotonically with scale (47.53 &rarr; 15.08).</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Tier 2: Syntax</td>
                    <td>AST &amp; JSON Parse Validity</td>
                    <td>Valid Python AST parse and RFC 8259 compliance (%)</td>
                    <td>Measures superficial grammar conformity; saturates early (87.5% in 50M).</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Tier 3: Execution</td>
                    <td>Functional Pass Rate (pass@1)</td>
                    <td>Deterministic execution against unit test suites</td>
                    <td>Exposes the capacity ceiling: 50M achieved 87.5% syntax but only 2.5% functional pass rate.</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Tier 4: Tool Chaining</td>
                    <td>Autonomous Chaining &amp; Decision Precision</td>
                    <td>Tool Intent Precision, Recall, and Sequential 2nd-turn Invocation (%)</td>
                    <td>Measures agency: Autonomous Chaining model achieved 50% 2nd-turn Python chaining (up from 0% in baseline).</td>
                  </tr>
                  <tr>
                    <td className={styles.metricName}>Tier 5: Grounding</td>
                    <td>Evidence Grounding &amp; Citation Precision</td>
                    <td>Proposition support score <MathView math="\ge 0.80" /> and exact bibliographic citation tag ([1], [2])</td>
                    <td>Measures factual honesty: Held-out benchmark achieved 100% full-support precision and 0.0% unsupported claims.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* 11. Empirical Findings: Held-Out Benchmark Analysis */}
      <section id="sec-11" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 11 &bull; Empirical Findings</div>
          <h2 className={styles.sectionTitle}>11. Quantitative Results: Held-Out Capability Benchmark</h2>
          <div className={styles.proseMeasure}>
            <p>
              In our blind capability evaluation on the untouched 100-record held-out benchmark split,
              we evaluated the single-turn dispatch parent baseline against the autonomous tool-chaining model:
            </p>

            <h3 className={styles.subheading}>Multi-Turn Chaining &amp; Synthesis Comparison</h3>
            <div className={styles.taxonomyTableWrap}>
              <table className={styles.taxonomyTable}>
                <thead>
                  <tr>
                    <th>Evaluation Metric</th>
                    <th>Parent Baseline (Single-Turn)</th>
                    <th>Autonomous Tool-Chaining (Step 100)</th>
                    <th>Absolute Delta</th>
                    <th>Academic Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {stage15BlindEvalData.map((row, idx) => (
                    <tr key={idx}>
                      <td className={styles.metricName}>{row.metric}</td>
                      <td>{row.parentD} ({row.parentDValue.toFixed(1)}%)</td>
                      <td><strong>{row.pilotF100} ({row.pilotF100Value.toFixed(1)}%)</strong></td>
                      <td style={{ color: row.pilotF100Value >= row.parentDValue ? 'var(--color-accent-sage)' : 'var(--color-accent-coral)' }}>
                        {row.pilotF100Value >= row.parentDValue ? `+${(row.pilotF100Value - row.parentDValue).toFixed(1)}%` : `${(row.pilotF100Value - row.parentDValue).toFixed(1)}%`}
                      </td>
                      <td style={{ fontSize: '11px', color: 'var(--color-ink-secondary)' }}>{row.interpretation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p style={{ marginTop: 'var(--space-6)' }}>
              <strong>Runtime Evidence Grounding &amp; State Isolation:</strong> Complementing model weights, runtime harness engineering
              introduced structured evidence records ([E1], [E2]), proposition-level grounding validation, and cross-request memory isolation.
              All regression tests in the repository test suite pass with 100% precision.
            </p>

            <h3 className={styles.subheading} style={{ marginTop: 'var(--space-8)' }}>
              11.2 Controlled Tool-Augmentation Ablation: Testing the Central Research Hypothesis
            </h3>
            <p>
              To definitively answer our central research question—<em>"Do external tools actually improve conversational LM performance when the underlying model is held constant, and can external retrieval compensate for smaller model capacity?"</em>—we conducted a controlled ablation on the 100-record Stage 13 held-out test split.
            </p>

            <div className={styles.calloutBox}>
              <div style={{ fontWeight: 700, color: 'var(--color-accent-indigo)', marginBottom: '4px' }}>
                FORMAL HYPOTHESIS &amp; STATISTICAL FRAMEWORK
              </div>
              <p style={{ fontSize: '13px', margin: '4px 0 8px 0', color: 'var(--color-ink-primary)' }}>
                Let <MathView math="Y_i \in \{0, 1\}" /> denote task success on record <MathView math="i \in \{1, \dots, N\}" />. We test:
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '13px', fontFamily: 'var(--font-mono)' }}>
                <div>H₀: P(Y = 1 | Tools Enabled) = P(Y = 1 | Tools Disabled) &nbsp;[Δ = 0]</div>
                <div>H₁: P(Y = 1 | Tools Enabled) &gt; P(Y = 1 | Tools Disabled) &nbsp;[Δ &gt; 0]</div>
              </div>
              <p style={{ fontSize: '12px', margin: '8px 0 0 0', color: 'var(--color-ink-secondary)' }}>
                Because observations are paired on identical tasks, we assess significance using McNemar's paired test with continuity correction:
              </p>
              <div style={{ margin: '6px 0' }}>
                <MathView math="\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} = \frac{(|26 - 5| - 1)^2}{26 + 5} = \frac{400}{31} \approx 12.9032 \implies p = 0.000328" />
              </div>
            </div>

            <ControlledAblationFigure figureNumber={8} />

            <div style={{ marginTop: 'var(--space-4)' }}>
              <h4 style={{ fontFamily: 'var(--font-serif)', fontSize: '16px', color: 'var(--color-ink-primary)', margin: 'var(--space-3) 0 var(--space-2) 0' }}>
                Key Empirical Insights from the Controlled Ablation
              </h4>
              <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.8, fontSize: 'var(--text-sm)' }}>
                <li>
                  <strong>Statistically Significant 500M Tool Advantage:</strong> Enabling tools elevated overall task success from <strong>43.0% (43/100)</strong> [95% CI: 34–53%] to <strong>64.0% (64/100)</strong> [95% CI: 55–74%], representing an absolute gain of <strong>+21.0% pts</strong> (relative gain +48.8%, McNemar <MathView math="p = 0.000328" />).
                </li>
                <li>
                  <strong>Refutation of the Simple Compensation Hypothesis (The 100M Ceiling):</strong> When tested on QORIN-100M, tool availability caused a net regression from <strong>36.0% (36/100)</strong> down to <strong>29.0% (29/100)</strong> (&minus;7.0% pts). At 100M parameters, the model lacks sufficient representational capacity to format tool calls and parse retrieved snippets, leading to token distraction and malformed syntax. External tools <em>cannot compensate for missing representational capacity below the critical threshold</em>.
                </li>
                <li>
                  <strong>Disproportionate Domain Gains:</strong> Tools unlocked massive improvements in <em>Parametric Knowledge</em> (12.5% &rarr; 100.0%, +87.5% pts with 0% false positive tool triggers), <em>Multi-Step Reasoning</em> (36.7% &rarr; 66.7%, +30.0% pts), and <em>Conversational Follow-Ups</em> (25.0% &rarr; 100.0%, +75.0% pts).
                </li>
                <li>
                  <strong>Persistent Edge-Case Blindspots:</strong> On tasks requiring epistemic calibration (Contradictory Evidence, Insufficient Evidence, and Partial Evidence), the model scored 0% success in both conditions, proving that external tools alone cannot resolve epistemic ambiguity without specialized hedging architectures.
                </li>
                <li>
                  <strong>Latency Cost:</strong> Enabling tools increased mean latency from <strong>1.291s</strong> to <strong>2.483s</strong> (a <MathView math="1.92\times" /> overhead factor), driven by multi-turn autoregressive token generation across tool observation cycles (subprocess runtime was &lt;15ms).
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 12. Threats to Validity */}
      <section id="sec-12" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 12 &bull; Methodological Rigor</div>
          <h2 className={styles.sectionTitle}>12. Threats to Validity and Mitigation Protocols</h2>
          <div className={styles.proseMeasure}>
            <p>
              We systematically identified and mitigated potential threats to experimental validity:
            </p>
            <ul style={{ paddingLeft: '1.2rem', lineHeight: 1.7, fontSize: 'var(--text-sm)' }}>
              <li>
                <strong>Construct Validity (Metric Leakage):</strong> Training cross-entropy loss is a poor proxy for downstream agency.
                Mitigated by instituting Tier 3 (functional execution) and Tier 5 (evidence proposition matching) evaluations.
              </li>
              <li>
                <strong>Internal Validity (Data Contamination &amp; Cache Bleed):</strong> Test data leakage could artificially inflate scores.
                Mitigated by strict cryptographic SHA-256 checks between training shards and evaluation sets.
                Furthermore, during live inference testing, we detected and remediated cross-request cache bleed by enforcing ephemeral execution UUIDs.
              </li>
              <li>
                <strong>External Validity (Distributional Generalizability):</strong> Evaluations on 100 curated tasks do not guarantee open-ended conversational competence.
                We explicitly disclose this boundary: task-specific pilot evaluations demonstrate mechanism viability, not universal intelligence.
              </li>
              <li>
                <strong>Benchmark Count Disambiguation (100 vs. 90 Tasks):</strong> The held-out evaluation dataset (<code>modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/test.jsonl</code>, SHA-256: <code>523ec496...</code>) contains exactly <strong>100 unique records</strong> (40 Multi-Turn, 30 Calibration, 30 Synthesis). The subset of tasks evaluating core sequential tool execution and calculation numbers exactly <strong>90 tasks</strong> (30 standard chaining + 30 calibration + 30 synthesis), while 10 records evaluate early-exit or tool-failure edge cases. Both numbers are officially reconciled in <a href="https://github.com/imVynor/Qorin/blob/main/docs/data/DATASET_REGISTRY.md" target="_blank" rel="noopener noreferrer">DATASET_REGISTRY.md</a>.
              </li>
              <li>
                <strong>Unequal Training Budgets &amp; Scaling Attribution:</strong> The four evaluated model configurations received different training budgets (100M Base: 1.5B tokens; 100M SFT v3: 29M tokens / 8,764 steps; 500M Base Pilot: 252M tokens / 7,694 steps; 500M SFT Exp F: 100k tokens / 100 steps). Therefore, performance differences between 100M and 500M are descriptive of these specific checkpoints and do not establish an isolated causal scaling law. Full audit details are documented in <a href="https://github.com/imVynor/Qorin/blob/main/docs/REPRODUCIBILITY_AUDIT.md" target="_blank" rel="noopener noreferrer">REPRODUCIBILITY_AUDIT.md</a>.
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* 13. Limitations and Future Work */}
      <section id="sec-13" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 13 &bull; System Boundaries</div>
          <h2 className={styles.sectionTitle}>13. Limitations and Open Research Frontiers</h2>
          <div className={styles.proseMeasure}>
            <p>
              In alignment with scientific integrity, we explicitly report remaining failure modes in our current live system:
            </p>
            <div className={styles.calloutBox} style={{ borderLeftColor: '#dc2626' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, color: '#dc2626', marginBottom: '6px' }}>
                <AlertTriangle size={13} />
                <span>UNRESOLVED LIVE SYSTEM LIMITATIONS</span>
              </div>
              <ul style={{ paddingLeft: '1.2rem', margin: 0, lineHeight: 1.6, fontSize: 'var(--text-xs)' }}>
                <li><strong>Contradictory Evidence Failure:</strong> When retrieved sources contain conflicting dates or claims, the 514M model arbitrarily selects one source rather than acknowledging epistemic ambiguity (0% success on contradictory held-out tasks).</li>
                <li><strong>Affirmative Tool Confirmation:</strong> When prompted with &ldquo;Yes, execute that calculation&rdquo;, the model outputs conversational affirmation instead of dispatching the tool token block.</li>
                <li><strong>Multi-Hop Pronoun Resolution:</strong> Low-temperature generation can confuse entity antecedents across extended multi-turn conversations.</li>
                <li><strong>Single-GPU Hardware Boundary:</strong> Pretraining throughput on a single RTX 5060 (8GB VRAM) averages ~2,800 tokens/sec, requiring careful compute-optimal scheduling.</li>
              </ul>
            </div>

            <h3 className={styles.subheading} style={{ marginTop: 'var(--space-8)' }}>
              13.2 The Next Frontier: Five Billion Tokens Pretraining
            </h3>
            <p>
              Having empirically established that tool augmentation acts as a capability multiplier only above a critical representational capacity (emerging at 500M but failing at 100M), our next research campaign targets the foundational pretraining bottleneck.
            </p>
            <p>
              The current QORIN-500M base consumed 1.12B tokens. While this established stable causal next-token convergence, Chinchilla scaling laws dictate that a 500M parameter model is severely undertrained at 1.12B tokens (compute-optimal ratio is ~20 tokens per parameter, suggesting at least 10B tokens). We formulate our next research campaign around a targeted <strong>5.0 Billion Token pretraining budget</strong>:
            </p>
            <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-rule)', borderRadius: 'var(--radius-sm)', padding: 'var(--space-4)', margin: 'var(--space-4) 0' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', textTransform: 'uppercase', color: 'var(--color-accent-indigo)', fontWeight: 700 }}>
                Next Research Hypothesis (Phase 3)
              </div>
              <p style={{ fontSize: '13px', margin: '6px 0 0 0', color: 'var(--color-ink-primary)' }}>
                <em>&ldquo;Does extending pretraining from 1.12B to 5.0B tokens provide the internal representational density required to resolve epistemic ambiguity, multi-hop pronoun disambiguation, and contradictory source arbitration natively prior to tool synthesis?&rdquo;</em>
              </p>
              <ul style={{ paddingLeft: '1.2rem', marginTop: '8px', lineHeight: 1.6, fontSize: '12px', color: 'var(--color-ink-secondary)' }}>
                <li><strong>Corpus Expansion:</strong> 5.0B tokens across FineWeb-Edu (45%), Starcoder-Python (25%), Cosmopedia synthetic reasoning (15%), and curated Wikipedia/Wikibooks (15%).</li>
                <li><strong>Context Window Scaling:</strong> Native 4,096-token pretraining with YaRN RoPE frequency scaling to accommodate 10-turn tool observation chains.</li>
                <li><strong>Reinforcement Learning with Verifiable Rewards (RLVR):</strong> Direct optimization of tool invocation decisions and evidence proposition citations using deterministic sandbox verification feedback.</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* 14. Academic References */}
      <section id="sec-14" className={styles.section}>
        <div className={styles.container}>
          <div className={styles.kicker}>Section 14 &bull; Citations</div>
          <h2 className={styles.sectionTitle}>14. References &amp; Foundational Literature</h2>
          <div className={styles.proseMeasure}>
            <ol style={{ paddingLeft: '1.2rem', lineHeight: 1.8, fontSize: 'var(--text-xs)', color: 'var(--color-ink-secondary)' }}>
              <li>Vaswani, A., et al. (2017). <em>Attention Is All You Need</em>. Advances in Neural Information Processing Systems (NeurIPS 2017).</li>
              <li>Hoffmann, J., et al. (2022). <em>Training Compute-Optimal Large Language Models</em> (Chinchilla). arXiv:2203.15556.</li>
              <li>Touvron, H., et al. (2023). <em>LLaMA: Open and Efficient Foundation Language Models</em>. arXiv:2302.13971.</li>
              <li>Shazeer, N. (2020). <em>GLU Variants Improve Transformer</em>. arXiv:2002.05202.</li>
              <li>Su, J., et al. (2021). <em>RoFormer: Enhanced Transformer with Rotary Position Embedding</em>. arXiv:2104.09864.</li>
              <li>Ainslie, J., et al. (2023). <em>GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints</em>. EMNLP 2023.</li>
              <li>Lewis, P., et al. (2020). <em>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</em>. NeurIPS 2020.</li>
              <li>Schick, T., et al. (2023). <em>Toolformer: Language Models Can Teach Themselves to Use Tools</em>. NeurIPS 2023.</li>
            </ol>

            <div style={{ marginTop: 'var(--space-8)', display: 'flex', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
              <Link to="/experiments" className={styles.cardTag} style={{ textDecoration: 'none', fontWeight: 600 }}>
                Inspect the Complete Experiments Archive &rarr;
              </Link>
              <Link to="/harness" className={styles.cardTag} style={{ textDecoration: 'none', fontWeight: 600 }}>
                Test the Model in the Interactive Harness &rarr;
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
};
