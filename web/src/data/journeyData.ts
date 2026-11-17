export interface JourneyChapter {
  id: number;
  slug: string;
  title: string;
  subtitle: string;
  tag: string;
  question: string;
  keyHypothesis: string;
  evidence: string[];
  findings: string[];
  architecturalDecisions: string[];
  failureInsight?: string;
  es335Takeaway: string;
}

export const journeyChaptersData: JourneyChapter[] = [
  {
    id: 1,
    slug: 'the-question',
    title: 'Chapter 1: The Research Question',
    subtitle: 'Parametric Capacity vs. Tool-Augmented Agency',
    tag: 'Problem Formulation',
    question: 'How does increasing the capacity of a small decoder-only Transformer affect conversational LM performance, and can external retrieval compensate for smaller model capacity?',
    keyHypothesis: 'External non-parametric retrieval and deterministic execution tools can bridge the factual and reasoning gap between small models (<1B parameters) and large frontier models without prohibitive compute costs.',
    evidence: [
      'Small language models (<1B parameters) suffer severe factual hallucination and catastrophic forgetting due to constrained parameter storage capacity.',
      'Parametric knowledge is static (frozen at training time) and lossy; non-parametric retrieval provides exact, up-to-date source documents dynamically at inference.',
      'Symbolic and arithmetic tasks fail catastrophically under next-token autoregression, but execute with 100% precision when delegated to a Python subprocess sandbox.'
    ],
    findings: [
      'Identified the three computational pillars of modern language modeling: Parametric weights, Dynamic Context Retrieval, and External Subprocess Tool Execution.',
      'Established the experimental unit: Multi-turn dialogue sessions evaluated across factual accuracy, AST code validity, and evidence citation fidelity.'
    ],
    architecturalDecisions: [
      'Focus on decoder-only Transformers trained completely from scratch rather than downloading third-party weights, ensuring total control over the data pipeline and checkpoint lineage.',
      'Strictly enforce local consumer hardware training boundaries (Single NVIDIA RTX 5060 8GB VRAM).'
    ],
    es335Takeaway: 'Formulating a rigorous comparative machine-learning inquiry rather than an engineering demo: Isolating capacity as an independent variable and external augmentation as a treatment variable.'
  },
  {
    id: 2,
    slug: 'model-foundation',
    title: 'Chapter 2: Establishing the Model Foundation',
    subtitle: 'From Scratch Transformer Architecture Design',
    tag: 'Baseline Engineering',
    question: 'What architectural components enable maximum training stability, memory efficiency, and inference throughput on consumer GPU hardware?',
    keyHypothesis: 'Combining modern Transformer architectural innovations (RoPE, RMSNorm, SwiGLU, Causal SDPA) provides significantly better loss convergence per parameter than traditional GPT-2 architectures.',
    evidence: [
      'Pre-norm RMSNorm (eps=1e-6) eliminated gradient explosion during mixed-precision training, enabling 2x higher learning rates than LayerNorm.',
      'SwiGLU activation with a 2.5x-2.67x hidden expansion ratio achieved 8-12% lower validation perplexity than standard GELU at equivalent parameter counts.',
      'Rotary Position Embeddings (RoPE) enabled zero-degradation relative position extrapolation compared to absolute learned embeddings.',
      'Tied input-output token embeddings reduced memory by 16M-32M parameters with negligible loss impact.'
    ],
    findings: [
      'Built custom Byte-Pair Encoding (BPE) tokenizers (16K vocabulary for early pilots, 32K vocabulary for QORIN-500M) with domain-specific tokens (<tool_call>, </tool_call>, ```python).',
      'Established the initial 16.8M baseline model (6 layers, 384 dim, 6 heads) which achieved 3.8614 validation loss on 100M tokens.'
    ],
    architecturalDecisions: [
      'Discarded absolute positional embeddings in favor of RoPE.',
      'Standardized on Pre-RMSNorm and SwiGLU across all model sizes (16M, 50M, 100M, 500M).'
    ],
    es335Takeaway: 'Sound deep learning engineering requires rigorous ablation of core components before scaling model capacity.'
  },
  {
    id: 3,
    slug: 'increasing-capacity',
    title: 'Chapter 3: Increasing Model Capacity',
    subtitle: 'Scaling from 16M to 514M Parameters',
    tag: 'Model Scaling',
    question: 'How do perplexity, syntax adherence, and downstream reasoning scale across 16.8M, 49.1M, 100.4M, and 514.4M parameters?',
    keyHypothesis: 'Increasing parameter capacity reduces validation perplexity logarithmically (Chinchilla-style power law), but complex structured reasoning (AST validity, schema compliance) exhibits sharp emergent threshold behavior.',
    evidence: [
      'Validation perplexity dropped monotonically: 16.8M (47.53 ppl) -> 49.1M (22.54 ppl) -> 100.4M (15.08 ppl).',
      'Under strictly matched 100M token pretraining budgets (Iso-Data), 50M reduced validation loss by 0.4047 (-33.3% ppl) compared to 16M from parameter capacity alone.',
      'QORIN-500M (514,389,504 verified parameters: 20 layers, 1536 dim, 24 query heads, 4 KV heads, GQA 6:1, SwiGLU 3840, 2048 ctx) reached 0.8124 validation loss in Experiment D and 0.8947 in Experiment F.'
    ],
    findings: [
      'AST code syntax generation jumped from 77.5% in 16M to 87.5% in 50M and 96.5% in 500M.',
      'Functional code execution without external execution remained near 0% across ALL model scales (0.0% in 16M, 2.5% in 50M), proving that raw parameter scaling alone DOES NOT solve arithmetic and execution tasks.'
    ],
    architecturalDecisions: [
      'Introduced Grouped Query Attention (GQA 6:1 in 500M; 24 Q-heads, 4 KV-heads) to slash KV-cache footprint by 83.3%, allowing 2048-token context batches to fit inside 8GB VRAM.',
      'Adopted flash/memory-efficient scaled dot-product attention (SDPA).'
    ],
    es335Takeaway: 'A crucial empirical result: Increasing model capacity easily improves surface language perplexity and syntax validity, but fails to yield reliable deterministic computation.'
  },
  {
    id: 4,
    slug: 'why-retrieval',
    title: 'Chapter 4: Why Retrieval Became Necessary',
    subtitle: 'The Inherent Limits of Parametric Knowledge',
    tag: 'RAG Architecture',
    question: 'Why can small language models never store the long tail of human factual knowledge in weights alone?',
    keyHypothesis: 'Decoupling knowledge storage (external indexed text corpus) from language processing (Transformer reasoning) allows small models to answer factual inquiries with zero parametric hallucination.',
    evidence: [
      'Even our 514M parameter model consistently hallucinated specific biographical dates, recent electoral votes, and technical numbers when forced to answer from weights alone.',
      'Integrating non-parametric BM25 retrieval over segmented knowledge chunks provided high-similarity context passages (avg similarity > 0.72) within 15-35ms.',
      'When supplied with relevant chunks in prompt context, factual generation accuracy improved by over 40% on benchmark prompts.'
    ],
    findings: [
      'Small models are highly susceptible to "distractor noise": If retrieved snippets contain extraneous entities, the model easily attributes facts to the wrong entity.',
      'Retrieval without strict evidence formatting leads to "cherry-picking" where the model ignores the retrieved text and hallucinates anyway.'
    ],
    architecturalDecisions: [
      'Implemented dual-mode retrieval: Fast local BM25 indexing for domain documentation and live web search for open-domain queries.',
      'Formatted retrieved chunks with explicit document titles, sections, and similarity scores.'
    ],
    es335Takeaway: 'External retrieval shifts the learning burden from brute-force memorization to context-aware comprehension and synthesis.'
  },
  {
    id: 5,
    slug: 'from-retrieval-to-tools',
    title: 'Chapter 5: From Retrieval to Tool Augmentation',
    subtitle: 'Unifying Web Retrieval and Python Execution',
    tag: 'Tool Augmented LM',
    question: 'Can a small Transformer learn to autonomously select, parameterize, and invoke distinct external tools to solve multi-stage problems?',
    keyHypothesis: 'Supervising the model on structured tool call tokens (<tool_call>{"name": ..., "arguments": ...}</tool_call>) enables dual-tool orchestration: Web research for factual discovery and Python for mathematical evaluation.',
    evidence: [
      'Dual-Tool Autonomous Dispatch (Exp C): The model successfully learned to produce valid JSON tool dispatch payloads for both web research and python math.',
      'Decision Calibration (Exp D): The model learned when to invoke tools (98.0% precision) and when to answer directly without tools (96.5% accuracy), preventing tool overtriggering.',
      'Autonomous Sequential Chaining (Exp F Step 100 pilot): After receiving web research observations, the model autonomously invoked a second-turn Python calculation in 50% of evaluation cases (up from 0% in the baseline).'
    ],
    findings: [
      'Tool augmentation effectively turns the Transformer into a reasoning engine that delegates computation to deterministic environments (Python sandbox) and real-time knowledge bases (Web search).'
    ],
    architecturalDecisions: [
      'Designed a unified JSON tool contract with strict schema validation.',
      'Used specialized token framing for tool calls and observation injection.'
    ],
    es335Takeaway: 'Tool augmentation is not an external wrapper; it requires specialized supervised fine-tuning with causal loss masking on assistant decisions.'
  },
  {
    id: 6,
    slug: 'harness-evolves',
    title: 'Chapter 6: The Harness Evolves',
    subtitle: 'From Simple Dispatch to Robust State Machine',
    tag: 'Harness Engineering',
    question: 'How must the runtime orchestration harness be engineered to guarantee deterministic tool execution, citation verification, and session isolation?',
    keyHypothesis: 'The harness is not merely utility software; it is an active experimental component whose state validation and isolation directly determine system reliability.',
    evidence: [
      'State-Machine Orchestration Harness: Created a formal 4-state transition graph (IDLE -> TOOL_REQUEST -> EXECUTION -> SYNTHESIS) preventing infinite loops.',
      'Runtime Evidence Grounding: Built an EvidenceGroundingValidator requiring proposition-level evidence support (>=0.80 cosine/token overlap) and valid bibliographic citation tags ([1], [2]).',
      'Runtime Context Isolation: Discovered that live endpoints differed from test harnesses; resolved cross-turn contamination, intent misrouting, and session bleed through ephemeral execution UUIDs and turn scoping.'
    ],
    findings: [
      'Software engineering rigor is indispensable in machine learning systems: Without state isolation, valid model outputs can be overwritten by stale execution cache buffers.'
    ],
    architecturalDecisions: [
      'Enforced stateless execution: Python code runs in fresh subprocesses with strict execution timeouts (5000ms) and forbidden module AST validation.',
      'Added conservative fallback synthesis: When model output fails proposition grounding against retrieved sources, the harness synthesizes a faithful cited summary directly from source evidence.'
    ],
    es335Takeaway: 'A complete ML system consists of the trained neural model PLUS the deterministic verification harness that governs its interactions with the real world.'
  },
  {
    id: 7,
    slug: 'learning-from-failures',
    title: 'Chapter 7: Learning from Failures',
    subtitle: 'Forensic Autopsies of Actual Project Anomalies',
    tag: 'Error Analysis',
    question: 'What failed during development, and what did each failure teach us about small language models and runtime harnesses?',
    keyHypothesis: 'Systematic failure diagnosis yields more scientific insight than celebrating isolated benchmark successes.',
    evidence: [
      'Failure 1 (The Starvation Paradox): SFT v3 hit lowest validation loss (0.765) but suffered 0.026% greeting collapse due to dataset imbalance.',
      'Failure 2 (The Combinations Bug): Stale math.comb(3, 5) output leaked across requests due to missing execution-ID scoping in session memory.',
      'Failure 3 (Factual Hallucination after Retrieval): QORIN-500M retrieved accurate Trump Wikipedia pages but generated a fabricated "2-2 vote in 2021 election" due to parametric prior dominance.',
      'Failure 4 (Cross-Turn Contamination): Live endpoint preserved previous turn observations, causing a Jensen Huang query to answer with facts about Donald Trump.',
      'Failure 5 (Identity Mirroring): Uncalibrated intent routing led the model to answer "You are donald trump. I am donald trump."'
    ],
    findings: [
      'Every major architectural and harness enhancement in QORIN was directly motivated by an honest post-mortem of a reproducible failure.'
    ],
    architecturalDecisions: [
      'Implemented ephemeral UUID4 execution IDs.',
      'Introduced strict sentence-level evidence proposition matching.',
      'Built turn-isolated dialogue context buffers.'
    ],
    es335Takeaway: 'The ES 335 assessment rewards honest scientific error analysis over cherry-picked demonstrations.'
  },
  {
    id: 8,
    slug: 'what-we-learned',
    title: 'Chapter 8: What We Learned & Future Frontiers',
    subtitle: 'Empirical Conclusions from the QORIN Investigation',
    tag: 'Conclusions',
    question: 'What are the definitive answers to our central research question, and what remain open unsolved challenges?',
    keyHypothesis: 'External retrieval and deterministic execution tools can compensate for small model capacity on factual and computational tasks, but cannot substitute for capacity in multi-hop reasoning, epistemic uncertainty, and nuance.',
    evidence: [
      'Where Tools Compensate: On arithmetic and Python execution, QORIN-500M + Python sandbox achieves 100% computational accuracy, vastly outperforming even multi-billion parameter models operating purely autoregressively.',
      'Where Tools Compensate: On factual inquiries with explicit sources, QORIN-500M + Web retrieval achieved 100% full-support precision and 0.0% unsupported claims in held-out benchmark synthesis testing.',
      'Where Capacity Still Matters: On contradictory evidence, subtle entity disambiguation, and multi-turn affirmative reference resolution, small models struggle due to limited attention bandwidth and reasoning depth.'
    ],
    findings: [
      '1. Parameter scaling improves language fluency, syntactic adherence, and single-turn comprehension, but does not solve factual hallucination or symbolic arithmetic.',
      '2. Non-parametric retrieval completely changes the economics of language modeling: A 514M parameter model with a grounding harness can deliver factual accuracy comparable to models 10x-50x larger.',
      '3. The runtime harness is an essential co-processor that must be evaluated, audited, and isolated with the same rigor as the neural weights.'
    ],
    architecturalDecisions: [
      'Document all active live limitations clearly (repetitive follow-ups, affirmative tool confirmation failure, contradictory source resolution).',
      'Preserve all checkpoints, training curves, and evaluation logs for full scientific reproducibility.'
    ],
    es335Takeaway: 'The central research question is answered with empirical nuance: External tools dramatically compensate for small capacity in knowledge acquisition and deterministic computation, but require architectural scaffolding to overcome reasoning bottlenecks.'
  }
];
