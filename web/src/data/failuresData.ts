export interface FailureCase {
  id: string;
  title: string;
  category: 'model_hallucination' | 'tool_isolation' | 'cross_turn_leakage' | 'harness_dispatch' | 'data_starvation' | 'live_limitation';
  categoryLabel: string;
  query: string;
  observedOutput: string;
  rootCause: string;
  remedy: string;
  status: 'resolved_in_harness' | 'resolved_in_training' | 'active_limitation';
  investigationContext: string;
  scientificTakeaway: string;
}

export const failureCasesData: FailureCase[] = [
  {
    id: 'FAIL-01-STALE-PYTHON',
    title: 'Cross-Request Python State Bleed (The Combinations Bug)',
    category: 'tool_isolation',
    categoryLabel: 'Tool Execution State',
    query: 'write a python function that adds two numbers / whats 1+2?',
    observedOutput: 'Combinations: 0 (The Python execution yielded: Combinations)',
    rootCause: 'Subprocess execution outputs were assigned to shared session memory without unique request/execution IDs. An earlier execution of math.comb(3, 5) remained in the execution buffer and leaked into subsequent requests.',
    remedy: 'Introduced ephemeral UUID4 execution IDs, stateless Python subprocess dispatch with clean stdout/stderr capture, and strict request-scoped execution registries.',
    status: 'resolved_in_harness',
    investigationContext: 'Execution Isolation Audit',
    scientificTakeaway: 'Proves that tool-augmented language model reliability depends as much on backend isolation contracts as on the weights of the Transformer itself.'
  },
  {
    id: 'FAIL-02-EVIDENCE-HALLUCINATION',
    title: 'Factual Hallucination Despite Accurate Web Retrieval',
    category: 'model_hallucination',
    categoryLabel: 'Evidence Grounding',
    query: 'who is donald trump?',
    observedOutput: 'The United States presidential election held on January 20, 2021 concluded with a 2-2 vote. Donald Trump is the seventh president of the United States and a member of the Democratic Party.',
    rootCause: 'Retrieved Wikipedia and Britannica documents were passed as unstructured raw text. The 514M parameter model prioritized strong parametric priors over the retrieved text tokens, generating wildly fabricated historical claims.',
    remedy: 'Engineered Grounding Framework: Implemented EvidenceGroundingValidator with structured evidence objects ([E1], [E2]), strict sentence-level proposition matching (threshold >= 0.80), and deterministic conservative fallback synthesis.',
    status: 'resolved_in_harness',
    investigationContext: 'Evidence Grounding Verification',
    scientificTakeaway: 'Demonstrates the parametric vs non-parametric tension: Small LMs will hallucinate even in the presence of facts unless the harness enforces rigid grounding constraints.'
  },
  {
    id: 'FAIL-03-WRONG-ENTITY',
    title: 'Entity Attribute Misattribution from Mixed Snippets',
    category: 'harness_dispatch',
    categoryLabel: 'Entity Attribution',
    query: 'Who is Jensen Huang?',
    observedOutput: 'Jensen Huang is a Japanese-born computer science student who works in computational and statistical physics at the University of Tokyo.',
    rootCause: 'Search snippet contained multi-entity search results. The model extracted biographical facts from an adjacent passage on the page and attributed them to the query entity.',
    remedy: 'Introduced entity-scoped proposition verification in the grounding validator and rejected dangling pronoun sentences ("He", "She", "It") unless the target entity name is explicitly present.',
    status: 'resolved_in_harness',
    investigationContext: 'Entity Extraction Audit',
    scientificTakeaway: 'Shows that passage retrieval without fine-grained entity linking can poison downstream generation.'
  },
  {
    id: 'FAIL-04-CROSS-TURN-LEAK',
    title: 'Cross-Turn Conversational Observation Contamination',
    category: 'cross_turn_leakage',
    categoryLabel: 'State Management',
    query: 'Turn 1: "Who is Jensen Huang?" -> Turn 2: "Who is Donald Trump?"',
    observedOutput: 'Turn 2: "The University of Tokyo is known for its impressive research in computation and statistics."',
    rootCause: 'Live endpoint preserved raw tool observation strings in the active dialogue history without turn boundaries, causing Turn 2 to synthesize Turn 1 observation artifacts.',
    remedy: 'Enforced conversation turn scoping in backend/app.py: Prior tool observations are pruned or summarized, ensuring each turn operates with a clean retrieval slate.',
    status: 'resolved_in_harness',
    investigationContext: 'Dialogue State Management Audit',
    scientificTakeaway: 'Multi-turn conversational tool use requires strict boundary isolation between dialogue context and tool scratchpads.'
  },
  {
    id: 'FAIL-05-IDENTITY-MIRRORING',
    title: 'Identity Mirroring Degeneration ("You are donald trump")',
    category: 'harness_dispatch',
    categoryLabel: 'Intent Routing',
    query: 'who is donald trump?',
    observedOutput: 'You are donald trump. I am donald trump.',
    rootCause: 'Intent classifier treated the prompt as casual conversational dialogue rather than a factual research query, bypassing web retrieval entirely and triggering low-entropy repetitive decoding on the small base model.',
    remedy: 'Added FACTUAL_INQUIRY intent classification in ToolPolicyClassifier to detect biographical, geographical, and historical inquiries and force web research before model generation.',
    status: 'resolved_in_harness',
    investigationContext: 'Intent Routing Calibration',
    scientificTakeaway: 'Tool policies must proactively classify user intent to route queries to appropriate tools rather than relying on the small LM to spontaneously decide.'
  },
  {
    id: 'FAIL-06-GREETING-STARVATION',
    title: 'The Starvation Paradox & Conversational Collapse',
    category: 'data_starvation',
    categoryLabel: 'Dataset Distribution',
    query: 'Hello! How are you?',
    observedOutput: 'Model generated synthetic Wikipedia article section headers and random Python function signatures instead of a conversational greeting.',
    rootCause: 'SFT v3 achieved record-low validation loss (0.765), but greetings constituted only 0.026% of training records (9 / 34,700). High training loss optimization caused catastrophic forgetting of simple conversational greetings.',
    remedy: 'Rebalanced training mix in balanced tool fine-tuning with calibrated negative controls, dialogue anchors, and direct answering supervision.',
    status: 'resolved_in_training',
    investigationContext: 'Conversational Alignment Audit',
    scientificTakeaway: 'A vital machine-learning lesson: Validation loss alone is an incomplete metric; severe capability starvation can occur even as cross-entropy loss reaches all-time lows.'
  },
  {
    id: 'FAIL-07-CONTRADICTORY-EVAL',
    title: 'Contradictory and Insufficient Evidence Failure',
    category: 'live_limitation',
    categoryLabel: 'Current Limitation',
    query: 'Historical query with conflicting dates in retrieved sources',
    observedOutput: 'Model arbitrarily selected one source date and stated it with 100% confidence, completely failing to acknowledge conflicting sources.',
    rootCause: 'In the held-out capability evaluation, contradictory and insufficient evidence cases scored 0% success. The 514M model has not yet learned epistemic humility or dialectical hedging.',
    remedy: 'Currently handled by conservative fallback in the harness; end-to-end parametric hedging remains an active research frontier.',
    status: 'active_limitation',
    investigationContext: 'Held-Out Capability Evaluation',
    scientificTakeaway: 'Small models lack the parameter capacity for epistemic uncertainty calibration without explicit architectural scaffolding.'
  },
  {
    id: 'FAIL-08-AFFIRMATIVE-TOOL',
    title: 'Affirmative Tool Confirmation Failure',
    category: 'live_limitation',
    categoryLabel: 'Current Limitation',
    query: 'User: "Yes, please execute that Python calculation now."',
    observedOutput: '"Certainly, I will execute the calculation for you right away." (No actual tool call block emitted)',
    rootCause: 'The model generates a natural language confirmation instead of emitting a structured tool invocation token block when prompted with colloquial affirmations.',
    remedy: 'Active limitation in current live model; requires multi-turn dialogue state tracking in the harness to map confirmations to prior pending tool actions.',
    status: 'active_limitation',
    investigationContext: 'Interactive Agent Verification',
    scientificTakeaway: 'Demonstrates the gap between single-turn tool calling and conversational multi-turn agency in small language models.'
  },
  {
    id: 'FAIL-09-100M-TOOL-DEGRADATION',
    title: 'Tool Distraction & Schema Ingestion Breakdown at 100M Scale',
    category: 'live_limitation',
    categoryLabel: 'Capacity Threshold',
    query: 'Controlled 100-record ablation test on QORIN-100M (best_val.pt)',
    observedOutput: 'Tools Disabled: 36.0% (36/100) -> Tools Enabled: 29.0% (29/100) [Net -7.0% regression]',
    rootCause: 'At 100M parameters (14L, 768D), the model lacks representational bandwidth to ingest tool schemas, format arguments reliably, and filter retrieved passage noise. Tool availability introduced syntax hallucination and context distraction, degrading accuracy.',
    remedy: 'Discovered critical capacity threshold: External tools only act as a capability multiplier when base model capacity reaches >=500M parameters (where gain was +21.0% pts).',
    status: 'active_limitation',
    investigationContext: 'Controlled Tool-Augmentation Ablation (Central Question)',
    scientificTakeaway: 'Direct empirical refutation of the compensation hypothesis: External tools do NOT compensate for inadequate model capacity below the critical parametric threshold.'
  }
];
