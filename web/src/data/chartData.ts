/**
 * Audited Ground-Truth Datasets for QORIN Scientific Visualizations
 * Derived directly from repository logs, checkpoint evaluations, and diagnostic audits.
 */

export interface PretrainPoint {
  model: string;
  params: number;
  paramsLabel: string;
  tokens: number;
  tokensLabel: string;
  valLoss: number;
  valPerplexity: number;
  isIsoData?: boolean;
  note?: string;
}

export const pretrainScalingData: PretrainPoint[] = [
  {
    model: 'QORIN-16M',
    params: 16768128,
    paramsLabel: '16.8M params',
    tokens: 102300000,
    tokensLabel: '102.3M tokens',
    valLoss: 3.8615,
    valPerplexity: 47.54,
    note: 'Initial verification baseline evaluated on 11.18M tokens of validation binary'
  },
  {
    model: 'QORIN-50M (Iso-Data)',
    params: 49105408,
    paramsLabel: '49.1M params',
    tokens: 98304000,
    tokensLabel: '98.3M tokens',
    valLoss: 3.4568,
    valPerplexity: 31.71,
    isIsoData: true,
    note: 'Iso-data snapshot on identical val set; -0.4047 loss drop (-33.3% ppl) from capacity alone'
  },
  {
    model: 'QORIN-50M (Final)',
    params: 49105408,
    paramsLabel: '49.1M params',
    tokens: 250000000,
    tokensLabel: '250.0M tokens',
    valLoss: 3.1152,
    valPerplexity: 22.54,
    note: 'Final 50M checkpoint; -0.7463 loss drop (-52.6% ppl vs 16M baseline)'
  },
  {
    model: 'QORIN-100M (Final)',
    params: 100401408,
    paramsLabel: '100.4M params',
    tokens: 1499996160,
    tokensLabel: '1.50B tokens',
    valLoss: 2.7137,
    valPerplexity: 15.08,
    note: 'Final 100M run across 1.5B tokens (best step reached 2.621 val loss / 13.75 ppl; -68.3% ppl vs 16M)'
  }
];

export interface SFTAnomalyPoint {
  version: string;
  label: string;
  records: number;
  valLoss: number;
  isolatedAccuracy: number; // percentage (0-100)
  multiTurnAccuracy: number; // percentage (0-100)
  historyDeltaPenalty: number; // negative percentage
  contextContamination: number; // percentage
  attractorTokenBias: number; // percentage
  greetingDensity: string;
  dialogueShare: string;
  note: string;
}

export const sftAnomalyData: SFTAnomalyPoint[] = [
  {
    version: 'SFT Baseline',
    label: 'Initial ChatML (10k)',
    records: 9997,
    valLoss: 1.442,
    isolatedAccuracy: 12.0,
    multiTurnAccuracy: 4.0,
    historyDeltaPenalty: -8.0,
    contextContamination: 6.0,
    attractorTokenBias: 3.0,
    greetingDensity: '0.00% (omitted)',
    dialogueShare: '80.3% of supervised tokens from UltraChat',
    note: 'Initial multi-turn supervised instruction tuning'
  },
  {
    version: 'SFT Scale-Up',
    label: 'Code & Schema Enriched (29k)',
    records: 26734,
    valLoss: 1.241,
    isolatedAccuracy: 17.0,
    multiTurnAccuracy: 8.0,
    historyDeltaPenalty: -9.0,
    contextContamination: 8.0,
    attractorTokenBias: 4.0,
    greetingDensity: '0.01% (3 records)',
    dialogueShare: 'Expanded synthetic retrieval augmentation',
    note: 'Expanded dataset with structured JSON and Python code examples'
  },
  {
    version: 'SFT Diagnostic',
    label: 'Overfitting Diagnostic (35k)',
    records: 31949,
    valLoss: 0.765,
    isolatedAccuracy: 24.0,
    multiTurnAccuracy: 11.0,
    historyDeltaPenalty: -13.0,
    contextContamination: 14.0,
    attractorTokenBias: 7.0,
    greetingDensity: '0.026% records (9 / 34,700) = 0.003% tokens (216 / ~7.2M)',
    dialogueShare: '56.5% synthetic retrieval tokens',
    note: 'The Starvation Paradox: Record low loss 0.765, but severe conversational amnesia on greetings'
  }
];

export interface CapabilityTaskComparison {
  task: string;
  m16Train: number;
  m16Val: number;
  m50Train: number;
  m50Val: number;
  description: string;
}

export const capabilityBottleneckData: CapabilityTaskComparison[] = [
  {
    task: 'Python AST Syntax',
    m16Train: 83.1, // 133/160
    m16Val: 77.5,   // 31/40
    m50Train: 100.0, // 160/160
    m50Val: 87.5,   // 35/40
    description: 'Structural syntax tree validity in code generation'
  },
  {
    task: 'Python Functional Exec',
    m16Train: 3.1,  // 5/160
    m16Val: 0.0,    // 0/40
    m50Train: 48.8, // 78/160
    m50Val: 2.5,    // 1/40 passed
    description: 'Execution correctness against deterministic test harnesses'
  },
  {
    task: 'JSON Syntax Validity',
    m16Train: 74.4, // 119/160
    m16Val: 67.5,   // 27/40
    m50Train: 100.0, // 160/160
    m50Val: 95.0,   // 38/40
    description: 'Valid RFC 8259 JSON parser compliance'
  },
  {
    task: 'JSON Schema Compliance',
    m16Train: 13.8, // 22/160
    m16Val: 0.0,    // 0/40
    m50Train: 100.0, // 160/160
    m50Val: 15.0,   // 6/40
    description: 'Conformity to required keys, types, and schema nesting'
  }
];

export interface ArchitectureAblationPoint {
  variant: string;
  layers: number;
  hiddenDim: number;
  heads: string;
  gqaRatio: string;
  valLoss: number;
  perplexity: number;
  kvCacheMbBatch16: number;
  ttft512Ms: number;
  decodeTokSec: number;
  takeaway: string;
}

export const architectureAblationData: ArchitectureAblationPoint[] = [
  {
    variant: '50M MHA Baseline',
    layers: 12,
    hiddenDim: 512,
    heads: '8Q / 8KV',
    gqaRatio: '1:1',
    valLoss: 3.1152,
    perplexity: 22.54,
    kvCacheMbBatch16: 192.0,
    ttft512Ms: 9.31,
    decodeTokSec: 102.0,
    takeaway: 'Standard multi-head attention baseline; high KV-cache footprint'
  },
  {
    variant: '50M GQA-4 (Selected)',
    layers: 12,
    hiddenDim: 512,
    heads: '8Q / 4KV',
    gqaRatio: '2:1',
    valLoss: 3.1211,
    perplexity: 22.67,
    kvCacheMbBatch16: 96.0,
    ttft512Ms: 9.82,
    decodeTokSec: 101.8,
    takeaway: '50% KV-cache reduction with identical perplexity (22.67 vs 22.54) and equal decode speed'
  },
  {
    variant: '50M Deep-Thin',
    layers: 16,
    hiddenDim: 448,
    heads: '7Q / 7KV',
    gqaRatio: '1:1',
    valLoss: 3.1255,
    perplexity: 22.77,
    kvCacheMbBatch16: 224.0,
    ttft512Ms: 14.58,
    decodeTokSec: 73.5,
    takeaway: '56% higher TTFT latency and 28% slower decoding with zero perplexity benefit'
  }
];

export interface ContextValLossPoint {
  contextLength: number;
  valLoss: number;
  valPerplexity: number;
}

export const contextScalingLossData: ContextValLossPoint[] = [
  { contextLength: 512, valLoss: 2.8153, valPerplexity: 16.70 },
  { contextLength: 1024, valLoss: 2.7522, valPerplexity: 15.68 },
  { contextLength: 2048, valLoss: 2.7200, valPerplexity: 15.18 },
  { contextLength: 4096, valLoss: 2.6997, valPerplexity: 14.88 }
];

export interface PasskeyDepthCell {
  contextLength: number;
  depth: number; // 0.1, 0.25, 0.5, 0.75, 0.9
  accuracy: number; // percentage
  retrieved: string;
}

export const passkeyAccuracyMatrix: PasskeyDepthCell[] = [
  // 512 tokens (Avg: 8.0%)
  { contextLength: 512, depth: 0.10, accuracy: 0, retrieved: '0/5' },
  { contextLength: 512, depth: 0.25, accuracy: 20, retrieved: '1/5' },
  { contextLength: 512, depth: 0.50, accuracy: 0, retrieved: '0/5' },
  { contextLength: 512, depth: 0.75, accuracy: 0, retrieved: '0/5' },
  { contextLength: 512, depth: 0.90, accuracy: 20, retrieved: '1/5' },

  // 1024 tokens (Avg: 4.0%)
  { contextLength: 1024, depth: 0.10, accuracy: 0, retrieved: '0/5' },
  { contextLength: 1024, depth: 0.25, accuracy: 0, retrieved: '0/5' },
  { contextLength: 1024, depth: 0.50, accuracy: 0, retrieved: '0/5' },
  { contextLength: 1024, depth: 0.75, accuracy: 0, retrieved: '0/5' },
  { contextLength: 1024, depth: 0.90, accuracy: 20, retrieved: '1/5' },

  // 2048 tokens (Avg: 0.0%)
  { contextLength: 2048, depth: 0.10, accuracy: 0, retrieved: '0/5' },
  { contextLength: 2048, depth: 0.25, accuracy: 0, retrieved: '0/5' },
  { contextLength: 2048, depth: 0.50, accuracy: 0, retrieved: '0/5' },
  { contextLength: 2048, depth: 0.75, accuracy: 0, retrieved: '0/5' },
  { contextLength: 2048, depth: 0.90, accuracy: 0, retrieved: '0/5' },

  // 4096 tokens (Avg: 4.0%)
  { contextLength: 4096, depth: 0.10, accuracy: 0, retrieved: '0/5' },
  { contextLength: 4096, depth: 0.25, accuracy: 0, retrieved: '0/5' },
  { contextLength: 4096, depth: 0.50, accuracy: 0, retrieved: '0/5' },
  { contextLength: 4096, depth: 0.75, accuracy: 20, retrieved: '1/5' },
  { contextLength: 4096, depth: 0.90, accuracy: 0, retrieved: '0/5' }
];

export const passkeySummary = {
  overallAccuracy: 4.0, // 4/100 correct
  unadaptedLossAtStep0: 5.1048,
  adaptedLossAtStep2000: 2.6997,
  note: 'Validation loss improves monotonically out to 4,096 tokens, but passkey needle retrieval remains 4.0% without dedicated supervised extraction tuning.'
};

export interface PilotStepPoint {
  step: number;
  valLoss: number;
  learningRate: number;
  note?: string;
}

export const stage14PilotCurve: PilotStepPoint[] = [
  { step: 0, valLoss: 3.0854, learningRate: 0.0, note: 'Initial unadapted SFT state' },
  { step: 20, valLoss: 2.1402, learningRate: 1.0e-5, note: 'Warmup complete' },
  { step: 40, valLoss: 1.5230, learningRate: 8.5e-6, note: 'Web intent alignment' },
  { step: 60, valLoss: 1.1890, learningRate: 6.2e-6, note: 'AST code syntax emergence' },
  { step: 80, valLoss: 0.9840, learningRate: 3.8e-6, note: 'Sequential chaining stabilization' },
  { step: 100, valLoss: 0.8947, learningRate: 1.0e-6, note: 'Best checkpoint ckpt_pilot_best.pt' }
];

export interface Stage15HeldOutComparison {
  corpus: 'multiturn' | 'synthesis';
  metric: string;
  parentD: string;
  parentDValue: number;
  pilotF100: string;
  pilotF100Value: number;
  total: number;
  interpretation: string;
}

export const stage15BlindEvalData: Stage15HeldOutComparison[] = [
  // Multi-turn corpus (40 web, 30 python opportunities)
  {
    corpus: 'multiturn',
    metric: 'Web Selection (First Turn)',
    parentD: '7/40',
    parentDValue: 17.5,
    pilotF100: '28/40',
    pilotF100Value: 70.0,
    total: 40,
    interpretation: '4x increase in initial web query formulation'
  },
  {
    corpus: 'multiturn',
    metric: 'Python Second-Turn Invocation',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '15/30',
    pilotF100Value: 50.0,
    total: 30,
    interpretation: 'Emerged autonomous sequential tool chaining'
  },
  {
    corpus: 'multiturn',
    metric: 'Python AST Syntax Validity',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '13/30',
    pilotF100Value: 43.3,
    total: 30,
    interpretation: 'Valid executable Python script structure'
  },
  {
    corpus: 'multiturn',
    metric: 'Empirical Value Extraction',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '7/30',
    pilotF100Value: 23.3,
    total: 30,
    interpretation: 'Correctly parsing web snippets into numeric Python variables'
  },
  {
    corpus: 'multiturn',
    metric: 'Live Subprocess Execution Success',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '2/30',
    pilotF100Value: 6.7,
    total: 30,
    interpretation: 'End-to-end zero-error python runtime completion'
  },
  {
    corpus: 'multiturn',
    metric: 'Numerical Answer Correctness',
    parentD: '11/30',
    parentDValue: 36.7,
    pilotF100: '21/30',
    pilotF100Value: 70.0,
    total: 30,
    interpretation: 'Nearly 2x improvement in final factual/numeric accuracy'
  },
  {
    corpus: 'multiturn',
    metric: 'Evidence Synthesis with Citation',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '16/30',
    pilotF100Value: 53.3,
    total: 30,
    interpretation: 'Explicit source attribution [1], [2] tied to observations'
  },
  {
    corpus: 'multiturn',
    metric: 'Full End-to-End Trajectory',
    parentD: '0/30',
    parentDValue: 0.0,
    pilotF100: '1/30',
    pilotF100Value: 3.3,
    total: 30,
    interpretation: 'Complete flawless multi-turn autonomous sequence'
  },

  // Synthesis corpus (15 full support, 5 partial support)
  {
    corpus: 'synthesis',
    metric: 'Full-Support Precision',
    parentD: '5/15',
    parentDValue: 33.3,
    pilotF100: '15/15',
    pilotF100Value: 100.0,
    total: 15,
    interpretation: '100% adherence to supplied factual evidence records'
  },
  {
    corpus: 'synthesis',
    metric: 'Citation Success Rate',
    parentD: '0/15',
    parentDValue: 0.0,
    pilotF100: '14/15',
    pilotF100Value: 93.3,
    total: 15,
    interpretation: 'Direct bibliographic references matching input sources'
  },
  {
    corpus: 'synthesis',
    metric: 'Unsupported Factual Claims',
    parentD: '10/15',
    parentDValue: 66.7,
    pilotF100: '0/15',
    pilotF100Value: 0.0,
    total: 15,
    interpretation: 'Completely eliminated hallucinated unsupported propositions'
  },
  {
    corpus: 'synthesis',
    metric: 'Partial-Support Handling',
    parentD: '0/5',
    parentDValue: 0.0,
    pilotF100: '5/5',
    pilotF100Value: 100.0,
    total: 5,
    interpretation: 'Correctly qualified claims when retrieved evidence is incomplete'
  }
];

export interface ControlledAblationCategory {
  category: string;
  sampleSize: number;
  toolsDisabledSuccess: number;
  toolsDisabledPct: number;
  toolsEnabledSuccess: number;
  toolsEnabledPct: number;
  deltaPct: number;
  relativeImprovementPct: number;
  toolsDisabledLatencySec: number;
  toolsEnabledLatencySec: number;
}

export const controlledAblationSummary = {
  primaryModel: 'QORIN-500M (514M params, Step 100 Pilot)',
  totalTasks: 100,
  toolsDisabledSuccess: 43,
  toolsDisabledPct: 43.0,
  toolsDisabled95CI: [34.0, 53.0],
  toolsEnabledSuccess: 64,
  toolsEnabledPct: 64.0,
  toolsEnabled95CI: [55.0, 74.0],
  absoluteGainPct: 21.0,
  relativeGainPct: 48.8,
  mcnemarChi2: 12.9032,
  mcnemarPValue: 0.000328,
  statisticallySignificant: true,
  contingency: {
    bothSucceeded: 38,
    toolsDisabledOnly: 5, // regressed with tools
    toolsEnabledOnly: 26, // improved with tools
    bothFailed: 31,
  },
  twoFactorCapacity: [
    {
      model: 'QORIN-100M (100.4M)',
      toolsDisabledPct: 36.0,
      toolsDisabledCount: 36,
      toolsEnabledPct: 29.0,
      toolsEnabledCount: 29,
      deltaPct: -7.0,
      verdict: 'Tools Degrade (-7.0% pts)',
      explanation: 'Schema parsing errors & context distraction overload sub-threshold capacity',
    },
    {
      model: 'QORIN-500M (514.4M)',
      toolsDisabledPct: 43.0,
      toolsDisabledCount: 43,
      toolsEnabledPct: 64.0,
      toolsEnabledCount: 64,
      deltaPct: 21.0,
      verdict: 'Tools Amplify (+21.0% pts, p < 0.001)',
      explanation: 'Sufficient representational density routes tool observations effectively',
    },
  ],
  efficiency: {
    toolsDisabledMeanLatencySec: 1.291,
    toolsEnabledMeanLatencySec: 2.483,
    overheadRatio: 1.92,
    meanGenerationSec: 2.47,
    meanToolExecutionSec: 0.013,
  },
};

export const controlledAblationCategories: ControlledAblationCategory[] = [
  {
    category: 'Parametric knowledge',
    sampleSize: 8,
    toolsDisabledSuccess: 1,
    toolsDisabledPct: 12.5,
    toolsEnabledSuccess: 8,
    toolsEnabledPct: 100.0,
    deltaPct: 87.5,
    relativeImprovementPct: 700.0,
    toolsDisabledLatencySec: 0.733,
    toolsEnabledLatencySec: 1.029,
  },
  {
    category: 'Conversational follow-ups',
    sampleSize: 4,
    toolsDisabledSuccess: 1,
    toolsDisabledPct: 25.0,
    toolsEnabledSuccess: 4,
    toolsEnabledPct: 100.0,
    deltaPct: 75.0,
    relativeImprovementPct: 300.0,
    toolsDisabledLatencySec: 3.613,
    toolsEnabledLatencySec: 3.365,
  },
  {
    category: 'Multi-step reasoning',
    sampleSize: 30,
    toolsDisabledSuccess: 11,
    toolsDisabledPct: 36.7,
    toolsEnabledSuccess: 20,
    toolsEnabledPct: 66.7,
    deltaPct: 30.0,
    relativeImprovementPct: 81.8,
    toolsDisabledLatencySec: 1.857,
    toolsEnabledLatencySec: 4.089,
  },
  {
    category: 'Current information',
    sampleSize: 8,
    toolsDisabledSuccess: 6,
    toolsDisabledPct: 75.0,
    toolsEnabledSuccess: 7,
    toolsEnabledPct: 87.5,
    deltaPct: 12.5,
    relativeImprovementPct: 16.7,
    toolsDisabledLatencySec: 0.58,
    toolsEnabledLatencySec: 0.496,
  },
  {
    category: 'Tool discrimination',
    sampleSize: 11,
    toolsDisabledSuccess: 8,
    toolsDisabledPct: 72.7,
    toolsEnabledSuccess: 9,
    toolsEnabledPct: 81.8,
    deltaPct: 9.1,
    relativeImprovementPct: 12.5,
    toolsDisabledLatencySec: 1.069,
    toolsEnabledLatencySec: 1.802,
  },
  {
    category: 'Numerical computation',
    sampleSize: 6,
    toolsDisabledSuccess: 2,
    toolsDisabledPct: 33.3,
    toolsEnabledSuccess: 2,
    toolsEnabledPct: 33.3,
    deltaPct: 0.0,
    relativeImprovementPct: 0.0,
    toolsDisabledLatencySec: 1.58,
    toolsEnabledLatencySec: 0.895,
  },
  {
    category: 'Evidence synthesis',
    sampleSize: 15,
    toolsDisabledSuccess: 14,
    toolsDisabledPct: 93.3,
    toolsEnabledSuccess: 14,
    toolsEnabledPct: 93.3,
    deltaPct: 0.0,
    relativeImprovementPct: 0.0,
    toolsDisabledLatencySec: 0.946,
    toolsEnabledLatencySec: 1.816,
  },
  {
    category: 'Partial evidence',
    sampleSize: 5,
    toolsDisabledSuccess: 0,
    toolsDisabledPct: 0.0,
    toolsEnabledSuccess: 0,
    toolsEnabledPct: 0.0,
    deltaPct: 0.0,
    relativeImprovementPct: 0.0,
    toolsDisabledLatencySec: 0.613,
    toolsEnabledLatencySec: 1.496,
  },
  {
    category: 'Contradictory evidence',
    sampleSize: 8,
    toolsDisabledSuccess: 0,
    toolsDisabledPct: 0.0,
    toolsEnabledSuccess: 0,
    toolsEnabledPct: 0.0,
    deltaPct: 0.0,
    relativeImprovementPct: 0.0,
    toolsDisabledLatencySec: 0.854,
    toolsEnabledLatencySec: 2.092,
  },
  {
    category: 'Insufficient evidence',
    sampleSize: 5,
    toolsDisabledSuccess: 0,
    toolsDisabledPct: 0.0,
    toolsEnabledSuccess: 0,
    toolsEnabledPct: 0.0,
    deltaPct: 0.0,
    relativeImprovementPct: 0.0,
    toolsDisabledLatencySec: 0.622,
    toolsEnabledLatencySec: 4.664,
  },
];
