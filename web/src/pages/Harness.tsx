import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  GitCompare,
  Activity,
  History,
  Cpu,
  RefreshCw,
  Database,
  Zap,
  AlertCircle,
  Lock,
  Unlock,
  Loader2,
} from 'lucide-react';
import type { ModelInfo, TelemetryInfo, GenerationSettings } from '../types/harness';
import {
  getHealth,
  getModels,
  selectModel,
  unloadModel,
  isBackendConfigured,
  isComputeAuthorized,
} from '../services/api';
import { HarnessControls } from '../components/harness/HarnessControls';
import { ChatArea } from '../components/harness/ChatArea';
import { InspectionPanel } from '../components/harness/InspectionPanel';
import { RetrievalComparisonView } from '../components/harness/RetrievalComparisonView';
import { TrainingDashboardView } from '../components/harness/TrainingDashboardView';
import { ExperimentTracesView } from '../components/harness/ExperimentTracesView';
import { ComputeAuthModal } from '../components/auth/ComputeAuthModal';
import styles from '../components/harness/Harness.module.css';

const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 40,
  max_new_tokens: 256,
  repetition_penalty: 1.15,
  system_prompt: 'You are QORIN, an open-weights small language model research assistant developed by the QORIN research project. Answer accurately, thoughtfully, and concisely.',
  retrieval_mode: 'auto',
  retrieval_top_k: 3,
  retrieval_threshold: 0.15,
  web_research_enabled: false,
  python_sandbox_enabled: false,
};

const FALLBACK_MODELS: ModelInfo[] = [
  {
    id: 'qorin-500m-sft-exp-f',
    name: 'QORIN-500M SFT (Autonomous Multi-Turn)',
    provider: 'qorin',
    generation: '500M',
    parameters: 514389504,
    context_length: 2048,
    architecture: '20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, BPE-32k',
    status: 'production_sft',
    description: 'Flagship production candidate fine-tuned for autonomous sequential multi-turn tool chaining (Web -> Python -> Grounded Synthesis).',
    is_native: true,
    supported_params: ['temperature', 'top_p', 'top_k', 'max_new_tokens', 'repetition_penalty'],
    vram_estimate_mb: 2100,
  },
  {
    id: 'qorin-100m-sft-v3',
    name: 'QORIN-100M SFT v3 (4K Context)',
    provider: 'qorin',
    generation: '100M',
    parameters: 100401408,
    context_length: 4096,
    architecture: '14L-12Q-4KV (GQA 3:1), SwiGLU, RMSNorm, RoPE s=8.0',
    status: 'production_sft',
    description: 'Fine-tuned 100M model with 4,096-token sequence capability and ChatML alignment.',
    is_native: true,
    supported_params: ['temperature', 'top_p', 'top_k', 'max_new_tokens', 'repetition_penalty'],
    vram_estimate_mb: 650,
  },
  {
    id: 'qorin-50m-sft',
    name: 'QORIN-50M SFT v2',
    provider: 'qorin',
    generation: '50M',
    parameters: 49105408,
    context_length: 512,
    architecture: '12L-8Q-4KV (GQA 2:1), SwiGLU, RMSNorm, RoPE base',
    status: 'completed',
    description: '50M parameter SFT checkpoint fine-tuned on multi-domain instruction mixture.',
    is_native: true,
    supported_params: ['temperature', 'top_p', 'top_k', 'max_new_tokens'],
    vram_estimate_mb: 350,
  },
  {
    id: 'qorin-16m-sft',
    name: 'QORIN-16M SFT Baseline',
    provider: 'qorin',
    generation: '16M',
    parameters: 16768128,
    context_length: 512,
    architecture: '6L-6H Standard MHA, GELU, LayerNorm, RoPE base',
    status: 'completed',
    description: 'Lightweight 16M proof-of-concept checkpoint used in capability diagnosis.',
    is_native: true,
    supported_params: ['temperature', 'top_p', 'top_k', 'max_new_tokens'],
    vram_estimate_mb: 180,
  },
];

export const Harness: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'compare' | 'training' | 'traces'>('chat');
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(FALLBACK_MODELS[0]);
  const [telemetry, setTelemetry] = useState<TelemetryInfo | null>(null);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);
  const [isBackendAvailable, setIsBackendAvailable] = useState<boolean>(true);
  const [connectionState, setConnectionState] = useState<'online' | 'offline' | 'waking' | 'checking'>('checking');
  const [wakeAttempt, setWakeAttempt] = useState<number>(0);
  const [authModalOpen, setAuthModalOpen] = useState<boolean>(false);
  const [isAuthorized, setIsAuthorized] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<{
    ttft_ms?: number;
    tok_per_sec?: number;
    latency_ms?: number;
    token_count?: number;
    injected_prompt?: string;
  }>({});

  useEffect(() => {
    const checkAuth = () => setIsAuthorized(isComputeAuthorized());
    checkAuth();
    window.addEventListener('qorin-compute-auth-changed', checkAuth);
    return () => window.removeEventListener('qorin-compute-auth-changed', checkAuth);
  }, []);

  useEffect(() => {
    initHarness();
  }, []);

  const pollAwakening = async () => {
    setConnectionState('waking');
    const MAX_ATTEMPTS = 12;
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      setWakeAttempt(attempt);
      await new Promise((r) => setTimeout(r, 4000));
      try {
        const healthData = await getHealth();
        if (healthData.status === 'healthy') {
          const modelsList = await getModels().catch(() => []);
          setIsBackendAvailable(true);
          setConnectionState('online');
          setWakeAttempt(0);
          if (healthData.telemetry) setTelemetry(healthData.telemetry);
          const activeList = modelsList.length > 0 ? modelsList : FALLBACK_MODELS;
          setModels(activeList);
          const defaultM =
            activeList.find((m) => m.id === 'qorin-500m-sft-exp-f') ||
            activeList.find((m) => m.id === 'qorin-100m-sft-v3') ||
            activeList[0] ||
            null;
          setSelectedModel(defaultM);
          return;
        }
      } catch {
        // Space is still awakening, continue polling
      }
    }

    setIsBackendAvailable(false);
    setConnectionState('offline');
    setWakeAttempt(0);
    setModels(FALLBACK_MODELS);
    setSelectedModel(FALLBACK_MODELS[0]);
  };

  const initHarness = async () => {
    if (!isBackendConfigured) {
      setIsBackendAvailable(false);
      setConnectionState('offline');
      setTelemetry(null);
      setModels(FALLBACK_MODELS);
      setSelectedModel(FALLBACK_MODELS[0]);
      return;
    }

    setConnectionState('checking');

    try {
      const [healthData, modelsList] = await Promise.all([
        getHealth().catch(() => ({ status: 'offline', telemetry: null as any })),
        getModels().catch(() => [] as ModelInfo[]),
      ]);

      const online = healthData.status === 'healthy' || modelsList.length > 0;
      if (online) {
        setIsBackendAvailable(true);
        setConnectionState('online');
        setWakeAttempt(0);

        if (healthData.telemetry) {
          setTelemetry(healthData.telemetry);
        } else {
          setTelemetry(null);
        }

        const activeList = modelsList.length > 0 ? modelsList : FALLBACK_MODELS;
        setModels(activeList);

        const defaultM =
          activeList.find((m) => m.id === 'qorin-500m-sft-exp-f') ||
          activeList.find((m) => m.id === 'qorin-100m-sft-v3') ||
          activeList[0] ||
          null;
        setSelectedModel(defaultM);
        return;
      }
    } catch (err) {
      console.warn('Initial connection probe failed:', err);
    }

    // Configured backend failed initial ping: could be sleeping Hugging Face Space
    pollAwakening();
  };

  const handleSelectModel = async (modelId: string) => {
    const target = models.find((m) => m.id === modelId) || null;
    setSelectedModel(target);

    if (isBackendAvailable) {
      setIsLoadingModel(true);
      try {
        const res = await selectModel(modelId);
        setTelemetry(res.telemetry);
      } catch (err: any) {
        console.warn('Model selection notice:', err.message);
        if (err.message && (err.message.includes('compute access protected') || err.message.includes('401'))) {
          setAuthModalOpen(true);
        }
      } finally {
        setIsLoadingModel(false);
      }
    }
  };

  const handleUnloadModel = async () => {
    if (!isBackendAvailable) return;
    setIsLoadingModel(true);
    try {
      const res = await unloadModel();
      setTelemetry(res.telemetry);
    } catch (err: any) {
      console.warn('Failed to unload model:', err.message);
    } finally {
      setIsLoadingModel(false);
    }
  };

  const handleUpdateSettings = (newSettings: Partial<GenerationSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  };

  const isGpuActive = telemetry?.qorin?.cuda_available;

  return (
    <div className={styles.harnessWrap}>
      {/* Editorial Header */}
      <header className={styles.pageHeader}>
        <div className={styles.container}>
          <div className={styles.kicker}>QORIN Research Infrastructure</div>
          <h1 className={styles.title}>Model Experimentation Harness</h1>
          <p className={styles.lead}>
            A functional inference testbed bridging native small language model checkpoints with
            grounded TF-IDF retrieval, isolated Python tool execution, real-world web research, and
            transparent GPU training observability.
          </p>

          {/* Tab Navigation */}
          <nav className={styles.tabsBar} aria-label="Harness Navigation Tabs">
            <button
              type="button"
              className={`${styles.tabBtn} ${activeTab === 'chat' ? styles.tabBtnActive : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <MessageSquare size={14} />
              <span>Interactive Chat &amp; Tools</span>
            </button>

            <button
              type="button"
              className={`${styles.tabBtn} ${activeTab === 'compare' ? styles.tabBtnActive : ''}`}
              onClick={() => setActiveTab('compare')}
            >
              <GitCompare size={14} />
              <span>Retrieval A/B Lab</span>
            </button>

            <button
              type="button"
              className={`${styles.tabBtn} ${activeTab === 'training' ? styles.tabBtnActive : ''}`}
              onClick={() => setActiveTab('training')}
            >
              <Activity size={14} />
              <span>Training Observability</span>
            </button>

            <button
              type="button"
              className={`${styles.tabBtn} ${activeTab === 'traces' ? styles.tabBtnActive : ''}`}
              onClick={() => setActiveTab('traces')}
            >
              <History size={14} />
              <span>Experiment Traces</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className={styles.container}>
        {/* Top Hardware Telemetry Strip */}
        <div className={styles.topBar}>
          <div className={styles.modelSelectGroup}>
            <span className={styles.selectLabel}>ACTIVE CHECKPOINT:</span>
            <strong style={{ fontSize: '13px', fontFamily: 'var(--font-mono)' }}>
              {selectedModel?.name || 'No model loaded'}
            </strong>
            <span style={{ fontSize: '11px', color: 'var(--color-ink-muted)' }}>
              ({selectedModel ? (selectedModel.parameters / 1e6).toFixed(1) : 0}M params &bull; {selectedModel?.context_length} ctx)
            </span>
          </div>

          <div className={styles.telemetryGroup}>
            <div className={`${styles.telemetryPill} ${isBackendAvailable && isGpuActive ? styles.telemetryPillActive : ''}`}>
              <Cpu size={12} />
              <span>
                {isBackendAvailable
                  ? telemetry?.qorin?.device_name || (isGpuActive ? 'RTX 5060 (CUDA)' : 'CPU Mode')
                  : 'Backend Offline'}
              </span>
            </div>

            {isBackendAvailable ? (
              <>
                <div className={styles.telemetryPill}>
                  <Zap size={12} />
                  <span>
                    VRAM: {telemetry?.qorin?.vram_allocated_mb?.toFixed(0) || 0} MB
                  </span>
                </div>

                <div className={styles.telemetryPill}>
                  <Database size={12} />
                  <span>RAG: {settings.retrieval_mode.toUpperCase()}</span>
                </div>

                {isAuthorized ? (
                  <button
                    type="button"
                    onClick={() => setAuthModalOpen(true)}
                    className={styles.telemetryPillAuthUnlocked}
                    title="Compute Unlocked: Prof. Udit Bhatia. Click to inspect or revoke authorization."
                  >
                    <Unlock size={11} />
                    <span>Compute: Authorized</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => setAuthModalOpen(true)}
                    className={styles.telemetryPillAuthLocked}
                    title="Workstation compute locked. Click to enter professor's name."
                  >
                    <Lock size={11} />
                    <span>Compute: Locked</span>
                  </button>
                )}
              </>
            ) : (
              <div className={styles.telemetryPill} style={{ color: 'var(--color-ink-muted)' }}>
                <span title="Research showcase running in static mode">Static Showcase Mode</span>
              </div>
            )}

            <button
              type="button"
              onClick={initHarness}
              className={styles.clearBtn}
              style={{ padding: '3px 8px', fontSize: '10px' }}
              title="Refresh telemetry and backend connection"
            >
              <RefreshCw size={11} /> Sync
            </button>
          </div>
        </div>

        {/* Remote Hugging Face Space Awakening Banner */}
        {connectionState === 'waking' && (
          <div
            style={{
              padding: '10px 14px',
              marginBottom: 'var(--space-4)',
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '12px',
              lineHeight: 1.5,
              color: 'var(--color-ink-secondary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Loader2 size={15} className={styles.spin} style={{ color: 'var(--color-accent-indigo)', flexShrink: 0 }} />
              <div>
                <strong style={{ color: 'var(--color-ink-primary)' }}>Awakening Hugging Face CPU Container:</strong>{' '}
                The remote inference backend is spinning up from idle sleep (probe {wakeAttempt}/12). Container initialization typically takes 30–50 seconds.
              </div>
            </div>
          </div>
        )}

        {/* Backend Offline / Static Showcase Banner */}
        {connectionState !== 'waking' && !isBackendAvailable && (
          <div
            style={{
              padding: '10px 14px',
              marginBottom: 'var(--space-4)',
              background: 'var(--color-paper-subtle)',
              border: '1px solid var(--color-rule)',
              borderRadius: 'var(--radius-xs)',
              fontSize: '12px',
              lineHeight: 1.5,
              color: 'var(--color-ink-secondary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
              flexWrap: 'wrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertCircle size={15} style={{ color: 'var(--color-accent-amber)', flexShrink: 0 }} />
              <div>
                <strong style={{ color: 'var(--color-ink-primary)' }}>Inference Backend Service Offline:</strong>{' '}
                {isBackendConfigured
                  ? 'Unable to reach the remote Hugging Face CPU backend. The website is operating in static showcase mode.'
                  : 'The website is operating in static mode. Architecture blueprints, experiment archives, and research narratives are fully accessible.'}
              </div>
            </div>
            <button
              type="button"
              onClick={initHarness}
              style={{
                background: 'var(--color-surface)',
                border: '1px solid var(--color-rule)',
                borderRadius: '4px',
                padding: '4px 10px',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                cursor: 'pointer',
                color: 'var(--color-ink-primary)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
              }}
            >
              <RefreshCw size={11} /> Reconnect
            </button>
          </div>
        )}

        {/* Tab 1: Interactive Chat & Tool Harness */}
        {activeTab === 'chat' && (
          <div>
            <div className={styles.workspaceLayout}>
              <HarnessControls
                models={models}
                selectedModel={selectedModel}
                onSelectModel={handleSelectModel}
                onUnloadModel={handleUnloadModel}
                settings={settings}
                onUpdateSettings={handleUpdateSettings}
                telemetry={telemetry}
                isLoadingModel={isLoadingModel}
              />

              <ChatArea
                selectedModel={selectedModel}
                settings={settings}
                onUpdateSettings={handleUpdateSettings}
                onTraceRecorded={() => {}}
                onActivePromptChanged={(prompt) => setLastStats((prev) => ({ ...prev, injected_prompt: prompt }))}
                isBackendAvailable={isBackendAvailable}
                onRequireComputeAuth={() => setAuthModalOpen(true)}
              />

              <InspectionPanel
                telemetry={telemetry}
                activeModel={selectedModel}
                lastStats={lastStats}
              />
            </div>
          </div>
        )}

        {/* Tab 2: Retrieval A/B Comparison */}
        {activeTab === 'compare' && (
          <RetrievalComparisonView
            models={models}
            selectedModelId={selectedModel?.id || 'qorin-500m-sft-exp-f'}
          />
        )}

        {/* Tab 3: Training Dashboard */}
        {activeTab === 'training' && (
          <TrainingDashboardView />
        )}

        {/* Tab 4: Historical Traces */}
        {activeTab === 'traces' && (
          <ExperimentTracesView />
        )}
      </main>

      {/* Personal Compute Access Protection Modal */}
      <ComputeAuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthorized={() => setIsAuthorized(true)}
      />
    </div>
  );
};
