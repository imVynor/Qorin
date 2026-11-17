import React from 'react';
import { Sliders, Database, Terminal, Cpu } from 'lucide-react';
import type { GenerationSettings, ModelInfo, TelemetryInfo } from '../../types/harness';
import styles from './Harness.module.css';

interface HarnessControlsProps {
  models: ModelInfo[];
  selectedModel: ModelInfo | null;
  onSelectModel: (modelId: string) => void;
  onUnloadModel: () => void;
  settings: GenerationSettings;
  onUpdateSettings: (newSettings: Partial<GenerationSettings>) => void;
  telemetry: TelemetryInfo | null;
  isLoadingModel: boolean;
}

export const HarnessControls: React.FC<HarnessControlsProps> = ({
  models,
  selectedModel,
  onSelectModel,
  onUnloadModel,
  settings,
  onUpdateSettings,
  telemetry: _telemetry,
  isLoadingModel,
}) => {
  return (
    <aside className={styles.sidebarCard}>
      {/* 1. Model & Provider Selection */}
      <div>
        <div className={styles.cardSectionTitle}>
          <Cpu size={13} color="var(--color-accent-indigo)" />
          <span>Model Architecture</span>
        </div>

        <div className={styles.settingItem}>
          <select
            className={styles.modelSelect}
            value={selectedModel?.id || ''}
            onChange={(e) => onSelectModel(e.target.value)}
            disabled={isLoadingModel}
            style={{ width: '100%' }}
          >
            <optgroup label="Native QORIN Checkpoints">
              {models
                .filter((m) => m.is_native)
                .map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} ({m.context_length} ctx)
                  </option>
                ))}
            </optgroup>

            {models.some((m) => !m.is_native) && (
              <optgroup label="External Benchmark Models (Ollama)">
                {models
                  .filter((m) => !m.is_native)
                  .map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
              </optgroup>
            )}
          </select>
        </div>

        {selectedModel && (
          <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
            <div>Arch: {selectedModel.architecture}</div>
            <div>Params: {selectedModel.parameters.toLocaleString()} &bull; Context: {selectedModel.context_length} tokens</div>
            <div>Status: <strong style={{ color: selectedModel.status === 'ready' ? 'var(--color-accent-sage)' : 'var(--color-ink-primary)' }}>{selectedModel.status}</strong></div>
          </div>
        )}

        <div style={{ display: 'flex', gap: '6px', marginTop: '8px' }}>
          {selectedModel && selectedModel.status === 'ready' && (
            <button
              type="button"
              className={styles.clearBtn}
              onClick={onUnloadModel}
              style={{ fontSize: '10px', padding: '4px 8px' }}
            >
              Unload VRAM
            </button>
          )}
        </div>
      </div>

      {/* 2. Generation Parameters */}
      <div>
        <div className={styles.cardSectionTitle}>
          <Sliders size={13} color="var(--color-accent-indigo)" />
          <span>Generation Controls</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div className={styles.settingItem}>
            <div className={styles.settingLabel}>
              <span>Temperature</span>
              <span>{settings.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={settings.temperature}
              onChange={(e) => onUpdateSettings({ temperature: parseFloat(e.target.value) })}
              className={styles.sliderInput}
            />
          </div>

          <div className={styles.settingItem}>
            <div className={styles.settingLabel}>
              <span>Top-P (Nucleus)</span>
              <span>{settings.top_p.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={settings.top_p}
              onChange={(e) => onUpdateSettings({ top_p: parseFloat(e.target.value) })}
              className={styles.sliderInput}
            />
          </div>

          <div className={styles.settingItem}>
            <div className={styles.settingLabel}>
              <span>Max New Tokens</span>
              <span>{settings.max_new_tokens}</span>
            </div>
            <input
              type="range"
              min="32"
              max={selectedModel?.context_length ? Math.min(1024, selectedModel.context_length) : 512}
              step="32"
              value={settings.max_new_tokens}
              onChange={(e) => onUpdateSettings({ max_new_tokens: parseInt(e.target.value, 10) })}
              className={styles.sliderInput}
            />
          </div>

          <div className={styles.settingItem}>
            <div className={styles.settingLabel}>
              <span>Repetition Penalty</span>
              <span>{settings.repetition_penalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="1.5"
              step="0.05"
              value={settings.repetition_penalty}
              onChange={(e) => onUpdateSettings({ repetition_penalty: parseFloat(e.target.value) })}
              className={styles.sliderInput}
            />
          </div>
        </div>
      </div>

      {/* 3. Non-Parametric Retrieval Controls */}
      <div>
        <div className={styles.cardSectionTitle}>
          <Database size={13} color="var(--color-accent-indigo)" />
          <span>Knowledge Retrieval (RAG)</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div className={styles.segmentedBtns}>
            {(['off', 'auto', 'always'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`${styles.segBtn} ${settings.retrieval_mode === mode ? styles.segBtnActive : ''}`}
                onClick={() => onUpdateSettings({ retrieval_mode: mode })}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>

          {settings.retrieval_mode !== 'off' && (
            <>
              <div className={styles.settingItem} style={{ marginTop: '4px' }}>
                <div className={styles.settingLabel}>
                  <span>Top-K Chunks</span>
                  <span>{settings.retrieval_top_k}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={settings.retrieval_top_k}
                  onChange={(e) => onUpdateSettings({ retrieval_top_k: parseInt(e.target.value, 10) })}
                  className={styles.sliderInput}
                />
              </div>

              <div className={styles.settingItem}>
                <div className={styles.settingLabel}>
                  <span>Similarity Cutoff</span>
                  <span>{settings.retrieval_threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="0.30"
                  step="0.02"
                  value={settings.retrieval_threshold}
                  onChange={(e) => onUpdateSettings({ retrieval_threshold: parseFloat(e.target.value) })}
                  className={styles.sliderInput}
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* 4. External Tool Orchestration */}
      <div>
        <div className={styles.cardSectionTitle}>
          <Terminal size={13} color="var(--color-accent-sage)" />
          <span>Capability Controls</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <label className={styles.checkboxItem} title="Advertises Web Researcher to QORIN and enables live retrieval with source attribution">
            <input
              type="checkbox"
              className={styles.checkboxInput}
              checked={settings.web_research_enabled}
              onChange={(e) => onUpdateSettings({ web_research_enabled: e.target.checked })}
            />
            <span style={{ fontWeight: 500 }}>Web retrieval</span>
            <span style={{ marginLeft: 'auto', fontSize: '9px', padding: '1px 5px', borderRadius: '3px', background: settings.web_research_enabled ? 'rgba(74, 222, 128, 0.15)' : 'var(--color-surface)', color: settings.web_research_enabled ? 'var(--color-accent-sage)' : 'var(--color-ink-muted)' }}>
              {settings.web_research_enabled ? 'ACTIVE' : 'OFF'}
            </span>
          </label>

          <label className={styles.checkboxItem} title="Advertises Python Sandbox to QORIN and dispatches code execution for mathematics and data analysis">
            <input
              type="checkbox"
              className={styles.checkboxInput}
              checked={settings.python_sandbox_enabled}
              onChange={(e) => onUpdateSettings({ python_sandbox_enabled: e.target.checked })}
            />
            <span style={{ fontWeight: 500 }}>Python execution</span>
            <span style={{ marginLeft: 'auto', fontSize: '9px', padding: '1px 5px', borderRadius: '3px', background: settings.python_sandbox_enabled ? 'rgba(74, 222, 128, 0.15)' : 'var(--color-surface)', color: settings.python_sandbox_enabled ? 'var(--color-accent-sage)' : 'var(--color-ink-muted)' }}>
              {settings.python_sandbox_enabled ? 'ACTIVE' : 'OFF'}
            </span>
          </label>
        </div>

        <div style={{ marginTop: '8px', fontSize: '10px', color: 'var(--color-ink-muted)', lineHeight: 1.4 }}>
          {!settings.web_research_enabled && !settings.python_sandbox_enabled && (
            <span>Standalone Model Mode (Both tools disabled)</span>
          )}
          {settings.web_research_enabled && !settings.python_sandbox_enabled && (
            <span>Web Researcher Active (Python disabled)</span>
          )}
          {!settings.web_research_enabled && settings.python_sandbox_enabled && (
            <span>Python Sandbox Active (Web disabled)</span>
          )}
          {settings.web_research_enabled && settings.python_sandbox_enabled && (
            <span>Two-Tool Orchestrator Mode (Python + Web)</span>
          )}
        </div>
      </div>
    </aside>
  );
};
