import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  RotateCcw,
  BookOpen,
  Terminal,
  Globe,
  CloudSun,
  ChevronDown,
  ChevronUp,
  ArrowDown,
} from 'lucide-react';
import type { ChatMessage, GenerationSettings, ModelInfo, RetrievalTrace, ToolTrace } from '../../types/harness';
import { streamChat } from '../../services/api';
import styles from './Harness.module.css';

interface ChatAreaProps {
  selectedModel: ModelInfo | null;
  settings: GenerationSettings;
  onUpdateSettings?: (newSettings: Partial<GenerationSettings>) => void;
  onTraceRecorded?: () => void;
  onActivePromptChanged?: (prompt: string) => void;
  isBackendAvailable?: boolean;
  onRequireComputeAuth?: () => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  selectedModel,
  settings,
  onUpdateSettings,
  onTraceRecorded,
  onActivePromptChanged,
  isBackendAvailable = true,
  onRequireComputeAuth,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveStreamText, setLiveStreamText] = useState('');
  const [liveStats, setLiveStats] = useState<{ ttft_ms?: number; tok_per_sec?: number; token_count?: number }>({});
  const [expandedRetrievalId, setExpandedRetrievalId] = useState<string | null>(null);
  const [activeRetrievalTrace, setActiveRetrievalTrace] = useState<RetrievalTrace | null>(null);
  const [activeToolTraces, setActiveToolTraces] = useState<ToolTrace[]>([]);

  const conversationIdRef = useRef<string>(`conv-${Date.now()}`);
  const abortRef = useRef<(() => void) | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isAutoScrollLocked, setIsAutoScrollLocked] = useState(true);
  const [showJumpButton, setShowJumpButton] = useState(false);

  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const distanceToBottom = scrollHeight - (scrollTop + clientHeight);
    const isNearBottom = distanceToBottom <= 60;
    setIsAutoScrollLocked(isNearBottom);
    setShowJumpButton(!isNearBottom);
  };

  const scrollToBottom = () => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
      setIsAutoScrollLocked(true);
      setShowJumpButton(false);
    }
  };

  useEffect(() => {
    if (isAutoScrollLocked && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [messages, liveStreamText, activeToolTraces, isAutoScrollLocked]);

  const handleSend = () => {
    if (!inputPrompt.trim() || isGenerating) return;

    if (!isBackendAvailable) {
      const userMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        content: inputPrompt.trim(),
      };
      const offlineMsg: ChatMessage = {
        id: `msg-offline-${Date.now() + 1}`,
        role: 'assistant',
        content: `[Backend Inference Service Offline]\n\nLive token streaming and tool dispatch require the local Python FastAPI server (http://127.0.0.1:8000).\n\nThe research monograph, architecture specifications, interactive learning lab, and experiment archives remain fully functional in static mode.\n\nTo run live model generation locally:\n1. Start the API backend: .venv\\Scripts\\python -m uvicorn backend.app:app --port 8000\n2. Click "Sync" in the top telemetry strip to connect.`,
      };
      setMessages((prev) => [...prev, userMsg, offlineMsg]);
      setInputPrompt('');
      return;
    }

    if (!selectedModel) return;

    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: inputPrompt.trim(),
    };

    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInputPrompt('');
    setIsGenerating(true);
    setLiveStreamText('');
    setLiveStats({});
    setActiveRetrievalTrace(null);
    setActiveToolTraces([]);
    setIsAutoScrollLocked(true);
    setShowJumpButton(false);
    setTimeout(() => {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
      }
    }, 10);

    if (onActivePromptChanged) {
      onActivePromptChanged(userMsg.content);
    }

    const payload = {
      model_id: selectedModel.id,
      messages: newHistory.map((m) => ({ role: m.role, content: m.content })),
      temperature: settings.temperature,
      top_p: settings.top_p,
      top_k: settings.top_k,
      max_new_tokens: settings.max_new_tokens,
      repetition_penalty: settings.repetition_penalty,
      system_prompt: settings.system_prompt || undefined,
      retrieval_mode: settings.retrieval_mode,
      retrieval_top_k: settings.retrieval_top_k,
      retrieval_threshold: settings.retrieval_threshold,
      web_research_enabled: settings.web_research_enabled,
      python_sandbox_enabled: settings.python_sandbox_enabled,
      conversation_id: conversationIdRef.current,
    };

    let fullText = '';
    let currentRetrieval: RetrievalTrace | null = null;
    const currentTools: ToolTrace[] = [];

    const stopFn = streamChat(payload, {
      onRetrieval: (trace) => {
        currentRetrieval = trace;
        setActiveRetrievalTrace(trace);
      },
      onToolStart: (_toolData) => {
        // Track tool initiation
      },
      onToolResult: (trace) => {
        currentTools.push(trace);
        setActiveToolTraces([...currentTools]);
      },
      onToken: (tok) => {
        fullText += tok.text;
        setLiveStreamText(fullText);
        setLiveStats({
          ttft_ms: tok.ttft_ms,
          tok_per_sec: tok.tok_per_sec,
          token_count: tok.token_count,
        });
      },
      onDone: (doneData) => {
        const assistantMsg: ChatMessage = {
          id: `msg-${Date.now() + 1}`,
          role: 'assistant',
          content: fullText,
          retrieval_trace: currentRetrieval || undefined,
          tool_traces: currentTools.length > 0 ? currentTools : undefined,
          ttft_ms: doneData.ttft_ms,
          tok_per_sec: doneData.tok_per_sec,
          latency_ms: doneData.total_latency_ms,
          token_count: doneData.total_tokens,
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setIsGenerating(false);
        setLiveStreamText('');
        abortRef.current = null;
        if (onTraceRecorded) onTraceRecorded();
      },
      onError: (err) => {
        const isAuthError =
          err.message &&
          (err.message.includes('compute access protected') ||
            err.message.includes('401') ||
            err.message.includes('ALL CAPS'));

        const errorMsg: ChatMessage = {
          id: `msg-err-${Date.now()}`,
          role: 'assistant',
          content: isAuthError
            ? `[Personal Compute Protected]\n\nAccess to the author's local GPU inference and execution runtime is restricted to Prof. Udit Bhatia.\n\nHint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX).\n\nPlease authorize compute access to run live inference.`
            : `[Inference Error: ${err.message}]`,
        };
        setMessages((prev) => [...prev, errorMsg]);
        setIsGenerating(false);
        abortRef.current = null;

        if (isAuthError && onRequireComputeAuth) {
          onRequireComputeAuth();
        }
      },
    });

    abortRef.current = stopFn;
  };

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setIsGenerating(false);
  };

  const handleClear = () => {
    handleStop();
    conversationIdRef.current = `conv-${Date.now()}`;
    setMessages([]);
    setLiveStreamText('');
    setActiveRetrievalTrace(null);
    setActiveToolTraces([]);
    setIsAutoScrollLocked(true);
    setShowJumpButton(false);
    setTimeout(() => {
      if (scrollContainerRef.current) {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
      }
    }, 10);
  };

  const PRESET_QUERIES = [
    'What are the architecture specs of QORIN-500M?',
    'Explain the RoPE frequency scaling applied in the 4K pilot.',
    'Why did SFT v3 experience conversational collapse?',
    'What is the hardware thermal duty cycle on the RTX 5060?',
    '```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\nprint([fib(i) for i in range(8)])\n```',
  ];

  return (
    <div className={styles.chatMainArea}>
      {/* Messages Scroll Area */}
      <div
        className={styles.chatMessages}
        ref={scrollContainerRef}
        onScroll={handleScroll}
      >
        {messages.length === 0 && !isGenerating ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyTitle}>QORIN Research Inference Harness</div>
            <p style={{ maxWidth: '480px', lineHeight: 1.5, fontSize: '13px' }}>
              Select a native QORIN model or external test model to investigate raw generation, non-parametric knowledge retrieval, or isolated tool invocation.
            </p>

            <div className={styles.presetGroup}>
              {PRESET_QUERIES.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  className={styles.presetBtn}
                  onClick={() => setInputPrompt(q)}
                >
                  {q.length > 55 ? q.substring(0, 52) + '...' : q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              className={`${styles.messageRow} ${m.role === 'user' ? styles.userRow : styles.assistantRow}`}
            >
              {/* Optional Inline Retrieval Trace Pill */}
              {m.retrieval_trace && m.retrieval_trace.triggered && m.retrieval_trace.retrieved_chunks && m.retrieval_trace.retrieved_chunks.length > 0 && (
                <div className={styles.inlineCard}>
                  <div
                    className={styles.inlineRetrievalHeader}
                    onClick={() =>
                      setExpandedRetrievalId(expandedRetrievalId === m.id ? null : m.id)
                    }
                  >
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                      <BookOpen size={12} />
                      <span>Retrieved {m.retrieval_trace.retrieved_chunks.length} Knowledge Chunks</span>
                    </span>
                    {expandedRetrievalId === m.id ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </div>

                  {expandedRetrievalId === m.id && (
                    <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid var(--color-rule-subtle)' }}>
                      <div style={{ color: 'var(--color-ink-muted)', marginBottom: '4px' }}>
                        Reason: {m.retrieval_trace.decision_reason} ({m.retrieval_trace.retrieval_latency_ms}ms)
                      </div>
                      {m.retrieval_trace.retrieved_chunks.map((c) => (
                        <div key={c.chunk_id} style={{ marginTop: '4px', padding: '4px 8px', background: 'var(--color-surface)', borderRadius: '2px' }}>
                          <strong>{c.doc_title} ({c.section})</strong> &bull; Score: {c.similarity_score}
                          <p style={{ margin: '2px 0 0', color: 'var(--color-ink-secondary)' }}>{c.snippet}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Optional Inline Tool Traces */}
              {m.tool_traces && m.tool_traces.map((t, tIdx) => (
                <div key={t.execution_id || `${m.id}-tool-${tIdx}`} className={styles.inlineCard}>
                  <div className={styles.inlineToolHeader}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                      {t.tool_name === 'python' ? <Terminal size={12} /> : t.tool_name === 'weather' ? <CloudSun size={12} /> : <Globe size={12} />}
                      <span>Tool Invocation: {t.tool_name} ({t.status}) {t.execution_id ? `[${t.execution_id.slice(-6)}]` : ''}</span>
                    </span>
                    <span>{t.latency_ms}ms</span>
                  </div>
                  {t.tool_name === 'python' && (
                    <div style={{ marginTop: '4px' }}>
                      <pre style={{ margin: 0, padding: '4px', fontSize: '10px' }}>
                        {t.output_payload?.stdout || t.output_payload?.stderr || 'No output.'}
                      </pre>
                    </div>
                  )}
                  {(t.tool_name === 'web_search' || t.tool_name === 'web_research') && (
                    <div style={{ marginTop: '4px' }}>
                      {t.output_payload?.sources && t.output_payload.sources.map((s: any, sIdx: number) => (
                        <div key={sIdx} style={{ fontSize: '10px' }}>
                          [{sIdx + 1}] <a href={s.url} target="_blank" rel="noreferrer" style={{ color: 'var(--color-accent-indigo)' }}>{s.title}</a>
                        </div>
                      ))}
                    </div>
                  )}
                  {t.tool_name === 'weather' && (
                    <div style={{ marginTop: '4px', fontSize: '11px', color: 'var(--color-ink-secondary)' }}>
                      {t.output_payload?.temperature_c !== undefined ? (
                        <div>
                          <strong>{t.output_payload.location_name || t.input_payload?.location}:</strong>{' '}
                          {t.output_payload.temperature_c}°C ({t.output_payload.weather_condition}) &bull; Humidity: {t.output_payload.relative_humidity_pct}% &bull; Wind: {t.output_payload.wind_speed_kmh} km/h
                          <div style={{ fontSize: '9px', color: 'var(--color-ink-muted)', marginTop: '2px' }}>
                            Source: {t.output_payload.source || 'Open-Meteo Weather API'} &bull; Observed: {t.output_payload.observation_time}
                          </div>
                        </div>
                      ) : (
                        <div>{t.error || t.output_payload?.error || 'Weather data unavailable.'}</div>
                      )}
                    </div>
                  )}
                </div>
              ))}

              <div className={m.role === 'user' ? styles.userBubble : styles.assistantBubble}>
                {m.content}
              </div>

              {/* Measured Metrics Ticker */}
              {m.role === 'assistant' && (m.ttft_ms !== undefined || m.tok_per_sec !== undefined) && (
                <div className={styles.statsTicker}>
                  {m.ttft_ms !== undefined && (
                    <span title="Time to first token (measured from API dispatch to first token yield)">
                      TTFT: {m.ttft_ms.toFixed(0)}ms
                    </span>
                  )}
                  {m.tok_per_sec !== undefined && (
                    <span title="Generation speed (measured tokens divided by active generation duration)">
                      &bull; {m.tok_per_sec.toFixed(1)} tok/s
                    </span>
                  )}
                  {m.token_count !== undefined && (
                    <span>&bull; {m.token_count} tokens</span>
                  )}
                  {m.latency_ms !== undefined && (
                    <span>&bull; {m.latency_ms.toFixed(0)}ms total</span>
                  )}
                </div>
              )}
            </div>
          ))
        )}

        {/* Live Streaming Turn */}
        {isGenerating && (
          <div className={`${styles.messageRow} ${styles.assistantRow}`}>
            {activeRetrievalTrace && activeRetrievalTrace.triggered && activeRetrievalTrace.retrieved_chunks && activeRetrievalTrace.retrieved_chunks.length > 0 && (
              <div className={styles.inlineCard}>
                <div className={styles.inlineRetrievalHeader}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                    <BookOpen size={12} />
                    <span>Injected {activeRetrievalTrace.retrieved_chunks.length} Knowledge Chunks ({activeRetrievalTrace.retrieval_latency_ms}ms)</span>
                  </span>
                </div>
              </div>
            )}

            {activeToolTraces.map((t, idx) => (
              <div key={t.execution_id || `live-tool-${idx}`} className={styles.inlineCard}>
                <div className={styles.inlineToolHeader}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                    {t.tool_name === 'python' ? <Terminal size={12} /> : t.tool_name === 'weather' ? <CloudSun size={12} /> : <Globe size={12} />}
                    <span>Tool: {t.tool_name} ({t.status}) {t.execution_id ? `[${t.execution_id.slice(-6)}]` : ''}</span>
                  </span>
                  <span>{t.latency_ms}ms</span>
                </div>
              </div>
            ))}

            <div className={styles.assistantBubble}>
              {liveStreamText || <span style={{ color: 'var(--color-ink-muted)' }}>Generating tokens on {selectedModel?.name}...</span>}
            </div>

            {liveStats.tok_per_sec !== undefined && (
              <div className={styles.statsTicker}>
                {liveStats.ttft_ms !== undefined && <span>TTFT: {liveStats.ttft_ms.toFixed(0)}ms</span>}
                {liveStats.tok_per_sec !== undefined && <span>&bull; {liveStats.tok_per_sec.toFixed(1)} tok/s</span>}
                {liveStats.token_count !== undefined && <span>&bull; {liveStats.token_count} tokens</span>}
              </div>
            )}
          </div>
        )}

      </div>

      {/* Floating Jump to Latest Button */}
      {showJumpButton && (
        <button
          type="button"
          className={styles.jumpToBottomBtn}
          onClick={scrollToBottom}
          title="Jump to latest generation"
        >
          <ArrowDown size={13} />
          <span>Jump to latest</span>
        </button>
      )}

      {/* Input Action Bar */}
      <div className={styles.inputBar}>
        <div className={styles.inputControlsRow}>
          <span>
            Target: <strong>{selectedModel?.name || 'No Model Loaded'}</strong>
            {selectedModel && ` (${selectedModel.parameters.toLocaleString()} params)`}
          </span>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            {messages.length > 0 && (
              <button
                type="button"
                className={styles.clearBtn}
                onClick={handleClear}
                disabled={isGenerating}
              >
                <RotateCcw size={11} /> Clear
              </button>
            )}
          </div>
        </div>

        {/* Capability Controls Toolbar (Web retrieval & Python execution) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px', background: 'var(--color-surface)', border: '1px solid var(--color-rule-subtle)', borderRadius: '4px', fontSize: '11px', flexWrap: 'wrap' }}>
          <span style={{ color: 'var(--color-ink-muted)', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.04em', fontWeight: 600 }}>
            Capabilities:
          </span>

          <button
            type="button"
            onClick={() => onUpdateSettings?.({ web_research_enabled: !settings.web_research_enabled })}
            disabled={isGenerating}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 8px',
              borderRadius: '3px',
              border: settings.web_research_enabled ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid var(--color-rule-subtle)',
              background: settings.web_research_enabled ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
              color: settings.web_research_enabled ? 'var(--color-accent-indigo)' : 'var(--color-ink-muted)',
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              fontWeight: 500,
              fontSize: '11px',
              transition: 'all 0.15s ease',
            }}
            title="When enabled, advertises Web Researcher to QORIN and executes live web retrieval with source evidence"
          >
            <Globe size={11} />
            <span>Web retrieval</span>
            <span style={{ fontSize: '9px', opacity: 0.85, fontWeight: 700 }}>
              {settings.web_research_enabled ? 'ON' : 'OFF'}
            </span>
          </button>

          <button
            type="button"
            onClick={() => onUpdateSettings?.({ python_sandbox_enabled: !settings.python_sandbox_enabled })}
            disabled={isGenerating}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px',
              padding: '3px 8px',
              borderRadius: '3px',
              border: settings.python_sandbox_enabled ? '1px solid rgba(74, 222, 128, 0.4)' : '1px solid var(--color-rule-subtle)',
              background: settings.python_sandbox_enabled ? 'rgba(74, 222, 128, 0.12)' : 'transparent',
              color: settings.python_sandbox_enabled ? 'var(--color-accent-sage)' : 'var(--color-ink-muted)',
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              fontWeight: 500,
              fontSize: '11px',
              transition: 'all 0.15s ease',
            }}
            title="When enabled, advertises Python Sandbox to QORIN and executes Python in an isolated subprocess"
          >
            <Terminal size={11} />
            <span>Python execution</span>
            <span style={{ fontSize: '9px', opacity: 0.85, fontWeight: 700 }}>
              {settings.python_sandbox_enabled ? 'ON' : 'OFF'}
            </span>
          </button>

          <span style={{ marginLeft: 'auto', fontSize: '10px', color: 'var(--color-ink-muted)', fontFamily: 'var(--font-mono)' }}>
            {!settings.web_research_enabled && !settings.python_sandbox_enabled && 'Standalone Mode (No Tools)'}
            {settings.web_research_enabled && !settings.python_sandbox_enabled && 'Web Retrieval Only'}
            {!settings.web_research_enabled && settings.python_sandbox_enabled && 'Python Execution Only'}
            {settings.web_research_enabled && settings.python_sandbox_enabled && 'Two-Tool Orchestrator'}
          </span>
        </div>

        <textarea
          className={styles.inputTextarea}
          placeholder="Ask a technical research question, enter a prompt, or run Python with ```python ... ```"
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={isGenerating || !selectedModel}
        />

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 'var(--space-2)' }}>
          {isGenerating ? (
            <button type="button" className={styles.stopBtn} onClick={handleStop}>
              <Square size={12} /> Stop Generation
            </button>
          ) : (
            <button
              type="button"
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!inputPrompt.trim() || !selectedModel}
            >
              <Send size={12} /> Generate Response
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
