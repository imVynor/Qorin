import { useState, useRef, useCallback, useEffect } from 'react';
import type {
  ChatMessage,
  GenerationSettings,
  ModelInfo,
  RetrievalTrace,
  ToolTrace,
} from '../types/harness';
import {
  globalWebSocketClient,
  type WsServerEvent,
  type WsChatPayload,
} from '../services/websocketClient';
import { getComputeToken } from '../services/api';

export interface UseStreamingChatOptions {
  onRequireComputeAuth?: () => void;
  onTraceRecorded?: () => void;
  onActivePromptChanged?: (prompt: string) => void;
}

export function useStreamingChat(options: UseStreamingChatOptions = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveStreamText, setLiveStreamText] = useState('');
  const [liveStats, setLiveStats] = useState<{
    ttft_ms?: number;
    tok_per_sec?: number;
    token_count?: number;
  }>({});
  const [activeRetrievalTrace, setActiveRetrievalTrace] = useState<RetrievalTrace | null>(null);
  const [activeToolTraces, setActiveToolTraces] = useState<ToolTrace[]>([]);
  const [modelLoadingNotice, setModelLoadingNotice] = useState<string | null>(null);

  const conversationIdRef = useRef<string>('');
  useEffect(() => {
    if (!conversationIdRef.current) {
      conversationIdRef.current = `conv-${Date.now()}`;
    }
  }, []);
  const cancelFnRef = useRef<(() => void) | null>(null);
  const activeRequestIdRef = useRef<string | null>(null);

  const cancelGeneration = useCallback(() => {
    if (cancelFnRef.current) {
      cancelFnRef.current();
      cancelFnRef.current = null;
    }
    if (activeRequestIdRef.current) {
      globalWebSocketClient.cancelRequest(activeRequestIdRef.current, conversationIdRef.current);
      activeRequestIdRef.current = null;
    }
    setIsGenerating(false);
    setModelLoadingNotice(null);
  }, []);

  const clearConversation = useCallback(() => {
    cancelGeneration();
    conversationIdRef.current = `conv-${Date.now()}`;
    setMessages([]);
    setLiveStreamText('');
    setLiveStats({});
    setActiveRetrievalTrace(null);
    setActiveToolTraces([]);
    setModelLoadingNotice(null);
  }, [cancelGeneration]);

  const sendMessage = useCallback(
    (
      prompt: string,
      model: ModelInfo | null,
      settings: GenerationSettings,
      isBackendAvailable: boolean
    ) => {
      const trimmed = prompt.trim();
      if (!trimmed || isGenerating) return;

      if (!isBackendAvailable) {
        const userMsg: ChatMessage = {
          id: `msg-${Date.now()}`,
          role: 'user',
          content: trimmed,
        };
        const offlineMsg: ChatMessage = {
          id: `msg-offline-${Date.now() + 1}`,
          role: 'assistant',
          content: `[Backend Inference Service Offline]\n\nLive WebSocket token streaming and tool dispatch require the local Python FastAPI server (http://127.0.0.1:8000) or authorized remote deployment.\n\nThe research monograph, architecture specifications, interactive learning lab, and experiment archives remain fully accessible.\n\nTo run live model generation locally:\n1. Start the API backend: .venv\\Scripts\\python -m uvicorn backend.app:app --port 8000\n2. Click "Sync" in the top telemetry strip to connect.`,
        };
        setMessages((prev) => [...prev, userMsg, offlineMsg]);
        return;
      }

      if (!model) return;

      const userMsg: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: 'user',
        content: trimmed,
      };

      const newHistory = [...messages, userMsg];
      setMessages(newHistory);
      setIsGenerating(true);
      setLiveStreamText('');
      setLiveStats({});
      setActiveRetrievalTrace(null);
      setActiveToolTraces([]);
      setModelLoadingNotice(null);

      if (options.onActivePromptChanged) {
        options.onActivePromptChanged(trimmed);
      }

      const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      activeRequestIdRef.current = requestId;

      const payload: WsChatPayload = {
        request_id: requestId,
        conversation_id: conversationIdRef.current,
        message: trimmed,
        model_id: model.id,
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
        weather_enabled: true,
        compute_token: getComputeToken() || undefined,
      };

      let accumulatedText = '';
      let currentRetrieval: RetrievalTrace | null = null;
      let currentTools: ToolTrace[] = [];

      const handleEvent = (event: WsServerEvent) => {
        switch (event.type) {
          case 'model_loading':
            setModelLoadingNotice(`Loading weights for ${event.model_id}...`);
            break;

          case 'model_ready':
            setModelLoadingNotice(null);
            break;

          case 'generation_started':
            setModelLoadingNotice(null);
            break;

          case 'retrieval':
            currentRetrieval = event.trace;
            setActiveRetrievalTrace(event.trace);
            break;

          case 'token':
            accumulatedText += event.text;
            setLiveStreamText(accumulatedText);
            setLiveStats({
              ttft_ms: event.ttft_ms,
              tok_per_sec: event.tok_per_sec,
              token_count: event.token_count,
            });
            break;

          case 'tool_call_started': {
            const newToolStub: ToolTrace = {
              tool_name: event.tool_name,
              call_id: event.call_id,
              status: 'started',
              requested_by: 'model',
              input_payload: {},
              output_payload: {},
              latency_ms: 0,
            };
            currentTools = [...currentTools, newToolStub];
            setActiveToolTraces(currentTools);
            break;
          }

          case 'tool_call_arguments': {
            currentTools = currentTools.map((t) =>
              t.call_id === event.call_id ? { ...t, input_payload: event.arguments } : t
            );
            setActiveToolTraces(currentTools);
            break;
          }

          case 'tool_execution_started': {
            currentTools = currentTools.map((t) =>
              t.call_id === event.call_id ? { ...t, status: 'started' } : t
            );
            setActiveToolTraces(currentTools);
            break;
          }

          case 'tool_execution_completed':
          case 'tool_execution_failed': {
            const completedTrace = event.trace;
            const existingIndex = currentTools.findIndex((t) => t.call_id === event.call_id);
            if (existingIndex >= 0) {
              currentTools = currentTools.map((t, idx) => (idx === existingIndex ? completedTrace : t));
            } else {
              currentTools = [...currentTools, completedTrace];
            }
            setActiveToolTraces(currentTools);
            break;
          }

          case 'generation_completed': {
            const assistantMsg: ChatMessage = {
              id: `msg-${Date.now() + 1}`,
              role: 'assistant',
              content: accumulatedText,
              retrieval_trace: currentRetrieval || undefined,
              tool_traces: event.tool_traces && event.tool_traces.length > 0 ? event.tool_traces : (currentTools.length > 0 ? currentTools : undefined),
              ttft_ms: event.ttft_ms,
              tok_per_sec: event.tok_per_sec,
              latency_ms: event.total_latency_ms,
              token_count: event.total_tokens,
            };

            setMessages((prev) => [...prev, assistantMsg]);
            setIsGenerating(false);
            setLiveStreamText('');
            setModelLoadingNotice(null);
            cancelFnRef.current = null;
            activeRequestIdRef.current = null;

            if (options.onTraceRecorded) {
              options.onTraceRecorded();
            }
            break;
          }

          case 'generation_cancelled': {
            if (accumulatedText.trim().length > 0) {
              const partialMsg: ChatMessage = {
                id: `msg-${Date.now() + 1}`,
                role: 'assistant',
                content: accumulatedText + ' [Generation stopped by user]',
                retrieval_trace: currentRetrieval || undefined,
                tool_traces: currentTools.length > 0 ? currentTools : undefined,
              };
              setMessages((prev) => [...prev, partialMsg]);
            }
            setIsGenerating(false);
            setLiveStreamText('');
            setModelLoadingNotice(null);
            cancelFnRef.current = null;
            activeRequestIdRef.current = null;
            break;
          }

          case 'error': {
            handleError(new Error(event.error));
            break;
          }

          default:
            break;
        }
      };

      const handleError = (err: Error) => {
        const isAuthError =
          err.message &&
          (err.message.includes('compute access protected') ||
            err.message.includes('AUTH_REQUIRED') ||
            err.message.includes('401') ||
            err.message.includes('ALL CAPS'));

        const isTimeout =
          Boolean(err.message && (
            err.message.includes('timeout') ||
            err.message.includes('DEADLINE_EXCEEDED') ||
            err.message.includes('timed out')
          ));

        const errorMsg: ChatMessage = {
          id: `msg-err-${Date.now()}`,
          role: 'assistant',
          is_error: true,
          is_timeout: isTimeout,
          retry_prompt: trimmed,
          content: isAuthError
            ? `[Protected Compute Locked]\n\nAccess to live model inference is restricted to protect author compute resources.\n\nPlease activate compute access via the Options menu (Unlock Compute) to run live queries.`
            : isTimeout
            ? `[Execution Timeout]\n\n${err.message || 'Global execution deadline exceeded.'}\n\nYou can click Retry below to reload this prompt or refine settings.`
            : `[Inference Error: ${err.message}]`,
        };

        setMessages((prev) => [...prev, errorMsg]);
        setIsGenerating(false);
        setLiveStreamText('');
        setModelLoadingNotice(null);
        cancelFnRef.current = null;
        activeRequestIdRef.current = null;

        if (isAuthError && options.onRequireComputeAuth) {
          options.onRequireComputeAuth();
        }
      };

      const cancelFn = globalWebSocketClient.sendChat(payload, handleEvent, handleError);
      cancelFnRef.current = cancelFn;
    },
    [isGenerating, messages, options]
  );

  return {
    messages,
    setMessages,
    isGenerating,
    liveStreamText,
    liveStats,
    activeRetrievalTrace,
    activeToolTraces,
    modelLoadingNotice,
    sendMessage,
    cancelGeneration,
    clearConversation,
  };
}
