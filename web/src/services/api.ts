import type {
  ModelInfo,
  TelemetryInfo,
  TrainingRunSummary,
  TrainingRunDetail,
  ExperimentTraceRecord,
  RetrievalTrace,
  ToolTrace,
} from '../types/harness';

// Configurable API base URL via Vite environment variables
// In development, defaults to '/api' which Vite proxies to http://127.0.0.1:8000
// In production builds (e.g. GitHub Pages), NEVER silently fall back to '/api' when no backend URL is configured
const configuredEnvUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;

export const isBackendConfigured: boolean = Boolean(
  configuredEnvUrl || (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE)
);

export const API_BASE: string = configuredEnvUrl
  ? configuredEnvUrl.replace(/\/$/, '')
  : (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE ? '/api' : '');

export function getApiBaseUrl(): string {
  return API_BASE;
}

export async function getHealth(): Promise<{ status: string; telemetry: TelemetryInfo }> {
  if (!isBackendConfigured || !API_BASE) {
    return {
      status: 'offline',
      telemetry: {
        active_model_id: null,
        active_provider: null,
        qorin: {
          active_model_id: null,
          device: 'client-static',
          is_loaded: false,
          cuda_available: false,
        },
        ollama: {
          is_reachable: false,
          base_url: '',
          active_model_id: null,
        },
      },
    };
  }
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function getModels(): Promise<ModelInfo[]> {
  if (!isBackendConfigured || !API_BASE) {
    return [];
  }
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to list models: ${res.statusText}`);
  return res.json();
}

export async function selectModel(modelId: string, device?: string): Promise<{ success: boolean; telemetry: TelemetryInfo }> {
  const res = await fetch(`${API_BASE}/models/select`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, device }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to select model');
  }
  return res.json();
}

export async function unloadModel(): Promise<{ success: boolean; telemetry: TelemetryInfo }> {
  const res = await fetch(`${API_BASE}/models/unload`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to unload model');
  return res.json();
}

export interface StreamCallbacks {
  onRetrieval?: (trace: RetrievalTrace) => void;
  onToolStart?: (toolData: any) => void;
  onToolResult?: (trace: ToolTrace) => void;
  onToken?: (tokenData: { text: string; is_finished: boolean; ttft_ms?: number; tok_per_sec?: number; token_count?: number; elapsed_ms?: number }) => void;
  onDone?: (doneData: any) => void;
  onError?: (err: Error) => void;
}

export function streamChat(
  payload: any,
  callbacks: StreamCallbacks,
  abortSignal?: AbortSignal
): () => void {
  if (!isBackendConfigured || !API_BASE) {
    setTimeout(() => {
      callbacks.onError?.(
        new Error(
          'Inference backend is not configured in this deployment. The research website is operating in Static Showcase Mode.'
        )
      );
    }, 0);
    return () => {};
  }

  const controller = new AbortController();
  const signal = abortSignal || controller.signal;

  (async () => {
    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      });

      if (!res.ok) {
        throw new Error(`Chat stream request failed: ${res.status} ${res.statusText}`);
      }

      if (!res.body) {
        throw new Error('Response body is missing.');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = 'message';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) {
            currentEvent = 'message';
            continue;
          }

          if (trimmed.startsWith('event:')) {
            currentEvent = trimmed.replace('event:', '').trim();
          } else if (trimmed.startsWith('data:')) {
            const dataStr = trimmed.replace('data:', '').trim();
            if (!dataStr) continue;

            try {
              const parsed = JSON.parse(dataStr);
              if (currentEvent === 'retrieval' && callbacks.onRetrieval) {
                callbacks.onRetrieval(parsed);
              } else if (currentEvent === 'tool_start' && callbacks.onToolStart) {
                callbacks.onToolStart(parsed);
              } else if (currentEvent === 'tool_result' && callbacks.onToolResult) {
                callbacks.onToolResult(parsed);
              } else if (currentEvent === 'token' && callbacks.onToken) {
                callbacks.onToken(parsed);
              } else if (currentEvent === 'done' && callbacks.onDone) {
                callbacks.onDone(parsed);
              } else if (currentEvent === 'error') {
                if (callbacks.onError) callbacks.onError(new Error(parsed.error || 'Server error'));
              }
            } catch (parseErr) {
              console.warn('Failed to parse SSE event payload:', dataStr);
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError' && callbacks.onError) {
        callbacks.onError(err);
      }
    }
  })();

  return () => controller.abort();
}

export async function getTrainingRuns(): Promise<TrainingRunSummary[]> {
  if (!isBackendConfigured || !API_BASE) return [];
  const res = await fetch(`${API_BASE}/training/runs`);
  if (!res.ok) throw new Error('Failed to load training runs');
  return res.json();
}

export async function getTrainingRunDetail(runId: string): Promise<TrainingRunDetail> {
  if (!isBackendConfigured || !API_BASE) throw new Error('Backend not configured (Static Showcase Mode)');
  const res = await fetch(`${API_BASE}/training/runs/${runId}`);
  if (!res.ok) throw new Error(`Failed to load run detail for ${runId}`);
  return res.json();
}

export async function compareRetrieval(modelId: string, prompt: string): Promise<any> {
  if (!isBackendConfigured || !API_BASE) throw new Error('Backend not configured (Static Showcase Mode)');
  const res = await fetch(`${API_BASE}/retrieval/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, prompt }),
  });
  if (!res.ok) throw new Error('Failed to execute retrieval comparison');
  return res.json();
}

export async function getExperiments(limit = 50): Promise<ExperimentTraceRecord[]> {
  if (!isBackendConfigured || !API_BASE) return [];
  const res = await fetch(`${API_BASE}/experiments?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to load experiments');
  return res.json();
}

export function getExportUrl(format: 'json' | 'jsonl'): string {
  if (!isBackendConfigured || !API_BASE) return '#';
  return `${API_BASE}/experiments/export?format=${format}`;
}
