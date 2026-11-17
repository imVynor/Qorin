import type {
  ModelInfo,
  TelemetryInfo,
  TrainingRunSummary,
  TrainingRunDetail,
  ExperimentTraceRecord,
  RetrievalTrace,
  ToolTrace,
} from '../types/harness';
import type { UsageSummaryResponse } from '../types/usage';

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

// Compute Authorization Token Management (Author's Personal Compute Protection)
const COMPUTE_TOKEN_STORAGE_KEY = 'qorin_compute_token';

export function getComputeToken(): string | null {
  try {
    return localStorage.getItem(COMPUTE_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setComputeToken(token: string): void {
  try {
    localStorage.setItem(COMPUTE_TOKEN_STORAGE_KEY, token);
    window.dispatchEvent(new Event('qorin-compute-auth-changed'));
  } catch {
    // Ignore storage errors in restricted contexts
  }
}

export function clearComputeToken(): void {
  try {
    localStorage.removeItem(COMPUTE_TOKEN_STORAGE_KEY);
    window.dispatchEvent(new Event('qorin-compute-auth-changed'));
  } catch {
    // Ignore
  }
}

export function isComputeAuthorized(): boolean {
  const token = getComputeToken();
  return Boolean(token && (token.toUpperCase() === 'UDIT BHATIA' || token.toUpperCase() === 'UDIT_BHATIA_AUTHORIZED'));
}

export async function verifyComputeAccess(password: string): Promise<{ success: boolean; token?: string; message: string }> {
  if (!isBackendConfigured || !API_BASE) {
    // In static mode, validate locally for demonstration
    const clean = password.trim().toUpperCase();
    if (clean === 'UDIT BHATIA') {
      setComputeToken('UDIT_BHATIA_AUTHORIZED');
      return {
        success: true,
        token: 'UDIT_BHATIA_AUTHORIZED',
        message: 'Compute access authorized: Welcome, Prof. Udit Bhatia! (Static Mode Simulation)',
      };
    }
    throw new Error("Invalid compute access password. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)");
  }

  const res = await fetch(`${API_BASE}/auth/verify-compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Invalid compute access password. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)");
  }

  const data = await res.json();
  if (data.token) {
    setComputeToken(data.token);
  }
  return data;
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
  const token = getComputeToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['X-Compute-Token'] = token;

  const res = await fetch(`${API_BASE}/models/select`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model_id: modelId, device, compute_token: token }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to select model');
  }
  return res.json();
}

export async function unloadModel(): Promise<{ success: boolean; telemetry: TelemetryInfo }> {
  const token = getComputeToken();
  const headers: Record<string, string> = {};
  if (token) headers['X-Compute-Token'] = token;

  const res = await fetch(`${API_BASE}/models/unload`, { method: 'POST', headers });
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
      const token = getComputeToken();
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (token) headers['X-Compute-Token'] = token;

      const outgoingPayload = {
        ...payload,
        compute_token: payload.compute_token || token,
      };

      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers,
        body: JSON.stringify(outgoingPayload),
        signal,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errJson.detail || `Chat stream request failed: ${res.status} ${res.statusText}`);
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
  const token = getComputeToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['X-Compute-Token'] = token;

  const res = await fetch(`${API_BASE}/retrieval/compare`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ model_id: modelId, prompt, compute_token: token }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to execute retrieval comparison');
  }
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

export const FALLBACK_USAGE_REPORT: UsageSummaryResponse = {
  providers: [
    {
      provider_id: 'huggingface',
      provider_name: 'Hugging Face',
      status: 'not_connected',
      account_name: null,
      plan_tier: 'Static Showcase Mode',
      is_live: false,
      is_cached: true,
      last_synchronized: null,
      inference_credits: {
        available_via_api: false,
        status: 'backend_offline',
        remaining: null,
        used: null,
        currency: 'USD',
        billing_period: 'Static Mode',
        limitation_notice: 'The research showcase is running in static mode. Start the local backend with HF_TOKEN to query live provider status.',
        billing_dashboard_url: 'https://huggingface.co/settings/billing',
      },
      other_resources: [
        {
          category_id: 'inference_providers',
          name: 'Serverless Inference Providers',
          status: 'unavailable_offline',
          note: 'Requires local backend connection.',
          dashboard_url: 'https://huggingface.co/settings/billing',
        },
        {
          category_id: 'hub_storage',
          name: 'Model Hub Storage & LFS',
          status: 'available',
          note: 'Public model weights & checkpoints hosted on Hugging Face.',
          dashboard_url: 'https://huggingface.co/settings/billing',
        },
        {
          category_id: 'spaces_hardware',
          name: 'Spaces Compute & ZeroGPU',
          status: 'available',
          note: 'Hardware grants tied to active Hugging Face Spaces instances.',
          dashboard_url: 'https://huggingface.co/settings/billing',
        },
      ],
      dashboard_url: 'https://huggingface.co/settings/billing',
      documentation_url: 'https://huggingface.co/docs/hub/billing',
      error_message: 'Inference backend is offline. Live usage accounting requires running the backend server with HF_TOKEN configured.',
    },
  ],
  timestamp: new Date().toISOString(),
  backend_mode: 'static_showcase',
};

export async function getProviderUsage(forceRefresh = false): Promise<UsageSummaryResponse> {
  if (!isBackendConfigured || !API_BASE) {
    return FALLBACK_USAGE_REPORT;
  }
  try {
    const endpoint = forceRefresh ? `${API_BASE}/usage/refresh` : `${API_BASE}/usage`;
    const res = await fetch(endpoint, {
      method: forceRefresh ? 'POST' : 'GET',
    });
    if (!res.ok) throw new Error(`Usage query failed: ${res.statusText}`);
    return await res.json();
  } catch (err) {
    console.warn('Backend usage endpoint unreachable, using static fallback:', err);
    return FALLBACK_USAGE_REPORT;
  }
}

