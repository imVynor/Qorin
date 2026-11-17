import { getComputeToken } from './api';
import type { RetrievalTrace, ToolTrace } from '../types/harness';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export interface WsChatPayload {
  message: string;
  model_id: string;
  conversation_id?: string;
  request_id?: string;
  messages?: Array<{ role: string; content: string }>;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_new_tokens?: number;
  repetition_penalty?: number;
  system_prompt?: string;
  retrieval_mode?: string;
  retrieval_top_k?: number;
  retrieval_threshold?: number;
  web_research_enabled?: boolean;
  python_sandbox_enabled?: boolean;
  weather_enabled?: boolean;
  manual_tool_override?: string;
  compute_token?: string;
}

export interface WsEventMap {
  connection_ack: { type: 'connection_ack'; timestamp: number; active_model?: string; device?: string; version?: string };
  request_accepted: { type: 'request_accepted'; request_id: string; conversation_id: string; model_id: string; timestamp: number };
  model_loading: { type: 'model_loading'; model_id: string; timestamp: number };
  model_ready: { type: 'model_ready'; model_id: string; timestamp: number };
  retrieval: { type: 'retrieval'; request_id: string; conversation_id: string; trace: RetrievalTrace; timestamp: number };
  generation_started: { type: 'generation_started'; request_id: string; conversation_id: string; model_id: string; timestamp: number };
  token: { type: 'token'; request_id: string; conversation_id: string; text: string; token_count?: number; ttft_ms?: number; tok_per_sec?: number; timestamp: number };
  tool_call_started: { type: 'tool_call_started'; request_id: string; conversation_id: string; call_id: string; tool_name: string; timestamp: number };
  tool_call_arguments: { type: 'tool_call_arguments'; request_id: string; conversation_id: string; call_id: string; tool_name: string; arguments: Record<string, unknown>; timestamp: number };
  tool_execution_started: { type: 'tool_execution_started'; request_id: string; conversation_id: string; call_id: string; tool_name: string; timestamp: number };
  tool_execution_completed: { type: 'tool_execution_completed'; request_id: string; conversation_id: string; call_id: string; tool_name: string; trace: ToolTrace; timestamp: number };
  tool_execution_failed: { type: 'tool_execution_failed'; request_id: string; conversation_id: string; call_id: string; tool_name: string; trace: ToolTrace; timestamp: number };
  generation_completed: { type: 'generation_completed'; request_id: string; conversation_id: string; trace_id: string; total_tokens: number; ttft_ms: number; total_latency_ms: number; tok_per_sec: number; tools_executed: number; tool_traces: ToolTrace[]; timestamp: number };
  generation_cancelled: { type: 'generation_cancelled'; request_id: string; conversation_id?: string; timestamp: number };
  error: { type: 'error'; request_id?: string; conversation_id?: string; error: string; code?: string; timestamp: number };
  heartbeat: { type: 'heartbeat'; timestamp: number };
}

export type WsServerEvent = WsEventMap[keyof WsEventMap];

export type EventCallback<T = WsServerEvent> = (event: T) => void;
export type StatusCallback = (status: ConnectionStatus) => void;

export function getWebSocketChatUrl(): string {
  const configuredEnvUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;
  if (configuredEnvUrl) {
    const clean = configuredEnvUrl.replace(/\/$/, '').replace(/\/api$/, '');
    if (clean.startsWith('https://')) return clean.replace('https://', 'wss://') + '/ws/chat';
    if (clean.startsWith('http://')) return clean.replace('http://', 'ws://') + '/ws/chat';
    return clean + '/ws/chat';
  }
  if (import.meta.env.DEV) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/chat`;
  }
  // Default production Hugging Face Space WebSocket
  return 'wss://vynor-stt4.hf.space/ws/chat';
}

export class QorinWebSocketClient {
  private socket: WebSocket | null = null;
  private status: ConnectionStatus = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 6;
  private reconnectTimer: number | null = null;
  private heartbeatInterval: number | null = null;
  private explicitDisconnect = false;
  private statusListeners = new Set<StatusCallback>();
  private globalEventListeners = new Set<EventCallback>();
  private requestCallbacks = new Map<string, { onEvent: EventCallback; onError: (err: Error) => void }>();
  private urlGetter: () => string;

  constructor(urlGetter: () => string = getWebSocketChatUrl) {
    this.urlGetter = urlGetter;
  }

  public get currentStatus(): ConnectionStatus {
    return this.status;
  }

  public get isConnected(): boolean {
    return this.status === 'connected' && this.socket?.readyState === WebSocket.OPEN;
  }

  public connect(): void {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.explicitDisconnect = false;
    this.setStatus(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');

    try {
      const url = this.urlGetter();
      this.socket = new WebSocket(url);

      this.socket.onopen = () => {
        this.reconnectAttempts = 0;
        this.setStatus('connected');
        this.startHeartbeat();
      };

      this.socket.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data) as WsServerEvent;
          this.handleServerEvent(parsed);
        } catch (err) {
          console.warn('[QorinWS] Malformed message received:', event.data, err);
        }
      };

      this.socket.onerror = (err) => {
        console.warn('[QorinWS] WebSocket error encountered:', err);
        this.setStatus('error');
      };

      this.socket.onclose = () => {
        this.stopHeartbeat();
        this.socket = null;
        if (!this.explicitDisconnect) {
          this.scheduleReconnect();
        } else {
          this.setStatus('disconnected');
        }
      };
    } catch (err) {
      console.warn('[QorinWS] Connection construction failed:', err);
      this.setStatus('error');
      this.scheduleReconnect();
    }
  }

  public disconnect(): void {
    this.explicitDisconnect = true;
    this.clearReconnectTimer();
    this.stopHeartbeat();
    if (this.socket) {
      try {
        this.socket.close();
      } catch {
        // Ignore close errors
      }
      this.socket = null;
    }
    this.setStatus('disconnected');
  }

  public onStatusChange(callback: StatusCallback): () => void {
    this.statusListeners.add(callback);
    callback(this.status);
    return () => this.statusListeners.delete(callback);
  }

  public onEvent(callback: EventCallback): () => void {
    this.globalEventListeners.add(callback);
    return () => this.globalEventListeners.delete(callback);
  }

  public sendChat(
    payload: WsChatPayload,
    onEvent: EventCallback,
    onError: (err: Error) => void
  ): () => void {
    const requestId = payload.request_id || `req_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const conversationId = payload.conversation_id || `conv_${Date.now()}`;
    const token = payload.compute_token || getComputeToken();

    const requestPayload = {
      type: 'chat',
      request_id: requestId,
      conversation_id: conversationId,
      message: payload.message,
      model_id: payload.model_id,
      messages: payload.messages,
      temperature: payload.temperature,
      top_p: payload.top_p,
      top_k: payload.top_k,
      max_new_tokens: payload.max_new_tokens,
      repetition_penalty: payload.repetition_penalty,
      system_prompt: payload.system_prompt,
      retrieval_mode: payload.retrieval_mode,
      retrieval_top_k: payload.retrieval_top_k,
      retrieval_threshold: payload.retrieval_threshold,
      web_research_enabled: payload.web_research_enabled,
      python_sandbox_enabled: payload.python_sandbox_enabled,
      weather_enabled: payload.weather_enabled,
      manual_tool_override: payload.manual_tool_override,
      compute_token: token,
    };

    // Ensure connection
    if (!this.isConnected) {
      this.connect();
    }

    this.requestCallbacks.set(requestId, { onEvent, onError });

    const sendWhenReady = () => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify(requestPayload));
      } else {
        setTimeout(sendWhenReady, 100);
      }
    };

    sendWhenReady();

    // Return cancellation function
    return () => {
      this.cancelRequest(requestId, conversationId);
      this.requestCallbacks.delete(requestId);
    };
  }

  public cancelRequest(requestId: string, conversationId?: string): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(
        JSON.stringify({
          type: 'cancel',
          request_id: requestId,
          conversation_id: conversationId,
        })
      );
    }
  }

  public ping(): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
    }
  }

  private handleServerEvent(event: WsServerEvent): void {
    // Notify global event listeners
    this.globalEventListeners.forEach((listener) => {
      try {
        listener(event);
      } catch (err) {
        console.error('[QorinWS] Listener error:', err);
      }
    });

    // Check request-specific callbacks
    if ('request_id' in event && event.request_id) {
      const handler = this.requestCallbacks.get(event.request_id);
      if (handler) {
        if (event.type === 'error') {
          handler.onError(new Error(event.error));
          this.requestCallbacks.delete(event.request_id);
        } else {
          handler.onEvent(event);
          if (event.type === 'generation_completed' || event.type === 'generation_cancelled') {
            this.requestCallbacks.delete(event.request_id);
          }
        }
      }
    }
  }

  private setStatus(newStatus: ConnectionStatus): void {
    if (this.status !== newStatus) {
      this.status = newStatus;
      this.statusListeners.forEach((listener) => {
        try {
          listener(this.status);
        } catch (err) {
          console.error('[QorinWS] Status listener error:', err);
        }
      });
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.setStatus('error');
      return;
    }

    this.setStatus('reconnecting');
    this.clearReconnectTimer();

    // Exponential backoff: 1s, 2s, 4s, 8s, 15s max
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 15000) + Math.random() * 500;
    this.reconnectAttempts++;

    this.reconnectTimer = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = window.setInterval(() => {
      this.ping();
    }, 15000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval !== null) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

// Global singleton instance for shared connection across pages and tabs
export const globalWebSocketClient = new QorinWebSocketClient();
