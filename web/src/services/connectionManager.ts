import { getHealth, isBackendConfigured, isComputeAuthorized } from './api';
import { globalWebSocketClient, type ConnectionStatus } from './websocketClient';
import type { TelemetryInfo } from '../types/harness';

export interface QorinConnectionState {
  isBackendAvailable: boolean;
  overallStatus: 'online' | 'offline' | 'waking' | 'checking';
  wsStatus: ConnectionStatus;
  telemetry: TelemetryInfo | null;
  isComputeAuthorized: boolean;
  activeModelId: string | null;
  lastChecked: number;
  wakeAttempts: number;
}

export type ConnectionStateListener = (state: QorinConnectionState) => void;

class ConnectionManager {
  private state: QorinConnectionState = {
    isBackendAvailable: false,
    overallStatus: 'checking',
    wsStatus: 'disconnected',
    telemetry: null,
    isComputeAuthorized: false,
    activeModelId: null,
    lastChecked: 0,
    wakeAttempts: 0,
  };

  private listeners = new Set<ConnectionStateListener>();
  private checkIntervalTimer: number | null = null;
  private isChecking = false;

  constructor() {
    this.state.isComputeAuthorized = isComputeAuthorized();

    // Listen to WS status
    globalWebSocketClient.onStatusChange((wsStatus) => {
      this.updateState({ wsStatus });
    });

    // Listen to WS server events for active model synchronization
    globalWebSocketClient.onEvent((event) => {
      if (event.type === 'connection_ack' && event.active_model) {
        this.updateState({ activeModelId: event.active_model });
      } else if (event.type === 'model_ready' && event.model_id) {
        this.updateState({ activeModelId: event.model_id });
      }
    });

    // Listen for window compute authorization changes
    if (typeof window !== 'undefined') {
      window.addEventListener('qorin-compute-auth-changed', () => {
        this.updateState({ isComputeAuthorized: isComputeAuthorized() });
      });
    }

    // Auto-initialize connection
    this.initialize();
  }

  public getState(): QorinConnectionState {
    return { ...this.state };
  }

  public subscribe(listener: ConnectionStateListener): () => void {
    this.listeners.add(listener);
    listener(this.getState());
    return () => this.listeners.delete(listener);
  }

  public async checkHealth(): Promise<QorinConnectionState> {
    if (this.isChecking) return this.state;
    this.isChecking = true;

    try {
      if (!isBackendConfigured) {
        this.updateState({
          isBackendAvailable: false,
          overallStatus: 'offline',
          lastChecked: Date.now(),
        });
        return this.state;
      }

      const health = await getHealth();
      const isHealthy = health.status === 'healthy';

      if (isHealthy) {
        this.updateState({
          isBackendAvailable: true,
          overallStatus: 'online',
          telemetry: health.telemetry,
          activeModelId: health.telemetry.active_model_id,
          lastChecked: Date.now(),
          wakeAttempts: 0,
        });

        // If backend is online, ensure WebSocket connects
        if (!globalWebSocketClient.isConnected) {
          globalWebSocketClient.connect();
        }
      } else {
        const isWaking = this.state.wakeAttempts < 4;
        this.updateState({
          isBackendAvailable: false,
          overallStatus: isWaking ? 'waking' : 'offline',
          lastChecked: Date.now(),
          wakeAttempts: isWaking ? this.state.wakeAttempts + 1 : this.state.wakeAttempts,
        });
      }
    } catch {
      const isWaking = this.state.wakeAttempts < 4;
      this.updateState({
        isBackendAvailable: false,
        overallStatus: isWaking ? 'waking' : 'offline',
        lastChecked: Date.now(),
        wakeAttempts: isWaking ? this.state.wakeAttempts + 1 : this.state.wakeAttempts,
      });
    } finally {
      this.isChecking = false;
    }

    return this.state;
  }

  public reconnectWebSocket(): void {
    globalWebSocketClient.disconnect();
    globalWebSocketClient.connect();
  }

  private initialize(): void {
    this.checkHealth();

    // Check periodically: every 30s
    if (typeof window !== 'undefined') {
      this.checkIntervalTimer = window.setInterval(() => {
        this.checkHealth();
      }, 30000);
    }
  }

  private updateState(partial: Partial<QorinConnectionState>): void {
    this.state = { ...this.state, ...partial };
    this.listeners.forEach((listener) => {
      try {
        listener(this.state);
      } catch (err) {
        console.error('[ConnectionManager] Listener error:', err);
      }
    });
  }

  public destroy(): void {
    if (this.checkIntervalTimer !== null) {
      clearInterval(this.checkIntervalTimer);
      this.checkIntervalTimer = null;
    }
    this.listeners.clear();
  }
}

export const connectionManager = new ConnectionManager();
