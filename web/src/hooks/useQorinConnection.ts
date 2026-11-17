import { useState, useEffect, useCallback } from 'react';
import { connectionManager, type QorinConnectionState } from '../services/connectionManager';

export function useQorinConnection() {
  const [state, setState] = useState<QorinConnectionState>(connectionManager.getState());

  useEffect(() => {
    const unsubscribe = connectionManager.subscribe((newState) => {
      setState(newState);
    });
    return unsubscribe;
  }, []);

  const refreshHealth = useCallback(() => {
    return connectionManager.checkHealth();
  }, []);

  const reconnectWebSocket = useCallback(() => {
    connectionManager.reconnectWebSocket();
  }, []);

  return {
    ...state,
    refreshHealth,
    reconnectWebSocket,
  };
}
