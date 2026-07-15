/**
 * WebSocket hook — manages connection lifecycle for a run.
 *
 * Connects when runId is set, reconnects on disconnect,
 * dispatches all events to the Zustand store via handleEvent().
 */

import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '../stores/useStore';

const WS_URL = 'ws://localhost:8000';

export function useWebSocket() {
  const runId = useStore((s) => s.currentRunId);
  const handleEvent = useStore((s) => s.handleEvent);
  const setWsConnected = useStore((s) => s.setWsConnected);
  const status = useStore((s) => s.status);

  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const connect = useCallback(() => {
    if (!runId) return;
    // Don't connect again if already completed/failed/cancelled
    if (['completed', 'failed', 'cancelled'].includes(status)) return;

    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_URL}/ws/${runId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      console.log(`[WS] Connected to run ${runId}`);
    };

    ws.onmessage = (evt) => {
      try {
        const event = JSON.parse(evt.data);
        handleEvent(event);

        // If the run terminated, close the WS gracefully
        if (['orchestration_completed', 'orchestration_failed', 'orchestration_cancelled']
          .includes(event.type)) {
          setTimeout(() => ws.close(), 1000);
        }
      } catch (e) {
        console.warn('[WS] Parse error:', e);
      }
    };

    ws.onclose = () => {
      setWsConnected(false);
      console.log(`[WS] Disconnected from run ${runId}`);
      // Auto-reconnect if run is still active
      const currentStatus = useStore.getState().status;
      if (!['completed', 'failed', 'cancelled', 'idle'].includes(currentStatus)) {
        reconnectTimerRef.current = setTimeout(connect, 2000);
      }
    };

    ws.onerror = (err) => {
      console.warn('[WS] Error:', err);
    };
  }, [runId, handleEvent, setWsConnected]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [runId]);

  // Expose send for control commands
  const send = useCallback((cmd) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  return { send };
}
