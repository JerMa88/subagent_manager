/**
 * StatusBar — top bar showing connection status, model, run status, and global controls.
 */

import React from 'react';
import { useStore } from '../stores/useStore';

const STATUS_LABELS = {
  idle: 'Idle',
  planning: 'Planning…',
  plan_review: 'Plan Ready',
  executing: 'Executing',
  synthesizing: 'Synthesizing…',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export default function StatusBar() {
  const status = useStore((s) => s.status);
  const model = useStore((s) => s.config.model);
  const wsConnected = useStore((s) => s.wsConnected);
  const currentRunId = useStore((s) => s.currentRunId);
  const subtasks = useStore((s) => s.subtasks);
  const cancelRun = useStore((s) => s.cancelRun);
  const toggleConfig = useStore((s) => s.toggleConfigPanel);

  const totalTokens = Object.values(subtasks).reduce((a, s) => a + (s.tokens || 0), 0);
  const totalTools = Object.values(subtasks).reduce((a, s) => a + (s.toolCallsMade || 0), 0);
  const isActive = ['planning', 'executing', 'synthesizing'].includes(status);

  return (
    <header style={{
      height: 'var(--statusbar-h)',
      background: 'var(--bg-secondary)',
      borderBottom: '1px solid var(--border)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 var(--sp-md)',
      gap: 'var(--sp-md)',
      flexShrink: 0,
      zIndex: 100,
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 8 }}>
        <div style={{
          width: 28, height: 28, borderRadius: 8,
          background: 'linear-gradient(135deg, var(--accent), #a855f7)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 14, fontWeight: 700, color: '#fff',
          boxShadow: 'var(--glow-accent)',
        }}>
          S
        </div>
        <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-0.2px' }}>
          SubAgent
        </span>
      </div>

      {/* Separator */}
      <div style={{ width: 1, height: 20, background: 'var(--border)' }} />

      {/* Status badge */}
      <span className={`badge badge-${status}`}>
        {isActive && <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} />}
        {STATUS_LABELS[status] || status}
      </span>

      {/* Model */}
      <span style={{ color: 'var(--text-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
        {model}
      </span>

      {/* Stats */}
      {currentRunId && (
        <>
          <div style={{ width: 1, height: 20, background: 'var(--border)' }} />
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {totalTokens.toLocaleString()} tok · {totalTools} calls
          </span>
        </>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* WS indicator */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <div style={{
          width: 7, height: 7, borderRadius: '50%',
          background: wsConnected ? 'var(--success)' : 'var(--muted)',
          boxShadow: wsConnected ? '0 0 6px var(--success)' : 'none',
          transition: 'all 0.3s',
        }} />
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {wsConnected ? 'Live' : 'Offline'}
        </span>
      </div>

      {/* New Run Button */}
      <button
        className="btn btn-primary btn-sm"
        onClick={useStore((s) => s.startNewRun)}
        title="Start New Run"
      >
        + New
      </button>

      {/* Cancel button */}
      {isActive && (
        <button
          id="cancel-run-btn"
          className="btn btn-danger btn-sm"
          onClick={cancelRun}
        >
          ✕ Cancel
        </button>
      )}

      {/* Config button */}
      <button
        id="open-config-btn"
        className="btn btn-ghost btn-sm"
        onClick={toggleConfig}
        title="Settings"
      >
        ⚙
      </button>
    </header>
  );
}
