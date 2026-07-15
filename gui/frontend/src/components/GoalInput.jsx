/**
 * GoalInput — goal entry bar with Run/Cancel controls and optional context.
 */

import React, { useState } from 'react';
import { useStore } from '../stores/useStore';

export default function GoalInput() {
  const goal = useStore((s) => s.goal);
  const setGoal = useStore((s) => s.setGoal);
  const status = useStore((s) => s.status);
  const startRun = useStore((s) => s.startRun);
  const cancelRun = useStore((s) => s.cancelRun);
  const [context, setContext] = useState('');
  const [showContext, setShowContext] = useState(false);

  const isActive = ['planning', 'executing', 'synthesizing'].includes(status);
  const isIdle = status === 'idle' || status === 'completed' || status === 'failed' || status === 'cancelled';

  const handleRun = async () => {
    if (!goal.trim()) return;
    await startRun(goal.trim(), context.trim() || undefined);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleRun();
    }
  };

  return (
    <div style={{
      padding: 'var(--sp-md)',
      borderBottom: '1px solid var(--border)',
      background: 'var(--bg-primary)',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <div style={{ flex: 1 }}>
          <textarea
            id="goal-input"
            className="input"
            placeholder="Enter your goal… (⌘Enter to run)"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            disabled={isActive}
            style={{
              resize: 'none',
              minHeight: 0,
              fontSize: 15,
              lineHeight: 1.5,
              background: 'var(--bg-elevated)',
            }}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {isIdle ? (
            <button
              id="run-btn"
              className="btn btn-primary"
              onClick={handleRun}
              disabled={!goal.trim()}
              style={{ minWidth: 80 }}
            >
              ▶ Run
            </button>
          ) : (
            <button
              id="cancel-run-btn-2"
              className="btn btn-danger"
              onClick={cancelRun}
            >
              ✕ Cancel
            </button>
          )}
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setShowContext((v) => !v)}
            title="Add context"
          >
            {showContext ? '∧ Context' : '+ Context'}
          </button>
        </div>
      </div>

      {showContext && (
        <textarea
          className="input animate-fade-in"
          placeholder="Optional context for the orchestrator…"
          value={context}
          onChange={(e) => setContext(e.target.value)}
          style={{ marginTop: 8, minHeight: 64 }}
          disabled={isActive}
        />
      )}
    </div>
  );
}
