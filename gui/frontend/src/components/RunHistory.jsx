/**
 * RunHistory — left sidebar listing past runs from SQLite.
 */

import React, { useEffect } from 'react';
import { useStore } from '../stores/useStore';

const STATUS_ICONS = {
  running: '◎',
  completed: '✓',
  failed: '✕',
  cancelled: '—',
};

function RunRow({ run, isActive }) {
  const loadRun = useStore((s) => s.loadRun);

  return (
    <div
      id={`run-history-${run.id}`}
      onClick={() => loadRun(run.id)}
      style={{
        padding: '8px 12px',
        borderRadius: 'var(--radius-sm)',
        cursor: 'pointer',
        border: `1px solid ${isActive ? 'var(--border-active)' : 'transparent'}`,
        background: isActive ? 'var(--accent-dim)' : 'transparent',
        transition: 'all var(--transition-fast)',
        marginBottom: 2,
      }}
      onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--bg-elevated)'; }}
      onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <span style={{
          fontSize: 11,
          color: run.status === 'completed' ? 'var(--success)'
               : run.status === 'failed'    ? 'var(--error)'
               : 'var(--text-muted)',
        }}>
          {STATUS_ICONS[run.status] || '○'}
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {run.goal}
        </span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', paddingLeft: 17 }}>
        {run.created_at ? new Date(run.created_at).toLocaleString() : ''}
      </div>
    </div>
  );
}

export default function RunHistory() {
  const runs = useStore((s) => s.runs);
  const fetchRuns = useStore((s) => s.fetchRuns);
  const currentRunId = useStore((s) => s.currentRunId);
  const status = useStore((s) => s.status);

  useEffect(() => {
    fetchRuns();
  }, []);

  // Refresh history when run completes
  useEffect(() => {
    if (['completed', 'failed', 'cancelled'].includes(status)) {
      fetchRuns();
    }
  }, [status]);

  return (
    <aside style={{
      width: 'var(--sidebar-w)',
      flexShrink: 0,
      background: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <div style={{ padding: '12px var(--sp-md)', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <h3>Run History</h3>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
        {runs.length === 0 ? (
          <div style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12, textAlign: 'center' }}>
            No runs yet
          </div>
        ) : (
          runs.map((r) => (
            <RunRow key={r.id} run={r} isActive={r.id === currentRunId} />
          ))
        )}
      </div>
    </aside>
  );
}
