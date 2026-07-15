/**
 * AgentPanel — right-side inspection panel for the selected subtask.
 *
 * Shows: agent name/status header, tabs for Output / Tool Calls / Context.
 * Controls: Pause / Resume / Cancel subtask.
 * Context injection textarea.
 */

import React, { useState } from 'react';
import { useStore } from '../stores/useStore';
import ToolCallLog from './ToolCallLog';
import ContextViewer from './ContextViewer';

export default function AgentPanel() {
  const selectedId = useStore((s) => s.selectedSubtaskId);
  const subtasks = useStore((s) => s.subtasks);
  const setSelected = useStore((s) => s.setSelectedSubtask);
  const pauseSubtask = useStore((s) => s.pauseSubtask);
  const resumeSubtask = useStore((s) => s.resumeSubtask);
  const injectContext = useStore((s) => s.injectContext);
  const cancelRun = useStore((s) => s.cancelRun);

  const [tab, setTab] = useState('output');
  const [injecting, setInjecting] = useState(false);
  const [injectText, setInjectText] = useState('');

  const subtask = selectedId != null ? subtasks[selectedId] : null;

  if (!subtask) {
    return (
      <aside style={{
        width: 340,
        flexShrink: 0,
        background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'var(--text-muted)',
        fontSize: 13,
        flexDirection: 'column',
        gap: 12,
        userSelect: 'none',
      }}>
        <div style={{ fontSize: 32, opacity: 0.3 }}>◈</div>
        <span>Click a node to inspect</span>
      </aside>
    );
  }

  const { agent_name, status, task, answer, toolCalls, messages, sources, tokens, error } = subtask;
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isActive = isRunning || isPaused;

  const AGENT_COLORS = {
    researcher: 'var(--color-researcher)',
    analyzer:   'var(--color-analyzer)',
    coder:      'var(--color-coder)',
    verifier:   'var(--color-verifier)',
  };
  const agentColor = AGENT_COLORS[agent_name] || 'var(--color-default)';

  const handleInject = async () => {
    if (!injectText.trim()) return;
    await injectContext(selectedId, injectText.trim());
    setInjectText('');
    setInjecting(false);
  };

  return (
    <aside
      id="agent-panel"
      className="animate-slide-right"
      style={{
        width: 340,
        flexShrink: 0,
        background: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        padding: '12px var(--sp-md)',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 10, height: 10, borderRadius: '50%',
            background: agentColor, flexShrink: 0,
            boxShadow: isRunning ? `0 0 8px ${agentColor}` : 'none',
          }} />
          <span style={{ fontWeight: 700, fontSize: 14, color: agentColor }}>
            {agent_name || 'agent'}
          </span>
          <span className={`badge badge-${status}`} style={{ marginLeft: 'auto' }}>
            {isRunning && <span className="spinner" style={{ width: 8, height: 8, borderWidth: 1.5 }} />}
            {status}
          </span>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setSelected(null)}
            style={{ padding: '2px 6px', fontSize: 14 }}
          >
            ✕
          </button>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.4 }}>
          #{selectedId} · {tokens > 0 ? `${tokens.toLocaleString()} tokens` : 'no tokens yet'}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          {task}
        </div>

        {/* Controls */}
        {isActive && (
          <div style={{ display: 'flex', gap: 6 }}>
            {isRunning && (
              <button className="btn btn-warning btn-sm" onClick={() => pauseSubtask(selectedId)}>
                ⏸ Pause
              </button>
            )}
            {isPaused && (
              <button className="btn btn-success btn-sm" onClick={() => resumeSubtask(selectedId)}>
                ▶ Resume
              </button>
            )}
            <button className="btn btn-ghost btn-sm" onClick={() => setInjecting(!injecting)}>
              + Inject Context
            </button>
            <button className="btn btn-danger btn-sm" onClick={cancelRun}>
              ✕
            </button>
          </div>
        )}
      </div>

      {/* Context injection */}
      {injecting && (
        <div className="animate-fade-in" style={{ padding: 10, borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}>
          <textarea
            className="input"
            placeholder="Additional context to inject…"
            value={injectText}
            onChange={(e) => setInjectText(e.target.value)}
            rows={3}
            style={{ fontSize: 12 }}
          />
          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
            <button className="btn btn-primary btn-sm" onClick={handleInject}>Inject</button>
            <button className="btn btn-ghost btn-sm" onClick={() => setInjecting(false)}>Cancel</button>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs" style={{ flexShrink: 0 }}>
        {[
          { key: 'output', label: 'Output' },
          { key: 'tools', label: `Tools (${toolCalls?.length || 0})` },
          { key: 'context', label: 'Context' },
        ].map(({ key, label }) => (
          <button
            key={key}
            id={`agent-tab-${key}`}
            className={`tab ${tab === key ? 'active' : ''}`}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 'output' && (
          <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--sp-md)' }}>
            {error && (
              <div style={{ background: 'var(--error-dim)', border: '1px solid var(--border-error)', borderRadius: 'var(--radius-sm)', padding: 10, marginBottom: 12, fontSize: 12, color: 'var(--error)' }}>
                ✕ {error}
              </div>
            )}
            {answer ? (
              <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {answer}
              </div>
            ) : isRunning ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: 'var(--text-muted)', fontSize: 13 }}>
                <span className="spinner" style={{ color: 'var(--accent)' }} />
                Running…
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No output yet.</div>
            )}
          </div>
        )}
        {tab === 'tools' && (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <ToolCallLog toolCalls={toolCalls || []} />
          </div>
        )}
        {tab === 'context' && (
          <div style={{ flex: 1, overflowY: 'auto' }}>
            <ContextViewer messages={messages || []} sources={sources || []} />
          </div>
        )}
      </div>
    </aside>
  );
}
