/**
 * ToolCallLog — scrollable log of tool calls for a subtask.
 *
 * Each row shows: tool name, arguments (truncated), result snippet, timing.
 * Expandable for full content.
 */

import React, { useState } from 'react';

function ToolRow({ tc }) {
  const [open, setOpen] = useState(false);

  let argsStr = '';
  try {
    argsStr = typeof tc.arguments === 'string'
      ? tc.arguments
      : JSON.stringify(tc.arguments, null, 2);
  } catch { argsStr = String(tc.arguments); }

  return (
    <div
      onClick={() => setOpen(!open)}
      style={{
        background: open ? 'var(--bg-elevated)' : 'transparent',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '8px 10px',
        cursor: 'pointer',
        transition: 'all var(--transition-fast)',
        marginBottom: 4,
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--info)',
          fontWeight: 600,
        }}>
          ⚙ {tc.tool_name}
        </span>
        <span style={{ flex: 1, fontSize: 11, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {argsStr.slice(0, 80)}
        </span>
        {tc.duration_ms != null && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)', flexShrink: 0 }}>
            {tc.duration_ms}ms
          </span>
        )}
        {tc.result == null && (
          <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5, color: 'var(--info)' }} />
        )}
        {tc.result != null && (
          <span style={{ color: 'var(--success)', fontSize: 12 }}>✓</span>
        )}
        <span style={{ color: 'var(--text-muted)', fontSize: 12, flexShrink: 0 }}>{open ? '▲' : '▼'}</span>
      </div>

      {/* Expanded content */}
      {open && (
        <div style={{ marginTop: 10 }} className="animate-fade-in">
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Arguments</div>
            <pre style={{
              background: 'var(--bg-primary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              padding: '8px 10px',
              fontSize: 11,
              color: 'var(--text-code)',
              overflowX: 'auto',
              margin: 0,
            }}>
              {argsStr}
            </pre>
          </div>

          {tc.result != null && (
            <div>
              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Result</div>
              <pre style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 10px',
                fontSize: 11,
                color: 'var(--text-secondary)',
                overflowX: 'auto',
                maxHeight: 200,
                overflow: 'auto',
                margin: 0,
              }}>
                {tc.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function ToolCallLog({ toolCalls = [] }) {
  if (toolCalls.length === 0) {
    return (
      <div style={{ padding: 'var(--sp-md)', color: 'var(--text-muted)', fontSize: 13, textAlign: 'center' }}>
        No tool calls yet.
      </div>
    );
  }

  return (
    <div style={{ padding: 'var(--sp-sm)', overflow: 'auto' }}>
      {toolCalls.map((tc, i) => (
        <ToolRow key={tc.id || i} tc={tc} />
      ))}
    </div>
  );
}
