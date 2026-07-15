/**
 * ContextViewer — shows full message history / context for a subtask.
 */

import React, { useRef, useEffect } from 'react';

const ROLE_COLORS = {
  system: 'var(--text-muted)',
  user: 'var(--info)',
  assistant: 'var(--accent)',
  tool: 'var(--success)',
};

function Message({ msg }) {
  const color = ROLE_COLORS[msg.role] || 'var(--text-secondary)';
  return (
    <div style={{
      padding: '8px 12px',
      borderLeft: `2px solid ${color}`,
      marginBottom: 8,
      borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
      background: 'var(--bg-elevated)',
    }}>
      <div style={{ fontSize: 10, color, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>
        {msg.role}
        {msg.ts && <span style={{ marginLeft: 8, color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>{new Date(msg.ts).toLocaleTimeString()}</span>}
      </div>
      <pre style={{
        fontFamily: 'var(--font-sans)',
        fontSize: 12,
        color: 'var(--text-secondary)',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        margin: 0,
        lineHeight: 1.6,
      }}>
        {msg.content}
      </pre>
    </div>
  );
}

export default function ContextViewer({ messages = [], sources = [] }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div style={{ padding: 'var(--sp-md)', color: 'var(--text-muted)', fontSize: 13, textAlign: 'center' }}>
        No messages yet.
      </div>
    );
  }

  return (
    <div style={{ padding: 'var(--sp-sm)', overflowY: 'auto' }}>
      {messages.map((m, i) => <Message key={i} msg={m} />)}

      {sources.length > 0 && (
        <div style={{ marginTop: 12, padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.1em' }}>Sources</div>
          {sources.map((s, i) => (
            <div key={i} style={{ fontSize: 11, color: 'var(--info)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              <a href={s} target="_blank" rel="noopener noreferrer" style={{ color: 'inherit' }}>{s}</a>
            </div>
          ))}
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
