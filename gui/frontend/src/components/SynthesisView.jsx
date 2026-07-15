/**
 * SynthesisView — shows the final synthesized answer after all subtasks complete.
 */

import React from 'react';
import { useStore } from '../stores/useStore';

export default function SynthesisView() {
  const result = useStore((s) => s.synthesisResult);
  const streaming = useStore((s) => s.synthesisStreaming);
  const status = useStore((s) => s.status);

  if (!result && !streaming && status !== 'synthesizing') return null;

  return (
    <div
      id="synthesis-view"
      className="animate-fade-in"
      style={{
        margin: 'var(--sp-md)',
        padding: 'var(--sp-lg)',
        background: 'var(--bg-glass)',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--border-success)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--glow-success)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        {streaming ? (
          <span className="spinner" style={{ color: 'var(--accent)' }} />
        ) : (
          <span style={{ color: 'var(--success)', fontSize: 18 }}>✓</span>
        )}
        <h2 style={{ color: 'var(--success)', fontWeight: 700 }}>
          {streaming ? 'Synthesizing…' : 'Final Answer'}
        </h2>
      </div>
      {result && (
        <div style={{
          color: 'var(--text-primary)',
          fontSize: 14,
          lineHeight: 1.8,
          whiteSpace: 'pre-wrap',
          maxHeight: 400,
          overflowY: 'auto',
        }}>
          {result}
        </div>
      )}
    </div>
  );
}
