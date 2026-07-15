/**
 * DagNode — a single agent node in the SVG DAG.
 *
 * Renders a glassmorphism card at the given position.
 * Animates based on status: pulsing glow when running, shake on fail.
 * Clicking selects the subtask and opens the agent panel.
 */

import React from 'react';
import { useStore } from '../stores/useStore';

const STATUS_COLORS = {
  pending:   'var(--muted)',
  running:   'var(--accent)',
  paused:    'var(--warning)',
  completed: 'var(--success)',
  failed:    'var(--error)',
  cancelled: 'var(--muted)',
};

const AGENT_COLORS = {
  researcher: 'var(--color-researcher)',
  analyzer:   'var(--color-analyzer)',
  coder:      'var(--color-coder)',
  verifier:   'var(--color-verifier)',
};

const STATUS_ICONS = {
  pending:   '○',
  running:   '◎',
  paused:    '⏸',
  completed: '✓',
  failed:    '✕',
  cancelled: '—',
};

export default function DagNode({ id, x, y, width, height, subtask }) {
  const selectedId = useStore((s) => s.selectedSubtaskId);
  const setSelected = useStore((s) => s.setSelectedSubtask);
  const pauseSubtask = useStore((s) => s.pauseSubtask);
  const resumeSubtask = useStore((s) => s.resumeSubtask);

  if (!subtask) return null;

  const { status = 'pending', agent_name = '', task = '', tokens = 0 } = subtask;
  const color = STATUS_COLORS[status] || 'var(--muted)';
  const agentColor = AGENT_COLORS[agent_name] || 'var(--color-default)';
  const isSelected = selectedId === id;
  const isRunning = status === 'running';
  const isPaused = status === 'paused';

  const handleClick = () => setSelected(isSelected ? null : id);

  const handlePauseResume = (e) => {
    e.stopPropagation();
    if (isPaused) resumeSubtask(id);
    else if (isRunning) pauseSubtask(id);
  };

  // Truncate task text for node
  const shortTask = task.length > 50 ? task.slice(0, 48) + '…' : task;

  return (
    <g
      id={`dag-node-${id}`}
      transform={`translate(${x}, ${y})`}
      onClick={handleClick}
      style={{ cursor: 'pointer' }}
    >
      {/* Running glow animation */}
      {isRunning && (
        <rect
          width={width} height={height} rx={10}
          fill="none"
          stroke={color}
          strokeWidth={2}
          opacity={0.5}
          style={{ animation: 'pulse-glow-accent 2s ease-in-out infinite' }}
        />
      )}
      {isPaused && (
        <rect
          width={width} height={height} rx={10}
          fill="none"
          stroke={color}
          strokeWidth={2}
          opacity={0.4}
          style={{ animation: 'pulse-glow-warning 2s ease-in-out infinite' }}
        />
      )}

      {/* Card background */}
      <rect
        width={width} height={height} rx={10}
        fill={isSelected ? 'rgba(99,102,241,0.18)' : 'rgba(26,34,53,0.85)'}
        stroke={isSelected ? 'var(--border-active)' : color}
        strokeWidth={isSelected ? 2 : 1.5}
        strokeOpacity={isSelected ? 1 : 0.5}
      />

      {/* Agent type accent bar */}
      <rect x={0} y={0} width={4} height={height} rx={2} fill={agentColor} opacity={0.9} />

      {/* Status dot */}
      <circle cx={width - 14} cy={14} r={5} fill={color}
        style={isRunning ? { filter: `drop-shadow(0 0 4px ${color})` } : {}}
      />

      {/* Agent name */}
      <text
        x={12} y={22}
        fill={agentColor}
        fontSize={10}
        fontWeight={600}
        fontFamily="var(--font-mono)"
        textAnchor="start"
      >
        {agent_name || 'agent'}
      </text>

      {/* Task text */}
      <foreignObject x={10} y={28} width={width - 20} height={height - 44}>
        <div xmlns="http://www.w3.org/1999/xhtml" style={{
          color: 'var(--text-secondary)',
          fontSize: 11,
          lineHeight: '1.4',
          overflow: 'hidden',
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
        }}>
          {shortTask}
        </div>
      </foreignObject>

      {/* Bottom status + token count */}
      <text x={12} y={height - 8} fontSize={10} fill="var(--text-muted)" fontFamily="var(--font-mono)">
        {STATUS_ICONS[status]} {tokens > 0 ? `${tokens}t` : status}
      </text>

      {/* Pause/resume mini button */}
      {(isRunning || isPaused) && (
        <g onClick={handlePauseResume} style={{ cursor: 'pointer' }}>
          <circle cx={width - 14} cy={height - 13} r={10} fill="rgba(0,0,0,0.5)" />
          <text cx={width - 14} cy={height - 9} x={width - 14} y={height - 9}
            fontSize={11} textAnchor="middle" fill={isRunning ? 'var(--warning)' : 'var(--success)'}>
            {isRunning ? '⏸' : '▶'}
          </text>
        </g>
      )}
    </g>
  );
}
