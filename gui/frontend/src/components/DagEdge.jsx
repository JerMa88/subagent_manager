/**
 * DagEdge — animated SVG edge between two nodes.
 *
 * Draws a cubic bezier curve with a dashed "flow" animation
 * when the downstream node is running. Static when pending/done.
 */

import React from 'react';

export default function DagEdge({ fromX, fromY, toX, toY, fromStatus, toStatus }) {
  const isActive = toStatus === 'running';
  const isCompleted = fromStatus === 'completed' && toStatus !== 'pending';

  // Cubic bezier control points for smooth S-curve
  const cy1 = fromY + (toY - fromY) * 0.5;
  const cy2 = toY - (toY - fromY) * 0.5;
  const d = `M ${fromX} ${fromY} C ${fromX} ${cy1}, ${toX} ${cy2}, ${toX} ${toY}`;

  const strokeColor = isCompleted
    ? 'var(--success)'
    : isActive
    ? 'var(--accent)'
    : 'var(--muted)';

  return (
    <>
      {/* Base edge */}
      <path
        d={d}
        fill="none"
        stroke={strokeColor}
        strokeWidth={isActive ? 2 : 1.5}
        strokeOpacity={isActive ? 0.9 : isCompleted ? 0.6 : 0.25}
      />

      {/* Animated flow overlay */}
      {isActive && (
        <path
          d={d}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={2}
          strokeOpacity={0.7}
          strokeDasharray="8 12"
          strokeDashoffset={0}
          style={{ animation: 'dash-flow 0.8s linear infinite' }}
        />
      )}

      {/* Arrowhead */}
      <polygon
        points={`${toX},${toY} ${toX - 5},${toY - 8} ${toX + 5},${toY - 8}`}
        fill={strokeColor}
        opacity={isActive ? 1 : isCompleted ? 0.6 : 0.25}
      />
    </>
  );
}
