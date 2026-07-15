/**
 * DagView — full SVG DAG visualization.
 *
 * Computes layout from the current plan + subtask states,
 * renders all nodes and animated edges, supports click-to-select.
 * Shows an empty state when no plan is available yet.
 */

import React, { useMemo } from 'react';
import { useStore } from '../stores/useStore';
import { layoutDag } from '../utils/dagLayout';
import DagNode from './DagNode';
import DagEdge from './DagEdge';

const PADDING = 40;

export default function DagView() {
  const plan = useStore((s) => s.plan);
  const subtasks = useStore((s) => s.subtasks);
  const status = useStore((s) => s.status);

  // Merge plan with live subtask state
  const enrichedTasks = useMemo(() => {
    return plan.map((raw) => ({
      ...raw,
      ...(subtasks[raw.id] || {}),
    }));
  }, [plan, subtasks]);

  const { nodes, edges, canvasW = 600, canvasH = 400 } = useMemo(
    () => layoutDag(enrichedTasks),
    [enrichedTasks]
  );

  const svgW = canvasW + PADDING * 2;
  const svgH = canvasH + PADDING * 2;

  // Empty state
  if (plan.length === 0) {
    return (
      <div style={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 16,
        color: 'var(--text-muted)',
        userSelect: 'none',
      }}>
        {status === 'planning' ? (
          <>
            <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3, color: 'var(--accent)' }} />
            <span style={{ fontSize: 14 }}>Planning orchestration…</span>
          </>
        ) : status === 'idle' ? (
          <>
            <div style={{ fontSize: 48, opacity: 0.3 }}>◈</div>
            <span style={{ fontSize: 14 }}>Enter a goal to get started</span>
          </>
        ) : (
          <>
            <div className="spinner" style={{ width: 32, height: 32, borderWidth: 3, color: 'var(--accent)' }} />
            <span>{status}…</span>
          </>
        )}
      </div>
    );
  }

  return (
    <div
      id="dag-view"
      className="scroll-x"
      style={{
        flex: 1,
        overflow: 'auto',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        padding: 8,
      }}
    >
      <svg
        width={svgW}
        height={svgH}
        style={{ display: 'block', overflow: 'visible' }}
      >
        <g transform={`translate(${PADDING}, ${PADDING})`}>
          {/* Render edges first (below nodes) */}
          {edges.map((e, i) => {
            const fromTask = subtasks[e.from] || {};
            const toTask = subtasks[e.to] || {};
            return (
              <DagEdge
                key={i}
                fromX={e.fromX}
                fromY={e.fromY}
                toX={e.toX}
                toY={e.toY}
                fromStatus={fromTask.status}
                toStatus={toTask.status}
              />
            );
          })}

          {/* Render nodes */}
          {nodes.map((n) => {
            const raw = plan.find((p) => p.id === n.id) || {};
            const live = subtasks[n.id] || {};
            return (
              <DagNode
                key={n.id}
                id={n.id}
                x={n.x}
                y={n.y}
                width={n.width}
                height={n.height}
                subtask={{ ...raw, ...live }}
              />
            );
          })}
        </g>
      </svg>
    </div>
  );
}
