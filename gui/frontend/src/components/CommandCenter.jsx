/**
 * CommandCenter — main single-page layout assembling all panels.
 *
 * Layout:
 *   StatusBar (top)
 *   ┌──────────┬──────────────────────────┬──────────┐
 *   │ Run      │ Goal Input (top)          │ Agent    │
 *   │ History  │ DAG View                  │ Panel    │
 *   │ Sidebar  │ Synthesis View (bottom)   │          │
 *   └──────────┴──────────────────────────┴──────────┘
 *   ConfigPanel (overlay, slide-in from right)
 */

import React from 'react';
import StatusBar from './StatusBar';
import GoalInput from './GoalInput';
import DagView from './DagView';
import AgentPanel from './AgentPanel';
import SynthesisView from './SynthesisView';
import RunHistory from './RunHistory';
import ConfigPanel from './ConfigPanel';
import { useWebSocket } from '../hooks/useWebSocket';

export default function CommandCenter() {
  // Activate WebSocket for the current run
  useWebSocket();

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-base)',
      overflow: 'hidden',
    }}>
      {/* Top status bar */}
      <StatusBar />

      {/* Main content row */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Left: run history sidebar */}
        <RunHistory />

        {/* Center: goal input + DAG + synthesis */}
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
        }}>
          {/* Goal input */}
          <GoalInput />

          {/* DAG fills remaining space */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>
            <DagView />
          </div>

          {/* Synthesis answer (bottom, scrollable) */}
          <div style={{ maxHeight: '40%', overflowY: 'auto', flexShrink: 0 }}>
            <SynthesisView />
          </div>
        </div>

        {/* Right: agent inspection panel */}
        <AgentPanel />
      </div>

      {/* Config overlay */}
      <ConfigPanel />
    </div>
  );
}
