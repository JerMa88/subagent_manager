/**
 * Central Zustand store for the SubAgent Manager GUI.
 *
 * Tracks orchestration state, per-subtask agent state, UI state,
 * run history, and exposes all action functions.
 */

import { create } from 'zustand';

const API = 'http://localhost:8000';

// ─── Initial subtask state ────────────────────────────────────────────────
const makeSubtask = (raw) => ({
  id: raw.id,
  task: raw.task || '',
  agent_name: raw.agent || raw.agent_name || '',
  depends_on: raw.depends_on || [],
  context: raw.context || '',
  status: 'pending',   // pending | running | paused | completed | failed | cancelled
  messages: [],
  toolCalls: [],
  answer: '',
  sources: [],
  tokens: 0,
  toolCallsMade: 0,
  error: null,
  startedAt: null,
  completedAt: null,
});

// ─── Store ────────────────────────────────────────────────────────────────
export const useStore = create((set, get) => ({
  // Connection
  wsConnected: false,

  // Current run
  currentRunId: null,
  goal: '',
  status: 'idle',   // idle | planning | plan_review | executing | synthesizing | completed | failed | cancelled

  // Plan & subtasks
  plan: [],
  subtasks: {},     // { [subtask_id]: subtask_state }

  // Config
  config: {
    model: 'ollama/qwen3',
    orchestrator_model: '',
    strategy: 'adaptive',
    max_subtasks: 10,
    api_key: '',
    api_base: '',
    agents: [],
  },

  // UI state
  selectedSubtaskId: null,
  isConfigPanelOpen: false,
  isPlanEditorOpen: false,

  // Synthesis
  synthesisResult: null,
  synthesisStreaming: false,

  // Run history (from SQLite)
  runs: [],

  // ─── Actions ───────────────────────────────────────────────────────────

  setGoal: (goal) => set({ goal }),
  setWsConnected: (v) => set({ wsConnected: v }),
  setSelectedSubtask: (id) => set({ selectedSubtaskId: id }),
  toggleConfigPanel: () => set((s) => ({ isConfigPanelOpen: !s.isConfigPanelOpen })),
  togglePlanEditor: () => set((s) => ({ isPlanEditorOpen: !s.isPlanEditorOpen })),
  updateConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),

  // Reset state for a fresh run
  resetForRun: (runId, goal) => set({
    currentRunId: runId,
    goal,
    status: 'planning',
    plan: [],
    subtasks: {},
    selectedSubtaskId: null,
    synthesisResult: null,
    synthesisStreaming: false,
  }),

  // ─── Event handler: dispatches incoming WS events to store ─────────────
  handleEvent: (event) => {
    const { type, subtask_id, agent_name, data, timestamp } = event;

    set((state) => {
      const patch = {};

      switch (type) {
        case 'orchestration_started':
          patch.status = 'planning';
          break;

        case 'plan_created': {
          const subtasks = {};
          (data.plan || []).forEach((raw) => {
            subtasks[raw.id] = makeSubtask(raw);
          });
          patch.plan = data.plan || [];
          patch.subtasks = subtasks;
          patch.status = 'executing';
          break;
        }

        case 'subtask_started': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id] || makeSubtask({ id: subtask_id, agent_name });
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: {
              ...prev,
              status: 'running',
              agent_name: agent_name || prev.agent_name,
              startedAt: timestamp,
              messages: [
                ...prev.messages,
                { role: 'system', content: data.task, ts: timestamp },
              ],
            },
          };
          break;
        }

        case 'tool_call_started': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          const tc = {
            id: Date.now(),
            tool_name: data.tool_name,
            arguments: data.arguments,
            result: null,
            duration_ms: null,
            startedAt: timestamp,
          };
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: {
              ...prev,
              toolCalls: [...prev.toolCalls, tc],
            },
          };
          break;
        }

        case 'tool_call_completed': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          // Update last open tool call
          const tcs = [...prev.toolCalls];
          const idx = tcs.findLastIndex((t) => t.tool_name === data.tool_name && t.result === null);
          if (idx >= 0) {
            tcs[idx] = {
              ...tcs[idx],
              result: data.result,
              duration_ms: data.duration_ms,
            };
          }
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: { ...prev, toolCalls: tcs, toolCallsMade: prev.toolCallsMade + 1 },
          };
          break;
        }

        case 'llm_call_completed': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: {
              ...prev,
              tokens: prev.tokens + (data.tokens || 0),
            },
          };
          break;
        }

        case 'subtask_paused': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: { ...prev, status: 'paused' },
          };
          break;
        }

        case 'subtask_resumed': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: { ...prev, status: 'running' },
          };
          break;
        }

        case 'subtask_completed': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: {
              ...prev,
              status: 'completed',
              answer: data.answer || '',
              sources: data.sources || [],
              tokens: data.tokens_used || prev.tokens,
              toolCallsMade: data.tool_calls_made || prev.toolCallsMade,
              completedAt: timestamp,
            },
          };
          break;
        }

        case 'subtask_failed': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: {
              ...prev,
              status: 'failed',
              error: data.error || 'Unknown error',
              completedAt: timestamp,
            },
          };
          break;
        }

        case 'subtask_cancelled': {
          if (subtask_id == null) break;
          const prev = state.subtasks[subtask_id];
          if (!prev) break;
          patch.subtasks = {
            ...state.subtasks,
            [subtask_id]: { ...prev, status: 'cancelled', completedAt: timestamp },
          };
          break;
        }

        case 'synthesis_started':
          patch.status = 'synthesizing';
          patch.synthesisStreaming = true;
          break;

        case 'synthesis_completed':
          patch.synthesisStreaming = false;
          break;

        case 'orchestration_completed':
          patch.status = 'completed';
          patch.synthesisStreaming = false;
          break;

        case 'orchestration_failed':
          patch.status = 'failed';
          patch.synthesisStreaming = false;
          break;

        case 'orchestration_cancelled':
          patch.status = 'cancelled';
          patch.synthesisStreaming = false;
          break;

        default:
          break;
      }

      return patch;
    });
  },

  // ─── API actions ────────────────────────────────────────────────────────

  startRun: async (goal) => {
    const { config, resetForRun } = get();
    try {
      const res = await fetch(`${API}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal, config }),
      });
      const data = await res.json();
      resetForRun(data.run_id, goal);
      return data.run_id;
    } catch (e) {
      console.error('startRun failed:', e);
      return null;
    }
  },

  cancelRun: async () => {
    const { currentRunId } = get();
    if (!currentRunId) return;
    await fetch(`${API}/api/run/${currentRunId}/cancel`, { method: 'POST' });
  },

  pauseSubtask: async (subtaskId) => {
    const { currentRunId } = get();
    if (!currentRunId) return;
    await fetch(`${API}/api/run/${currentRunId}/subtask/${subtaskId}/pause`, { method: 'POST' });
  },

  resumeSubtask: async (subtaskId) => {
    const { currentRunId } = get();
    if (!currentRunId) return;
    await fetch(`${API}/api/run/${currentRunId}/subtask/${subtaskId}/resume`, { method: 'POST' });
  },

  injectContext: async (subtaskId, context) => {
    const { currentRunId } = get();
    if (!currentRunId) return;
    await fetch(`${API}/api/run/${currentRunId}/subtask/${subtaskId}/inject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context }),
    });
  },

  fetchRuns: async () => {
    try {
      const res = await fetch(`${API}/api/runs`);
      const runs = await res.json();
      set({ runs });
    } catch (e) {
      console.error('fetchRuns failed:', e);
    }
  },

  loadRun: async (runId) => {
    try {
      const res = await fetch(`${API}/api/runs/${runId}`);
      const data = await res.json();
      // Replay events to rebuild subtask state
      const { plan, result, events = [] } = data;
      const subtasks = {};
      (plan || []).forEach((raw) => { subtasks[raw.id] = makeSubtask(raw); });

      // Apply persisted results
      if (result?.subtask_results) {
        result.subtask_results.forEach((r) => {
          const st = Object.values(subtasks).find((s) => s.task === r.task);
          if (st) {
            Object.assign(st, {
              status: r.success ? 'completed' : 'failed',
              answer: r.answer,
              sources: r.sources,
              error: r.error,
            });
          }
        });
      }

      set({
        currentRunId: runId,
        goal: data.goal,
        status: data.status,
        plan: plan || [],
        subtasks,
        synthesisResult: result?.answer || null,
      });
    } catch (e) {
      console.error('loadRun failed:', e);
    }
  },

  setSynthesisResult: (answer) => set({ synthesisResult: answer }),
}));
