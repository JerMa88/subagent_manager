"""
Active run lifecycle manager.

Bridges the FastAPI layer to SubAgentManager. Each call to start_run()
creates a fresh SubAgentManager, wires up an EventBus, WebSocket broadcast,
and SQLite persistence, then launches the orchestration as a background task.

All control signals (pause, resume, cancel) flow through asyncio.Event objects
that are checked at each tool-loop checkpoint inside the agent pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from subagent_manager import EventBus, EventType, Event, SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool, URLReaderTool, PythonExecTool, FileReaderTool

from gui.backend import db

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Active run state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ActiveRun:
    """Holds all live state for a single orchestration run."""

    run_id: str
    goal: str
    config: dict[str, Any]

    # The event bus bridges the orchestration pipeline to WebSocket clients
    event_bus: EventBus = field(default_factory=EventBus)

    # Connected WebSocket clients (each receives a copy of every event)
    ws_clients: set[Any] = field(default_factory=set)

    # Per-subtask pause events: SET = running, CLEARED = paused
    # Populated after plan is received.
    pause_events: dict[int, asyncio.Event] = field(default_factory=dict)

    # Global cancel event: SET = cancel requested
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # The background asyncio.Task
    task: asyncio.Task | None = None

    # Current status
    status: str = "starting"  # starting | planning | executing | synthesizing | completed | failed | cancelled

    # Plan (set after PLAN_CREATED event)
    plan: list[dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Global registry of active runs
# ─────────────────────────────────────────────────────────────────────

_active_runs: dict[str, ActiveRun] = {}


def get_active_run(run_id: str) -> ActiveRun | None:
    return _active_runs.get(run_id)


def get_all_active_runs() -> list[dict[str, Any]]:
    return [
        {"run_id": r.run_id, "goal": r.goal, "status": r.status}
        for r in _active_runs.values()
    ]


# ─────────────────────────────────────────────────────────────────────
# Tool name → instance mapping
# ─────────────────────────────────────────────────────────────────────

_TOOL_REGISTRY = {
    "web_search": WebSearchTool,
    "read_url": URLReaderTool,
    "python_exec": PythonExecTool,
    "read_file": FileReaderTool,
}


def _build_agent_configs(agents_config: list[dict]) -> list[SubAgentConfig]:
    """Convert GUI config dicts to SubAgentConfig instances."""
    configs = []
    for ag in agents_config:
        tools = []
        for tool_name in ag.get("tools", []):
            cls = _TOOL_REGISTRY.get(tool_name)
            if cls:
                tools.append(cls())
            else:
                logger.warning(f"Unknown tool: {tool_name!r}")
        configs.append(SubAgentConfig(
            name=ag["name"],
            description=ag.get("description", ""),
            tools=tools,
            model=ag.get("model"),
            system_prompt=ag.get("system_prompt"),
            max_tool_iterations=ag.get("max_tool_iterations", 5),
            max_answer_tokens=ag.get("max_answer_tokens", 512),
            temperature=ag.get("temperature", 0.0),
        ))
    return configs


# ─────────────────────────────────────────────────────────────────────
# WebSocket broadcast
# ─────────────────────────────────────────────────────────────────────

async def _broadcast(run: ActiveRun, event_dict: dict[str, Any]) -> None:
    """Send an event to all connected WebSocket clients for this run."""
    dead: set[Any] = set()
    payload = json.dumps(event_dict)
    for ws in run.ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    run.ws_clients -= dead


# ─────────────────────────────────────────────────────────────────────
# Main run lifecycle
# ─────────────────────────────────────────────────────────────────────

async def start_run(
    goal: str,
    config: dict[str, Any],
) -> str:
    """
    Launch a new orchestration run in the background.

    Returns the run_id immediately. Events stream via WebSocket.
    """
    run_id = str(uuid.uuid4())
    run = ActiveRun(run_id=run_id, goal=goal, config=config)
    _active_runs[run_id] = run
    run.event_bus.set_run_id(run_id)

    # Persist run start
    await db.create_run(run_id, goal, config)

    # Wire up: event bus → WebSocket broadcast + SQLite
    def _on_event(event: Event) -> None:
        event_dict = event.to_dict()
        # Update run status based on event type
        if event.type == EventType.ORCHESTRATION_STARTED:
            run.status = "planning"
        elif event.type == EventType.PLAN_CREATED:
            run.status = "executing"
            run.plan = event.data.get("plan", [])
            # Create pause events for each subtask
            for subtask in run.plan:
                sid = subtask.get("id")
                if sid is not None:
                    e = asyncio.Event()
                    e.set()  # start in running state
                    run.pause_events[sid] = e
            # Persist plan
            asyncio.get_event_loop().create_task(
                db.update_run_plan(run_id, run.plan)
            )
        elif event.type == EventType.SYNTHESIS_STARTED:
            run.status = "synthesizing"
        elif event.type == EventType.ORCHESTRATION_COMPLETED:
            run.status = "completed"
        elif event.type == EventType.ORCHESTRATION_FAILED:
            run.status = "failed"
        elif event.type == EventType.ORCHESTRATION_CANCELLED:
            run.status = "cancelled"

        # Persist event
        asyncio.get_event_loop().create_task(db.save_event(
            run_id=run_id,
            event_type=event.type.value,
            timestamp=event.timestamp,
            data=event.data,
            subtask_id=event.subtask_id,
            agent_name=event.agent_name,
        ))

        # Broadcast to WebSocket clients
        asyncio.get_event_loop().create_task(_broadcast(run, event_dict))

    run.event_bus.subscribe(_on_event)

    # Build the manager from config
    model = config.get("model", "ollama/qwen3")
    orchestrator_model = config.get("orchestrator_model")
    strategy = config.get("strategy", "adaptive")
    max_subtasks = config.get("max_subtasks", 10)
    api_key = config.get("api_key")
    api_base = config.get("api_base")

    agents_config_list = config.get("agents", [])
    subagents = _build_agent_configs(agents_config_list) if agents_config_list else None

    manager = SubAgentManager(
        model=model,
        subagents=subagents,
        strategy=strategy,
        max_subtasks=max_subtasks,
        api_key=api_key,
        api_base=api_base,
        orchestrator_model=orchestrator_model,
    )

    async def _run_task() -> None:
        try:
            result = await manager.run_with_events(
                goal=goal,
                event_bus=run.event_bus,
                pause_events=run.pause_events,
                cancel_event=run.cancel_event,
            )
            result_dict = {
                "answer": result.answer,
                "total_tokens": result.total_tokens,
                "total_tool_calls": result.total_tool_calls,
                "sources": result.sources,
                "subtask_results": [
                    {
                        "agent_name": r.agent_name,
                        "task": r.task,
                        "answer": r.answer,
                        "sources": r.sources,
                        "tool_calls_made": r.tool_calls_made,
                        "success": r.success,
                        "error": r.error,
                        "tokens_used": r.tokens_used,
                    }
                    for r in result.subtask_results
                ],
            }
            await db.complete_run(run_id, "completed", result_dict)
        except asyncio.CancelledError:
            run.status = "cancelled"
            await db.complete_run(run_id, "cancelled")
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}", exc_info=True)
            run.status = "failed"
            await db.complete_run(run_id, "failed", {"error": str(e)})
        finally:
            # Clean up after a short delay (give WS clients time to receive final events)
            await asyncio.sleep(5)
            _active_runs.pop(run_id, None)

    run.task = asyncio.create_task(_run_task())
    logger.info(f"Started run {run_id} for goal: {goal[:80]}...")
    return run_id


# ─────────────────────────────────────────────────────────────────────
# Control signals
# ─────────────────────────────────────────────────────────────────────

def cancel_run(run_id: str) -> bool:
    """Cancel the entire orchestration. Returns False if run not found."""
    run = _active_runs.get(run_id)
    if not run:
        return False
    run.cancel_event.set()
    if run.task and not run.task.done():
        run.task.cancel()
    return True


def pause_subtask(run_id: str, subtask_id: int) -> bool:
    """Pause a subtask at its next tool-loop checkpoint."""
    run = _active_runs.get(run_id)
    if not run:
        return False
    ev = run.pause_events.get(subtask_id)
    if not ev:
        # Create it pre-cleared so any future agent also starts paused
        ev = asyncio.Event()
        run.pause_events[subtask_id] = ev
    ev.clear()  # CLEARED = paused
    logger.info(f"Paused subtask {subtask_id} in run {run_id}")
    return True


def resume_subtask(run_id: str, subtask_id: int) -> bool:
    """Resume a paused subtask."""
    run = _active_runs.get(run_id)
    if not run:
        return False
    ev = run.pause_events.get(subtask_id)
    if not ev:
        return False
    ev.set()  # SET = running
    logger.info(f"Resumed subtask {subtask_id} in run {run_id}")
    return True


def inject_context(run_id: str, subtask_id: int, context: str) -> bool:
    """
    Inject additional context into a subtask before it resumes.

    Since the agent's conversation is immutable once started, this
    actually modifies the pause_events registry and emits a synthetic
    event. Full mid-stream injection requires deeper refactoring (v2).
    """
    run = _active_runs.get(run_id)
    if not run:
        return False
    # Emit a synthetic event so the GUI reflects the injection
    run.event_bus.emit(Event(
        type=EventType.SUBTASK_STARTED,  # reuse as context update notification
        subtask_id=subtask_id,
        data={"injected_context": context},
    ))
    logger.info(f"Context injected for subtask {subtask_id} in run {run_id}")
    return True
