"""
FastAPI server — REST + WebSocket bridge for the React GUI.

Endpoints:
  GET  /api/config           — get current default config
  PUT  /api/config           — update default config
  POST /api/run              — start a new orchestration run
  POST /api/run/{id}/cancel  — cancel entire run
  POST /api/run/{id}/subtask/{sid}/pause   — pause subtask
  POST /api/run/{id}/subtask/{sid}/resume  — resume subtask
  POST /api/run/{id}/subtask/{sid}/cancel  — cancel subtask (via cancel_event)
  POST /api/run/{id}/subtask/{sid}/inject  — inject context
  GET  /api/runs             — list past runs
  GET  /api/runs/{id}        — get full run details + events
  WS   /ws/{run_id}          — real-time event stream for a run
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gui.backend import db
from gui.backend import run_manager as rm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Default config (in-memory, overridable via PUT /api/config)
# ─────────────────────────────────────────────────────────────────────

_default_config: dict[str, Any] = {
    "model": "ollama/ornith:latest",
    "orchestrator_model": None,
    "strategy": "adaptive",
    "max_subtasks": 10,
    "api_key": None,
    "api_base": None,
    "agents": [],  # Empty → use SubAgentManager defaults
}


# ─────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="SubAgent Manager GUI API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev-only; lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    context: str = ""
    config: dict[str, Any] | None = None  # Overrides default config for this run


class InjectContextRequest(BaseModel):
    context: str


class ConfigUpdate(BaseModel):
    model: str | None = None
    orchestrator_model: str | None = None
    strategy: str | None = None
    max_subtasks: int | None = None
    api_key: str | None = None
    api_base: str | None = None
    agents: list[dict[str, Any]] | None = None


# ─────────────────────────────────────────────────────────────────────
# Config endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return _default_config


@app.put("/api/config")
async def update_config(update: ConfigUpdate) -> dict[str, Any]:
    for key, val in update.model_dump(exclude_none=True).items():
        _default_config[key] = val
    return _default_config


@app.get("/api/models")
async def list_models() -> list[str]:
    """Fetch locally available models from Ollama."""
    import urllib.request
    import json
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [f"ollama/{m['name']}" for m in data.get("models", [])]
            # Always ensure some common fallbacks are present
            fallbacks = ["gpt-4o-mini", "anthropic/claude-3-5-sonnet-20240620", "gemini/gemini-2.5-flash"]
            return models + fallbacks
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")
        return [
            "ollama/ornith:latest",
            "ollama/gemma4:e2b-mlx",
            "gpt-4o-mini",
            "anthropic/claude-3-5-sonnet-20240620",
            "gemini/gemini-2.5-flash"
        ]


# ─────────────────────────────────────────────────────────────────────
# Run endpoints
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def start_run(req: RunRequest) -> dict[str, str]:
    """Start a new orchestration run. Returns run_id immediately."""
    config = {**_default_config, **(req.config or {})}
    run_id = await rm.start_run(goal=req.goal, config=config)
    return {"run_id": run_id, "status": "started"}


@app.post("/api/run/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, str]:
    ok = rm.cancel_run(run_id)
    if not ok:
        raise HTTPException(404, f"Run {run_id} not found or already finished.")
    return {"status": "cancelling"}


@app.post("/api/run/{run_id}/subtask/{subtask_id}/pause")
async def pause_subtask(run_id: str, subtask_id: int) -> dict[str, str]:
    ok = rm.pause_subtask(run_id, subtask_id)
    if not ok:
        raise HTTPException(404, f"Run {run_id} or subtask {subtask_id} not found.")
    return {"status": "pausing"}


@app.post("/api/run/{run_id}/subtask/{subtask_id}/resume")
async def resume_subtask(run_id: str, subtask_id: int) -> dict[str, str]:
    ok = rm.resume_subtask(run_id, subtask_id)
    if not ok:
        raise HTTPException(404, f"Run {run_id} or subtask {subtask_id} not found.")
    return {"status": "resuming"}


@app.post("/api/run/{run_id}/subtask/{subtask_id}/cancel")
async def cancel_subtask(run_id: str, subtask_id: int) -> dict[str, str]:
    """Cancel a single subtask by triggering the global cancel (conservative approach)."""
    ok = rm.cancel_run(run_id)
    if not ok:
        raise HTTPException(404, f"Run {run_id} not found.")
    return {"status": "cancelling"}


@app.post("/api/run/{run_id}/subtask/{subtask_id}/inject")
async def inject_context(
    run_id: str, subtask_id: int, req: InjectContextRequest
) -> dict[str, str]:
    ok = rm.inject_context(run_id, subtask_id, req.context)
    if not ok:
        raise HTTPException(404, f"Run {run_id} not found.")
    return {"status": "injected"}


# ─────────────────────────────────────────────────────────────────────
# Run history endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List past runs (summary) from SQLite."""
    return await db.get_runs()


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get full run details + all events."""
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found.")
    events = await db.get_events(run_id)
    run["events"] = events
    return run


# ─────────────────────────────────────────────────────────────────────
# WebSocket — real-time event stream
# ─────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    WebSocket connection for a specific run.

    On connect:
      1. Send all events stored so far (catch-up for late joiners).
      2. Register as a live subscriber for future events.

    On message received:
      Handle control commands from the client:
        { "command": "pause_subtask", "subtask_id": 1 }
        { "command": "resume_subtask", "subtask_id": 1 }
        { "command": "cancel_run" }
    """
    await websocket.accept()
    run = rm.get_active_run(run_id)

    # Send replay of past events (catch-up)
    past_events = await db.get_events(run_id)
    for event in past_events:
        try:
            import json
            await websocket.send_text(json.dumps(event))
        except Exception:
            break

    # Register for live events
    if run:
        run.ws_clients.add(websocket)

    try:
        while True:
            # Wait for commands from the client
            data = await websocket.receive_text()
            try:
                import json
                cmd = json.loads(data)
                command = cmd.get("command")

                if command == "pause_subtask":
                    rm.pause_subtask(run_id, cmd["subtask_id"])
                elif command == "resume_subtask":
                    rm.resume_subtask(run_id, cmd["subtask_id"])
                elif command == "cancel_run":
                    rm.cancel_run(run_id)
                elif command == "inject_context":
                    rm.inject_context(run_id, cmd["subtask_id"], cmd.get("context", ""))
                else:
                    logger.warning(f"Unknown WS command: {command!r}")

            except Exception as e:
                logger.warning(f"WS command error: {e}")

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected from run {run_id}")
    finally:
        if run:
            run.ws_clients.discard(websocket)


# ─────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
