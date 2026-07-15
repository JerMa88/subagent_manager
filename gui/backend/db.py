"""
SQLite persistence layer for orchestration run history.

Stores runs, their events, and full results so the user
can browse past orchestrations from the GUI sidebar.

Schema:
  runs:    id, goal, status, created_at, completed_at, config_json, plan_json, result_json
  events:  id, run_id, type, timestamp, subtask_id, agent_name, data_json
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

DB_PATH = Path(__file__).parent / "runs.db"

# ─────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    created_at  TEXT NOT NULL,
    completed_at TEXT,
    config_json TEXT,
    plan_json   TEXT,
    result_json TEXT
);
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    type        TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    subtask_id  INTEGER,
    agent_name  TEXT,
    data_json   TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""

_CREATE_EVENTS_IDX = """
CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id);
"""


async def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist. Called once at startup."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_CREATE_RUNS)
        await db.execute(_CREATE_EVENTS)
        await db.execute(_CREATE_EVENTS_IDX)
        await db.commit()


# ─────────────────────────────────────────────────────────────────────
# Run CRUD
# ─────────────────────────────────────────────────────────────────────

async def create_run(
    run_id: str,
    goal: str,
    config: dict[str, Any],
    db_path: Path = DB_PATH,
) -> None:
    """Insert a new run row at start of orchestration."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO runs (id, goal, status, created_at, config_json) "
            "VALUES (?, ?, 'running', ?, ?)",
            (run_id, goal, now, json.dumps(config)),
        )
        await db.commit()


async def update_run_plan(
    run_id: str,
    plan: list[dict],
    db_path: Path = DB_PATH,
) -> None:
    """Set the plan after the orchestrator returns it."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE runs SET plan_json = ? WHERE id = ?",
            (json.dumps(plan), run_id),
        )
        await db.commit()


async def complete_run(
    run_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Mark a run as completed/failed/cancelled."""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE runs SET status = ?, completed_at = ?, result_json = ? WHERE id = ?",
            (status, now, json.dumps(result) if result else None, run_id),
        )
        await db.commit()


async def get_runs(
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch recent runs (summary only, no events)."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, goal, status, created_at, completed_at "
            "FROM runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_run(
    run_id: str,
    db_path: Path = DB_PATH,
) -> dict[str, Any] | None:
    """Fetch a single run with plan and result."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    d = dict(row)
    # Deserialize JSON fields
    for key in ("config_json", "plan_json", "result_json"):
        val = d.pop(key, None)
        field = key.replace("_json", "")
        d[field] = json.loads(val) if val else None
    return d


# ─────────────────────────────────────────────────────────────────────
# Event storage
# ─────────────────────────────────────────────────────────────────────

async def save_event(
    run_id: str,
    event_type: str,
    timestamp: str,
    data: dict[str, Any],
    subtask_id: int | None = None,
    agent_name: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Persist a single event for a run."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO events (run_id, type, timestamp, subtask_id, agent_name, data_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, event_type, timestamp, subtask_id, agent_name, json.dumps(data)),
        )
        await db.commit()


async def get_events(
    run_id: str,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch all events for a run in chronological order."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, run_id, type, timestamp, subtask_id, agent_name, data_json "
            "FROM events WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["data"] = json.loads(d.pop("data_json") or "{}")
        result.append(d)
    return result
