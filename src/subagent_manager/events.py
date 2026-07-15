"""
Event bus for GUI integration.

Provides a lightweight, async-safe event system that the orchestration
pipeline can use to emit real-time events to subscribers (e.g., the
FastAPI WebSocket layer). No behavior changes — purely additive.

Usage:
    bus = EventBus()
    bus.subscribe(lambda e: print(e))
    bus.emit(Event(type=EventType.PLAN_CREATED, data={"plan": [...]}))
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """All event types emitted by the orchestration pipeline."""

    # Orchestration lifecycle
    ORCHESTRATION_STARTED = "orchestration_started"
    ORCHESTRATION_COMPLETED = "orchestration_completed"
    ORCHESTRATION_FAILED = "orchestration_failed"
    ORCHESTRATION_CANCELLED = "orchestration_cancelled"

    # Plan
    PLAN_CREATED = "plan_created"
    PLAN_AWAITING_REVIEW = "plan_awaiting_review"
    PLAN_APPROVED = "plan_approved"

    # Subtask lifecycle
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_PAUSED = "subtask_paused"
    SUBTASK_RESUMED = "subtask_resumed"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"
    SUBTASK_CANCELLED = "subtask_cancelled"

    # LLM calls
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"

    # Tool calls
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"

    # Synthesis
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_COMPLETED = "synthesis_completed"


@dataclass
class Event:
    """
    A single orchestration event.

    Emitted at each meaningful point in the orchestration pipeline
    and forwarded to all registered subscribers.
    """

    type: EventType
    """The event type."""

    data: dict[str, Any] = field(default_factory=dict)
    """Event-specific payload data."""

    run_id: str | None = None
    """The run ID this event belongs to."""

    subtask_id: int | None = None
    """The subtask ID this event belongs to (if applicable)."""

    agent_name: str | None = None
    """The agent name this event belongs to (if applicable)."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    """ISO 8601 timestamp of when this event occurred."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a JSON-serializable dict."""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "subtask_id": self.subtask_id,
            "agent_name": self.agent_name,
            "data": self.data,
        }


# Callback type: receives an Event, returns nothing
EventCallback = Callable[[Event], None]


class EventBus:
    """
    Lightweight, async-safe event bus.

    Thread-safe for adding/removing subscribers. Each emit() dispatches
    the event to all registered callbacks synchronously. For async
    callbacks, use `emit_async()` to schedule them on the event loop.

    This is NOT a message queue — events are not buffered or replayed.
    Subscribers only see events emitted after they subscribe.
    """

    def __init__(self) -> None:
        self._callbacks: list[EventCallback] = []
        self._run_id: str | None = None

    def set_run_id(self, run_id: str) -> None:
        """Set the run ID that will be attached to all emitted events."""
        self._run_id = run_id

    def subscribe(self, callback: EventCallback) -> None:
        """Register a callback to receive events."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Remove a previously registered callback."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def emit(self, event: Event) -> None:
        """
        Dispatch an event to all registered callbacks.

        If the callback raises, logs the error and continues to
        the next callback (one bad subscriber can't break the pipeline).
        """
        if self._run_id and event.run_id is None:
            event.run_id = self._run_id

        for callback in list(self._callbacks):
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"EventBus callback error ({type(e).__name__}): {e}")

    def emit_async(self, event: Event, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """
        Schedule an event emission on an event loop.

        Use this when emitting from a sync context that has an associated
        async event loop (e.g., from inside asyncio.gather callbacks).
        Falls back to sync emit() if no loop is available.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if loop and loop.is_running():
            loop.call_soon_threadsafe(self.emit, event)
        else:
            self.emit(event)

    def clear(self) -> None:
        """Remove all subscribers."""
        self._callbacks.clear()

    def __repr__(self) -> str:
        return f"<EventBus subscribers={len(self._callbacks)} run_id={self._run_id!r}>"
