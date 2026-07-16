"""
SubAgentManager — the conductor of the orchestration.

This is the main entry point for the framework. It:
1. Takes a high-level goal from the user
2. Uses an LLM to decompose it into atomic subtasks
3. Delegates each subtask to an isolated subagent
4. Collects results and synthesizes a final answer

The manager enforces SHORT-HORIZON reasoning:
- The manager ONLY plans and synthesizes — it never does direct work
- Each subagent gets ONE task in a FRESH context
- Subagent answers are concise and grounded
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from subagent_manager.logging_config import (
    configure_logging,
    format_tokens,
    log_phase,
    truncate_for_log,
    VERBOSE1,
    VERBOSE2,
)

from subagent_manager.events import Event, EventBus, EventType
from subagent_manager.llm_client import LLMClient
from subagent_manager.prompts.orchestrator import (
    build_orchestrator_system_prompt,
    build_synthesis_prompt,
)
from subagent_manager.strategies.adaptive import AdaptiveStrategy
from subagent_manager.strategies.base import BaseStrategy, ExecutionPlan, SubtaskDef
from subagent_manager.strategies.parallel import ParallelStrategy
from subagent_manager.strategies.sequential import SequentialStrategy
from subagent_manager.subagent import SubAgent, SubAgentConfig, SubAgentResult
from subagent_manager.tools.file_reader import FileReaderTool
from subagent_manager.tools.python_exec import PythonExecTool
from subagent_manager.tools.url_reader import URLReaderTool
from subagent_manager.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

# Default agent configurations that cover common use cases
DEFAULT_AGENTS: list[SubAgentConfig] = [
    SubAgentConfig(
        name="researcher",
        description=(
            "Searches the web and reads URLs to find factual information. "
            "Use for any task that requires current data, facts, statistics, "
            "or information from the internet."
        ),
        tools=[WebSearchTool(), URLReaderTool()],
    ),
    SubAgentConfig(
        name="analyzer",
        description=(
            "Analyzes information, compares options, evaluates tradeoffs, "
            "and draws conclusions. Use for tasks that require reasoning "
            "about provided context — but NOT for gathering new information."
        ),
        tools=[],  # Pure reasoning — no tools needed
    ),
    SubAgentConfig(
        name="coder",
        description=(
            "Executes Python code for calculations, data processing, "
            "and analysis. Also reads files for code review tasks."
        ),
        tools=[PythonExecTool(), FileReaderTool()],
    ),
    SubAgentConfig(
        name="verifier",
        description=(
            "Fact-checks claims by searching the web for corroborating "
            "or contradicting evidence. Use to verify information from "
            "other subtasks."
        ),
        tools=[WebSearchTool(), URLReaderTool()],
    ),
]


@dataclass
class ManagerResult:
    """
    The final result from the SubAgentManager.

    Contains the synthesized answer plus full metadata about
    the orchestration process.
    """

    answer: str
    """The final synthesized answer."""

    subtask_results: list[SubAgentResult] = field(default_factory=list)
    """Individual results from each subagent."""

    plan: list[dict[str, Any]] = field(default_factory=list)
    """The decomposition plan created by the orchestrator."""

    total_tokens: int = 0
    """Total tokens consumed across all agents."""

    total_tool_calls: int = 0
    """Total tool calls across all agents."""

    sources: list[str] = field(default_factory=list)
    """All sources discovered during execution."""


class SubAgentManager:
    """
    The conductor. Plans tasks, delegates to subagents, synthesizes results.

    This is the main entry point for the framework.

    Usage:
        manager = SubAgentManager(model="ollama/qwen3")
        result = manager.run_sync("What are the latest advances in quantum computing?")
        print(result.answer)
    """

    def __init__(
        self,
        model: str = "ollama/qwen3",
        subagents: list[SubAgentConfig] | None = None,
        strategy: str = "adaptive",
        max_subtasks: int = 10,
        api_key: str | None = None,
        api_base: str | None = None,
        orchestrator_model: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SubAgentManager.

        Args:
            model: Default LLM model for all agents (LiteLLM format).
                Examples: "ollama/qwen3", "gpt-4o-mini", "gemini/gemini-2.5-flash"
            subagents: Custom subagent configurations. If None, uses defaults.
            strategy: Execution strategy: "parallel", "sequential", or "adaptive".
            max_subtasks: Maximum subtasks the orchestrator can create.
            api_key: API key (optional, can use env vars).
            api_base: Custom API base URL (for Ollama, etc.).
            orchestrator_model: Override model for the orchestrator (can be
                smarter/larger than subagent models).
            verbose: Enable debug logging.
        """
        # Configure graduated verbose logging
        self._verbosity = verbose if isinstance(verbose, int) else (1 if verbose else 0)
        if self._verbosity > 0:
            configure_logging(verbosity=self._verbosity)

        self.model = model
        self.max_subtasks = max_subtasks

        # LLM clients
        self.llm_client = LLMClient(
            model=model,
            api_key=api_key,
            api_base=api_base,
        )
        self.orchestrator_client = (
            LLMClient(model=orchestrator_model, api_key=api_key, api_base=api_base)
            if orchestrator_model
            else self.llm_client
        )

        # Subagent configs and instances
        self.agent_configs = subagents or DEFAULT_AGENTS
        self.agents: dict[str, SubAgent] = {}
        for config in self.agent_configs:
            self.agents[config.name] = SubAgent(
                config=config,
                llm_client=self.llm_client,
            )

        # Execution strategy
        self.strategy = self._get_strategy(strategy)

        # Log full configuration at init
        logger.log(VERBOSE1, "[ORCHESTRATOR] SubAgentManager initialized:")
        logger.log(VERBOSE1, f"[ORCHESTRATOR]   model={self.model}")
        if orchestrator_model:
            logger.log(VERBOSE1, f"[ORCHESTRATOR]   orchestrator_model={orchestrator_model}")
        logger.log(VERBOSE1, f"[ORCHESTRATOR]   strategy={type(self.strategy).__name__}")
        logger.log(VERBOSE1, f"[ORCHESTRATOR]   max_subtasks={self.max_subtasks}")
        logger.log(VERBOSE1, f"[ORCHESTRATOR]   verbosity={self._verbosity}")
        for cfg in self.agent_configs:
            tool_names = [t.name for t in cfg.tools]
            logger.log(
                VERBOSE1,
                f"[ORCHESTRATOR]   agent '{cfg.name}': "
                f"tools={tool_names}, max_tool_iter={cfg.max_tool_iterations}, "
                f"max_answer_tokens={cfg.max_answer_tokens}, temp={cfg.temperature}",
            )

    def _get_strategy(self, name: str) -> BaseStrategy:
        """Get the execution strategy by name."""
        strategies: dict[str, BaseStrategy] = {
            "parallel": ParallelStrategy(),
            "sequential": SequentialStrategy(),
            "adaptive": AdaptiveStrategy(),
        }
        if name not in strategies:
            raise ValueError(
                f"Unknown strategy '{name}'. Choose from: {list(strategies.keys())}"
            )
        return strategies[name]

    async def run(
        self,
        goal: str,
        context: str = "",
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ManagerResult:
        """
        Execute a complex goal through subagent orchestration.

        This is the main method. It:
        1. Uses the orchestrator LLM to decompose the goal into subtasks
        2. Executes subtasks via the selected strategy
        3. Synthesizes results into a final answer

        Args:
            goal: The high-level goal or question from the user.
            context: Optional additional context for the orchestrator.
            event_bus: Optional event bus for GUI real-time streaming.
            pause_events: Per-subtask asyncio.Event for pause/resume.
            cancel_event: Global cancel signal.

        Returns:
            ManagerResult with the final answer and full metadata.
        """
        logger.log(VERBOSE1, f"[ORCHESTRATOR] ═══ Starting orchestration ═══")
        logger.log(VERBOSE1, f"[ORCHESTRATOR] Goal: {goal}")
        if context:
            logger.log(VERBOSE1, f"[ORCHESTRATOR] Context provided ({len(context)} chars)")
            logger.log(VERBOSE2, f"[ORCHESTRATOR] Context: {context}")

        orchestration_t0 = time.monotonic()

        if event_bus:
            event_bus.emit(Event(
                type=EventType.ORCHESTRATION_STARTED,
                data={"goal": goal, "model": self.model},
            ))

        # Step 1: Plan — decompose into subtasks
        with log_phase(logger, "Planning phase", "[PLAN]") as plan_meta:
            plan, raw_plan = await self._plan(goal, context)
            plan_meta["subtasks"] = len(plan.subtasks)

        if not plan.subtasks:
            # Fallback: if planning fails, use a single researcher
            logger.warning("[PLAN] Planning produced no subtasks. Using single-researcher fallback.")
            plan = ExecutionPlan(subtasks=[
                SubtaskDef(id=1, task=goal, agent_name="researcher"),
            ])
            raw_plan = [{"id": 1, "task": goal, "agent": "researcher"}]

        logger.log(VERBOSE1, f"[PLAN] Created {len(plan.subtasks)} subtasks:")
        for st in plan.subtasks:
            deps_str = f" depends_on={st.depends_on}" if st.depends_on else " (independent)"
            logger.log(
                VERBOSE1,
                f"[PLAN]   #{st.id} → agent='{st.agent_name}'{deps_str}: "
                f"{truncate_for_log(st.task, 120)}",
            )

        if event_bus:
            event_bus.emit(Event(
                type=EventType.PLAN_CREATED,
                data={"plan": raw_plan, "subtask_count": len(plan.subtasks)},
            ))

        # Step 2: Execute — run subtasks via strategy
        with log_phase(logger, "Execution phase", "[STRATEGY]") as exec_meta:
            subtask_results = await self.strategy.execute(
                plan, self.agents,
                event_bus=event_bus,
                pause_events=pause_events,
                cancel_event=cancel_event,
            )
            exec_meta["completed"] = len(subtask_results)
            exec_meta["succeeded"] = sum(1 for r in subtask_results if r.success)
            exec_meta["failed"] = sum(1 for r in subtask_results if not r.success)

        # Step 3: Synthesize — combine results into final answer
        if event_bus:
            event_bus.emit(Event(
                type=EventType.SYNTHESIS_STARTED,
                data={"subtask_count": len(subtask_results)},
            ))

        with log_phase(logger, "Synthesis phase", "[SYNTHESIS]") as synth_meta:
            final_answer = await self._synthesize(goal, subtask_results)
            synth_meta["answer_length"] = len(final_answer)

        if event_bus:
            event_bus.emit(Event(
                type=EventType.SYNTHESIS_COMPLETED,
                data={"answer_length": len(final_answer)},
            ))

        # Aggregate metadata
        all_sources: list[str] = []
        total_tokens = 0
        total_tool_calls = 0
        for r in subtask_results:
            total_tokens += r.tokens_used
            total_tool_calls += r.tool_calls_made
            for s in r.sources:
                if s not in all_sources:
                    all_sources.append(s)

        result = ManagerResult(
            answer=final_answer,
            subtask_results=subtask_results,
            plan=raw_plan,
            total_tokens=total_tokens,
            total_tool_calls=total_tool_calls,
            sources=all_sources,
        )

        if event_bus:
            event_bus.emit(Event(
                type=EventType.ORCHESTRATION_COMPLETED,
                data={
                    "total_tokens": total_tokens,
                    "total_tool_calls": total_tool_calls,
                    "sources_count": len(all_sources),
                },
            ))

        total_elapsed = time.monotonic() - orchestration_t0
        logger.log(
            VERBOSE1,
            f"[ORCHESTRATOR] ═══ Orchestration complete ═══  "
            f"wall={total_elapsed:.1f}s  tokens={total_tokens:,}  "
            f"tool_calls={total_tool_calls}  sources={len(all_sources)}",
        )

        return result

    def run_sync(self, goal: str, context: str = "") -> ManagerResult:
        """
        Synchronous wrapper for run().

        Convenience method for scripts and notebooks that don't
        want to deal with async/await.

        Args:
            goal: The high-level goal or question.
            context: Optional additional context.

        Returns:
            ManagerResult with the final answer and metadata.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g., Jupyter)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.run(goal, context))
                return future.result()
        else:
            return asyncio.run(self.run(goal, context))

    async def run_with_events(
        self,
        goal: str,
        context: str = "",
        event_bus: EventBus | None = None,
        pause_events: dict[int, asyncio.Event] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ManagerResult:
        """
        Execute a goal with event streaming support for the GUI.

        Identical to run() but explicitly exposes event_bus, pause_events,
        and cancel_event for the FastAPI backend to wire up.

        Args:
            goal: The high-level goal or question.
            context: Optional additional context.
            event_bus: Event bus to emit real-time events to.
            pause_events: Dict mapping subtask_id -> asyncio.Event for pause/resume.
            cancel_event: Global cancel signal (set to cancel entire orchestration).

        Returns:
            ManagerResult with the final answer and metadata.
        """
        try:
            return await self.run(
                goal=goal,
                context=context,
                event_bus=event_bus,
                pause_events=pause_events,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            if event_bus:
                event_bus.emit(Event(
                    type=EventType.ORCHESTRATION_CANCELLED,
                    data={"goal": goal},
                ))
            raise
        except Exception as e:
            if event_bus:
                event_bus.emit(Event(
                    type=EventType.ORCHESTRATION_FAILED,
                    data={"error": str(e)},
                ))
            raise

    async def _plan(
        self, goal: str, context: str = ""
    ) -> tuple[ExecutionPlan, list[dict[str, Any]]]:
        """
        Use the orchestrator LLM to decompose the goal into subtasks.

        Returns:
            Tuple of (ExecutionPlan, raw plan data for metadata).
        """
        # Build the orchestrator prompt
        agent_descriptions = [
            {"name": c.name, "description": c.description}
            for c in self.agent_configs
        ]
        system_prompt = build_orchestrator_system_prompt(
            available_agents=agent_descriptions,
            max_subtasks=self.max_subtasks,
        )

        user_content = f"## GOAL\n\n{goal}"
        if context:
            user_content = f"## ADDITIONAL CONTEXT\n\n{context}\n\n{user_content}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.log(VERBOSE2, f"[PLAN] Orchestrator system prompt ({len(system_prompt)} chars):\n{system_prompt}")
        logger.log(VERBOSE2, f"[PLAN] User content ({len(user_content)} chars):\n{user_content}")

        result = await self.orchestrator_client.complete(
            messages=messages,
            max_tokens=2048,
        )

        logger.log(
            VERBOSE1,
            f"[PLAN] Orchestrator LLM response: {len(result.content)} chars, "
            f"{format_tokens(result.usage)}",
        )
        logger.log(VERBOSE2, f"[PLAN] Raw orchestrator response:\n{result.content}")

        # Parse the JSON plan
        plan_data = self._parse_plan_json(result.content)

        subtasks = []
        for item in plan_data:
            subtasks.append(
                SubtaskDef(
                    id=item.get("id", len(subtasks) + 1),
                    task=item.get("task", ""),
                    agent_name=item.get("agent", "researcher"),
                    depends_on=item.get("depends_on", []),
                    context=item.get("context", ""),
                )
            )

        return ExecutionPlan(subtasks=subtasks), plan_data

    def _parse_plan_json(self, text: str) -> list[dict[str, Any]]:
        """
        Parse the orchestrator's JSON plan from its response.

        Handles various formats: pure JSON, JSON in code blocks,
        or JSON embedded in text.
        """
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "plan" in data:
                logger.log(VERBOSE1, "[PARSE] Plan JSON parsed via: direct JSON (dict with 'plan' key)")
                return data["plan"]
            if isinstance(data, list):
                logger.log(VERBOSE1, "[PARSE] Plan JSON parsed via: direct JSON (array)")
                return data
        except json.JSONDecodeError:
            logger.log(VERBOSE2, "[PARSE] Direct JSON parse failed, trying code blocks...")

        # Try extracting JSON from code blocks
        json_blocks = re.findall(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
        logger.log(VERBOSE2, f"[PARSE] Found {len(json_blocks)} code block(s) in response")
        for i, block in enumerate(json_blocks):
            try:
                data = json.loads(block)
                if isinstance(data, dict) and "plan" in data:
                    logger.log(VERBOSE1, f"[PARSE] Plan JSON parsed via: code block #{i+1} (dict with 'plan' key)")
                    return data["plan"]
                if isinstance(data, list):
                    logger.log(VERBOSE1, f"[PARSE] Plan JSON parsed via: code block #{i+1} (array)")
                    return data
            except json.JSONDecodeError:
                logger.log(VERBOSE2, f"[PARSE] Code block #{i+1} is not valid JSON")
                continue

        # Try finding JSON object/array in text
        logger.log(VERBOSE2, "[PARSE] Trying embedded JSON extraction...")
        for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict) and "plan" in data:
                        logger.log(VERBOSE1, "[PARSE] Plan JSON parsed via: embedded JSON object")
                        return data["plan"]
                    if isinstance(data, list):
                        logger.log(VERBOSE1, "[PARSE] Plan JSON parsed via: embedded JSON array")
                        return data
                except json.JSONDecodeError:
                    continue

        logger.warning(f"[PARSE] Could not parse plan JSON from response ({len(text)} chars): {text[:300]}...")
        return []

    async def _synthesize(
        self, goal: str, results: list[SubAgentResult]
    ) -> str:
        """
        Synthesize subagent results into a final answer.

        The synthesis prompt instructs the LLM to combine results
        without adding new information — it's a reducer, not a generator.
        """
        result_dicts = [
            {
                "task": r.task,
                "agent": r.agent_name,
                "answer": r.answer,
                "sources": r.sources,
                "success": r.success,
            }
            for r in results
        ]

        logger.log(
            VERBOSE1,
            f"[SYNTHESIS] Synthesizing {len(results)} subtask results "
            f"({sum(1 for r in results if r.success)} succeeded, "
            f"{sum(1 for r in results if not r.success)} failed)",
        )

        synthesis_prompt = build_synthesis_prompt(
            original_goal=goal,
            subtask_results=result_dicts,
        )

        logger.log(VERBOSE2, f"[SYNTHESIS] Synthesis prompt ({len(synthesis_prompt)} chars):\n{synthesis_prompt}")

        messages = [
            {"role": "system", "content": synthesis_prompt},
            {
                "role": "user",
                "content": "Please synthesize the results into a final answer.",
            },
        ]

        result = await self.orchestrator_client.complete(
            messages=messages,
            max_tokens=2048,
        )

        logger.log(
            VERBOSE1,
            f"[SYNTHESIS] Synthesis LLM response: {len(result.content)} chars, "
            f"{format_tokens(result.usage)}",
        )
        logger.log(VERBOSE2, f"[SYNTHESIS] Raw synthesis response:\n{result.content}")

        return result.content
