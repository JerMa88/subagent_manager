# subagent_manager

**A universal subagent orchestration framework that enhances any LLM's reasoning capabilities — especially edge/open-source models — by enforcing single-step reasoning chains through subagent delegation.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## The Problem

Large Language Models suffer from three interrelated failure modes that get worse with complexity:

1. **"Lost-in-the-Middle" Information Loss** ([Liu et al., 2024](https://arxiv.org/abs/2307.03172)): Models fail to attend to information in the middle of long contexts, showing a U-shaped performance curve where only the beginning and end of the context are reliably processed.

2. **Multi-Step Reasoning Decay**: Each additional reasoning step compounds error probability. A 5-step reasoning chain with 90% accuracy per step only has a 59% chance of reaching the correct final answer.

3. **Context Window Pollution**: Intermediate work (tool outputs, search results, scratch computations) consumes the context budget, displacing the original task and degrading overall performance.

## The Solution

Every major agentic system has independently converged on the same architectural insight:

> **Isolate subtasks into fresh context windows where each agent performs exactly ONE reasoning step.**

| System | Subagent Mechanism | Context Model |
|--------|-------------------|---------------|
| **Google Antigravity** | `invoke_subagent` | Fully isolated, async |
| **Claude Code** | `Agent` tool | Isolated, 5 levels deep |
| **GitHub Copilot** | `Task()` | Isolated git worktrees |
| **Cursor** | `Task()` | Background/foreground |
| **Google ADK** | `ParallelAgent` | Shared state, isolated history |
| **OpenAI Agents SDK** | `handoffs` | Sequential handoff |

**The problem**: None of these are portable. Each is locked into its own ecosystem.

**subagent_manager** provides a single, open-source Python package that:
- Works with **any LLM provider** via [LiteLLM](https://github.com/BerriAI/litellm) (100+ providers)
- Enforces **short-horizon reasoning**: each subagent gets ONE task, ONE context, ONE answer
- Is optimized for **edge deployment** with small models (3B-8B parameters)
- Requires **zero API keys** for web search (uses DuckDuckGo)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  User / Application              │
│                                                  │
│  manager = SubAgentManager(model="ollama/qwen3") │
│  result = manager.run_sync("Complex task...")    │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│             Orchestrator Agent                   │
│  • Decomposes goal into 2-10 atomic subtasks     │
│  • NEVER does direct work itself                 │
│  • Assigns each subtask to the best agent        │
│  • Synthesizes final answer from results         │
└──────┬──────────┬──────────┬────────────────────┘
       │          │          │
       ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ SubAgent │ │ SubAgent │ │ SubAgent │
│ FRESH    │ │ FRESH    │ │ FRESH    │
│ CONTEXT  │ │ CONTEXT  │ │ CONTEXT  │
│          │ │          │ │          │
│ 1 task   │ │ 1 task   │ │ 1 task   │
│ 1 answer │ │ 1 answer │ │ 1 answer │
│ tools ✓  │ │ tools ✓  │ │ tools ✓  │
└──────────┘ └──────────┘ └──────────┘
```

### How It Enforces Short-Horizon Reasoning

1. **The orchestrator ONLY plans** — it never attempts to solve subtasks directly
2. **Each subtask is atomic** — solvable with a single reasoning step
3. **Each subagent gets a FRESH context** — no history accumulation
4. **Answers are capped** — `max_answer_tokens=512` forces concise, focused responses
5. **Tool loops are bounded** — `max_tool_iterations=5` prevents runaway chains
6. **Synthesis is constrained** — the synthesizer combines results without adding new information

## SubAgent Command Center (GUI)

The framework includes a real-time, interactive React GUI for visualizing and controlling the orchestration process.

```bash
# Terminal 1: Start the FastAPI/WebSocket backend (SQLite persistent)
cd gui/backend
uvicorn server:app --port 8000

# Terminal 2: Start the Vite + React frontend
cd gui/frontend
npm install
npm run dev
```

**GUI Features:**
- **Visual DAG**: Watch the orchestrator's plan unfold as an animated dependency graph.
- **Agent Inspector**: Click any node to instantly view its live context window, thought iterations, tool calls, and exact token usage (context vs generation).
- **Execution Control**: Pause, resume, or cancel specific subagents mid-generation, or inject new instructions into their isolated context dynamically.
- **Run History**: Automatically persists all runs, plans, and streaming events to SQLite, allowing you to instantly replay and review any past orchestration task.

## Installation

```bash
# Clone the repository
git clone https://github.com/JerMa88/subagent_manager.git
cd subagent_manager

# Install in development mode
pip install -e ".[dev]"
```

### Prerequisites

You need at least one LLM provider:

- **Ollama (local, free)**: [Install Ollama](https://ollama.com), then `ollama pull qwen3`
- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Google Gemini**: Set `GEMINI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable
- **Any OpenAI-compatible API**: Provide `api_base` parameter

## Quick Start

### Minimal Example

```python
from subagent_manager import SubAgentManager

# Create a manager (defaults to Ollama/qwen3)
manager = SubAgentManager(model="ollama/qwen3")

# Or use OpenAI
# manager = SubAgentManager(model="gpt-4o-mini", api_key="sk-...")

# Run a complex query
result = manager.run_sync(
    "What are the three most significant breakthroughs in quantum computing "
    "from 2024-2025, and what are their practical implications?"
)

print(result.answer)
print(f"Sources: {result.sources}")
print(f"Tool calls: {result.total_tool_calls}")
```

### What Happens Under the Hood

1. **Planning**: The orchestrator decomposes the query into subtasks:
   ```json
   {
     "plan": [
       {"id": 1, "task": "Search for quantum computing breakthroughs 2024-2025", "agent": "researcher"},
       {"id": 2, "task": "Search for practical applications of recent quantum advances", "agent": "researcher"},
       {"id": 3, "task": "Verify the top 3 most impactful breakthroughs", "agent": "verifier", "depends_on": [1]},
       {"id": 4, "task": "Analyze practical implications", "agent": "analyzer", "depends_on": [1, 2, 3]}
     ]
   }
   ```

2. **Execution**: Subtasks 1 and 2 run in parallel (both independent). Subtask 3 waits for 1. Subtask 4 waits for 1, 2, and 3.

3. **Synthesis**: Results are combined into a single, grounded answer with citations.

### Custom Agents

```python
from subagent_manager import SubAgentManager, SubAgentConfig
from subagent_manager.tools import WebSearchTool, URLReaderTool, PythonExecTool

manager = SubAgentManager(
    model="gpt-4o-mini",
    subagents=[
        SubAgentConfig(
            name="web_researcher",
            description="Searches the web for current information and data",
            tools=[WebSearchTool(), URLReaderTool()],
        ),
        SubAgentConfig(
            name="data_analyst",
            description="Runs Python code for calculations and data analysis",
            tools=[PythonExecTool()],
            max_tool_iterations=3,
        ),
        SubAgentConfig(
            name="synthesizer",
            description="Combines findings into clear, structured summaries",
            tools=[],  # Pure reasoning
        ),
    ],
    strategy="adaptive",  # Auto-selects parallel vs sequential
)
```

### Hybrid Model Strategy (Edge Deployment)

Use a larger model for planning and smaller models for execution:

```python
manager = SubAgentManager(
    model="ollama/qwen3",              # Fast model for subagents
    orchestrator_model="ollama/qwen3",  # Larger model for planning
    subagents=[
        SubAgentConfig(
            name="searcher",
            description="Web search specialist",
            tools=[WebSearchTool()],
            max_tool_iterations=3,   # Fewer iterations = faster
            max_answer_tokens=256,   # Shorter answers = less generation
        ),
    ],
    max_subtasks=5,  # Limit subtasks for efficiency
)
```

## Built-in Tools

| Tool | Description | API Key Required? |
|------|-------------|-------------------|
| `WebSearchTool` | DuckDuckGo web search | **No** |
| `URLReaderTool` | Fetch & parse web pages to markdown | **No** |
| `PythonExecTool` | Sandboxed Python execution with timeout | **No** |
| `FileReaderTool` | Read local files with line ranges & sandboxing | **No** |

All tools are designed to work without any API keys, making the framework ideal for edge and offline-capable deployments.

## Execution Strategies

| Strategy | When to Use | How It Works |
|----------|-------------|--------------|
| `"adaptive"` (default) | Most cases | Auto-detects dependency structure |
| `"parallel"` | Independent tasks | Fan-out all at once, fan-in results |
| `"sequential"` | Dependent chains | Execute one-by-one, chain context |

## Comparison with Existing Frameworks

| Feature | subagent_manager | OpenAI SDK | CrewAI | LangGraph | smolagents |
|---------|-----------------|------------|--------|-----------|------------|
| **Provider agnostic** | ✅ (LiteLLM) | ✅ | ⚠️ | ⚠️ | ✅ |
| **Short-horizon enforcement** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Zero API keys needed** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Edge/Ollama optimized** | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| **Parallel execution** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Context isolation** | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Built-in tools** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Dependency tracking** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Complexity** | Low | Low | Medium | High | Very Low |

## API Reference

### `SubAgentManager`

```python
SubAgentManager(
    model="ollama/qwen3",        # LLM model (LiteLLM format)
    subagents=None,              # Custom SubAgentConfig list (or use defaults)
    strategy="adaptive",         # "parallel", "sequential", or "adaptive"
    max_subtasks=10,             # Max subtasks per plan
    api_key=None,                # API key (optional, uses env vars)
    api_base=None,               # Custom API base URL
    orchestrator_model=None,     # Override model for the planner
    verbose=False,               # Enable debug logging
)
```

### `SubAgentConfig`

```python
SubAgentConfig(
    name="researcher",           # Agent identifier
    description="...",           # What this agent does (used for routing)
    tools=[WebSearchTool()],     # Available tools
    model=None,                  # Model override (or use manager's model)
    system_prompt=None,          # Custom system prompt (or auto-generated)
    max_tool_iterations=5,       # Max tool call rounds
    max_answer_tokens=512,       # Force concise answers
    temperature=0.0,             # LLM temperature
)
```

### `ManagerResult`

```python
result = manager.run_sync("goal")
result.answer              # Final synthesized answer
result.subtask_results     # List of SubAgentResult
result.plan                # The decomposition plan
result.total_tokens        # Tokens consumed
result.total_tool_calls    # Tool calls made
result.sources             # All discovered sources
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/subagent_manager
```

## Examples

See the `examples/` directory:

- [`basic_usage.py`](examples/basic_usage.py) — Minimal example
- [`web_research.py`](examples/web_research.py) — Web research with grounding
- [`code_review.py`](examples/code_review.py) — Code analysis workflow
- [`ollama_edge.py`](examples/ollama_edge.py) — Edge deployment with Ollama

## Theoretical Foundation

This framework is grounded in research on LLM reasoning limitations:

- **"Lost in the Middle"** (Liu et al., 2024): Demonstrates that LLMs fail to attend to information in the middle of long contexts, following a U-shaped performance curve.
- **Short-horizon reasoning**: Research shows models excel at immediate, sequential tasks but degrade as dependency chains grow.
- **Context engineering**: By isolating each reasoning step into a fresh context, we eliminate positional bias and prevent information loss.
- **Agentic decomposition**: Hierarchical planning with verification loops mirrors how the best agentic systems (Antigravity, Claude Code, Copilot) handle complex tasks.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the GPL-3.0 License — see the [LICENSE](LICENSE) file for details.
