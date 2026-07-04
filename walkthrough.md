# SubAgent Manager — Walkthrough

## What Was Built

A complete, production-ready Python package that enables **any LLM** (especially edge/open-source models) to perform complex tasks through **short-horizon reasoning** via subagent delegation.

## Research Summary

Investigated how 7 major systems handle subagent orchestration:

| System | Mechanism | Key Insight Adopted |
|--------|-----------|-------------------|
| **Google Antigravity** | `invoke_subagent` | Async, non-blocking subagent lifecycle |
| **Claude Code** | `Agent` tool | Fully isolated fresh context per agent |
| **GitHub Copilot** | `Task()` + Fleet mode | Git worktree-based parallel isolation |
| **Cursor** | `Task()` | Background/foreground execution modes |
| **Google ADK** | `ParallelAgent` | Built-in parallel/sequential/loop primitives |
| **OpenAI Agents SDK** | `handoffs` | Lightweight, 4-primitive design |
| **HuggingFace smolagents** | Agents-as-tools | Code-first, minimal abstraction |

**Key finding**: All converged on **context isolation** as the solution to reasoning decay, but none are portable across providers.

## Architecture

```
User → SubAgentManager → Orchestrator LLM (plan only)
                             ↓
                      Execution Strategy
                    (parallel/sequential/adaptive)
                             ↓
                    SubAgent 1   SubAgent 2   SubAgent N
                    (FRESH ctx)  (FRESH ctx)  (FRESH ctx)
                    [tools]      [tools]      [tools]
                             ↓
                      Synthesis LLM (combine only)
                             ↓
                       Final Answer + Sources
```

## Files Created (30 total)

### Core Package (`src/subagent_manager/`)
| File | Purpose |
|------|---------|
| [__init__.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/__init__.py) | Public API exports |
| [manager.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/manager.py) | Main orchestrator (plan → delegate → synthesize) |
| [subagent.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/subagent.py) | Isolated worker execution |
| [llm_client.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/llm_client.py) | Universal LLM client via LiteLLM |

### Tools (`src/subagent_manager/tools/`)
| File | Purpose |
|------|---------|
| [base.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/tools/base.py) | Abstract tool with OpenAI schema generation |
| [web_search.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/tools/web_search.py) | DuckDuckGo search (no API key) |
| [url_reader.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/tools/url_reader.py) | HTML → markdown content extraction |
| [python_exec.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/tools/python_exec.py) | Sandboxed Python with timeout |
| [file_reader.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/tools/file_reader.py) | Local file reading with sandboxing |

### Strategies (`src/subagent_manager/strategies/`)
| File | Purpose |
|------|---------|
| [base.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/strategies/base.py) | Base strategy + ExecutionPlan + SubtaskDef |
| [parallel.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/strategies/parallel.py) | Wave-based parallel execution |
| [sequential.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/strategies/sequential.py) | Sequential chaining |
| [adaptive.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/strategies/adaptive.py) | Auto-selects best strategy |

### Prompts (`src/subagent_manager/prompts/`)
| File | Purpose |
|------|---------|
| [orchestrator.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/prompts/orchestrator.py) | Planning + synthesis prompts |
| [subagent.py](file:///Users/zma/Documents/programs/subagent_manager/src/subagent_manager/prompts/subagent.py) | Worker grounding prompts |

### Tests (`tests/`)
| File | Tests |
|------|-------|
| [test_manager.py](file:///Users/zma/Documents/programs/subagent_manager/tests/test_manager.py) | 16 tests: init, plan parsing, default agents |
| [test_subagent.py](file:///Users/zma/Documents/programs/subagent_manager/tests/test_subagent.py) | 9 tests: config, results, model overrides |
| [test_strategies.py](file:///Users/zma/Documents/programs/subagent_manager/tests/test_strategies.py) | 10 tests: plan analysis, parallel/seq/adaptive |
| [test_tools.py](file:///Users/zma/Documents/programs/subagent_manager/tests/test_tools.py) | 19 tests: schemas, execution, security, truncation |
| [test_llm_client.py](file:///Users/zma/Documents/programs/subagent_manager/tests/test_llm_client.py) | 6 tests: init, data classes, mock tools |

### Examples (`examples/`)
| File | Purpose |
|------|---------|
| [basic_usage.py](file:///Users/zma/Documents/programs/subagent_manager/examples/basic_usage.py) | Minimal quickstart |
| [web_research.py](file:///Users/zma/Documents/programs/subagent_manager/examples/web_research.py) | Grounded web research |
| [code_review.py](file:///Users/zma/Documents/programs/subagent_manager/examples/code_review.py) | Code analysis workflow |
| [ollama_edge.py](file:///Users/zma/Documents/programs/subagent_manager/examples/ollama_edge.py) | Edge deployment with hybrid models |

### Config
| File | Purpose |
|------|---------|
| [pyproject.toml](file:///Users/zma/Documents/programs/subagent_manager/pyproject.toml) | Package config, deps, tool configs |
| [README.md](file:///Users/zma/Documents/programs/subagent_manager/README.md) | Full documentation with theory, API, examples |

## Verification Results

### Tests: ✅ 60/60 passed
```
60 passed in 1.18s
```

### Lint: ✅ All checks passed
```
ruff check src/ → All checks passed!
```

### Package Installation: ✅ Successful
```
Successfully installed subagent-manager-0.1.0
```

### Import Verification: ✅ All modules load
```
Manager model: ollama/qwen3
Agents: ['researcher', 'analyzer', 'coder', 'verifier']
Strategy: AdaptiveStrategy
All 4 tools generate valid schemas
Version: 0.1.0
```

## Key Design Decisions

1. **LiteLLM for universality**: Supports 100+ providers with a single API, avoiding vendor lock-in
2. **DuckDuckGo for free search**: No API key needed, works offline after install
3. **Stateless LLM client**: Every call is independent — no history accumulation
4. **max_answer_tokens=512**: Forces concise subagent responses, preventing context pollution
5. **max_tool_iterations=5**: Hard cap prevents runaway reasoning chains
6. **Adaptive strategy as default**: Auto-analyzes dependency graph to maximize parallelism
7. **Robust JSON parsing**: Handles pure JSON, code blocks, and embedded JSON from various models
8. **Content truncation everywhere**: Tools, URLs, files all truncate long outputs to prevent context explosion
