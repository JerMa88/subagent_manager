# Task Tracker

## P0 — Skeptical Senior Programmer Orchestrator

- [x] Modify `src/subagent_manager/prompts/orchestrator.py` — add VERIFICATION RULES
- [x] Modify `bench/swe_bench_prompts.py` — skeptical orchestrator prompt for SWE-bench
- [x] Add `test_generator` agent to `bench/swe_bench_agents.py`
- [x] Run `python -m pytest tests/ -v --tb=short` → 75 passed
- [x] `git commit dad6107` — feat(bench/orchestrator): skeptical orchestrator + surgical str_replace editing

## P1 — StrReplaceTool + ViewFileTool

- [x] Add `StrReplaceTool` to `bench/swe_bench_tools.py`
- [x] Add `ViewFileTool` to `bench/swe_bench_tools.py`
- [x] Wire into `bench/swe_bench_agents.py` (patch_writer retool + new system prompt)
- [x] Add tests to `tests/test_tools.py` (TestStrReplaceTool×8, TestViewFileTool×7)
- [x] Run pytest → 75 passed (committed together with P0)
- [x] `git commit dad6107` — combined with P0

## P2 — Sliding-Window Context Pruning

- [x] Add `MAX_HISTORY_CHARS` + pruning logic to `llm_client.py`
- [x] Add WARN at 80% iteration budget in `llm_client.py`
- [x] max_answer_tokens already 2048 on all agents (set in P0)
- [x] Add context pruning unit tests to `tests/test_llm_client.py` (TestContextPruning×6)
- [x] Run pytest → 81 passed
- [x] `git commit 3ad96ad` — feat(llm): sliding-window context pruning + 80% budget warning

## P3 — Reproducer Agent + Validation Loop

- [x] Add `reproducer` agent to `bench/swe_bench_agents.py` (done in P0)
- [x] Update 5-step orchestrator plan in `bench/swe_bench_prompts.py` (done in P0)
- [x] Add reproduce.py validation to `run_instance()` in `bench/swe_bench_harness.py`
- [x] Run pytest → 81 passed
- [x] `git commit e567636` — feat(bench): independent reproduce.py validation in run_instance
- [/] **Smoke test running**: `python bench/swe_bench_harness.py --model ollama/ornith --instance-ids sympy__sympy-15345 --verbose 1`
