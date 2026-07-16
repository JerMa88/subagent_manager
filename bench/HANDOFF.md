# Next Agent Handoff: SWE-bench Harness Engineering

## Your Mission

You are continuing engineering work on `subagent_manager`, a multi-agent framework located at `/Users/zma/Documents/programs/subagent_manager`. A SWE-bench evaluation harness has been built and validated — the pipeline runs end-to-end and generates patches — but three fundamental limitations prevent it from achieving SOTA-level accuracy. Your job is to fix all three in order of priority, committing after each one.

**Strict workflow:** Task → Test → Debug → `git commit` → Next task. Do NOT move on until the current fix is committed and 60/60 tests pass.

---

## Current Codebase State

```
git log --oneline (most recent first):
fdfaaae feat(bench): add fallback code extraction for small models
ecf7e9f fix(critical): tool loop was blocking multi-step tool chains
d8256af fix: patch_writer custom prompt + configurable tool truncation
d1b2d70 fix: FileReaderTool path resolution + patch_writer tool enforcement
b3d971a feat(bench): add SWE-bench evaluation harness
```

**60/60 tests pass** (`python -m pytest tests/ -v`). Do not break this.

**Primary files you'll be modifying:**

| File | Role |
|------|------|
| `bench/swe_bench_tools.py` | Tool definitions: `ShellExecTool`, `FileWriterTool`, `GrepTool`, `DirectoryListTool` |
| `bench/swe_bench_agents.py` | Agent configs: `issue_analyzer`, `code_explorer`, `patch_writer`, `test_runner` |
| `bench/swe_bench_harness.py` | Harness: clone, run, extract patch, save JSONL |
| `bench/swe_bench_prompts.py` | Orchestrator and synthesis prompts |
| `src/subagent_manager/llm_client.py` | Tool loop, prompt injection, tool call parsing |
| `src/subagent_manager/tools/base.py` | `BaseTool`, `safe_execute`, `max_result_length` |
| `src/subagent_manager/tools/file_reader.py` | `FileReaderTool` with `working_dir` support |

**The LLM model being used:** `ollama/ornith` (pulled locally via Ollama). This is a small local model. It uses **prompt-based tool calling** (NOT native OpenAI function calling) because LiteLLM's Ollama tool support is broken. The tool call format injected into the system prompt is:

```json
{"name": "tool_name", "arguments": {"param": "value"}}
```

The response parser in `llm_client.py` (`_parse_tool_calls_from_content`) extracts these JSON blocks from model output.

---

## The Three Problems to Fix

### Problem 1 (HIGHEST PRIORITY): Surgical Code Editing via `str_replace`

**The root cause:** `FileWriterTool` writes the ENTIRE file content, requiring the model to reproduce a 4,000+ character file perfectly from memory. Small models hallucinate missing lines, drop imports, and corrupt indentation. In the last dry run, the model wrote to the *wrong file* entirely, deleting 824 lines from `sympy/functions/elementary/miscellaneous.py`.

**What SOTA harnesses do (SWE-agent, Moatless, Claude code):** They use a `str_replace_editor` that replaces only a specific block of text. The model provides: (a) old text, (b) new text. The tool does `file.read().replace(old, new)`. This is surgical, reliable, and requires zero full-file reproduction.

**Your implementation task:**

Create a new tool class `StrReplaceTool` in `bench/swe_bench_tools.py`:

```python
class StrReplaceTool(BaseTool):
    name = "str_replace"
    description = (
        "Replace an exact string in a file with new content. "
        "This is the PRIMARY way to fix bugs. Provide the exact "
        "old_str to replace (must match exactly, including indentation) "
        "and the new_str to replace it with."
    )
    parameters = [
        ToolParameter(name="path", type="string", description="File path relative to repo root."),
        ToolParameter(name="old_str", type="string", description="The exact string to replace. Must match exactly, including indentation and newlines."),
        ToolParameter(name="new_str", type="string", description="The replacement string."),
    ]
    
    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = working_dir
    
    async def execute(self, **kwargs) -> str:
        # 1. Resolve path using working_dir (same pattern as FileReaderTool)
        # 2. Read the file
        # 3. Check old_str appears EXACTLY once (return error if 0 or 2+ matches)
        # 4. Replace and write back
        # 5. Return confirmation: "Replaced 1 occurrence in path/to/file.py\n--- old ---\n+++ new ---"
```

Also create a companion `ViewFileTool` (exposes read_file with line range — name it `view_file` to match SWE-agent conventions that models trained on SWE-agent trajectories expect):

```python
class ViewFileTool(BaseTool):
    name = "view_file"
    description = "View file contents with optional line range."
    parameters = [
        ToolParameter(name="path", ...),
        ToolParameter(name="start_line", type="integer", required=False),
        ToolParameter(name="end_line", type="integer", required=False),
    ]
```

**Wire it into `swe_bench_agents.py`:** The `patch_writer` should use `[str_replace, view_file, grep_search, shell_exec]` as its tool set. Remove `write_file` from `patch_writer` — it is too destructive for small models.

**Update the `patch_writer` system prompt** in `swe_bench_agents.py` to say:

```
CRITICAL: To fix a bug, use this EXACT workflow:
1. Call view_file to read the relevant section (use start_line/end_line to zoom in)
2. Identify the exact lines that need changing
3. Call str_replace with old_str=<exact current code> and new_str=<corrected code>
4. Do NOT rewrite the entire file. Only change what is broken.

Common mistake: old_str must match EXACTLY including all whitespace and indentation.
Copy it directly from view_file output. Do not paraphrase.
```

---

### Problem 2 (HIGH PRIORITY): Context Budget Management

**The root cause:** The prompt-based tool loop in `llm_client.py` appends every message to the conversation. After 6 iterations with a 4,000-char file read, the model is processing 14,000+ chars of history. With `max_tokens=4096` on the LLM, it runs out of context mid-thought, producing empty responses (we see `WARN Empty response with 119 completion tokens` in the logs repeatedly). This will collapse completely on hard SWE-bench instances requiring 8-10 file reads.

**Your implementation task:**

In `llm_client.py`, in the `run_tool_loop_prompt_based` method (near line 597), add a **sliding window** strategy. Before each LLM call, measure total conversation chars and if over a threshold, prune the middle keeping: (a) the system prompt, (b) a one-line summary of pruned content, (c) the last 4 message pairs (2 rounds of tool call + result). Log it at VERBOSE1.

```python
MAX_HISTORY_CHARS = 10_000  # configurable via class attribute

total_chars = sum(len(m.get("content", "")) for m in conversation)
if total_chars > MAX_HISTORY_CHARS and len(conversation) > 6:
    # Always keep: [0]=system, [-6:]=last 3 message pairs
    pruned = conversation[1:-6]
    pruned_summary = f"[Context pruned: {len(pruned)} messages summarized to save space. Key findings from prior tool calls are in the most recent messages.]"
    conversation = [conversation[0], {"role": "user", "content": pruned_summary}] + conversation[-6:]
    logger.log(VERBOSE1, f"[LLM] Context pruned: {total_chars} → {sum(len(m.get('content','')) for m in conversation)} chars")
```

Also increase `max_answer_tokens` for all SWE-bench agents from 1024 to 2048 in `swe_bench_agents.py`, and add a `WARN` log when any agent uses more than 80% of its `max_tool_iterations` budget.

---

### Problem 3 (MEDIUM PRIORITY): Reproduction-Driven Patch Validation

**The root cause:** The current pipeline is zero-shot — it writes a patch and assumes it works. SOTA harnesses use a **reproduce-first** loop: write a failing test that demonstrates the bug, apply the patch, then run the test again to confirm it passes.

**Your implementation task:**

Add a `reproducer` agent to `swe_bench_agents.py`:

```python
SubAgentConfig(
    name="reproducer",
    description="Creates and runs a minimal Python script that reproduces the bug.",
    tools=[file_writer, shell_exec, view_file],
    system_prompt="""You write a reproduction script that demonstrates a bug.

WORKFLOW:
1. Write a minimal Python script to /tmp/reproduce.py that:
   - Imports the relevant module
   - Calls the buggy function
   - Prints 'BUG REPRODUCED' if the bug is present
   - Prints 'BUG FIXED' if the behavior is correct
2. Run: shell_exec('python /tmp/reproduce.py')
3. Confirm output contains 'BUG REPRODUCED'
4. Report what you found.
"""
)
```

Update the orchestrator prompt in `bench/swe_bench_prompts.py` to produce a 5-step plan:
1. `issue_analyzer` → understand the bug
2. `reproducer` → write and run reproduce.py (confirm BUG REPRODUCED)
3. `code_explorer` → find the exact file/function
4. `patch_writer` → apply surgical str_replace fix
5. `test_validator` (existing `test_runner`) → run `python /tmp/reproduce.py` and confirm BUG FIXED

Update `run_instance()` in `swe_bench_harness.py` to run `python /tmp/reproduce.py` after patching and log the result to `RunResult.diagnosis` as `REPRODUCE_PASS` or `REPRODUCE_FAIL`.

---

## Verification: How to Test Each Fix

After each fix, run:
```bash
cd /Users/zma/Documents/programs/subagent_manager
python -m pytest tests/ -v --tb=short 2>&1 | tail -10
```

Then validate end-to-end:
```bash
python bench/swe_bench_harness.py \
    --model ollama/ornith \
    --instance-ids sympy__sympy-15345 \
    --verbose 1 \
    --output bench/results/test_predictions.jsonl \
    --work-dir bench/repos 2>&1
```

**Expected signals after each fix:**
- Problem 1: `[LLM] Parsed 1 tool call(s): ['str_replace']` appears in the log
- Problem 2: `[LLM] Context pruned: XXXX → YYYY chars` appears; no more `WARN Empty response`
- Problem 3: 5 subtasks in plan; `REPRODUCE_PASS` or `REPRODUCE_FAIL` in summary

---

## Key Architecture Constraints

1. **Do NOT use LiteLLM's native tool calling for Ollama.** It is broken. All tool dispatch goes through the prompt-based loop in `llm_client.py`. The tool JSON format is parsed by `_parse_tool_calls_from_content`.

2. **`working_dir` must be threaded through ALL new tools** exactly as done in `FileReaderTool` — relative paths must resolve against `repo_dir`, not `os.getcwd()`.

3. **`BaseTool.max_result_length`** is the per-instance truncation limit for `safe_execute()`. `FileReaderTool` is currently set to 12,000 chars via `file_reader.max_result_length = 12000`. New tools should default to 8,000.

4. **`SubAgentConfig.system_prompt`** overrides the default subagent prompt entirely. Use it for `patch_writer` and `reproducer` — the default prompt says "be concise, plain text" which fights against tool usage.

5. **The predictions JSONL format** must remain: `{"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}` one per line. This is the SWE-bench official submission format consumed by `swe-bench evaluate`.

6. **The bench gitignore** at `bench/.gitignore` already excludes `bench/repos/` and `bench/results/`. Do not commit cloned repos.
