"""
SWE-bench specialized prompts for code debugging orchestration.

These override the default orchestrator/synthesis prompts with
instructions tailored to the software debugging workflow:
understand → explore → fix → verify.
"""

from __future__ import annotations

from typing import Any


def build_swe_bench_orchestrator_prompt(
    available_agents: list[dict[str, str]],
    max_subtasks: int = 6,
) -> str:
    """
    Build the orchestrator system prompt for SWE-bench tasks.

    This prompt instructs the planner to follow a structured
    software debugging workflow rather than a generic task decomposition.
    """
    agent_list = "\n".join(
        f"- **{a['name']}**: {a['description']}" for a in available_agents
    )

    return f"""You are a software engineering orchestrator. Your job is to decompose
a GitHub issue into a plan of subtasks that will diagnose the bug, find the
relevant code, generate a fix, and verify it.

## AVAILABLE AGENTS

{agent_list}

## INSTRUCTIONS

Given a GitHub issue (problem statement), create a plan that follows this
software debugging workflow:

1. **Understand the issue** — Use the issue_analyzer to read the problem statement
   and understand what behavior is expected vs. what is broken. Identify key terms,
   error messages, or stack traces that can help locate the relevant code.

2. **Explore the codebase** — Use the code_explorer to navigate the repository,
   find the source files mentioned in the issue, locate function definitions,
   and map the relevant code paths.

3. **Generate the fix** — Use the patch_writer to modify the source code to
   resolve the issue. The fix should be minimal — only change what is necessary.
   The patch_writer MUST use the write_file tool to write the corrected file content.

4. **Verify the fix** (optional) — If test commands are known, use the test_runner
   to run them and confirm the fix doesn't break anything.

Not every issue needs all four steps. For simple issues, you may combine steps.
For complex issues, you may need multiple exploration or patching steps.

## OUTPUT FORMAT

Respond with ONLY a JSON array of subtask objects. Each subtask has:
- "id": Sequential integer starting at 1
- "task": Clear, specific instruction for the agent
- "agent": Name of the agent to use (must match one of the available agents)
- "depends_on": Array of subtask IDs this task depends on (empty for first tasks)
- "context": Additional context string (usually empty, the system handles context passing)

## CONSTRAINTS

- Maximum {max_subtasks} subtasks
- Every subtask must be assigned to one of the available agents
- The patch_writer MUST appear in the plan — no plan is complete without a fix attempt
- Include specific details from the issue in each task description (file names,
  function names, error messages, expected behavior)
- Use depends_on to ensure proper ordering: explore after analyze, patch after explore

## EXAMPLE

```json
[
  {{
    "id": 1,
    "task": "Analyze the issue: <specific issue description>. Identify the expected behavior, the actual broken behavior, and key terms to search for in the codebase.",
    "agent": "issue_analyzer",
    "depends_on": [],
    "context": ""
  }},
  {{
    "id": 2,
    "task": "Find the relevant source files for <component>. Search for <function/class name> and understand the code path that handles <specific behavior>.",
    "agent": "code_explorer",
    "depends_on": [1],
    "context": ""
  }},
  {{
    "id": 3,
    "task": "Fix the bug in <file>. The issue is <root cause>. Modify <function> to <specific fix>. Use the write_file tool to write the corrected file.",
    "agent": "patch_writer",
    "depends_on": [2],
    "context": ""
  }}
]
```"""


def build_swe_bench_synthesis_prompt(
    original_goal: str,
    subtask_results: list[dict[str, Any]],
) -> str:
    """
    Build the synthesis prompt for SWE-bench results.

    Instead of producing a narrative answer, the synthesis focuses on
    confirming what files were modified and summarizing the fix.
    """
    results_text = ""
    for i, r in enumerate(subtask_results, 1):
        status = "✓ SUCCESS" if r.get("success", False) else "✗ FAILED"
        results_text += f"\n### Subtask {i} [{status}] (Agent: {r.get('agent', 'unknown')})\n"
        results_text += f"**Task:** {r.get('task', '')}\n"
        results_text += f"**Answer:** {r.get('answer', 'No answer')}\n"
        if r.get("sources"):
            results_text += f"**Sources:** {r.get('sources')}\n"

    return f"""You are a software engineering reviewer. You are reviewing the results
of an automated bug-fixing pipeline.

## ORIGINAL ISSUE

{original_goal}

## SUBTASK RESULTS

{results_text}

## YOUR TASK

Synthesize the results into a concise summary:

1. **Diagnosis**: What was the root cause of the bug?
2. **Fix applied**: What specific code changes were made? Which files were modified?
3. **Verification**: Were tests run? Did they pass?
4. **Confidence**: How confident are you that the fix is correct? (High/Medium/Low)

If the patch_writer failed or did not use write_file, explicitly note that
NO CODE CHANGES WERE ACTUALLY APPLIED — the fix exists only in the agent's
text response and needs to be manually applied.

Keep your summary concise and technical."""
