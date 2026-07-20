# Handoff: SWE-Bench Hierarchical Agent Development

Hello! You are picking up exactly where the previous agent left off in building and evaluating a hierarchical multi-agent framework on SWE-bench. 

## Current Implementation State
We have been developing a custom hierarchical agent framework (`subagent_manager`) designed to improve upon standard single-agent code generation.

### What We've Accomplished
- **SWE-bench Harness (`bench/swe_bench_harness.py`)**: Built a robust test harness that automatically clones SWE-bench repos, resets checkout states accurately (`git reset --hard HEAD`), and manages isolated execution environments.
- **Concurrency & Sandboxing**: Modified the harness to use parameterized paths (`/tmp/repo_{instance_id}` and `/tmp/reproduce_{instance_id}.py`) so mass parallel inference doesn't suffer from cross-contamination or race conditions.
- **Agent Tooling**: 
  - Overhauled `patch_writer` and its `str_replace` tool logic to gracefully strip absolute path hallucinations and resolve correct relative paths within the repo.
  - Built a fallback code extraction routine `_try_extract_and_write_code` that surgically extracts python code blocks from agent prose when the `str_replace` JSON tool call fails.
- **Evaluation Scripts (`bench/eval/compare.py`)**: A script designed to run both the hierarchical pipeline and a baseline single-agent pipeline head-to-head on the same instance, logging metrics and patches.

### Key Files & Locations
- **`bench/swe_bench_harness.py`**: The core execution engine. Handles Git operations, repo symlinking, instantiation of subagents, orchestrator running, fallback patch extraction, and objective test evaluation (`test_runner`).
- **`bench/swe_bench_agents.py`**: Defines the subagents in our current hierarchy (issue_analyzer, code_explorer, patch_writer, reproducer, test_runner) and wires their tool access.
- **`bench/swe_bench_prompts.py`**: Contains the system prompts for the orchestrator and the synthesis phase.
- **`bench/swe_bench_tools.py`**: Contains custom tools provided to the agents, including the highly-tuned `StrReplaceTool`.
- **`bench/eval/compare.py`**: Runs the comparative baseline vs hierarchical evaluation.
- **`src/subagent_manager/llm_client.py`**: Handles LLM interaction. We recently fixed a re-prompting bug here to use a generic JSON placeholder instead of hallucinating specific tool names, significantly improving tool calling compliance.

---

## 🎯 Directives for You (The New Agent)

You have access to an RTX 5080 GPU. Your overarching goal is to scale up our inference, expand the hierarchy, and make a critical architectural decision based on the results.

Please execute the following roadmap:

### 1. Set Up Inference Environment (vLLM & NVFP4)
You will need to run LLMs locally on the 5080.
- Install and configure a `vLLM` environment optimized for your hardware.
- Specifically, you should configure it to inference `nvfp4` (NVIDIA FP4) quantized models, taking advantage of the Blackwell/Ada hardware capabilities to maximize throughput and context window size.
- Ensure the `llm_client.py` or harness scripts are pointing to the correct local vLLM API endpoint.

### 2. Implement 3-Level Hierarchical Planning
Currently, the manager orchestrates a flat set of specialized subagents. Expand this into a true **3-level hierarchy**:
- **Level 1 (Singleton High Manager)**: Responsible for the global plan, allocating broad sub-tasks, and final synthesis. 
- **Level 2 (Mid-Level Managers)**: Responsible for specific domains (e.g., "Testing & Reproduction Manager", "Code Modification Manager").
- **Level 3 (Worker Subagents)**: Spawned by Level 2 managers. For instance, the "Testing Manager" might spawn a "Test Generator Subagent" and a "Test Execution Subagent". 

Modify `subagent_manager` and `swe_bench_agents.py` to support this dynamic multi-layered spawning. Ensure the Level 1 manager enforces a strict requirement to independently test the code via its subagents before trusting the patch.

### 3. Evaluate & Make a Pivot Decision
Run mass evaluations on SWE-bench using `compare.py` to test your 3-level hierarchy against the single-agent baseline.
Analyze the results closely. 
- **Decision Point**: Does our custom `subagent_manager` framework yield a statistically significant improvement justifying its maintenance? 
- **Forking `cline`**: If the custom framework is too brittle or tool compliance is too low, **fork `cline`** (a proven, robust coding agent system). Modify the `cline` fork to implement the 3-level agent hierarchy. Test this modified `cline` on SWE-bench to see if the combination of a proven foundation + hierarchical planning achieves state-of-the-art results.

Good luck! Start by provisioning the vLLM environment.
