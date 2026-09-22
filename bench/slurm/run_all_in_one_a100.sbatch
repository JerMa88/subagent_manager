#!/bin/bash
# ==============================================================================
# SLURM Job Script: All-In-One vLLM + SWE-bench Evaluation on A100 (80GB)
# Model: Qwen/Qwen2.5-Coder-32B-Instruct (unquantized BF16)
# ==============================================================================
#SBATCH --job-name=eval_qwen32b
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=bench/slurm/logs/all_in_one_%j.log
#SBATCH --error=bench/slurm/logs/all_in_one_%j.err

set -eo pipefail

echo "=============================================================================="
echo "Starting A100 All-In-One Evaluation on $(hostname) at $(date)"
echo "=============================================================================="

# ── 1. Setup Environment ─────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/projects/mhahsler/course_recomm/allocation001/hf_cache}"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export VLLM_PORT=8000
export VIRTUAL_ENV="/projects/mhahsler/course_recomm/allocation001/venvs/subagent_a100"
source "${VIRTUAL_ENV}/bin/activate"

PROJECT_ROOT="/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/subagent_manager"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"

OUTPUT_DIR="${PROJECT_ROOT}/bench/data/eval_results_qwen32b_bf16"
mkdir -p "${OUTPUT_DIR}" "bench/slurm/logs"

# ── 2. Launch Background vLLM Server ─────────────────────────────────────────
echo "[SLURM] Launching vLLM background server with ${MODEL_NAME} in BF16..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype bfloat16 \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --enforce-eager \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code > bench/slurm/logs/vllm_server_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "[SLURM] vLLM server launched with PID ${VLLM_PID}. Waiting for health check..."

cleanup() {
    echo "[SLURM] Tearing down background vLLM server (PID ${VLLM_PID})..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    echo "[SLURM] Finished cleanup at $(date)"
}
trap cleanup EXIT

# ── 3. Health Check Polling ──────────────────────────────────────────────────
MAX_RETRIES=120
RETRY_COUNT=0
READY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[SLURM] vLLM server is healthy and responding!"
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[SLURM] FATAL: vLLM server process died prematurely! Log tail:"
        tail -n 30 "bench/slurm/logs/vllm_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $((RETRY_COUNT % 6)) -eq 0 ]; then
        echo "[SLURM] Still waiting for vLLM model loading (${RETRY_COUNT}/${MAX_RETRIES})..."
    fi
done

if [ $READY -ne 1 ]; then
    echo "[SLURM] FATAL: vLLM server timed out after $((MAX_RETRIES * 5)) seconds."
    exit 1
fi

# ── 4. Execute Phase 1 Evaluation ────────────────────────────────────────────
echo "=============================================================================="
echo "[SLURM] Running Evaluation on Phase 1 Stratified Subset..."
echo "=============================================================================="

# Generate the 50-instance phase 1 set if not yet created
if [ ! -f "bench/phase1_instances.json" ]; then
    python bench/select_phase1_instances.py
fi

# Run comparison / ablation across levels 0, 1, 2, 3
python bench/eval/ablation.py \
    --model "openai/${MODEL_NAME}" \
    --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
    --instances-file "bench/phase1_instances.json" \
    --levels 0 1 2 3 \
    --output "${OUTPUT_DIR}/results_phase1.jsonl" \
    --work-dir "bench/repos" \
    --verbose 1

echo "[SLURM] Phase 1 evaluation finished successfully! Output: ${OUTPUT_DIR}/results_phase1.jsonl"

# ── 5. Generate Metrics & Summary Plots ───────────────────────────────────────
if [ -f "${OUTPUT_DIR}/results_phase1.jsonl" ]; then
    python bench/eval/plot_ablation.py \
        --results "${OUTPUT_DIR}/results_phase1.jsonl" \
        --output-dir "${OUTPUT_DIR}" || true
fi

echo "=============================================================================="
echo "All tasks finished successfully at $(date)"
echo "=============================================================================="
