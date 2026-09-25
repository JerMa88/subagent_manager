#!/bin/bash
# ==============================================================================
# SLURM Job Script: 2-Instance Smoke Test on A100 (80GB)
# Model: Qwen/Qwen2.5-Coder-32B-Instruct (unquantized BF16)
# Time: ~15-20 minutes
# ==============================================================================
#SBATCH --job-name=smoke_qwen32b
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=bench/slurm/logs/smoke_%j.log
#SBATCH --error=bench/slurm/logs/smoke_%j.err

set -eo pipefail

echo "=============================================================================="
echo "Starting A100 Smoke Test on $(hostname) at $(date)"
echo "=============================================================================="

export HF_HOME="${HF_HOME:-/projects/mhahsler/course_recomm/allocation001/hf_cache}"
export HF_TOKEN=$(cat /users/jerryma/.hf_token 2>/dev/null || true)
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export VLLM_PORT=8000
export VIRTUAL_ENV="/projects/mhahsler/course_recomm/allocation001/venvs/subagent_a100"
source "${VIRTUAL_ENV}/bin/activate"

PROJECT_ROOT="/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/subagent_manager"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"

OUTPUT_DIR="${PROJECT_ROOT}/bench/results"
mkdir -p "${OUTPUT_DIR}" "bench/slurm/logs"

# Launch vLLM in background
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
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code > bench/slurm/logs/vllm_smoke_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
cleanup() {
    echo "[SLURM] Tearing down vLLM server (PID ${VLLM_PID})..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
MAX_RETRIES=120
RETRY_COUNT=0
READY=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[SLURM] vLLM server is healthy!"
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[SLURM] vLLM server died! Log tail:"
        tail -n 30 "bench/slurm/logs/vllm_smoke_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $READY -ne 1 ]; then
    echo "[SLURM] Server timed out."
    exit 1
fi

echo "[SLURM] Running 2-instance smoke test..."
python bench/eval/ablation.py \
    --model "openai/${MODEL_NAME}" \
    --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
    --instance-ids astropy__astropy-12907 django__django-11099 \
    --levels 0 1 3 \
    --output "${OUTPUT_DIR}/smoke_qwen32b.jsonl" \
    --work-dir "bench/repos" \
    --verbose 1

echo "[SLURM] Smoke test completed successfully at $(date)"
