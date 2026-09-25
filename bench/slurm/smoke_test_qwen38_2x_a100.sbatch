#!/bin/bash
# ==============================================================================
# SLURM Job Script: 1-Instance Smoke Test on 2x A100 (80GB each)
# Model: Qwen/Qwen3.8-27B (unquantized BF16) with MTP Speculative Decoding
# Time: ~15-20 minutes
# ==============================================================================
#SBATCH --job-name=smoke_qwen38
#SBATCH --account=mhahsler_course_recomm_0001
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=bench/slurm/logs/smoke_qwen38_%j.log
#SBATCH --error=bench/slurm/logs/smoke_qwen38_%j.err

set -eo pipefail

echo "=============================================================================="
echo "Starting Qwen3.8-27B 2x A100 Smoke Test on $(hostname) at $(date)"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
nvidia-smi
echo "=============================================================================="

# ── 1. Setup Environment ─────────────────────────────────────────────────────
export VIRTUAL_ENV="/projects/mhahsler/course_recomm/allocation001/venvs/qwen38_env"
source "${VIRTUAL_ENV}/bin/activate"

export NV_LIB="/projects/mhahsler/course_recomm/allocation001/venvs/qwen38_env/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV_LIB}/cu13/lib:${NV_LIB}/cuda_runtime/lib:${NV_LIB}/cublas/lib:${NV_LIB}/cudnn/lib:${NV_LIB}/cufft/lib:${NV_LIB}/curand/lib:${NV_LIB}/cusolver/lib:${NV_LIB}/cusparse/lib:${NV_LIB}/nccl/lib:${NV_LIB}/nvtx/lib:${NV_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH}"
export PATH="/projects/mhahsler/course_recomm/allocation001/venvs/qwen38_env/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-12.8"

export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

PROJECT_ROOT="/projects/mhahsler/course_recomm/allocation001/AI_Club/paper/subagent_manager"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH}"

MODEL_PATH="/work/projects/mhahsler/course_recomm/allocation001/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL_NAME="Qwen/Qwen3.8-27B"
VLLM_PORT=8000

OUTPUT_DIR="${PROJECT_ROOT}/bench/results"
mkdir -p "${OUTPUT_DIR}" "bench/slurm/logs"

# ── 2. Launch Background vLLM Server ─────────────────────────────────────────
echo "[SLURM] Launching vLLM API server with Qwen3.8-27B (TP=2, MTP, BF16, 64k context)..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "attention_backend": "FLASHINFER"}' \
    --dtype bfloat16 \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --attention-backend FLASHINFER \
    --skip-mm-profiling \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code > bench/slurm/logs/vllm_smoke_qwen38_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "[SLURM] vLLM server launched with PID ${VLLM_PID}. Awaiting health check..."

cleanup() {
    echo "[SLURM] Cleaning up: terminating background vLLM server (PID ${VLLM_PID})..."
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
    echo "[SLURM] Cleanup complete at $(date)"
}
trap cleanup EXIT

# ── 3. Health Check Polling ──────────────────────────────────────────────────
MAX_RETRIES=240
RETRY_COUNT=0
READY=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "[SLURM] vLLM server is healthy and responding!"
        READY=1
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[SLURM] FATAL: vLLM server died prematurely! Log tail:"
        tail -n 40 "bench/slurm/logs/vllm_smoke_qwen38_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $((RETRY_COUNT % 6)) -eq 0 ]; then
        echo "[SLURM] Waiting for vLLM server initialization (${RETRY_COUNT}/${MAX_RETRIES})..."
    fi
done

if [ $READY -ne 1 ]; then
    echo "[SLURM] FATAL: Server startup timed out after $((MAX_RETRIES * 5))s."
    exit 1
fi

# ── 4. Run Smoke Test ────────────────────────────────────────────────────────
echo "[SLURM] Running 1-instance smoke test (django__django-11099) across Levels 0, 1, 3..."
python bench/eval/ablation.py \
    --model "openai/${SERVED_MODEL_NAME}" \
    --api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
    --instance-ids django__django-11099 \
    --levels 0 1 3 \
    --output "${OUTPUT_DIR}/smoke_qwen38_2x_a100.jsonl" \
    --work-dir "bench/repos" \
    --verbose 1

echo "=============================================================================="
echo "[SLURM] Qwen3.8 Smoke test completed successfully at $(date)"
echo "Output saved to: ${OUTPUT_DIR}/smoke_qwen38_2x_a100.jsonl"
echo "=============================================================================="
