#!/bin/bash
# ==============================================================================
# SLURM Job Script: Launch vLLM Server on NVIDIA A100 (80GB)
# Model: Qwen/Qwen2.5-Coder-32B-Instruct (unquantized BF16)
# ==============================================================================
#SBATCH --job-name=vllm_qwen32b
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=bench/slurm/logs/vllm_%j.log
#SBATCH --error=bench/slurm/logs/vllm_%j.err

set -eo pipefail

# ── Environment Configuration ────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/users/jerryma/hf_cache}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export VIRTUAL_ENV="/users/jerryma/.venvs/subagent_a100"
source "${VIRTUAL_ENV}/bin/activate"

# Determine host and port
HOST_IP=$(hostname -I | awk '{print $1}')
ENDPOINT_FILE="bench/slurm/vllm_endpoint.txt"
echo "http://${HOST_IP}:${VLLM_PORT}/v1" > "${ENDPOINT_FILE}"
echo "[vLLM SLURM] Node: $(hostname), IP: ${HOST_IP}, Port: ${VLLM_PORT}"
echo "[vLLM SLURM] Registered endpoint in ${ENDPOINT_FILE}"

# ── Launch vLLM Server ───────────────────────────────────────────────────────
# Hardware: NVIDIA A100 80GB SXM4
# Precision: bfloat16 (unquantized)
# Max Model Length: 32768 tokens (full native context)
# Prefix Caching: Enabled (RadixAttention style KV-cache reuse for multi-agent workflows)
# Chunked Prefill: Enabled (maximizes GPU throughput and eliminates compute spikes)
# Concurrency: max_num_seqs 16 (high throughput parallel agent execution)
echo "[vLLM SLURM] Launching vLLM server with ${MODEL_NAME} in BF16..."

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code
