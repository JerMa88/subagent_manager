#!/usr/bin/env bash
# =============================================================================
# run_qwen35_ablation.sh — Ablation with Qwen3.5-9B-NVFP4 using vLLM's native
# reasoning parser (--reasoning-parser qwen3).
#
# This replaces the custom vllm_server.py for Qwen3.5, which doesn't handle
# the model's <think>...</think> output correctly. vLLM's built-in serve
# command has native support for reasoning models.
#
# Usage:
#   bash bench/run_qwen35_ablation.sh
# =============================================================================
set -euo pipefail

cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="AxionML/Qwen3.5-9B-NVFP4"
OUTPUT="bench/results/ablation_qwen35_300.jsonl"

mkdir -p bench/results

# ── Kill any existing vLLM processes ──
pkill -f "vllm_server.py" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 2
pkill -9 -f "vllm_server.py" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$"); do
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmdline" | grep -qi "vllm\|EngineCore"; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 2

echo "[$(date)] Starting vLLM serve with model $MODEL and reasoning parser..."

# ── Use vLLM's native serve command with reasoning parser ──
# Key flags:
#   --reasoning-parser qwen3      : Properly separates <think> from content
#   --enable-thinking false       : DISABLED — AxionML/Qwen3.5-9B-NVFP4 produces
#                                   garbage Unicode tokens in thinking mode due to
#                                   NVFP4 quantization precision issues. The model
#                                   never exits <think> and content is always null.
#   --gpu-memory-utilization 0.75 : Fits on RTX 5080 16GB
#   --enforce-eager               : Required for NVFP4 (no CUDA graphs)
#
# IMPORTANT — CUDA_HOME must be the vllm conda env, NOT /usr/local/cuda.
# nvcc on this system lives in the conda env at:
#   /home/zma/anaconda3/envs/vllm/bin/nvcc
# /usr/local/cuda/bin/nvcc does NOT exist on this machine.
systemd-run --user --unit=vllm-qwen35 --remain-after-exit \
  --setenv=CUDA_HOME=/home/zma/anaconda3/envs/vllm \
  --setenv=PATH=/home/zma/anaconda3/envs/vllm/bin:/usr/bin:/bin \
  --setenv=LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:/home/zma/anaconda3/envs/vllm/lib64 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --working-directory=/home/zma/Documents/program/Github/subagent_manager \
  -- "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.75 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --cpu-offload-gb 4

echo "[$(date)] Waiting for vLLM to be ready (up to 600s — Triton warmup is slow on first boot)..."
for i in $(seq 1 600); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3.5"; then
        echo "[$(date)] vLLM ready after ${i}s!"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen3.5"; then
    echo "[$(date)] ERROR: vLLM not ready after 600s"
    exit 1
fi

# ── Launch ablation ──
echo "[$(date)] Starting ablation (300 instances, levels 0-4)..."
systemd-run --user --unit=qwen35-ablation --remain-after-exit \
  --setenv=PATH=/home/zma/anaconda3/envs/vllm/bin:/usr/bin:/bin \
  --setenv=PYTHONPATH=/home/zma/Documents/program/Github/subagent_manager/src:/home/zma/Documents/program/Github/subagent_manager \
  --working-directory=/home/zma/Documents/program/Github/subagent_manager \
  -- "$PYTHON" bench/eval/ablation.py \
    --model "openai/$MODEL" \
    --api-base http://localhost:8000/v1 \
    --api-key EMPTY \
    --all \
    --levels 0 1 2 3 4 \
    --output "$OUTPUT"

echo "[$(date)] Both services launched as systemd units."
echo "  Monitor: systemctl --user status vllm-qwen35 qwen35-ablation"
echo "  Logs:    journalctl --user -u qwen35-ablation -f"
echo "  Results: wc -l $OUTPUT"
