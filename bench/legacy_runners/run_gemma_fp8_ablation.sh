#!/usr/bin/env bash
set -euo pipefail
cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="google/gemma-4-12B-it"
OUTPUT="bench/results/ablation_gemma4_fp8_300.jsonl"
mkdir -p bench/results

systemctl --user stop vllm-qwen35 qwen35-ablation vllm-gemma4 gemma4-ablation 2>/dev/null || true
systemctl --user reset-failed 2>/dev/null || true

pkill -f "vllm_server.py" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 2
pkill -9 -f "vllm_server.py" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true

echo "[$(date)] Starting vLLM serve with model $MODEL..."

systemd-run --user --unit=vllm-gemma4 --remain-after-exit \
  --setenv=CUDA_HOME=/home/zma/anaconda3/envs/vllm \
  --setenv=PATH=/home/zma/anaconda3/envs/vllm/bin:/usr/bin:/bin \
  --setenv=LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:/home/zma/anaconda3/envs/vllm/lib64 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --working-directory=/home/zma/Documents/program/Github/subagent_manager \
  -- "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --kv-cache-dtype fp8

echo "[$(date)] Waiting for vLLM to be ready..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
        echo "[$(date)] vLLM ready after ${i}s!"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
    echo "[$(date)] ERROR: vLLM not ready."
    exit 1
fi

echo "[$(date)] Starting ablation..."
systemd-run --user --unit=gemma4-ablation --remain-after-exit \
  --setenv=PATH=/home/zma/anaconda3/envs/vllm/bin:/usr/bin:/bin \
  --setenv=PYTHONPATH=/home/zma/Documents/program/Github/subagent_manager/src:/home/zma/Documents/program/Github/subagent_manager \
  --working-directory=/home/zma/Documents/program/Github/subagent_manager \
  -- "$PYTHON" bench/eval/ablation.py \
    --model "openai/$MODEL" \
    --api-base http://localhost:8000/v1 \
    --api-key EMPTY \
    --instance-ids django__django-13315 django__django-13321 django__django-13401 django__django-13447 django__django-13448 django__django-13551 django__django-13590 django__django-13658 django__django-13660 django__django-13710 \
    --levels 0 1 2 3 4 \
    --output "$OUTPUT"

echo "[$(date)] Both services launched as systemd units."
