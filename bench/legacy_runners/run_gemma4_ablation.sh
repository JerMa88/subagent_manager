#!/usr/bin/env bash
# =============================================================================
# run_gemma4_ablation.sh — Ablation with unsloth/gemma-4-12b-it
# =============================================================================
set -euo pipefail

cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="unsloth/gemma-4-12b-it"
OUTPUT="bench/results/ablation_gemma4_300.jsonl"

mkdir -p bench/results

# ── Kill any existing vLLM processes ──
systemctl --user stop vllm-qwen35 qwen35-ablation vllm-gemma4 gemma4-ablation 2>/dev/null || true
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
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --quantization bitsandbytes --load-format bitsandbytes

echo "[$(date)] Waiting for vLLM to be ready (up to 600s)..."
for i in $(seq 1 600); do
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q "200"; then
        echo "[$(date)] vLLM ready after ${i}s!"
        break
    fi
    # fallback check in case /health doesn't return 200 properly but /v1/models works
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
        echo "[$(date)] vLLM ready after ${i}s!"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
    echo "[$(date)] ERROR: vLLM not ready after 600s"
    exit 1
fi

# ── Launch ablation ──
echo "[$(date)] Starting ablation (10 instances)..."
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
echo "  Monitor: systemctl --user status vllm-gemma4 gemma4-ablation"
echo "  Logs:    journalctl --user -u gemma4-ablation -f"
echo "  Results: wc -l $OUTPUT"
