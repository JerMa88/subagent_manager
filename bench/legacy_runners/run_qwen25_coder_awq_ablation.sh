#!/usr/bin/env bash
# =============================================================================
# run_qwen25_coder_awq_ablation.sh — 300-instance ablation evaluation
# =============================================================================
set -euo pipefail

cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
OUTPUT="bench/results/ablation_300.jsonl"

mkdir -p bench/results

echo "[$(date)] Launching Qwen2.5-Coder-7B-Instruct-AWQ ablation (300 instances, levels 0-4)..."
systemd-run --user --unit=qwen25-ablation --remain-after-exit \
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

echo "[$(date)] Qwen2.5 ablation unit launched."
