#!/usr/bin/env bash
# =============================================================================
# run_gemma3_ablation.sh — Ablation with google/gemma-3-4b-it
# =============================================================================
set -euo pipefail

cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="google/gemma-3-4b-it"
OUTPUT="bench/results/ablation_gemma3_300.jsonl"

mkdir -p bench/results

echo "[$(date)] Launching Gemma-3-4B-it ablation (300 instances, levels 0-4)..."
systemd-run --user --unit=gemma3-ablation --remain-after-exit \
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

echo "[$(date)] Gemma-3 ablation unit launched."
