#!/usr/bin/env bash
# =============================================================================
# run_nvfp4_ablation.sh — Crash-proof ablation runner
#
# Runs BOTH the vLLM server and the ablation script as independent background
# processes. Completely decoupled from Antigravity/IDE — survives IDE crashes,
# V8 OOM, and terminal disconnects.
#
# Usage:
#   bash bench/run_nvfp4_ablation.sh
#
# Monitor:
#   tail -f bench/results/nvfp4_vllm.log    # vLLM server logs
#   tail -f bench/results/nvfp4_ablation.log # Ablation progress
#   wc -l bench/results/ablation_nvfp4_300.jsonl  # Completed instances
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
MODEL="AxionML/Qwen3.5-9B-NVFP4"
OUTPUT="bench/results/ablation_nvfp4_300.jsonl"
VLLM_LOG="bench/results/nvfp4_vllm.log"
ABLATION_LOG="bench/results/nvfp4_ablation.log"

mkdir -p bench/results

# ── Kill any existing vLLM server AND its EngineCore child processes ──
# The EngineCore spawns as a separate process and holds GPU memory even
# after the parent is killed. We must kill both.
pkill -f "vllm_server.py" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 2
pkill -9 -f "vllm_server.py" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
# Also kill any orphaned vllm python processes holding GPU memory
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$"); do
    cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmdline" | grep -qi "vllm\|EngineCore"; then
        echo "[$(date)] Killing orphaned GPU process PID=$pid: $cmdline"
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 2

echo "[$(date)] Starting vLLM server with model $MODEL..."

# ── Start vLLM server in background ──
MODEL_NAME="$MODEL" GPU_UTIL=0.65 \
    nohup "$PYTHON" bench/vllm_server.py \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[$(date)] vLLM server PID: $VLLM_PID"

# ── Wait for vLLM to be ready ──
echo "[$(date)] Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[$(date)] vLLM server ready after ${i}s!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server died. Check $VLLM_LOG"
        exit 1
    fi
    sleep 1
done

# Final check
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[$(date)] ERROR: vLLM server not ready after 120s. Check $VLLM_LOG"
    exit 1
fi

# ── Launch ablation ──
echo "[$(date)] Starting ablation run (all 300 instances, levels 0-4)..."
echo "[$(date)] Output: $OUTPUT"
echo "[$(date)] Ablation log: $ABLATION_LOG"

nohup "$PYTHON" bench/eval/ablation.py \
    --model "openai/$MODEL" \
    --api-base http://localhost:8000/v1 \
    --api-key EMPTY \
    --all \
    --levels 0 1 2 3 4 \
    --output "$OUTPUT" \
    > "$ABLATION_LOG" 2>&1 &
ABLATION_PID=$!

echo "[$(date)] Ablation PID: $ABLATION_PID"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Both processes are now running in the background."
echo "  They will SURVIVE Antigravity/IDE crashes."
echo ""
echo "  Monitor:"
echo "    tail -f $VLLM_LOG"
echo "    tail -f $ABLATION_LOG"
echo "    wc -l $OUTPUT"
echo ""
echo "  PIDs: vLLM=$VLLM_PID  Ablation=$ABLATION_PID"
echo "════════════════════════════════════════════════════════════"
