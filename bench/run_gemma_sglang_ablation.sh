#!/bin/bash
set -eo pipefail

export CUDA_HOME=/home/zma/anaconda3/envs/sglang/lib/python3.11/site-packages/nvidia/cu13
export PATH=/home/zma/anaconda3/envs/sglang/lib/python3.11/site-packages/nvidia/cu13/nvvm/bin:/home/zma/anaconda3/envs/sglang/lib/python3.11/site-packages/nvidia/cu13/bin:/home/zma/anaconda3/envs/sglang/bin:$PATH
export CPATH=/home/zma/anaconda3/envs/sglang/lib/python3.11/site-packages/nvidia/cu13/include:${CPATH:-}
export C_INCLUDE_PATH=$CPATH
export CPLUS_INCLUDE_PATH=$CPATH
export CCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1"
export CFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1 ${CFLAGS:-}"
export CXXFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1 ${CXXFLAGS:-}"
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/sglang/lib/python3.11/site-packages/nvidia/cu13/lib:/home/zma/anaconda3/envs/sglang/lib:$LD_LIBRARY_PATH
export NINJAFLAGS="-j4"
export OPENAI_API_KEY="sk-no-key-required"
export OPENAI_API_BASE="http://localhost:8000/v1"

SGLANG_PYTHON="/home/zma/anaconda3/envs/sglang/bin/python"
VLLM_PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"

PORT=8000
MODEL="AxionML/Gemma-4-12B-NVFP4"
OUTPUT_DIR="bench/data/eval_results_gemma_sglang"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/results.jsonl"

echo "=== Cleaning GPU ==="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 $pid 2>/dev/null || true
done
pkill -f "sglang.launch_server" || true
sleep 2

echo "=== Starting SGLang server for $MODEL ==="
$SGLANG_PYTHON -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --port $PORT \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --disable-cuda-graph \
  --disable-flashinfer-autotune \
  --context-length 16384 \
  --max-running-requests 2 \
  --swa-full-tokens-ratio 0.80 \
  --mem-fraction-static 0.91 \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 > "$OUTPUT_DIR/sglang_server.log" 2>&1 &

SERVER_PID=$!
echo "SGLang server started with PID $SERVER_PID"

echo "Waiting for SGLang server health check on port $PORT..."
until curl -s "http://localhost:$PORT/health" > /dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: SGLang server exited prematurely! Log output:"
    cat "$OUTPUT_DIR/sglang_server.log" | tail -40
    exit 1
  fi
  sleep 3
done
echo "SGLang server is healthy and ready!"
sleep 5

echo "=== Launching Full 300-Instance Ablation Benchmark (L0-L4) ==="
$VLLM_PYTHON bench/eval/ablation.py \
  --model "openai/$MODEL" \
  --api-base "http://localhost:$PORT/v1" \
  --api-key "sk-no-key-required" \
  --all \
  --levels 0 1 2 3 4 \
  --output "$OUTPUT_DIR/results.jsonl"

echo "Benchmark evaluation completed successfully!"
