#!/bin/bash
set -eo pipefail

export CUDA_HOME=/home/zma/anaconda3/envs/vllm
export PATH=/home/zma/anaconda3/envs/vllm/bin:$PATH
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="sk-no-key-required"

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
PORT=8000
MODEL="AxionML/Qwen3.5-9B-NVFP4"
OUTPUT_DIR="bench/data/eval_results_qwen35_nvfp4"

mkdir -p "$OUTPUT_DIR"
touch "$OUTPUT_DIR/results.jsonl"

echo "=== Cleaning GPU ==="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  kill -9 $pid 2>/dev/null || true
done
sleep 2

echo "=== Starting vLLM server for $MODEL ==="
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" "openai/$MODEL" \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port $PORT \
  --enable-auto-tool-choice \
  --tool-call-parser hermes > "$OUTPUT_DIR/vllm_server.log" 2>&1 &

SERVER_PID=$!
echo "vLLM server started with PID $SERVER_PID"

echo "Waiting for vLLM server health check on port $PORT..."
until curl -s "http://localhost:$PORT/health" > /dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: vLLM server exited prematurely! Log output:"
    cat "$OUTPUT_DIR/vllm_server.log" | tail -40
    exit 1
  fi
  sleep 3
done
echo "vLLM server is healthy and ready!"
sleep 10

echo "=== Launching Ablation Benchmark (10 instances, L0-L4) ==="
$PYTHON bench/eval/ablation.py \
  --model "openai/$MODEL" \
  --api-base "http://localhost:$PORT/v1" \
  --api-key "sk-no-key-required" \
  --instance-ids astropy__astropy-12907 astropy__astropy-14182 astropy__astropy-14365 astropy__astropy-14995 astropy__astropy-6938 astropy__astropy-7746 django__django-10914 django__django-10924 django__django-11001 django__django-11019 \
  --levels 0 1 2 3 4 \
  --output "$OUTPUT_DIR/results.jsonl"

echo "Benchmark evaluation completed successfully!"
