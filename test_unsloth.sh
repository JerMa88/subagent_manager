#!/bin/bash
export CUDA_HOME=/home/zma/anaconda3/envs/vllm
export PATH=/home/zma/anaconda3/envs/vllm/bin:$PATH
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="sk-no-key-required"
PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"

$PYTHON -m vllm.entrypoints.openai.api_server \
  --model "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast" \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port 8000 > unsloth_vllm.log 2>&1 &
SERVER_PID=$!

for i in {1..60}; do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "Server healthy!"
    kill -9 $SERVER_PID
    exit 0
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server crashed!"
    cat unsloth_vllm.log
    exit 1
  fi
  sleep 3
done
echo "Timeout"
kill -9 $SERVER_PID
exit 1
