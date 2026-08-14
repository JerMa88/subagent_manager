#!/bin/bash
export CUDA_HOME=/home/zma/anaconda3/envs/vllm
export PATH=/home/zma/anaconda3/envs/vllm/bin:$PATH
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="sk-no-key-required"
PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"

$PYTHON -m vllm.entrypoints.openai.api_server \
  --model "google/gemma-4-12B-it" \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --port 8000 > gemma_kvfp8.log 2>&1 &
SERVER_PID=$!

for i in {1..200}; do
  if curl -s http://localhost:8000/health > /dev/null; then
    echo "Server healthy!"
    kill -9 $SERVER_PID
    exit 0
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server crashed!"
    tail -n 20 gemma_kvfp8.log
    exit 1
  fi
  sleep 3
done
echo "Timeout"
kill -9 $SERVER_PID
exit 1
