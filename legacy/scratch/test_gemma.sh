#!/bin/bash
export CUDA_HOME=/home/zma/anaconda3/envs/vllm
export PATH=/home/zma/anaconda3/envs/vllm/bin:$PATH
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
export OPENAI_API_KEY="sk-no-key-required"
PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"

echo "=== Starting vLLM server for AxionML/Gemma-4-12B-NVFP4 ==="
$PYTHON -m vllm.entrypoints.openai.api_server \
  --model "AxionML/Gemma-4-12B-NVFP4" \
  --served-model-name "AxionML/Gemma-4-12B-NVFP4" "openai/AxionML/Gemma-4-12B-NVFP4" \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --port 8000 > gemma_vllm.log 2>&1 &
SERVER_PID=$!

echo "Waiting for health check..."
until curl -s "http://localhost:8000/health" > /dev/null; do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server crashed!"
    cat gemma_vllm.log
    exit 1
  fi
  sleep 3
done
echo "Server healthy!"

cat << 'PYEOF' > test_gemma_llm.py
import asyncio
from subagent_manager.llm_client import LLMClient
async def main():
    llm = LLMClient(model="openai/AxionML/Gemma-4-12B-NVFP4", api_base="http://localhost:8000/v1", api_key="sk-no-key-required")
    messages = [{"role": "user", "content": "Hello, how are you? Please reply with a short sentence."}]
    res = await llm.complete(messages=messages, max_tokens=50)
    print("CONTENT START")
    print(repr(res.content))
    print("CONTENT END")
asyncio.run(main())
PYEOF

$PYTHON test_gemma_llm.py
kill -9 $SERVER_PID
