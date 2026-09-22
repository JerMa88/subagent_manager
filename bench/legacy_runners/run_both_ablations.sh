#!/usr/bin/env bash
set -euo pipefail
cd /home/zma/Documents/program/Github/subagent_manager

PYTHON="/home/zma/anaconda3/envs/vllm/bin/python"
export CUDA_HOME=/home/zma/anaconda3/envs/vllm
export PATH=/home/zma/anaconda3/envs/vllm/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/home/zma/anaconda3/envs/vllm/lib:/home/zma/anaconda3/envs/vllm/lib64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/zma/Documents/program/Github/subagent_manager/src:/home/zma/Documents/program/Github/subagent_manager

# Make sure old services are dead
systemctl --user stop vllm-qwen35 qwen35-ablation vllm-gemma4 gemma4-ablation 2>/dev/null || true
pkill -9 -f "vllm_server.py" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 2

echo "======================================"
echo " Starting Qwen/Qwen3.5-9B Evaluation "
echo "======================================"
MODEL="Qwen/Qwen3.5-9B"
OUTPUT="bench/results/ablation_qwen35_base_full_10.jsonl"
mkdir -p bench/results

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code > qwen_vllm_full.log 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen"; then
        echo "vLLM ready for Qwen!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM crashed!"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "Qwen"; then
    echo "ERROR: vLLM not ready for Qwen."
    kill -9 $SERVER_PID 2>/dev/null || true
else
    echo "Starting Qwen Ablation..."
    $PYTHON bench/eval/ablation.py \
        --model "openai/$MODEL" \
        --api-base http://localhost:8000/v1 \
        --api-key EMPTY \
        --instance-ids django__django-13315 django__django-13321 django__django-13401 django__django-13447 django__django-13448 django__django-13551 django__django-13590 django__django-13658 django__django-13660 django__django-13710 \
        --levels 0 1 2 3 4 \
        --output "$OUTPUT"
fi
kill -9 $SERVER_PID 2>/dev/null || true
sleep 5

echo "======================================"
echo " Starting google/gemma-4-12B-it Eval "
echo "======================================"
MODEL="google/gemma-4-12B-it"
OUTPUT="bench/results/ablation_gemma4_full_10.jsonl"

$PYTHON -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code > gemma_vllm_full.log 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
        echo "vLLM ready for Gemma!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM crashed!"
        break
    fi
    sleep 2
done

if ! curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "gemma"; then
    echo "ERROR: vLLM not ready for Gemma."
    kill -9 $SERVER_PID 2>/dev/null || true
else
    echo "Starting Gemma Ablation..."
    $PYTHON bench/eval/ablation.py \
        --model "openai/$MODEL" \
        --api-base http://localhost:8000/v1 \
        --api-key EMPTY \
        --instance-ids django__django-13315 django__django-13321 django__django-13401 django__django-13447 django__django-13448 django__django-13551 django__django-13590 django__django-13658 django__django-13660 django__django-13710 \
        --levels 0 1 2 3 4 \
        --output "$OUTPUT"
fi
kill -9 $SERVER_PID 2>/dev/null || true
echo "All Eval Complete."
