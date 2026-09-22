while true; do
  TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ GPU_UTIL=0.50 /home/zma/anaconda3/envs/vllm/bin/python bench/vllm_server.py
  echo "vLLM crashed! Restarting in 2 seconds..."
  sleep 2
done
