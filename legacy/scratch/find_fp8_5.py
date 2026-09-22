from huggingface_hub import HfApi
api = HfApi()
try:
    print(api.model_info("google/gemma-4-12B-it-FP8").id)
except Exception as e:
    print(e)
try:
    print(api.model_info("Qwen/Qwen3.5-9B-Instruct-FP8").id)
except Exception as e:
    print(e)
