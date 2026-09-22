from huggingface_hub import HfApi
api = HfApi()
print(api.model_info("Qwen/Qwen3.5-9B-Instruct").id)
print(api.model_info("google/gemma-4-12B-it").id)
