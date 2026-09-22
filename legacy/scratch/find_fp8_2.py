from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search="gemma-4-12B-it-FP8", limit=5)
for m in models:
    print(m.id)
print("---")
models = api.list_models(search="Qwen3.5-9B-Instruct-FP8", limit=5)
for m in models:
    print(m.id)
