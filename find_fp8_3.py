from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="neuralmagic", search="Qwen3.5", limit=5)
for m in models:
    print(m.id)
print("---")
models = api.list_models(author="neuralmagic", search="gemma-4", limit=5)
for m in models:
    print(m.id)
