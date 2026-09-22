from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search="gemma-4-12b", limit=10)
for m in models:
    print(m.id)
print("---")
models = api.list_models(search="Qwen3.5-9B-Instruct", limit=10)
for m in models:
    print(m.id)
