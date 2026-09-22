from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="Qwen", search="Qwen3.5", limit=50)
for m in models:
    print(m.id)
