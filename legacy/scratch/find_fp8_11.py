from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search="Qwen3.5-9B", limit=50)
for m in models:
    if "fp8" in m.id.lower():
        print(m.id)
