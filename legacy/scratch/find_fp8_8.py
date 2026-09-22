from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(author="Qwen", search="FP8", limit=20)
for m in models:
    if "3.5" in m.id:
        print(m.id)
