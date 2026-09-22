from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
messages = [{"role": "user", "content": "Hello"}]
tools = [{"type": "function", "function": {"name": "test_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}}]
prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
print(prompt)
