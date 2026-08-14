import asyncio
from subagent_manager.llm_client import LLMClient
async def main():
    llm = LLMClient(model="openai/AxionML/Qwen3.5-9B-NVFP4", api_base="http://localhost:8000/v1", api_key="sk-no-key-required")
    messages = [{"role": "user", "content": "Hello, how are you? Please reply with a short sentence."}]
    res = await llm.complete(messages=messages, max_tokens=100)
    print("CONTENT:")
    print(repr(res.content))
