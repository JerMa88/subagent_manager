import asyncio
from subagent_manager.llm_client import LLMClient

async def main():
    llm = LLMClient(model="openai/AxionML/Qwen3.5-9B-NVFP4", api_base="http://localhost:8000/v1", api_key="sk-no-key-required")
    prompt = """You are an expert software engineer.
You will be given a GitHub issue. Output ONLY a valid unified diff (git diff format)
that fixes the issue. No explanations, no markdown, just the diff."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": "## GITHUB ISSUE\nFix a bug where 1+1=3."}]
    res = await llm.complete(messages=messages, max_tokens=100)
    print("CONTENT START")
    print(repr(res.content))
    print("CONTENT END")

asyncio.run(main())
