"""Test: which LiteLLM model prefix uses /api/generate vs /api/chat?"""
import asyncio
import litellm
import logging

# Only show LiteLLM's request logging
logging.basicConfig(level=logging.DEBUG, format="%(message)s")

async def main():
    # Test with "ollama/" prefix
    print("=== Testing ollama/ prefix ===")
    try:
        resp = await litellm.acompletion(
            model="ollama/gemma4:e2b-mlx",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
        )
        print(f"content: {repr(resp.choices[0].message.content)}")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
