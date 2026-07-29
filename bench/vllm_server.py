"""
Lightweight, ultra-fast OpenAI-compatible vLLM server for RTX 5080 with AsyncLLMEngine real-time token streaming.
Bypasses FlashInfer JIT compilation using PyTorch Marlin and FlashAttention backends.
"""
import sys
import time
import os
import json
import uuid

os.environ["CUDA_HOME"] = "/home/zma/anaconda3/envs/vllm"
os.environ["PATH"] = "/home/zma/anaconda3/envs/vllm/bin:" + os.environ.get("PATH", "")
os.environ["CPATH"] = "/home/zma/anaconda3/envs/vllm/include:" + os.environ.get("CPATH", "")
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["FLASHINFER_SAMPLING"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1_ENGINE"] = "0"
os.environ["VLLM_NVFP4_MOE_BACKEND"] = "MARLIN"

import asyncio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# ── Verify CPU reorder patch is applied ──
import importlib.util
_fp4_path = os.path.join(
    os.path.dirname(importlib.util.find_spec("vllm").origin),
    "model_executor", "layers", "quantization", "utils", "flashinfer_fp4_moe.py"
)
with open(_fp4_path) as f:
    _src = f.read()
if "w1_cpu = w1.clone().cpu()" in _src:
    print("[FastServer] CPU reorder patch verified in flashinfer_fp4_moe.py")
else:
    print("[FastServer] WARNING: CPU reorder patch NOT found — may OOM during model load!")

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
GPU_UTIL = float(os.getenv("GPU_UTIL", "0.65"))  # Raised from 0.50: needed for KV cache of 4 concurrent seqs
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "16384"))
# SMOKE TEST NOTE (2026-07-29): max_num_seqs was 1 (serial) during the original 300-instance run.
# Increased to 4 to give asyncio.gather() real GPU parallelism.
# Smoke-tested at MAX_NUM_SEQS=4 vs MAX_NUM_SEQS=1 on 5 instances before full ablation;
# results are preserved in bench/results/smoke_concurrency_*.jsonl.
# WHY KEPT: With max_num_seqs=1, ParallelStrategy's asyncio.gather() is a no-op — all
# agent coroutines are serialized by the vLLM engine. Setting to 4 gives the AdaptiveStrategy
# real throughput benefit when independent workers (e.g., issue_analyzer + code_explorer)
# can overlap on the GPU.
# RISK: Higher peak VRAM. If OOM occurs, revert to MAX_NUM_SEQS=2 and GPU_UTIL=0.60.
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "4"))

app = FastAPI(title="vLLM FastServer")

engine: AsyncLLMEngine = None
tokenizer = None
is_ready = False

@app.on_event("startup")
async def startup_event():
    global engine, tokenizer, is_ready
    print(f"[FastServer] Initializing AsyncLLMEngine {MODEL_NAME} (util={GPU_UTIL}, max_len={MAX_MODEL_LEN})...")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_UTIL,
        max_num_seqs=MAX_NUM_SEQS,  # Was 1 (serial); now 4 for real asyncio.gather() parallelism
        enforce_eager=True,
        trust_remote_code=True,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=2,
        offload_prefetch_step=1,
        cpu_offload_gb=4,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    is_ready = True
    print("[FastServer Async] AsyncLLMEngine ready for real-time streaming on port 8000!")

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "vllm"},
            {"id": f"openai/{MODEL_NAME}", "object": "model", "created": int(time.time()), "owned_by": "vllm"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    while not is_ready:
        await asyncio.sleep(0.2)

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", MODEL_NAME)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 2048)
    stream = body.get("stream", False)

    tools = body.get("tools", [])
    
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tools=tools if tools else None, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    sampling_params = SamplingParams(
        temperature=temperature if temperature > 0 else 0.01,
        max_tokens=max_tokens,
    )

    request_id = f"cmpl-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    results_generator = engine.generate(prompt, sampling_params, request_id)

    if stream:
        async def sse_generator():
            previous_text = ""
            created = int(time.time())

            # Send initial role chunk immediately so connection stays alive
            initial_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            async for request_output in results_generator:
                text = request_output.outputs[0].text
                delta = text[len(previous_text):]
                previous_text = text
                if delta:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Final finish chunk
            final_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")
    else:
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        text = final_output.outputs[0].text if final_output else ""
        prompt_tokens = len(final_output.prompt_token_ids) if final_output else 0
        completion_tokens = len(final_output.outputs[0].token_ids) if final_output else 0

        parsed_content = text
        tool_calls = []
        import re
        
        # 1. Try to parse <tool_call> tags
        for match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
            try:
                tc = json.loads(match.group(1))
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else str(tc.get("arguments", "{}"))
                    }
                })
                parsed_content = parsed_content.replace(match.group(0), "")
            except Exception as e:
                pass

        # 2. Try markdown json block
        if not tool_calls:
            for match in re.finditer(r'```json\s*(\{.*?"name"\s*:\s*".*?".*?\})\s*```', text, re.DOTALL):
                try:
                    tc = json.loads(match.group(1))
                    if "name" in tc and "arguments" in tc:
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else str(tc["arguments"])
                            }
                        })
                        parsed_content = parsed_content.replace(match.group(0), "")
                except Exception as e:
                    pass

        # 3. Try parsing the entire text as JSON
        if not tool_calls:
            try:
                tc = json.loads(text.strip())
                if isinstance(tc, dict) and "name" in tc and "arguments" in tc:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else str(tc["arguments"])
                        }
                    })
                    parsed_content = ""
            except:
                pass
                
        message = {"role": "assistant", "content": parsed_content.strip() or None}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", loop="asyncio")
