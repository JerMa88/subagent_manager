import os
import multiprocessing

# Environment flags for multiprocess executor, FlashInfer, and all-reduce
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

# Add nvidia libraries to LD_LIBRARY_PATH if not already present
nv_lib = "/projects/mhahsler/course_recomm/allocation001/venvs/qwen38_env/lib/python3.11/site-packages/nvidia"
nv_dirs = [f"{nv_lib}/{pkg}/lib" for pkg in ["cu13", "cuda_runtime", "cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse", "nccl", "nvtx", "nvjitlink"]]
current_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join(nv_dirs) + (":" + current_ld if current_ld else "")
os.environ["PATH"] = "/projects/mhahsler/course_recomm/allocation001/venvs/qwen38_env/bin:/usr/local/cuda-12.8/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"

def main():
    import time
    from vllm import LLM, SamplingParams

    model_path = "/work/projects/mhahsler/course_recomm/allocation001/hf_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

    print(f"Initializing Qwen3.8-27B with TP=2 and MTP on 2x A100...")
    t0 = time.time()

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "attention_backend": "FLASHINFER",
        },
        max_model_len=65536,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
        attention_backend="FLASHINFER",
        skip_mm_profiling=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=True,
    )
    print(f"Model loaded successfully in {time.time() - t0:.2f} seconds.")

    prompts = [
        "Write a Python function to solve the two-sum problem efficiently with detailed comments and type hints.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
    )

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t1

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        tok_per_sec = num_tokens / gen_time if gen_time > 0 else 0
        print("\n" + "=" * 50)
        print(f"Generated {num_tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.2f} tok/s):")
        print(generated_text[:500] + ("..." if len(generated_text) > 500 else ""))
        print("=" * 50)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
