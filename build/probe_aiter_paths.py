#!/usr/bin/env python3
"""Probe which AITER fast paths actually engage on gfx90a under vLLM.

Loads a small model with VLLM_ROCM_USE_AITER=1 and reports:
  - which AITER dispatch predicates return True
  - which attention backend vLLM selects
  - whether the AITER ASM paged-attention kernel is reachable
"""
import os, sys

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

print("=" * 70)
print("AITER FAST-PATH PROBE — gfx90a")
print("=" * 70)

from vllm.platforms import current_platform
print(f"device: {current_platform.get_device_name()}")

from vllm import envs
print("\n--- AITER env flags ---")
for k in sorted(d for d in dir(envs) if "AITER" in d):
    try:
        print(f"  {k} = {getattr(envs, k)}")
    except Exception as e:
        print(f"  {k} = <err {e}>")

print("\n--- AITER dispatch predicates (vLLM rocm_aiter_* modules) ---")
probes = [
    ("paged attn (MHA)", "vllm.attention.ops.rocm_aiter_paged_attn", None),
    ("aiter MLA", "vllm.attention.ops.rocm_aiter_mla", None),
    ("aiter fa", "vllm.attention.ops.rocm_aiter_fa", None),
    ("aiter ops (norm/act)", "vllm.model_executor.layers.layernorm", None),
]
for label, mod, attr in probes:
    try:
        m = __import__(mod, fromlist=["*"])
        fns = [f for f in dir(m) if f.startswith("is_") or "enabled" in f.lower() or "support" in f.lower()]
        results = {}
        for f in fns:
            try:
                fn = getattr(m, f)
                if callable(fn):
                    try:
                        results[f] = fn()
                    except TypeError:
                        results[f] = "<needs args>"
            except Exception as e:
                results[f] = f"<err {type(e).__name__}>"
        print(f"  {label}: imported OK  {results if results else ''}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e)[:90]}")

print("\n--- direct aiter kernel availability ---")
try:
    import aiter
    for k in ["paged_attention_rocm", "pa_fwd_asm", "gemm_a8w8", "gemm_a16w4",
              "fmoe", "rms_norm", "silu_and_mul"]:
        print(f"  aiter.{k}: {'YES' if hasattr(aiter, k) else 'no'}")
except Exception as e:
    print(f"  aiter import failed: {e}")

print("\n--- what attention backend does vLLM pick? ---")
try:
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(model=sys.argv[1] if len(sys.argv) > 1 else "/models/probe",
                      max_model_len=2048, gpu_memory_utilization=0.35,
                      enforce_eager=True)
    cfg = args.create_engine_config()
    print(f"  model: {cfg.model_config.model}")
    print(f"  dtype: {cfg.model_config.dtype}")
    print(f"  quant: {cfg.model_config.quantization}")
except Exception as e:
    print(f"  config build: {type(e).__name__}: {str(e)[:120]}")
print("=" * 70)
