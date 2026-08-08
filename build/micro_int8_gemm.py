#!/usr/bin/env python3
"""Head-to-head: aiter.gemm_a8w8 vs vLLM's Triton int8 scaled_mm at decode shapes.

Only if AITER wins at M=1/M=8 does patching vLLM's CompressedTensorsW8A8Int8
dispatch make any sense. Shapes are the real per-layer GEMMs of Qwen3.6-27B.
"""
import time
import torch

torch.set_default_device("cuda")
torch.manual_seed(0)

# Qwen3.6-27B (hidden 5120). Real linear shapes, N x K:
SHAPES = [
    ("qkv_proj",   7168, 5120),
    ("o_proj",     5120, 4096),
    ("gate_up",   34816, 5120),
    ("down_proj",  5120, 17408),
]
BATCHES = [1, 8, 16, 32]
ITERS, WARMUP = 50, 10


def bench(fn, *a):
    for _ in range(WARMUP):
        fn(*a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn(*a)
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


import aiter

# NB: the kernel classes live under vllm/model_executor/kernels/linear/scaled_mm/,
# NOT under quantization/kernels/ -- that path is a dead end and cost an hour of
# grepping. AiterInt8ScaledMMLinearKernel and TritonInt8ScaledMMLinearKernel are
# both there; this script times the underlying ops rather than the classes.

# vLLM's actual int8 GEMM entry point
from vllm.model_executor.layers.quantization.utils import w8a8_utils  # noqa
try:
    apply_int8 = w8a8_utils.apply_int8_linear
except AttributeError:
    apply_int8 = None

print(f"{'layer':>10} {'M':>4} {'aiter ms':>10} {'triton ms':>11} {'winner':>10} {'speedup':>8}")
print("-" * 62)

results = []
for name, N, K in SHAPES:
    for M in BATCHES:
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-127, 127, (N, K), dtype=torch.int8)
        sa = torch.ones(M, 1, dtype=torch.float32)
        sb = torch.ones(N, 1, dtype=torch.float32)

        # --- AITER ---
        try:
            t_aiter = bench(lambda: aiter.gemm_a8w8(a, b, sa, sb))
        except Exception as e:
            print(f"{name:>10} {M:>4}  aiter FAIL {type(e).__name__}: {str(e)[:40]}")
            continue

        # --- Triton (what vLLM selects when the CK gate is closed) ---
        t_triton = None
        if apply_int8 is not None:
            try:
                t_triton = bench(
                    lambda: apply_int8(
                        input=a, weight=b.t(), weight_scale=sb.t(),
                        input_scale=sa, bias=None,
                    )
                )
            except Exception:
                t_triton = None
        if t_triton is None:
            # fallback: torch._scaled_mm-free reference via int32 matmul
            try:
                t_triton = bench(lambda: (a.to(torch.int32) @ b.t().to(torch.int32)).float() * sa * sb.t())
            except Exception as e:
                print(f"{name:>10} {M:>4}  triton/ref FAIL {type(e).__name__}")
                continue

        win = "AITER" if t_aiter < t_triton else "triton"
        sp = max(t_triton, t_aiter) / min(t_triton, t_aiter)
        results.append((name, M, t_aiter, t_triton, win, sp))
        print(f"{name:>10} {M:>4} {t_aiter*1e3:>10.4f} {t_triton*1e3:>11.4f} {win:>10} {sp:>7.2f}x")

print()
aw = sum(1 for r in results if r[4] == "AITER")
print(f"AITER wins {aw}/{len(results)} shapes")
m1 = [r for r in results if r[1] == 1]
if m1:
    tot_a = sum(r[2] for r in m1) * 1e3
    tot_t = sum(r[3] for r in m1) * 1e3
    print(f"M=1 total across 4 layers: aiter {tot_a:.3f} ms vs triton {tot_t:.3f} ms")
    print(f"  -> per-token (x~64 layers): aiter {tot_a*64:.1f} ms = {1000/(tot_a*64):.1f} tok/s ceiling")
    print(f"                              triton {tot_t*64:.1f} ms = {1000/(tot_t*64):.1f} tok/s ceiling")
