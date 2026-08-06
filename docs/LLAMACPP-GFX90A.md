# llama.cpp on gfx90a: ~6 tok/s for short-context GLM-5.2

The vLLM path in [DSA-GFX90A.md](DSA-GFX90A.md) tops out around **1 tok/s** because it
ships every active MoE expert over PCIe each forward pass. llama.cpp sidesteps that wall
by keeping the experts on the CPU and running them over the DDR4 bus instead — and on this
box that is a **~6x** short-context win, with correct output.

Measured 2026-08-06 on 2x MI210 (gfx90a), `unsloth/GLM-5.2-GGUF` UD-Q4_K_M (438 GiB, 11
splits), against the vLLM `glm52-int4int8` baseline on the same prompts.

| workload | vLLM (experts over PCIe) | llama.cpp `--cpu-moe` |
|---|---|---|
| short-context decode (mean of 6) | **~1.0 tok/s** | **5.9 tok/s** |
| open-ended | ~1.0 | 6.4 |
| structured (code) | ~1.0 | 6.2 |
| repetitive (settled) | ~1.0 | ~6.3 |
| decode @ 8.7k context | — | **6.9 tok/s** (holds — MLA compresses KV) |

## Why it is fast

`-cmoe` (`--cpu-moe`) pins the MoE **expert** weights to CPU while `-ngl 999` puts the
dense/attention/shared tensors on the GPUs. GLM-5.2 activates 8 of 256 experts per token
across ~76 MoE layers, so each token streams a few GiB of expert weights. Reading those
from host RAM over DDR4 (~120 GB/s) instead of pushing them over PCIe to the GPU (~25 GB/s)
is the entire ~5x bandwidth edge — it is the PCIe wall from DSA-GFX90A.md run in reverse.

The fastpaths that matter on gfx90a are on by default in the build; the two knobs people
usually reach for are red herrings here:

- `GGML_HIP_MMQ_MFMA=ON` — the CDNA quantized-GEMM fastpath (the one that counts).
- `-fa on` — native MFMA flash attention (`fattn-mma-f16`). rocWMMA flash attention
  (`-DGGML_HIP_ROCWMMA_FATTN`) was **removed** upstream (PR #26046, 2025-07-24) and was a
  measured loser on gfx90a anyway — leave it off.
- `ROCBLAS_USE_HIPBLASLT=1` — real but marginal; llama.cpp's MoE uses its own MMQ kernels,
  not rocBLAS, so this only touches FP16 prompt-processing GEMM. Set it, do not expect much.

## The build

A working build already existed on the bench box as `llama-rocm714-rpc` (ROCm 7.14):
`AMDGPU_TARGETS=gfx90a`, `GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_ROCWMMA_FATTN=OFF`. It shipped
only `llama-server`/`llama-bench`; `llama-perplexity` and `llama-cli` were added from the
cached build dir and committed as `llama-rocm714-rpc:full`:

```bash
docker run -d --name llama-build --entrypoint /bin/sh llama-rocm714-rpc:latest -c "sleep infinity"
docker exec llama-build bash -lc 'cd /src && cmake --build build --target llama-perplexity llama-cli -j6'
docker commit llama-build llama-rocm714-rpc:full && docker rm -f llama-build
```

GLM-5.2's `glm-dsa` architecture is **natively supported** (dedicated `llama_model_glm_dsa`
loader; `LLM_ARCH_GLM_DSA` in `llama-arch.cpp`). The GGUF also carries the DSA *indexer*
tensors and the MTP/NextN block (`blk.78.nextn.*`); the loader ignores the MTP block, which
is fine for standard (non-speculative) decode.

## The run

`build/start-llama-glm52.sh`. The load line:

```bash
HIP_VISIBLE_DEVICES=0,1 ROCBLAS_USE_HIPBLASLT=1 llama-server \
  -m GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf \
  -ngl 999 -cmoe -fa on -sm layer \
  -t 24 -tb 24 -c 8192 --temp 0
```

- `-sm layer`, not `-sm row`. `-sm row` fails with `device ROCm0 does not support split
  buffers` on the ROCm backend; layer split (whole layers per device) works.
- **438 GiB on a 499 GB box is a tight fit, so no `--mlock`.** It relies on the kernel page
  cache. Run it only with the vLLM container stopped (vLLM holds ~160 GB of offload).
- Launch the container with `--init`, or `docker rm -f` leaves a zombie `llama-server`
  (its `bash -lc` wrapper does not reap children). This is an operational wart, not a
  correctness issue.

## Correctness

Output matches a vLLM reference on the same prompts (same "social/economic/political"
structure for a history question; a valid `fib(n)` with docstring). The #26027-style
garbage-on-GPU-offload bug is **not present** on gfx90a. Dropping the MTP block is benign.
This is a generation-coherence check against a same-model reference, not a perplexity gate —
a numeric perplexity comparison against a non-offloaded baseline is still outstanding.

## Durability (does the 6x survive a cold start?)

Yes. After evicting the page cache and re-loading from NVMe, the 6-prompt bench reproduces
**5.9 tok/s** — identical to the warm run. The model fits in RAM (438 < 499), so once loaded
its mmap'd pages stay resident (no eviction without pressure); the only cost is a one-time
**~2-3 min cold load** (438 GiB faulted in from NVMe at ~3 GB/s).

## Limits, plainly

**Long-context prefill is slow.** ~170 s for 8.7k tokens, because `-cmoe` runs the prefill's
expert compute on CPU too. vLLM (GPU prefill + DSA sparse indexer) wins on time-to-first-token
for large inputs. Decode rate is unaffected at length (6.9 tok/s @ 8.7k).

**Long-context *quality* is unverified.** A probe with a pathological repeated-sentence
prompt produced echoed text; that is the prompt, not the engine (short-context output is
clean), but a natural-text long-context test was not run.

**RAM starvation collapses it.** The 6x holds only while the 438 GiB stays resident. If
something else squeezes RAM below ~440 GiB the experts spill to NVMe (~3 GB/s) and
throughput falls apart. As the sole server it fits; alongside another big consumer it does
not.

## When to use which

- **Short / interactive context, low concurrency:** llama.cpp (`-cmoe`) — ~6 tok/s.
- **Long prompts (large prefill), or concurrent batch:** vLLM — GPU prefill + DSA sparse
  attention, and it does not need the whole model resident in the way llama.cpp does here.
