# Qwen3.8-27B on a single AMD MI210 — 80 tok/s

Self-contained setup notes for running Qwen3.8-27B (INT8) on one MI210 (gfx90a / CDNA2)
under vLLM, with speculative decoding.

Measured on one MI210, single stream, short context:

| | |
|---|---|
| decode | **~80 tok/s** |
| prefill | ~2,070 tok/s at 3.5k ctx, ~1,775 at 12.4k |
| KV cache | ~375,000 tokens |
| VRAM | 28 GB weights of 64 GB |

For reference, the same card runs **36 tok/s** with no speculation, and **20 tok/s** if the
AITER INT8 GEMM silently fails to engage — which is the default outcome on some hosts. See
"Verify" below; that check is the whole difference between 20 and 80.

---

## 1. Requirements

- **1x AMD Instinct MI210** (gfx90a). TP=1 — a second card is better spent on a second
  instance than on tensor parallelism (TP=2 measured only +10.6% for double the hardware,
  because decode all-reduce is latency-bound on PCIe).
- **A vLLM image with AITER built for gfx90a.** This is the part you cannot skip: AMD's
  AITER ships kernels for gfx942/gfx950 only, and vLLM's INT8 path falls back to a generic
  Triton kernel at roughly a third of the rate without it.
  Build one from https://github.com/davetha/mi210-vllm (`build/build.sh`, then
  `build/add-aiter.sh`). ROCm 7.x.

## 2. Get the weights

```bash
# target model: INT8 W8A8, plus INT8 GDN projections and INT8 lm_head  (28 GB)
hf download davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8 \
    --local-dir /models/qwen38-27b-int8

# speculative draft: DFlash2, 5 layers, keep it BF16  (3.85 GB)
hf download z-lab/Qwen3.8-27B-DFlash2 \
    --local-dir /models/qwen38-dflash2
```

Notes on the target model: it is an **abliterated** derivative (refusal behaviour removed —
that property is inherited, be aware of it), and it quantizes the gated-delta-net
projections and `lm_head`, which a stock W8A8 recipe leaves in BF16. That is worth ~30% and
costs ~2.3% perplexity; long-context recall was unchanged (9/9 needle at 107k tokens). A
plain W8A8 build of the same model is the conservative choice if you would rather not take
that trade.

## 3. Find the right GPU device node

**Skip this if every GPU in the machine is an MI210.** If the host has mixed GPUs, this is
worth 2.4x and is the most expensive thing to get wrong:

```bash
for n in /dev/dri/renderD*; do
  echo "$n $(cat /sys/class/drm/${n##*/}/device/device)"
done
# MI210 / Aldebaran = 0x740f   (also 0x740c, 0x7408)
```

vLLM resolves the GPU architecture **once at import**, from amdsmi, for *physical device 0*
— and it ignores `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`. If another vendor's card
is in slot 0, vLLM concludes the whole box is that architecture and silently disables every
gfx9 code path. Passing only the MI210's render node is the fix.

## 4. Serve

```bash
docker run -d --name qwen38 --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri/renderD128 \
  --group-add video --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16g \
  -v /models:/models:ro \
  -e VLLM_ROCM_USE_AITER=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GPU_PINNED_MIN_XFER_SIZE=67108864 \
  -p 8000:8000 \
  <your-image> /models/qwen38-27b-int8 \
    --served-model-name qwen38-27b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "dflash",
                           "model": "/models/qwen38-dflash2",
                           "num_speculative_tokens": 12}'
```

Replace `renderD128` with your MI210's node from step 3.

**First start takes 7-9 minutes** (torch.compile plus CUDA-graph capture). That is normal
on this hardware — do not kill it at minute four.

### Why these flags

| flag | why |
|---|---|
| `VLLM_ROCM_USE_AITER=1` | selects AITER's CK INT8 GEMM. Inert unless the image has gfx90a AITER kernels. |
| `cudagraph_mode: FULL_DECODE_ONLY` | vLLM otherwise drops to PIECEWISE graphs under speculation. Graphs are worth 1.7x here. |
| `method: dflash`, `num_speculative_tokens: 12` | 12 is the measured optimum: N=8 gives 78 tok/s, N=12 gives 80.6, N=16 falls to 74.9 as draft cost outruns acceptance. |
| `HSA_NO_SCRATCH_RECLAIM`, `HIP_FORCE_DEV_KERNARG` | ROCm runtime tunables, not optional in practice. |
| `GPU_PINNED_MIN_XFER_SIZE` | load-time: without it, weight loading can take hours instead of seconds. |

## 5. Verify it took the fast paths

All three of these fail **silently** and each costs 2-3x. Check them once, on first boot:

```bash
docker logs qwen38 2>&1 | grep 'Selected.*ScaledMMLinearKernel'
#   want: AiterInt8ScaledMMLinearKernel
#   if it says TritonInt8...  -> wrong GPU arch detected (step 3), or the image has no AITER

docker logs qwen38 2>&1 | grep -o 'Capturing CUDA graphs ([A-Z_]*)'
#   want FULL to appear, not PIECEWISE alone

docker logs qwen38 2>&1 | grep -ci dflash
#   want: non-zero
```

Then measure the real rate. **Do not count streaming chunks** — vLLM coalesces several
tokens per SSE chunk under speculation, which under-reports by ~3x:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","prompt":"Explain virtual memory paging.","max_tokens":200,"temperature":0}' \
  -w '\n%{time_total}\n'
# tok/s = usage.completion_tokens / wall seconds
```

## 6. Serving more than one user

Single-stream is not the ceiling. Decode reads the weights once per step regardless of how
many sequences ride along, so concurrency is close to free:

| concurrent | aggregate | per stream |
|---|---|---|
| 1 | 51 tok/s | 51 |
| 4 | 162 | 41 |
| **8** | **312** | **39** |
| 16 | 413 | 26 |

(measured before the last two optimizations, so treat as a lower bound on the shape)

8-way is the knee. With two MI210s, run **two independent TP=1 instances** rather than one
TP=2 instance.

## 7. If something looks wrong

| symptom | cause |
|---|---|
| ~⅓ the expected decode rate, no error anywhere | AITER INT8 GEMM not engaged — step 3 and step 5 |
| slower *with* speculation than without | wrong depth, or a quantized draft. Keep the draft BF16: INT8 makes each step 7.7% cheaper but costs 13.7% of accepted tokens, net −6.5% |
| empty responses | it is a reasoning model and can spend a small `max_tokens` budget entirely inside `<think>`. Budget generously. |
| image inputs rejected | this is a vision-language model; don't pass `--limit-mm-per-prompt '{"image":0,"video":0}'` unless you want text-only |
| "it got slower after I upgraded" | compare a run with speculation **disabled** first. tok/s is tokens-per-step × steps-per-second and those move independently; a change in acceptance rate looks exactly like a slowdown but has a completely different cause. |

---

## More detail

This page is deliberately self-contained so it can be pasted somewhere on its own. The
reasoning behind each choice, with the measurements, is split across:

- [SPEC-DECODE.md](SPEC-DECODE.md) — the depth sweep, why the draft stays BF16, and why
  speculation depth has to be re-measured after anything that moves step time
- [INT8-GFX90A.md](INT8-GFX90A.md) — confirming the AITER kernel is selected, what a stock
  W8A8 recipe leaves in BF16, and the `targets: ["Linear", "re:.*lm_head$"]` form that makes
  a quantized `lm_head` work
- [RUNNING.md](RUNNING.md) — the general case for any model, and the mixed-GPU
  architecture-detection problem in full

Tooling: https://github.com/davetha/mi210-vllm
