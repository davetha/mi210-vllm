# Recipe: Qwen3.8-27B INT8 + dflash on one MI210

The fastest known configuration for this model on this hardware, end to end.
**80.6 tok/s** single-stream decode on a single MI210, ~2,070 tok/s prefill.

Every other doc here explains *why* one piece is the way it is. This one is the
copy-pasteable version.

## 1. What you need

| | | |
|---|---|---|
| image | with AITER for gfx90a | `build/build.sh`, then `build/add-aiter.sh` |
| model | INT8 W8A8 + INT8 GDN + INT8 `lm_head` | `davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8` (28 GB) |
| draft | DFlash2, **BF16** | `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB) |

```bash
hf download davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8 --local-dir /models/qwen38-27b-int8
hf download z-lab/Qwen3.8-27B-DFlash2                    --local-dir /models/qwen38-dflash2
```

A stock W8A8 build of this model works too and is the conservative choice — it is ~30%
slower here, because it leaves the gated-delta-net projections and `lm_head` in BF16.
[INT8-GFX90A.md](INT8-GFX90A.md) has the trade, including the perplexity cost per step.

## 2. Find the gfx90a render nodes

Skip if every GPU in this host is gfx90a — `run.sh` handles it and the default is right.
Otherwise this is worth **2.4x** and is the single most expensive thing to get wrong:

```bash
for n in /dev/dri/renderD*; do
  echo "$n $(cat /sys/class/drm/${n##*/}/device/device)"
done
# gfx90a (Aldebaran) = 0x740f / 0x740c / 0x7408
```

Why: vLLM resolves the GPU architecture once at import, from amdsmi, for **physical device
0** — ignoring `HIP_VISIBLE_DEVICES` *and* `ROCR_VISIBLE_DEVICES`. A foreign card in slot 0
makes it disable every gfx9 path silently.
[RUNNING.md](RUNNING.md#which-gpus-the-container-sees).

## 3. Serve

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
  <image> /models/qwen38-27b-int8 \
    --served-model-name qwen38-27b \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"method": "dflash",
                           "model": "/models/qwen38-dflash2",
                           "num_speculative_tokens": 12}'
```

`run.sh` does the device selection and ROCm environment for you; the explicit form above is
for compose, Kubernetes or Slurm.

Startup is **7-9 minutes** (torch.compile plus cudagraph capture). That is normal on this
hardware, not a hang.

## 4. Check it actually took the fast paths

Three greps. Each failure is silent and costs 2-3x.

```bash
docker logs qwen38 2>&1 | grep 'Selected.*ScaledMMLinearKernel'
#   want: AiterInt8ScaledMMLinearKernel        not: TritonInt8ScaledMMLinearKernel

docker logs qwen38 2>&1 | grep -o 'Capturing CUDA graphs ([A-Z_]*)'
#   want: FULL                                 not: PIECEWISE only

docker logs qwen38 2>&1 | grep -c 'DFlash2\|dflash'
#   want: non-zero
```

Then confirm the real rate — **not** by counting SSE chunks, which vLLM coalesces under
speculation:

```bash
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-27b","prompt":"Explain virtual memory paging.","max_tokens":200,"temperature":0}' \
  -w '\n%{time_total}\n'
# tok/s = completion_tokens / wall
```

## 5. What to expect

Single MI210, short context:

| | |
|---|---|
| decode | ~80 tok/s |
| prefill | ~2,070 tok/s at 3.5k, ~1,775 at 12.4k |
| KV cache | ~375k tokens |
| tokens/step | ~3.5 |

Serving several users at once is a different axis and much larger: 8-way concurrency
measured **6.1x aggregate** while each stream still saw 39 tok/s. Two independent TP=1
instances (one per card) beat one TP=2 instance — TP=2 was +10.6% for double the hardware,
because decode all-reduce is latency-bound on this PCIe pair.

## 6. If a number looks wrong

| symptom | look at |
|---|---|
| ~⅓ the decode rate, no error | `Selected TritonInt8...` — wrong arch detected, or the image has no AITER. [RUNNING.md](RUNNING.md#which-gpus-the-container-sees) |
| ~40% slower with speculation on | wrong depth, or a quantized draft. Keep the draft BF16. [SPEC-DECODE.md](SPEC-DECODE.md) |
| slower after an image change | compare a **no-spec** arm first: tok/s is tokens/step x steps/s and the two move independently |
| empty responses | reasoning models can spend the whole budget inside `<think>`; raise `max_tokens` |
| image/video requests rejected | `--limit-mm-per-prompt '{"image":0,"video":0}'` is a benchmark flag; drop it to serve a VL model |
