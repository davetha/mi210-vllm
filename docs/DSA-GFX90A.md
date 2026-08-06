# DSA sparse attention on gfx90a (GLM-5.2)

GLM-5.2 serves on 2x MI210. It generates coherent text, and it is **~1.0 tok/s**
(expert-selective offload; 0.81 before it) — proof the architecture runs on CDNA2,
not a usable deployment. For ~6 tok/s short-context, run it under llama.cpp instead
([LLAMACPP-GFX90A.md](LLAMACPP-GFX90A.md)); why ~1 is the ceiling is in
[ROUTING.md](ROUTING.md).

```
$ curl .../v1/completions -d '{"prompt":"The capital of France is","max_tokens":24}'
" Paris. Distance from Paris to Lyon is 391 km, while distance from
  Marseille to Lyon is 278 km"

"Q: What is 17 times 23? A:"      -> " 391."          (correct)
"def fibonacci(n):"               -> valid Python
"The three primary colors are"    -> " red, blue, and yellow."
```

Measured 2026-08-05 on `QuantTrio/GLM-5.2-Int4-Int8Mix` (378 GiB, TP=2).

---

## Why it did not work

GLM-5.2 uses DeepSeek-style sparse attention (DSA): an *indexer* scores cached
keys per token and takes the top-k. Five things blocked it, in order:

| # | blocker | cause |
|---|---|---|
| 1 | OOM in weight load | `repack_int4_to_int32` needs ~36 GiB of scratch for a 1.5 GiB result |
| 2 | `only supported on AITER` | indexer's ROCm path is gated `_ON_GFX942` and calls an FP8 kernel |
| 3 | `no attribute 'num_decodes'` (x4) | the AITER sparse builder never populated the decode/prefill split |
| 4 | `NotImplementedError` in `forward_mha` | short prefills take a dense-MHA shortcut the backend lacks |
| 5 | `no attribute 'rocm_aiter_mla_decode_fwd'` | AITER op registration is gated off on gfx90a |

Only #4 needed no code: upstream already has a config flag for it.

## The recipe

```bash
./build/add-aiter.sh   local/vllm-mi210:latest local/vllm-mi210:aiter
docker run --rm -v $PWD/build:/b --entrypoint bash local/vllm-mi210:aiter -lc '
  cp /b/mqa_logits_gfx9.py \
     /opt/python/lib/python3.14/site-packages/vllm/model_executor/layers/ &&
  python3 /b/patch_sparse_indexer_gfx9.py'
# then commit that container, preserving ENTRYPOINT (see add-aiter.sh)
```

Serving, with the numbers that actually worked:

```bash
docker run -d --name glm52 \
  --memory=465g --memory-swap=465g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 440g --security-opt seccomp=unconfined --ulimit memlock=-1 \
  -p 8000:8000 -e VLLM_ROCM_USE_AITER=1 \
  -v /path/to/GLM-5.2-Int4-Int8Mix:/model:ro -v ./cache:/cache \
  <image> /model \
  --tensor-parallel-size 2 \
  --cpu-offload-gb 162 --gpu-memory-utilization 0.97 \
  --max-model-len 32768 --max-num-batched-tokens 512 --max-num-seqs 4 \
  --enforce-eager \
  --attention-config '{"sparse_mla_force_mqa": true}'
```

**The `--memory` cap is not optional.** Without it an overshoot thrashes the host
until ssh and every service stop responding — that happened here and needed a
hard reset. With the cap the container is OOM-killed and the box stays up.

`--cpu-offload-gb 162` is a narrow window. 152 leaves the GPU 2.98 GiB free and
it dies; 172 pushes the host past 465 GB and the container is killed. 162 gives
~50 GiB resident per card, which is what fits.

## What the patches do

**`mqa_logits_gfx9.py`** — the indexer's logits in fp32 instead of FP8. Both
upstream paths are unusable here: `forward_hip` is gated `_ON_GFX942`,
`forward_cuda` needs DeepGEMM (CUDA-only). FP8 is a *storage* format, and gfx90a
stores and converts all four FP8 dtypes exactly (measured), so the arithmetic
runs in fp32 — more accurate than the path it replaces, at the cost of reading
keys as fp32.

The head dimension collapses. Multi-**query** attention shares one k:

```
sum_h w[m,h] * sum_d q[m,h,d]*k[n,d]  ==  sum_d ( sum_h w[m,h]*q[m,h,d] ) * k[n,d]
```

so folding weights into q first makes it one `[M,D] @ [D,chunk]` GEMM — **64x
fewer FLOPs and a 64x smaller intermediate** for GLM-5.2 than per-head logits
reduced afterwards. Validated against a naive oracle: max deviation 5.5e-04,
masks exact, peak ~100 MB flat.

**`patch_sparse_indexer_gfx9.py`** — wires it in and fills the gaps:

- `forward_hip` delegates to `forward_cuda` rather than raising, so vLLM's
  paging, chunking and top-k logic is reused and only the kernel is replaced
- `num_decodes`, `num_prefills`, `num_decode_tokens`, `prefill_max_seq_len` are
  **computed** from `split_decodes_and_prefills` and the same expression
  `sparse_mla_attention.py` uses — never defaulted. A wrong value here does not
  raise; it routes tokens through the wrong kernel and returns plausible,
  wrong output.
- `is_mla_enabled` and `register_ops_once` move to the gfx90a attention
  carve-out so `rocm_aiter_mla_decode_fwd` exists

**The chunked repack** (`fix/moe-wna16-repack-memory` on the vLLM fork) is what
makes the weights load. It is required.

## Limits, plainly

**~1.0 tok/s** (0.81 before expert-selective offload). 162 GiB per card is streamed
from system RAM over PCIe every forward pass. The model runs; it is not a deployment.

**~420 GB of host RAM.** The offloader maps the whole per-GPU shard into shared
memory regardless of the offload value, and the ~21 GiB of non-expert weights
are replicated per rank. 378 GiB of weights needs ~420 GB of RAM plus ~100 GiB
of VRAM — about 545-600 GB combined on a 627 GB machine.

**The MLA kernels are enabled but unvalidated.** `aiter-cdna2`'s
`port-matrix.md` records 11 of 24 `mla` kernels as portable and *"not yet
enabled or validated"*. This turns them on. Coherent output across four prompts
— including correct arithmetic — is evidence, not proof. A numeric comparison
against a non-AITER MLA reference is the outstanding work before trusting this
for anything real.

**32768 context, not 1M.** Untested above that; the KV cache reported 369,856
tokens, so headroom exists, but nothing here validates long context.

## If you want it faster

The answer found is to stop shipping experts over PCIe: llama.cpp `-cmoe` keeps them on the
CPU and reaches **~6 tok/s** short-context — see [LLAMACPP-GFX90A.md](LLAMACPP-GFX90A.md).
What follows is about the vLLM-internal knobs.

The fp32 logits kernel is a reference implementation — a Triton version would
cut the bandwidth cost, and upstream's own `# TODO: move and optimize below
logic with triton kernels` marks the same gap. But the dominant cost is the
PCIe offload, not the logits, so the real fix is more VRAM or more RAM (768 GB
would allow `--cpu-offload-gb 172+`, leaving ~24 GiB free per card).

For a model that actually serves on this hardware, `glm-awq` (GLM-4.6,
compressed-tensors W4A16) needs ~66 GB of offload instead of ~420 GB and scores
6 OK / 0 unsupported in `model-fastpath`.
