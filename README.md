# vLLM on AMD MI210 (gfx90a / CDNA2)

Deployment stack for running vLLM on 2x AMD Instinct MI210. It carries the
patches this hardware needs — several upstream, several local — pinned, verified
at build time, and documented with the measurements that justify them.

> **Status: scaffold.** The layout, build gates, patch registry and tuning
> manifest are real. `DERIVED_IMAGE` is empty until a verified build runs, and
> `VERSIONS` references fork tags that do not exist yet. Do not deploy this yet.

---

## What this carries

| patch | origin | what it does |
|---|---|---|
| multi-pass paged attention | [vllm#39001](https://github.com/vllm-project/vllm/pull/39001) (Eugene Kuznetsov) | removes the 131,072-token context ceiling |
| gfx90a free-kernel guards | local | declines shapes that compute silently wrong attention on CDNA2 |
| single-pass 131k–262k | local | keeps the common long-context range off the multi-pass path |
| int4 interleave packing | [vllm#43389](https://github.com/vllm-project/vllm/pull/43389) (amd-xavierwang), gate widened | 1.45–4.8x on the int4 MoE kernel, bit-identical output |
| wvSplitK stride guard | [vllm#50618](https://github.com/vllm-project/vllm/pull/50618) (John Qin / Yanyuan Qin) | fixes an out-of-bounds read on strided activations |
| sharded_state TP guard | local | rejects checkpoints saved at a different TP size instead of half-loading them |
| benchmark_moe int8_w8a16 | local, grouping from [vllm#31011](https://github.com/vllm-project/vllm/pull/31011) | the tuner could not run at all before this |

Measurements behind each are in the commit messages on
[davetha/vllm](https://github.com/davetha/vllm), and the investigation history is
in [mi210-llm-stack](https://github.com/davetha/mi210-llm-stack).

## Prior art

The gfx90a work here builds on people who got there first:

- **Eugene Kuznetsov** ([vllm#39001](https://github.com/vllm-project/vllm/pull/39001)) — the multi-pass reduction. Developed on gfx942; this repo validated it on gfx90a and added the CDNA2 guards.
- **amd-xavierwang** ([vllm#43389](https://github.com/vllm-project/vllm/pull/43389)) — int4 interleave packing, gated to RDNA upstream. Measured here on CDNA2 and widened.
- **rlrs** ([vllm#49888](https://github.com/vllm-project/vllm/pull/49888) and others) — upstreaming AITER attention for gfx90a on MI250X.
- The packaging pattern — digest-pinned base, overlays as source of truth, diffs as documentation — is taken from
  [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x).

## Layout

```text
VERSIONS              every pin; base image by DIGEST, never a tag
compose.yaml          the stack consumers run
build/                the ONE compiled layer + its gates
  Dockerfile          rebuilds _rocm_C for gfx90a. NEEDS NO GPU.
  build.sh            build, verify on real cards, record the digest
  add-aiter.sh        OPTIONAL, needs the cards: AITER + gfx90a ASM kernels
  verify.sh           static markers -> runtime gates -> numeric tests
patches/              Python-only overlays (empty: all patches are in the fork)
  registry.yaml       per-patch "is this still needed?" predicates
tuning/               tuned fused_moe configs, and why there are none
probe/                probe_image_patches.sh, also shipped inside the image
upgrade.sh            triage every patch against a new upstream tag
```

## Why the build gates exist

A half-patched image **fails by being slow, not by erroring**. This project has
already published a throughput ratio it had not earned, because a benchmark
client image was patched and the server image was not — nothing errored, and the
number looked reasonable.

So `build/verify.sh` runs during the build and refuses to produce an image
unless it passes, in three tiers that are not substitutes for one another:

1. **static markers** — could this image possibly do X
2. **runtime gates** — does the gate actually select it
3. **numeric acceptance** — 11 + 4 + 8 tests against reference implementations

Two runtime gates encode findings specific to this hardware:

```
gate ACCEPTS  head_size 256    Qwen3-Next needs it; stock vLLM refuses it
gate DECLINES block_size 544   Qwen3-Next safety; silently wrong otherwise
```

## Building it

```bash
./build/build.sh                 # no GPU required; builds anywhere
./build/add-aiter.sh <image>     # optional, requires gfx90a cards
```

The core image needs no GPU to build, which matters because most people who
want it do not have a spare MI210 to build it on. Every patch this project
carries works without AITER.

AITER cannot be a Dockerfile stage: `import aiter` probes the GPU through
`rocminfo`, and `docker build` exposes no `/dev/kfd`, so the import fails during
the build however the stage is written. `add-aiter.sh` therefore runs in a
container that has the cards and commits the result.

## Known limits

- **Model load needs `GPU_PINNED_MIN_XFER_SIZE=67108864`** (compose sets it).
  Above HIP's ~1 MiB pin threshold, `.to(device)` page-locks the caller's buffer
  and `hsa_amd_memory_lock_to_pool` costs ~1 s while the DMA is 14 ms.
  GLM-4.5-Air: **22 s** with it, hours without. See `docs/LOAD-TIME.md`.
  Unreported upstream.

## Licence

Patches derived from vLLM carry vLLM's Apache-2.0 headers; those derived from
AITER carry AITER's MIT header. See individual file headers.
