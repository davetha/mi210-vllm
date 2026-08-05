# vLLM on AMD MI210 (gfx90a / CDNA2)

Deployment stack for running vLLM on 2x AMD Instinct MI210. It carries the
patches this hardware needs — several upstream, several local — pinned, verified
at build time, and documented with the measurements that justify them.

> **Status: builds and verifies.** Built on 2x MI210 on 2026-08-04; all three
> verification tiers pass with `hardware_validated=true` (23 numeric tests).
> `DERIVED_IMAGE` in `VERSIONS` is still empty because the image has not been
> pushed to a registry — run `build/build.sh` to produce your own, or publish
> one and record its digest.

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
run.sh                serve any model in one command -- docs/RUNNING.md
compose.yaml          the stack consumers run
build/                the ONE compiled layer + its gates
  Dockerfile          rebuilds _rocm_C for gfx90a. NEEDS NO GPU.
  entrypoint.sh       the image's own dispatch + arch preamble
  build.sh            build, verify on real cards, record the digest
  add-aiter.sh        OPTIONAL, needs the cards: AITER + gfx90a ASM kernels
  verify.sh           static markers -> runtime gates -> numeric tests
patches/              NO diffs here -- the patches are branches on the fork,
  registry.yaml       already merged into VLLM_REF. This is their index, with
  README.md           an obsolete_when predicate per patch. Read patches/README.md.
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

## Running a model

```bash
./run.sh /path/to/your/model        # or an HF repo id
```

Serves an OpenAI-compatible API on `:8000`. Needs only `docker run` — no compose
plugin, no config file.

The rest of the CLI is there too, with the GPUs, mounts and ROCm environment
already correct:

```bash
./run.sh bench latency --model /path/to/model
./run.sh complete --url http://localhost:8000/v1 --model /path/to/model --quick "hello"
./run.sh exec probe-image-patches      # or shell, or any command
```

Before downloading something large, ask what it will actually hit — a local
path, an HF repo id or a pasted Hub URL, reading `config.json` only:

```bash
./run.sh exec model-fastpath https://huggingface.co/Qwen/Qwen3-8B --tp 2
./run.sh exec model-convert /path/to/model --to W4A16 --out /models/out
```

`model-fastpath` answers by calling vLLM's own gate predicates inside the image,
so it cannot drift from the code that will actually run. `model-convert` turns
its recommendation into a runnable llm-compressor recipe. Both in
[docs/RUNNING.md](docs/RUNNING.md).

`run.sh` is a convenience, not a requirement — the image dispatches for itself
and carries its own ROCm settings, so a plain `docker run <image> /path/to/model`
behaves identically under Kubernetes or Slurm. It prints what it is at every
start (patch markers, arch check), because a half-patched image fails by being
slow rather than by erroring.

`compose.yaml` is the other way in, for a deployment you run repeatedly. Both
are covered in [docs/RUNNING.md](docs/RUNNING.md), along with the settings that
are not optional on ROCm and the errors worth recognising.

On an HPC site without Docker, see [docs/APPTAINER.md](docs/APPTAINER.md) —
Frontier's MI250X is gfx90a, the same architecture this image targets.

## Everything is built from git sources

This repo stores no binaries and no build artifacts. Every compiled thing in the
image is produced during the build from a git clone at an immutable tag:

| component | source | how it is built |
|---|---|---|
| `_rocm_C` (attention.cu) | `davetha/vllm` @ `v0.26.1rc0+mi210.1` | `pip wheel` in `build/Dockerfile` |
| AITER Python + C++ | `ROCm/aiter` @ `v0.1.19` | `pip install --no-build-isolation .` |
| gfx90a ASM code objects | `davetha/aiter-cdna2` @ `v1.0` | `repatch_gfx942_to_gfx90a.py`, at build time |

`.github/workflows/sources-only.yml` enforces this rather than asserting it: it
rejects any committed binary or file over 256 KiB, requires `BASE_IMAGE` to be
digest-pinned, and checks that all three refs still resolve as tags.

### The one exception, and it is upstream's

AITER's ASM kernels are the exception, and no build anywhere compiles them from
source. `ROCm/aiter` @ `v0.1.19` contains **2,863 prebuilt `.co` code objects and
zero `.s`/`.S` files** — AMD ships these kernels only as binaries, for `gfx942`,
`gfx950` and `gfx1250`. There is no gfx90a build to run and no assembly to run it
on.

So `aiter-cdna2` transforms the code objects rather than compiling them,
re-assembling every instruction to prove portability and reporting the kernels
that do not translate instead of skipping them quietly. Those binaries are
fetched from upstream's git during `add-aiter.sh`; none are stored here.

If AMD ever publishes the sources, this step becomes a compile and the repatcher
can go.

## Building it

```bash
./build/build.sh local/vllm-mi210:dev    # no GPU needed to build
./build/add-aiter.sh local/vllm-mi210:dev  # optional, requires gfx90a cards
```

`build.sh` runs tier 0 inside the build, then tiers 1-2 against the cards, and
records the digest in `VERSIONS` only if everything passes. The qualification
record it writes looks like this:

```json
{ "result": "pass", "max_tier": 2, "arch": "gfx90a:sramecc+:xnack-",
  "hardware_validated": true,
  "unvalidated_claims": [
    "gfx12/RDNA4 head_size rule: derived from constexpr, not measured",
    "gfx942 block_size behaviour: inferred from upstream test cases, not measured"
  ] }
```

`hardware_validated` is false when the build host has no GPU, and the claims
this hardware cannot check are listed rather than asserted.

The core image needs no GPU to build, which matters because most people who
want it do not have a spare MI210 to build it on. Every patch this project
carries works without AITER.

AITER is a separate step by choice. `import aiter` probes the GPU through
`rocminfo` and plain `docker build` exposes no `/dev/kfd` — but BuildKit CDI
*can* pass the cards into a build, verified working here and written up in
`docs/GPU-IN-BUILD.md`. It is kept out so the core image needs no CDI setup, no
labs Dockerfile frontend and no GPU, which is the difference between "anyone can
rebuild this" and "anyone with an MI210 can rebuild this".

## Known limits

- **Model load needs `GPU_PINNED_MIN_XFER_SIZE=67108864`** (compose sets it).
  Above HIP's ~1 MiB pin threshold, `.to(device)` page-locks the caller's buffer
  and `hsa_amd_memory_lock_to_pool` costs ~1 s while the DMA is 14 ms.
  GLM-4.5-Air: **22 s** with it, hours without. See `docs/LOAD-TIME.md`.
  Unreported upstream.

## Licence

Patches derived from vLLM carry vLLM's Apache-2.0 headers; those derived from
AITER carry AITER's MIT header. See individual file headers.
