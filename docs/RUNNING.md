# Running a model

```bash
./run.sh /path/to/your/model
```

That is the whole thing. It serves an OpenAI-compatible API on `:8000`.

Everything on this page was run end-to-end on 2x MI210 on 2026-08-04 against
`local/vllm-mi210@sha256:d83b8a77`; the outputs are copied from those runs.

---

## Without this repo

`run.sh` is a convenience, not a requirement. The image dispatches for itself,
so it works the same under plain `docker run`, Kubernetes or Slurm:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 16G --security-opt seccomp=unconfined \
  -p 8000:8000 -v /mnt/models:/mnt/models:ro -v ./cache:/cache \
  <image> /mnt/models/my-model --tensor-parallel-size 2
```

```bash
<image> /mnt/models/my-model     # a model path serves
<image> bench latency --model …  # a subcommand runs that subcommand
<image> bash                     # an executable is exec'd
```

The settings that are not optional on ROCm are **baked into the image**, not
into this script — `GPU_PINNED_MIN_XFER_SIZE` in particular is the difference
between a 22-second load and a 4.6-hour one, and someone pulling the image does
not have our wrapper. `--host 0.0.0.0` is defaulted for the same reason: a
container that binds loopback starts cleanly and then answers nothing.

Every start prints what the image actually is, because this project's failure
mode is being slow rather than erroring:

```
=== vllm-mi210 (gfx90a / CDNA2)
    vllm_sha=491c3e1960b81e6929f083fa92522388dd38a055
    reduction_extended=6
    pytorch_rocm_arch=gfx90a
    GPU_PINNED_MIN_XFER_SIZE=67108864  VLLM_ROCM_USE_AITER=0
    arch: gfx90a  OK
```

`arch:` reports `no GPU visible` when the devices are missing, and warns when
the card is not gfx90a — this image carries gfx90a code objects only.
`MI210_QUIET=1` silences the banner.

## Before you serve: what will this model actually hit?

Reads `config.json` only — no weights, no GPU allocation, seconds not minutes.
All four of these name the same thing and all four work:

```bash
./run.sh exec model-fastpath /mnt/models/glm-awq --tp 2          # local directory
./run.sh exec model-fastpath Qwen/Qwen3-0.6B                     # HF repo id
./run.sh exec model-fastpath https://huggingface.co/Qwen/Qwen3-0.6B
./run.sh exec model-fastpath https://huggingface.co/Qwen/Qwen3-0.6B/tree/main
```

A `/tree/<rev>` or `/resolve/<rev>/config.json` URL is accepted too, and the
revision in it is honoured rather than discarded. Local paths are mounted into
the container automatically — `run.sh` mounts every argument that names a path
on your machine, so no `-v` is needed.

A full run, against a compressed-tensors W4A16 MoE checkpoint:

```
$ ./run.sh exec model-fastpath /mnt/models/glm-awq --tp 2

/mnt/models/glm-awq
  glm4_moe  92 layers  hidden 5120  heads 96/8  head_size 128  dtype unknown
  running on gfx90a

attention
  block_size   16  custom PA  specialised kernel
  block_size   32  custom PA  specialised kernel
  block_size   64  custom PA  free kernel
  block_size  128  Triton     block_size > 64 is wrong on gfx90a; routed to Triton
  block_size  544  Triton     block_size > 64 is wrong on gfx90a; routed to Triton

context
  202752 tokens needs the multi-pass reduction in this image; stock vLLM caps
  at 131072 on this path

quantization
  compressed-tensors W4A16
  MoE W4A16 hits the int4 interleave path this image widens to GFX9
  (measured 1.45-4.8x on the int4 MoE kernel, bit-identical).

MoE
  160 experts, moe_intermediate_size 1536
  at TP=2 the tuned-config key is E=160,N=768
  the filename does not encode K (hidden=5120), so a folder holding another
  model's E=160,N=768 config would be silently misapplied

recommendations
  - Tuned fused_moe configs measured neutral or worse on this hardware across
    ~13 GPU-hours. Start untuned; see tuning/manifest.json.
```

The verdicts come from calling **vLLM's own predicates in the image**, not from
a written-down copy of the rules. A summary of the gates drifts the moment
either side changes, and a confident wrong answer here is worse than no tool.

The same model in a format that *misses* a fast path says so, and says what to
do instead:

```
quantization
  gptq 4-bit
  This misses the int4 interleave fast path. That patch lives in the
  compressed-tensors MoE path; the awq/gptq path (moe_wna16.py) has not been ported.

recommendations
  - A compressed-tensors W4A16 export of this model would hit the int4 interleave
    path (1.45-4.8x on the int4 MoE kernel, bit-identical). Same weights,
    different packing -- `model-convert --to W4A16`.
```

`--json` gives the same content machine-readably, for a gate in CI or a script
that picks a checkpoint:

```json
{
 "model": "/mnt/models/t35-fp8",
 "arch": "gfx90a",
 "attention": {
  "64":  { "custom_paged_attention": true,  "reason": "free kernel" },
  "544": { "custom_paged_attention": false,
           "reason": "block_size > 64 is wrong on gfx90a; routed to Triton" }
 },
 "recommendations": [
  "Prefer an int4 (AWQ / GPTQ / compressed-tensors W4A16) or int8 W8A8 export of
   the same model. Both have hardware support on gfx90a; fp8 does not."
 ]
}
```

Some verdicts worth knowing before you download 200 GB:

| checkpoint | verdict |
|---|---|
| FP8 anything | MI210 is CDNA2 and has **no FP8 datapath**. It may load; there is nothing to accelerate. |
| compressed-tensors W4A16 MoE | hits the int4 interleave path (1.45–4.8x, bit-identical) |
| AWQ / GPTQ 4-bit MoE | misses it — the port to `moe_wna16.py` is not done |
| compressed-tensors W4A8 | int-quantized path, not the wNa16 path the patch widens |
| W8A8 int8 | hardware support on gfx90a |

## Converting a checkpoint

```bash
./run.sh exec model-convert /path/to/model            # asks, then generates
./run.sh exec model-convert https://huggingface.co/Org/M --to W4A16 --out /models/m-w4a16
```

It picks a scheme from what the model *is* — MoE gets W4A16, because that is the
only route to the interleave kernel — and writes a complete, runnable
`quantize.py`. FP8 is deliberately not offered: there is no datapath for it here.

```
$ ./run.sh exec model-convert /mnt/models/t235-gptq4 --out /mnt/models/t235-w4a16 -y

/mnt/models/t235-gptq4
  qwen3_moe  hidden 4096  MoE, 128 experts  quant gptq

  - Source is already gptq 4-bit.
  - This is a MoE checkpoint, and W4A16 in compressed-tensors format is the only
    path that reaches the int4 interleave kernel.

scheme: W4A16 (recommended)

wrote /mnt/models/t235-w4a16/quantize.py
  scheme  W4A16
  ignore  lm_head, re:.*mlp.gate$, re:.*mlp.shared_expert_gate$
  calib   512 samples x 2048 tokens from HuggingFaceH4/ultrachat_200k

Review it, then run:
  model-convert /mnt/models/t235-gptq4 --to W4A16 --out /mnt/models/t235-w4a16 --run
  or: python3 /mnt/models/t235-w4a16/quantize.py
```

Without `-y` it asks, showing each scheme and marking the recommended one.
`--to W4A16` or `--to W8A8` skips the question entirely.

It sets the calibration ignore list for you, including the MoE router. That one
matters: quantising the gate degrades expert selection **quietly** rather than
erroring.

**`--run` does not work in this image, and says so rather than pretending.**
Measured 2026-08-04: `llmcompressor` installs but does not import under the base
image's Python 3.14 (pydantic 2.13.4 cannot evaluate `dict[str, Any]` in 3.14's
`annotationlib`), *and* installing it downgrades transformers 5.14.0 → 5.10.1 and
moves compressed-tensors past vLLM's `==0.17.0` pin. Either alone disqualifies
it from the serving image.

The generated `quantize.py` is standalone — run it anywhere llm-compressor
works. `build/add-convert.sh` re-checks both conditions and refuses to commit an
image unless both pass, so it will start working when the base image moves.

## Mount a cache directory

The image points the Triton, inductor, vLLM and AITER caches at `/cache`, so one
mount persists all four:

```bash
-v ./cache:/cache
```

Without it they still work but die with the container, and every start
recompiles. `run.sh` and `compose.yaml` both mount `./cache` for you.

## What you need

- **Docker.** That is it for `run.sh`.
- **An image.** `build/build.sh` records its digest in `VERSIONS` as
  `DERIVED_IMAGE`, and `run.sh` picks it up from there. Override with
  `IMAGE=<tag>` to use one you already have.
- **The compose plugin, only if you use `compose.yaml`.** It is frequently
  absent — the MI210 host this was developed on runs Docker 29.1.3 with no
  compose plugin at all. `run.sh` needs only `docker run`, which is why it is
  the documented path.

## Three ways to name a model

```bash
./run.sh /mnt/models/my-model          # a directory on this machine
./run.sh Qwen/Qwen3-8B                 # an HF repo id; downloads to HF_CACHE
./run.sh /mnt/models/big --tensor-parallel-size 2 --max-model-len 200000
```

Anything after the model goes to vLLM untouched, so the entire server CLI is
available without this script knowing about it.

A local directory is mounted **at its own path** inside the container. That is
deliberate: remapping to `/models` would make every log line and error name a
path that does not exist on your host.

## The rest of the vLLM CLI

Serving is the common case, not the only one. Any vLLM subcommand works, and so
does any command at all:

```bash
./run.sh bench latency --model /mnt/models/my-model --input-len 32 --output-len 8
./run.sh chat     --url http://localhost:8000/v1 --model /mnt/models/my-model --quick "hello"
./run.sh complete --url http://localhost:8000/v1 --model /mnt/models/my-model --quick "The capital of France is"
./run.sh run-batch -i prompts.jsonl -o out.jsonl --model /mnt/models/my-model
./run.sh collect-env

./run.sh shell                                  # interactive bash in the image
./run.sh exec probe-image-patches               # what patches does this image carry
./run.sh exec cat /usr/local/share/build-manifest.txt
./run.sh exec python3 -c 'import torch; print(torch.cuda.device_count())'
```

Two things happen automatically, because getting them wrong is the usual reason
these fail:

**Every argument that names a path on your machine is mounted**, not just the
model. `bench --model X --dataset-path Y` works with no special case, because
the scan looks at all arguments rather than a designated one.

**Commands that connect out get host networking.** `chat`, `complete` and
`bench serve` talk to a server rather than being one, so `http://localhost:8000`
has to mean your machine and not an empty container namespace. Commands that
listen (`serve`, `launch`) publish `$PORT` instead.

`./run.sh --help` prints the list. All of the above were run on 2x MI210 on
2026-08-04; `bench latency` on the 0.6B model reported `Avg latency: 0.0244 s`.

## What it sets for you, and why

| setting | value | why |
|---|---|---|
| `GPU_PINNED_MIN_XFER_SIZE` | 67108864 | Not optional on ROCm. Above HIP's ~1 MiB default, `.to(device)` page-locks the caller's buffer and `hsa_amd_memory_lock_to_pool` costs ~1 s per tensor against a 14 ms DMA. On a MoE checkpoint that is hours. See [LOAD-TIME.md](LOAD-TIME.md). |
| `VLLM_ROCM_USE_AITER` | 0 | The core image has no AITER. Setting 1 makes AITER's JIT compile a module that fails and takes engine startup with it. Run `build/add-aiter.sh` first, then set 1. |
| `--max-model-len` | **not passed** | vLLM derives it from the checkpoint. See below. |
| `--tensor-parallel-size` | 1 (vLLM default) | Always works. Raise it for models that need the memory. |
| tuned MoE configs | none | Unset means vLLM uses its own shipped configs — right for a model nobody here has tuned. `DEPLOYMENT=<name>` opts into `tuning/by-deployment/<name>/`. |

## Do not set `--max-model-len` unless you mean it

Leave it alone and vLLM reads the model's own limit:

```
INFO [model.py:1848] Using max model len 40960
```

Setting it **above** what the checkpoint supports is a hard startup failure, not
a clamp:

```
Value error, User-specified max_model_len (262144) is greater than the derived
max_model_len (max_position_embeddings=40960 ... in model's config.json).
To allow overriding this maximum, set the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
```

Set it *below* the model's limit to save KV-cache memory — that is the normal
reason to pass it. Values above 131,072 are what the paged-attention work in
this image exists for; stock vLLM cannot serve them on gfx90a.

## Checking it works

Startup is ~35-75 s for a small model, minutes to tens of minutes for a large
one, dominated by weight loading.

```bash
curl -sf http://localhost:8000/health          # 200 when ready
curl -s  http://localhost:8000/v1/models       # confirms max_model_len
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/your/model",
       "messages":[{"role":"user","content":"hello"}],
       "max_tokens":40,"temperature":0}'
```

The `model` field must be **exactly** the string you passed to `run.sh` — for a
local model that is the full path. `GET /v1/models` tells you what it is.

## run.sh or compose.yaml

Use **`run.sh`** for a model you want to serve now: any path, no config file, no
compose plugin.

Use **`compose.yaml`** for a deployment you run repeatedly — it pins the model,
TP size, context length and tuned-config folder in `.env`, and restarts on boot.
Two constraints to know:

- Its HF cache mount is **read-only**, so the model must already be in the
  cache. It will not download one.
- A model outside the HF cache needs its own bind mount; there is a commented
  line in `compose.yaml` for exactly that.

```bash
cp .env.example .env    # edit MODEL, DEPLOYMENT, TP_SIZE, MAX_MODEL_LEN
docker compose up -d
```

`DEPLOYMENT=generic` is the right choice for anything untuned — the folder is
empty and vLLM falls back to its own configs. Never point it at a folder holding
configs from several models: the tuned-config filename does not encode `K`, so
two models can collide. [tuning/README.md](../tuning/README.md) has the case
where that cost 7.5 GPU-hours.

## Multiple GPUs

`--tensor-parallel-size 2` uses both cards. It requires the model's attention
head count to divide by the TP size, so it is not a free knob — a model with 3
KV heads will not shard across 2 cards.

Raise it when the model does not fit in one card's 64 GiB, not by default.

## When something goes wrong

| symptom | cause |
|---|---|
| `exec: "/path/to/model": is a directory: permission denied` | An image built before the entrypoint existed. Rebuild, or prefix the command with `vllm serve`. |
| `the input device is not a TTY` | `docker run -it` under ssh/cron/CI. `run.sh` uses `-i` always and adds `-t` only when stdin is a terminal — dropping `-i` too would silently discard piped input instead. |
| `max_model_len (N) is greater than the derived max_model_len` | See above — omit the flag. |
| `Free memory 0.3/63.98 GiB` at startup | Something else is holding VRAM. `docker ps` — a forgotten container is the usual answer, not a leak. |
| `module_rmsnorm_quant ... build failed` | `VLLM_ROCM_USE_AITER=1` against an image without AITER. Run `build/add-aiter.sh`, or leave it 0. |
| Load takes hours | `GPU_PINNED_MIN_XFER_SIZE` is not set. [LOAD-TIME.md](LOAD-TIME.md). |
| `vllm serve --help` shows almost nothing | The default help is grouped. Use `--help=all`, or `--help=max-model-len` for one flag. |

To see what the image actually carries:

```bash
docker run --rm --entrypoint probe-image-patches <image>
docker run --rm --entrypoint cat <image> /usr/local/share/build-manifest.txt
```

## What this hardware will and will not do

MI210 is CDNA2 (`gfx90a`). It has no FP8 datapath — that arrived with CDNA3 —
so FP8 checkpoints do not get native acceleration here even when they load.

Models this project has actually served on these cards, with the measurements
recorded in [mi210-llm-stack](https://github.com/davetha/mi210-llm-stack):
GLM-4.5-Air (AWQ int4), Qwen3-Next-80B (head_size 256, which stock vLLM refuses
on this arch), Qwen3-235B (W8A8), Qwen3-30B-A3B.

Two guards in this image are gfx90a-specific and will decline shapes rather than
compute them wrongly. If a model reports a declined shape, that is the guard
doing its job — the free attention kernel silently drops part of the V dimension
when `head_size % 64 != 0`. See the README table.
