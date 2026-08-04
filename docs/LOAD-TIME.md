# Model load time on ROCm

**Set `GPU_PINNED_MIN_XFER_SIZE=67108864` in the container environment.**
Without it GLM-4.5-Air takes hours on this hardware. With it, 22 seconds.

`compose.yaml` sets it. This is a HIP runtime tunable, not a vLLM setting, and
it is not documented anywhere upstream.

---

## The mechanism, end to end

`rocprofv3 --hip-trace --hsa-amd-trace` over a single slow copy:

    hipMemcpyWithStream                     1015.74 ms
      hsa_amd_memory_lock_to_pool           1001.28 ms   <-- the cost
      hsa_amd_memory_async_copy_on_engine     14.24 ms   <-- the actual DMA

`.to(device)` on a host tensor larger than `GPU_PINNED_MIN_XFER_SIZE` makes HIP
**page-lock the caller's buffer** for DMA rather than staging through a
pre-pinned one. Locking safetensors' mapping costs a full second. The transfer
itself is 14 ms.

Raise the threshold above the tensor size and HIP uses the staging path:

    default                          1002.06 ms
    GPU_PINNED_MIN_XFER_SIZE=64MiB      0.38 ms      2,637x

## Why it lands on MoE checkpoints specifically

The default threshold is between 1 and 2 MiB. Sweeping transfer size against
safetensors memory:

     1024 KiB     0.30 ms
     2048 KiB  1000.43 ms

GLM-4.5-Air's expert `qweight` tensors are **2.75 MB** — just over the line.
There are 16,512 of them, and `16,512 x 1 s = 4.6 h`, which is the load time
that was actually observed. The `qzeros` and `scales` are 0.02 MB, sail under
the threshold, and cost nothing.

That is the whole reason this presents as a MoE problem. It is not about MoE, or
about quantization, or about the number of bytes. It is about how many
individual transfers land above a 1 MiB threshold. A dense checkpoint has a few
hundred and loses a couple of minutes; this one has sixteen thousand.

## Measured, on the real model

Same model, same serving path, only the environment differs:

| configuration | GLM-4.5-Air, 15 shards |
|---|---:|
| default | 3h18m to reach 9/15 |
| `--safetensors-load-strategy=eager` | 1m 02s |
| `GPU_PINNED_MIN_XFER_SIZE=64MiB` | **22 s** |

The environment variable is preferred over `eager`. It is faster, it needs no
extra RAM, and it fixes every host-to-device path rather than only the
safetensors one. The two compose if you want both.

## What it is not

Five hypotheses were tested and refuted before the real cause was found. They
are recorded because each is plausible and someone will think of them again.

**Not disk I/O.** Cold sequential read of the checkpoint runs at 1.0-1.5 GiB/s.
Streaming an entire shard into page cache *before* touching its tensors changed
nothing: 1004 ms/tensor before, 1002 ms/tensor after.

**Not page-cache misses.** Same experiment as above. The pages were resident and
the copy was still slow. Forcing a fault-in with `.sum()` costs 0.53 ms, so the
data is genuinely present before the slow copy begins.

**Not the AWQ to WNA16 conversion.** `convert_awq_tensor` was the leading
suspect, because `py-spy` put 5 of 5 samples in `moe_wna16_weight_loader`. Timed
in isolation at the real shapes, conversion plus host-to-device transfer accounts
for **10.3 seconds of a 5.5 hour load — 0.05%**.

**Not copy contiguity.** Three separate fixes were tried against
`expert_data.copy_()`, including a backport of upstream PR #47580. All three
measured no change, because they patched a function that is not where the time
goes.

## Why `prefetch` does not help

`--safetensors-load-strategy=prefetch` looks like the right flag and is not.
Reading `weight_utils.py`:

```python
if should_prefetch:
    _prefetch_all_checkpoints(...)   # warms the page cache
...
else:
    with safe_open(st_file, framework="pt") as f:
        param = f.get_tensor(name)   # STILL mmap-backed
```

Prefetch warms the page cache and then falls through to the same mmap path. Page
cache was never the problem, so it fixes nothing. `eager` is the only strategy
that materialises tensors in ordinary memory:

```python
if safetensors_load_strategy == "eager":
    state_dict = load(f.read())      # ordinary memory
```

## Cost of `eager`

It reads each shard fully into RAM before deserialising, so peak RAM must cover
the largest shard (about 5 GiB here) on top of the model. vLLM already refuses to
auto-enable prefetch when the checkpoint exceeds 90% of available RAM, and the
same consideration applies. On a machine with 499 GiB this is not a constraint.

Measured overhead: **5.1 s per shard** to read, against hours saved.

## The narrower fix

If reading whole shards into RAM is unacceptable, cloning each tensor out of the
mapping before the copy has the same effect:

```python
loaded_weight = loaded_weight.clone().to(device)   # 0.30 ms
loaded_weight = loaded_weight.to(device)           # 2002 ms
```

Measured at 163x on cold tensors. `eager` is preferred because it needs no code
change and amortises the read across a whole shard.

## Status upstream

Not reported. Searches of `vllm-project/vllm` for slow ROCm safetensors loading
returned nothing, and PR #46766 ("ModelOpt Llama-4 Checkpoints Take 5+ minutes to
load") is a different bug — a view-chain in the Llama-4 fused-expert path.

**Not file-backed mmap as such.** Mapping the same shard by hand and building a
tensor over it with `torch.frombuffer` is fast at every combination:

    r--s  read-only  MAP_SHARED     0.61 ms
    r--p  read-only  MAP_PRIVATE    0.53 ms
    rw-s  writable   MAP_SHARED     2.29 ms
    rw-p  writable   MAP_PRIVATE    1.16 ms

`/proc/self/smaps` shows safetensors maps `rw-p` where a hand-rolled map is
`r--s`, but reproducing `rw-p` by hand does not reproduce the stall. The
mapping is context; the transfer size crossing the pin threshold is the trigger.

## Alternatives, if the environment variable is unavailable

`--safetensors-load-strategy=eager` reads each shard into ordinary memory before
deserialising (1m 02s measured, ~5 GiB peak per shard). Cloning each tensor out
of the mapping before the copy has the same effect at the call site:

```python
loaded_weight.clone().to(device)   #   0.30 ms
loaded_weight.to(device)           # 2002 ms
```

Note that `--safetensors-load-strategy=prefetch` does **not** help. It warms the
page cache and then falls through to the same path; page cache was never the
problem.

## Reporting

Worth filing in two places, with different asks:

- **vLLM** — set `GPU_PINNED_MIN_XFER_SIZE` on ROCm, or document it. The repro
  is two lines and the fix costs nothing.
- **ROCm** — `hsa_amd_memory_lock_to_pool` taking one second to pin a 2.75 MB
  region is a runtime bug regardless of who calls it. The staging path is
  2,637x faster for the same transfer, which suggests the lock path is doing
  something pathological rather than merely being slower.

Minimal reproduction:

```python
from safetensors import safe_open
import torch
with safe_open(shard, framework="pt") as f:
    t = f.get_tensor(key_of_a_tensor_over_1MiB)
    t.to("cuda")             # ~1000 ms
    t.clone().to("cuda")     #    ~0.3 ms
# or run the whole process with GPU_PINNED_MIN_XFER_SIZE=67108864
```
