# Model load time on ROCm

**If you load a large MoE checkpoint, pass `--safetensors-load-strategy=eager`.**
Without it, GLM-4.5-Air takes hours on this hardware. With it, minutes.

This is not a tuning preference. It is a work-around for a ROCm behaviour that
makes the default load path pathologically slow, and it is not documented
upstream.

---

## The measurement

vLLM's default safetensors path yields tensors backed by safetensors' own
mapping, and then calls `.to(device)` on them. On ROCm that copy costs about
**one second per tensor, independent of tensor size**.

Measured on 2x MI210 (gfx90a), cold tensors from untouched shards of
`GLM-4.5-Air-AWQ`:

| source of the CPU tensor | `.to(device)` median |
|---|---:|
| mmap-backed (`safe_open` + `get_tensor`, the default) | **1001.93 ms** |
| ordinary memory (`eager`, i.e. `load(f.read())`) | **0.24 ms** |

That is a **4,226x** difference on the same data, same GPU, same shard family.

Because the cost is **per tensor and not per byte**, it scales with tensor count:

| checkpoint | expert tensors | projected load |
|---|---:|---:|
| GLM-4.5-Air AWQ, default | 50,826 | **14.15 h** |
| GLM-4.5-Air AWQ, eager | 50,826 | **1.47 min** (including 15 shard reads) |

A dense model has a few hundred tensors and pays a few minutes, which reads as
"loading is a bit slow". A MoE checkpoint has fifty thousand and pays hours.
That is why this looked like a MoE bug for a long time. It is not.

## What it is not

Four hypotheses were tested and refuted before the real cause was found. They
are recorded because each one is plausible and someone will think of them again.

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

## What the mechanism is, and what is still unknown

Narrowed as far as black-box testing goes. **The cost follows the storage, not
the tensor wrapper.** Any view of a safetensors tensor is slow; any copy out of
it is fast:

    safetensors tensor as-is                 1016.24 ms
      [:] slice        (same storage)        1002.07 ms
      .view_as         (same storage)        1000.73 ms
      from_numpy(...)  (same storage)        1001.24 ms
      .clone()         (new storage)            0.33 ms
      empty().copy_()  (new storage)            0.52 ms

**It is not simply "file-backed mmap".** Mapping the same shard by hand and
building a tensor over it with `torch.frombuffer` is fast at every combination
tested:

    r--s  read-only  MAP_SHARED     0.61 ms
    r--p  read-only  MAP_PRIVATE    0.53 ms
    rw-s  writable   MAP_SHARED     2.29 ms
    rw-p  writable   MAP_PRIVATE    1.16 ms

`/proc/self/smaps` shows safetensors maps `rw-p` where a hand-rolled read-only
map is `r--s`, but reproducing `rw-p` by hand does *not* reproduce the slowness.
So the mapping flags are necessary context, not the trigger.

**No syscall accounts for the second.** `strace -f -T` across the slow copy
shows nothing long except idle worker futexes; the time is spent in userspace
inside the HIP runtime. Combined with the consistency of the figure (1001.28,
1001.46, 1001.62, 1001.96 ms) this reads as a poll loop with a one-second
timeout rather than a bandwidth or fault cost.

The exact HIP call is not identified. `AMD_LOG_LEVEL=4` produces no output in
this build, so naming it needs a runtime with HIP API tracing.

## Reporting

There are three places this could go and the evidence does not yet settle which:

- **vLLM** — actionable today: default to `eager` on ROCm, or document it.
  This is the report worth filing first, because the fix is known and the repro
  is two lines.
- **safetensors** — it maps the checkpoint writable-private though it never
  writes. Read-only mapping may sidestep whatever the runtime dislikes.
- **ROCm** — a one-second userspace stall in a host-to-device copy is a runtime
  bug wherever the trigger lives. Filing this needs the HIP call named first.

Minimal reproduction:

```python
from safetensors import safe_open
import torch
with safe_open(shard, framework="pt") as f:
    t = f.get_tensor(some_key)
    t.to("cuda")            # ~1000 ms
    t.clone().to("cuda")    #    ~0.3 ms
```
