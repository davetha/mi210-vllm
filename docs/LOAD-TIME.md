# Model load time on ROCm

**If you load a large MoE checkpoint, pass `--safetensors-load-strategy=eager`.**
Without it, GLM-4.5-Air takes hours on this hardware. With it, minutes.

This is not a tuning preference. It is a work-around for a ROCm behaviour that
makes the default load path pathologically slow, and it is not documented
upstream.

---

## The measurement

vLLM's default safetensors path yields **mmap-backed** tensors and then calls
`.to(device)` on them. On ROCm that copy costs about **one second per tensor,
independent of tensor size**.

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

The underlying behaviour is a ROCm one: a host-to-device copy whose source is
file-backed appears to take a slow path that copies in ordinary memory do not.
The round timings observed (2001.89, 2002.01, 3003.19 ms) look like a retry or
backoff rather than a bandwidth limit, which is worth investigating before
reporting it as a vLLM bug rather than a ROCm one.
