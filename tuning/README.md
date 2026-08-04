# Tuned `fused_moe` configs

**There are no configs here.** That is the finding, not an omission — see
`manifest.json`. Every tuned config produced on this hardware measured neutral
or worse than the untuned heuristic, across roughly 13 GPU-hours. This directory
exists so the next person does not repeat that.

---

## How vLLM finds a config

`get_config_file_name(E, N, dtype, block_shape)` builds exactly:

```
E=<num_experts>,N=<moe_intermediate_size/TP>,device_name=<dev>,dtype=<tag>.json
```

and opens it inside `VLLM_TUNED_CONFIG_FOLDER`, **before** the configs vLLM
ships. If the file is not there, the heuristic runs and nothing is logged at
error level.

## Three rules, each of which has already cost real time

**1. The `dtype` tag is mandatory.** A file whose name lacks it is never opened.
`docs/41` in `mi210-llm-stack` lost 7.5 GPU-hours to exactly this: a full sweep
whose output could not be loaded. The tuned arm reported "no improvement" from a
config that was never read — a false negative indistinguishable from a real one.

**2. Nearest-M matching has no fallback.** A config covering only decode sizes
(M=1,2) is still applied to prefill shapes. That is how a tuned config reached
**0.786x prefill**: not bad tuning, a missing M range.

**3. The filename does not encode `K`.** This is the subtle one.

`K` is the hidden size and the GEMM's reduction dimension, and it is absent from
the key. Two of our own models collide:

| deployment | E | N | topk | **K (hidden)** | per-expert GEMM |
|---|---:|---:|---:|---:|---|
| Qwen3-30B-A3B @ TP=1 | 128 | 768 | 8 | **2048** | M x 2048 x 768 |
| Qwen3-235B @ TP=2 | 128 | 768 | 8 | **4096** | M x 4096 x 768 |

Identical filename. Double the reduction depth. `BLOCK_SIZE_K`, `num_stages` and
`waves_per_eu` are precisely the parameters that respond to K, so a config tuned
at one is not a near-miss at the other.

**Therefore: one folder per deployment, never a flat pile.** `by-deployment/`
subfolders are the only mechanism available, because the filename is the only
key vLLM has. `compose.yaml` points `VLLM_TUNED_CONFIG_FOLDER` at exactly one.

This is not specific to us — upstream ships 317 configs under the same scheme,
so any two models sharing `(E, N, device, dtype)` with different hidden sizes
have the same exposure.

## Before adding a config here

1. Name it with the `dtype` tag, or it will be silently ignored.
2. Cover prefill M sizes, not just decode, or accept a prefill regression.
3. Put it under `by-deployment/<name>/`, and record `K` in `manifest.json`.
4. A/B it against the untuned heuristic and record the numbers. If it is not
   better, say so in `manifest.json` and do not ship it — a config that is
   worse than none is the outcome this directory is documenting.
