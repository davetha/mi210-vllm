# Speculative decoding on MI210

Speculation is the single largest decode lever on this hardware — larger than any kernel
change measured on these cards. On Qwen3.8-27B it is worth **2.4x**, and the optimum is
not the setting most guides suggest.

Measured on **one** MI210 (gfx90a), `qwen38-27b-ablit-w8a8-gdnint8-lmhead`, TP=1,
`cudagraph_mode: FULL_DECODE_ONLY`:

| speculation | tokens/step | ms/step | tok/s |
|---|---|---|---|
| none | 1.00 | 27.4 | 36.4 |
| MTP N=1 | 1.85 | 32.4 | 57.1 |
| MTP N=2 | 2.38 | 35.0 | 68.2 |
| dflash N=4 | 2.94 | 45.3 | 65.0 |
| dflash N=8 | 3.23 | 41.4 | 78.0 |
| **dflash N=12** | **3.51** | 43.5 | **80.6** |
| dflash N=16 | 3.70 | 49.5 | 74.9 |

`dflash` at **N=12** is the pick. N=16 still accepts more tokens per step, but the draft
cost grows faster than the acceptance gain.

## The draft model

`dflash` needs a separate draft checkpoint. Nothing needs publishing — it is already on the
Hub:

```bash
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir /models/qwen38-dflash2
```

`DFlash2DraftModel`, 5 layers, 3.85 GB BF16. `incoai/Qwen3.8-27B-DFlash2` mirrors it at the
same byte size. Quantized variants exist (FP8, W4A16, NVFP4, MXFP4) — **do not use them**,
see below.

DFlash2 is native in vLLM from v0.28.0rc2; it needs no fork patch. The old `dflash2-int`
cherry-pick is retired (see `patches/registry.yaml`).

```bash
--speculative-config '{"method": "dflash",
                       "model": "/models/qwen38-dflash2",
                       "num_speculative_tokens": 12}'
```

## Keep the draft in BF16

Quantizing the draft to INT8 is tempting — it is 3.85 GB read several times per step, and
a draft cannot corrupt output (the target verifies every token, so the only thing at risk
is the acceptance rate). Measured:

| draft | ms/step | tokens/step | tok/s |
|---|---|---|---|
| BF16 | 43.9 | 3.57 | **81.3** |
| INT8 | **40.5** | 3.08 | 76.0 |

The step really does get 7.7% cheaper. Acceptance falls 13.7%, for a net **-6.5%**.

Acceptance is a *first-mismatch* statistic — everything after the first rejected token is
discarded — so it amplifies small errors. Working back from tokens/step, per-token
agreement fell from ~72% to ~68%: a 4.7-point drop cost 19% of accepted tokens. A 5-layer
draft has little redundancy to absorb quantization error, and its job is to *match another
model's* argmax, where any deviation is pure loss.

If you do quantize a dflash draft, `q/k/v` must stay BF16 regardless:
`qwen3_dflash.py` builds `_fused_kv_weight` from a raw slice of the fused `qkv_proj.weight`
and calls `F.linear` on it, bypassing the quantization method — INT8 there fails at load
with `expected mat1 and mat2 to have the same dtype: c10::BFloat16 != signed char`.

## Depth is not a fixed property — re-measure it

"MTP N=2 is worse" was true on this hardware and is now false. It was measured when a step
cost ~92 ms; after the AITER fix the step is ~32 ms, and the trade inverted, because the
marginal cost of a draft position stayed roughly fixed (~1.3 ms) while the value of an
accepted token fell with step time.

**Re-run the depth sweep after anything that moves step time**: a quantization change, an
image change, a kernel fix.

## Benchmark speculation with a no-spec arm

tok/s under speculation is `tokens/step x steps/s`, and those two factors move
independently. A comparison that only reports tok/s cannot tell "the step got slower" from
"acceptance dropped", and they have completely different causes.

This is not hypothetical. Comparing two images on the same checkpoint and flags:

| image | no spec, ms/token | dflash N=12 ms/step | tokens/step | tok/s |
|---|---|---|---|---|
| `mi210.6-aiter` | 27.44 | 43.55 | **3.51** | 80.6 |
| `rocm10-mi210.7-aiter` | **27.26** | 43.50 | **3.03** | 69.7 |

Read as tok/s alone, the newer image looks 14% slower. It is not: its model forward is
marginally *faster*, and the dflash step times match to within 0.1%. The entire gap is
acceptance — a spec-decode difference between vllm 0.28.0rc2 and 0.28.1rc0, not a kernel,
attention or ROCm-10 regression. **Do not chase it as one.**

So: always run a no-spec arm, and read `tokens/step` from the spec counters
(`vllm:spec_decode_num_drafts_total`) rather than inferring throughput.
