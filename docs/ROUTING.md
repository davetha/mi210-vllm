# Why vLLM GLM-5.2 is stuck near 1 tok/s (routing + speculative decode)

Two ways to push the vLLM path in [DSA-GFX90A.md](DSA-GFX90A.md) past ~1 tok/s were tried
and both came back negative. The bottleneck is bandwidth, and neither locality nor
speculation can buy it back on this hardware. The way out is to stop shipping experts over
PCIe — see [LLAMACPP-GFX90A.md](LLAMACPP-GFX90A.md).

Measured 2026-08-06 on 2x MI210, `glm52-int4int8`, the clean `start-glm52.sh` config
(~1.0 tok/s baseline).

## Would a small VRAM expert cache help? No.

Instrumented the MoE router (`build/route_probe_sitecustomize.py`, bind-mounted into both
workers via `PYTHONPATH`) to record per-layer expert-id selections on a mixed workload
(English / code / math / Chinese / Spanish / German, `build/drive_routing.py`), then
replayed them (`build/analyze_routing.py`).

The routing is only **mildly skewed**:

| metric | value | meaning |
|---|---|---|
| per-layer normalized entropy | ~0.91 | near-uniform (1.0 = uniform over 256) |
| per-layer Gini | ~0.51 | moderate skew (1.0 = one expert) |
| top-10% of experts per layer | covers ~42% of accesses | the hot set is real but not dominant |
| top-20% | ~60% | diminishing |

An LRU replay over the decode stream tracks the static top-K curve closely — there is
temporal locality, but not enough to concentrate the working set. GLM-5.2 activates **8 of
256 experts per token**, so even a perfect cache of the hottest ~26 experts/layer still
leaves most accesses uncached. And there is only **~2.6 GiB of VRAM free** per card after
the offload window, which is a handful of experts, not a useful cache.

**Verdict:** caching will not help. The cost is reading 8 experts × ~76 layers per token
over PCIe, regardless of which experts they are.

## Does speculative decode help? No — net loss.

Ran n-gram speculative decode (`build/start-glm52-spec.sh`,
`num_speculative_tokens=3, prompt_lookup_max=3`) and benched it (`build/bench_spec.py`):

| kind | baseline | n-gram spec |
|---|---|---|
| repetitive (where spec should win) | ~1.0 | **0.85** |
| structured | ~1.0 | 0.99 |
| open-ended | ~1.0 | 0.95 |
| **overall** | ~1.0 | **0.92** |

Speculative decode *loses* here, and loses worst on the repetitive prompts where it is
supposed to win. The win condition for spec decode is "verification of the drafted tokens
is cheap because the hot experts are already on the GPU." Here there is no hot expert set
on the GPU — every accepted draft still streams its experts over PCIe — so the extra
verification candidates just add bandwidth on the wrong side of the wall. Native MTP
(GLM-5.2 carries the `blk.78.nextn.*` block) would accept more broadly than n-grams but
hits the same bandwidth wall on accept. Door closed by data.

## Bottom line

The PCIe offload wall is architectural for this model-on-this-hardware, not a routing or
scheduling artifact. To go faster, keep the experts on the CPU side of the bus (llama.cpp
`-cmoe`) or add VRAM/RAM so more of the model is GPU-resident.
