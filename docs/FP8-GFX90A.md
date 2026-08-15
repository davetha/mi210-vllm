# FP8 W8A16 on gfx90a

**FINAL DRAFT — all three previously-open items are resolved.** The
decode-variant-race section ("Why bf16-native shipped as the default") is
rewritten on KERNEL time (the like-for-like basis; wall-clock comparison
across variants was invalid in this dataset, see that section for why) and
no longer conflicts with the formal #17 run. The M=32 double-decode tile
search landed as the final commit, `a28f57dea8`, which rekeys the whole
ladder on (M, N) with a third tile (`bm32`) that recovers most of the
M=17..32 range — see "Tile ladder" below, including a sampling-methodology
correction that turned out to matter more than the ladder numbers
themselves; what remains of that backlog item is narrower and folded into
the backlog section, not gating ship. The gate_up `BLOCK_K=32` "ISA
attribution" backlog item is **retracted**, not landed — ultrasearch's ISA
dump found the apparent ops/weight jump at M=32/`num_warps=4` is a
counter-normalization artifact, not a real cost; see the deleted item's
replacement note in "Documented backlog." Every TODO slot in this document
now has a real, sourced number. Branch: `rocm-wfp8a16-gfx90a-v0.27.2rc0`,
6 commits ending `a28f57dea8`. Style follows `docs/W4A16-GFX90A.md`
deliberately — same card, adjacent kernel family, and several findings
below either mirror or directly extend that document's.

## Why this exists

gfx90a has no FP8 datapath (no native fp8 matmul instruction, unlike
gfx942/gfx950). Two consequences follow, and this document is really about
both of them:

1. **Correctness**: an fp8 *checkpoint* still needs to load and run somehow.
   Before this work, a compressed-tensors FP8 checkpoint on gfx90a crashed at
   the first forward pass — see "The dispatcher mis-route" below.
2. **Performance**: since there is no hardware fp8 matmul to fall back to,
   "supporting fp8" on this card means *decoding* fp8 weights into a dtype
   the MFMA unit does understand (bf16 or fp16), in the GEMM inner loop, the
   same shape of problem the W4A16 Triton kernel already solved for int4.

## Which kernel actually runs

`TritonW8A16Fp8LinearKernel`, registered in `_POSSIBLE_WFP8A16_KERNELS[ROCM]`
ahead of the (previously empty) list. Confirmed at runtime the same three
ways the W4A16 doc insists on rather than inferring from gating logic:
`init_wfp8_a16_linear_kernel()` on gfx90a; a real `CompressedTensorsW8A16Fp8`
scheme driven through `create_weights -> process_weights_after_loading ->
apply_weights`; and a rocprofv3 trace, now real rather than a promise —
the formal #17 run's counter CSVs (`fp8_counters.py`'s manifest labels the
kernel rows) confirm `triton_fp8_w8a16_gemm_kernel` is what actually runs
and is what every VALUBusy/ops-weight/kernel-time number in "#17 formal
benchmark" below is measured against, not a proxy or an assumption.

`is_supported()` gates strictly on `on_gfx90a()`, same rationale as the
W4A16 kernel: the subnormal-flush workaround and the tile ladder are both
measured facts about this card, not proven on gfx942/RDNA. Widen once
someone measures there.

Two adjacent facts worth knowing before anyone re-quantizes, mirroring the
W4A16 doc's own two:

- Only the **weight-only** compressed-tensors NVFP4-style scheme
  (`CompressedTensorsW8A16Fp8`, selected when the checkpoint's
  `input_activations` is `null`) reaches this kernel. A checkpoint with
  activation quantization present resolves to `CompressedTensorsW8A8Fp8`
  instead, which is a different code path entirely (see the dispatcher
  section below for what that path does on gfx90a).
- The native (non-compressed-tensors) `Fp8Config`/`Fp8LinearMethod` path
  (`quant_method: "fp8"`, e.g. official Qwen FP8 releases) does **not**
  currently route here either, even for the same weight-only shape — see
  "Documented backlog: Fp8Config fallback."

## The dispatcher mis-route (patch `w8a8-cdna2-fallback`, commit d74ce3db1d)

Before this patch, a compressed-tensors FP8 checkpoint crashed on the first
forward pass:

    torch._scaled_mm is only supported on CUDA devices with compute
    capability >= 9.0 or 8.9, or ROCm MI300+

Root cause, traced rather than assumed: `_check_scheme_supported` compares
`current_platform.get_device_capability().to_int()` against a CUDA compute
capability threshold (89). ROCm reports gfx90a as `(9, 0) -> 90`, which
clears that threshold **by accident** — the check is CUDA-shaped and treats
gfx90a's version numbering as if it meant the same thing CUDA's does. That
routes the checkpoint to `CompressedTensorsW8A8Fp8`, which calls
`torch._scaled_mm` on hardware with no fp8 matmul at all. (Ampere, sm80,
fails the same numeric check for the opposite reason and correctly falls
through to the weight-only scheme — so the fallback the fix restores is
upstream-intended behavior for hardware without native fp8, and gfx90a had
simply landed on the wrong side of a threshold built for a different vendor's
numbering.)

Fix: gate the W8A8 branch on `get_cdna_version() <= 2 and not on_rdna4()`,
the same predicate `ROCmFP8ScaledMMLinearKernel` and
`RowWiseTorchFP8ScaledMMLinearKernel` already use for an analogous decision
(see the FP8 entry-points recon report). This is the same *class* of bug as
the NVFP4 dispatcher fix on the sibling `rocm-nvfp4-a16-gfx90a-v0.27.2rc0`
branch (a CUDA-shaped capability/backend check asserting hardware it doesn't
have) — worth remembering as a pattern, not a one-off, see "Documented
backlog: capability-threshold audit" below.

Separately, `775181e5bd` found the AITER fp8 W8A8 kernels' `is_supported`
messages were misleading on gfx90a for the same underlying reason (a real
`get_cdna_version() > 2` gate exists, just hidden behind an
`@if_aiter_supported` decorator that short-circuits before the message ever
names the architecture) — not a behavior change, a diagnostics fix so a
correctly-rejected kernel doesn't read as "just set this env var and it'll
work," which it will not on this card.

## The decode: two variants, one measured winner

Same shape of problem as the W4A16 magic-bias work: dequantize fp8 weights
into a dtype the MFMA unit accepts, inside the GEMM's inner loop, without
paying more than the load traffic is worth.

### Variant 1 (default): native `fp8e4nv -> bf16` cast, bf16 dot

`b.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)`. Triton lowers this on
AMD as a packed magic-bias sequence (mask `0x7fff7fff`, `lshr 4`, `fmul
2^120` — see `ConvertFpCastOpToLLVM.cpp OcpF8ToBf16SW`) at roughly 4.5 ops
per value. Produces TRUE values directly, so nothing needs folding back out
in the epilogue.

Exact including the e4m3 denormal codes — not obvious, verified rather than
assumed: a denormal code lands in the bf16 *subnormal* range after the
shift, and the `fmul 2^120` renormalizes it exactly, because bf16's 7
mantissa bits hold e4m3's 3 significant bits with room to spare. Measured
254/254 codes exact through the pure cast, and 254/254 exact through
`tl.dot` at all three tile shapes in the ladder.

### Variant 2: `e4m3fn -> fp16` by bit manipulation

Three integer ops (shift, carry-add, bitcast) instead of a conversion
instruction, exploiting that e4m3fn's and fp16's exponent/mantissa fields
are compatible up to a shift and a bias. The resulting decode is the true
value times `2^-8` (a bias-difference artifact of the field-width
mismatch), so a `* 256.0` is needed to recover the true weight — verified
bit-exact against a float64 oracle, 254/254 codes, with and without
per-channel scales.

### Why bf16-native shipped as the default — measured, not assumed

**The valid basis for this comparison is KERNEL time, not wall time — stated
up front because an earlier wall-clock reading of this same question gave a
materially different (and wrong) answer.** `apply_weights` has no `decode`
argument, so the fp16-variant rows in any harness that goes through the
normal dispatch path are forced through a raw wrapper that skips the
reshape/contiguity/dispatch overhead the bf16-native rows pay through
`apply_weights`. Kernel time on these shapes is only 25-79% of wall time, so
that asymmetry is large enough to manufacture systematic-looking fp16 wins
that are actually a harness artifact, not a kernel result. **Wall-clock
comparisons between decode variants are invalid in this dataset for that
reason; kernel time (rocprofv3, identical harness path for both variants) is
the like-for-like basis, and everything below uses it.**

Original 6-shape sample (interleaved A/B, median of 40, three independent
sweeps) that first justified shipping bf16-native, kernel time only:

| shape | M | sweep 1 | sweep 2 | sweep 3 |
|---|---|---|---|---|
| gate_up 34816x5120 | 1 | 1.013 | 1.018 | 1.010 |
| gate_up 34816x5120 | 8 | 1.014 | 1.021 | 1.010 |
| down_proj 5120x17408 | 1 | 1.080 | 1.079 | 1.101 |
| down_proj 5120x17408 | 8 | 1.097 | 1.100 | 1.083 |
| o_proj 4096x4096 | 1 | 1.054 | 1.052 | 1.056 |
| q_proj 6144x5120 | 1 | 1.059 | 1.062 | 1.063 |

(Ratio of fp16 to bf16; below 1.000 would mean the bit trick wins. Note
these are the ratios recomputed on kernel time, not the wall-clock numbers
an earlier draft of this section carried.)

**Confirmed at formal scale.** #17's full 27-row sweep, scored on kernel
time across all 9 dispatches per shape/M/model config: **bf16-native wins or
ties 26/27 rows, with 1 inconclusive** (8B down_proj at M=8 — inside
run-to-run noise, not a clear win either way, called out rather than forced
to a side). The bit trick does not win on kernel time anywhere in this
dataset. Corroborating fingerprint, independent of the timing comparison
entirely: fp16 costs exactly **bf16 + 1.00 ops/weight at M∈{1,8} and + 2.00
ops/weight at M=32** in every row of the ops/weight table below (see "#17
formal benchmark") — a decode variant that always costs strictly more VALU
work per weight loaded is not a plausible wall- or kernel-time winner on a
kernel that is VALU-bound or mixed at most shapes, and the measured result
is consistent with that.

Ties on gate_up (1.3%, inside serving noise) and loses 5-10% everywhere else
in the original sample — the shipping rule (fp16 only on a consistent >5%
win) selects bf16-native, and the formal run's 26/27 confirms the rule chose
correctly. Also the simpler kernel: no fold factor, no subnormal hazard on
either operand, no activation conversion. `decode="fp16"` is kept as an
explicit override, not deleted, because it was the original design, it is
fully verified, and it is the fallback if a future Triton release changes
the native cast lowering.

Why the cheaper decode doesn't win: at these shapes the kernel streams
weights rather than saturating the VALU, so ~1.5 extra ops/weight are mostly
hidden, while variant 2's extra bf16->fp16 activation convert and in-loop
`* 256.0` are not free. MFMA rate is not the tiebreaker either —
`v_mfma_f32_16x16x16bf16_1k` measures 165.5 TFLOP/s against `16x16x16f16`'s
161.5 on this card (spec claims both 181) — so the dtype choice is a
decode-cost question only, not an MFMA-throughput one.

## The MFMA subnormal-flush finding

Applies to variant 2 only; the default (variant 1) is immune (see below) but
the underlying hardware fact is permanent and worth recording for anyone
writing an fp16 MFMA kernel on this target.

The e4m3 denormal codes `0x01..0x07` are `1..7 * 2^-9`. Biased by variant
2's `2^-8` decode, they land at `2^-17..7*2^-17` — fp16 *subnormals* (fp16
min normal is `2^-14`).

Probed directly on this hardware (MI210, gfx90a, torch 2.11.0+rocm7.14.0,
triton 3.7.1): `v_mfma_f32_*f16` **flushes subnormal inputs to zero**.
Feeding the biased values straight to `tl.dot` returned exactly `0.0` for
every code `0x01..0x07` and the correct value from `0x08` (`= 2^-14`, first
normal) upward — identical across all five ladder tile shapes, so it's a
property of the MFMA path, not one instruction selection:

| operand | through tl.dot | through a plain VALU multiply |
|---|---|---|
| 2^-13 | 1.2207e-04 | 1.2207e-04 |
| 2^-14 (min normal) | 6.1035e-05 | 6.1035e-05 |
| 2^-15 (first subnormal) | **0.0000e+00** | 3.0518e-05 |
| 2^-17 | **0.0000e+00** | 7.6294e-06 |
| 2^-24 | **0.0000e+00** | 5.9605e-08 |

The VALU does **not** flush — the same bitcast, multiplied by `256.0` in
fp16 *before* the dot, returns the exact e4m3 value for every one of those
codes. So the fix is one fp16 multiply inside the loop, cancelling the
`2^-8` at the source and leaving no subnormal anywhere: `b_f16 =
bitcast(t, fp16) * 256.0`. Without it, the low seventh of the weight range
silently becomes zero — no crash, just fluent wrong text.

Variant 1 is structurally immune: it produces true bf16 values directly
(nothing to fold), and e4m3's smallest magnitude (`2^-9`) is nowhere near
bf16's subnormal floor (`2^-126`), so the flush cannot bite on the weight
side; the activation is consumed as bf16 with no conversion, so it cannot
bite there either.

**Same finding, independently reproduced for NVFP4** on the sibling branch
(`rocm-nvfp4-a16-gfx90a-v0.27.2rc0`) — e2m1's smaller dynamic range needs a
`2^-14` fold instead of fp8's `2^-8`, same underlying hardware behavior, same
fix shape (a VALU pre-multiply before the value reaches MFMA). Two
independent decode kernels hitting the identical hardware wall is why this
section treats it as a permanent fact about the card, not a kernel-specific
quirk.

Activation range for variant 2 (not applicable to the default): `bf16 ->
fp16` is exact for `|x|` in `[2^-14, 65504]`. Measured cost of the low end
scaling a random K=4096 activation vector against the float64 oracle:

| \|a\| magnitude | % of A subnormal | err |
|---|---|---|
| ~1e+00 | 0.0% | 9.2e-07 |
| ~1e-02 | 0.5% | 3.1e-04 |
| ~1e-04 | 45.8% | 2.5e-01 |
| ~1e-05 | 100.0% | 1.0e+00 |

An absolute floor, not a relative one — bites only once a large fraction of
the whole activation vector sits below `2^-14`, a regime post-RMSNorm hidden
states don't reach. Re-running the identical sweep on the bf16 path (the
shipped default) gives flat ~1e-06 error across the whole range — as
accurate at the 100%-subnormal case as at `|a|~1`. Removing this cliff is a
large part of why bf16-native ships as the default.

## MEASUREMENT BASIS (single source of truth for every GB/s figure)

Ported verbatim from the kernel module header (`4fd2463c87` collapsed two
drifting copies to one table — cite this table, do not restate the numbers
elsewhere). fp8 weight bytes moved per second, at M=1, median of 30, on a
card that is concurrently serving:

| shape | [N,K]-backed view | [K,N] contiguous copy |
|---|---|---|
| gate_up 34816x5120 | 611.7 GB/s | 454.7 GB/s |
| down_proj 5120x17408 | 373.7 GB/s (**stale**) | 224.2 GB/s (**stale**) |
| o_proj 4096x4096 | 169.1 GB/s (**stale**) | 142.3 GB/s (**stale**) |

**STALE rows**: down_proj and o_proj have N < 16384, so at measurement time
they ran the old `(16, 16, 128)` tile. The re-keyed ladder (final commit
`a28f57dea8`) now gives every narrow-rung shape `(16, 32, 64)`, measured
7-9% faster on exactly these shapes — both rows still understate the
current kernel, and their own GB/s figures are still literally unretaken.
Left in place (not deleted) because what they're cited for is the
view-vs-copy *ratio*, and both columns moved together, so the ratio itself
is unaffected by which tile they ran on. That's also why retaking them is
no longer a gating question: `a28f57dea8` directly re-measured the
view-vs-copy comparison at the wider rungs and confirmed it — the view
still wins decode (25-55%), the copy wins only at `(64, 64, 32)` and
`(128, 128, 32)` prefill-shaped tiles, unaffected by that commit's own
sampling-artifact correction (both arms of that specific comparison used
the same tile, so the interleaving effect cancels between them — see
"Tile ladder"). The stale label on these two GB/s rows stands; the
question they were cited to support does not need them retaken to answer.

**Measurement boundary**: at the `triton_fp8_w8a16_gemm` wrapper — host wall
clock, `torch.cuda.synchronize` inside the timed region, including output
allocation and per-call launch/sync. That is lower than the kernel alone and
higher than what a layer actually sees. For gate_up at M=1, the same work
measures roughly:

| boundary | GB/s |
|---|---|
| kernel only (profiler) | ~754 |
| wrapper (this table) | 611.7 |
| through `apply_weights` | ~517 |

Quoted from a separate profiling pass, shown here only to fix what the
wrapper number contains — a number taken at one boundary must never be set
against a number taken at another.

**Weight layout choice** (`[N,K]`-backed view wins over a forced-contiguous
`[K,N]` copy): measured at M=1 on the narrow rung, where the comparison was
actually run. An earlier draft of this section overreached, claiming this
"holds for every rung" — `a28f57dea8` corrected it with a real measurement
at the wider rungs rather than just flagging the gap: it fails at `(64, 64,
32)` and `(128, 128, 32)`, exactly the comparison's own documented kill
condition (a wider `BLOCK_N` than `BLOCK_K` inverts which layout coalesces
better) predicted — the contiguous copy wins there instead, gate_up 1.02x
at M=32 and 1.11x at M=64 on `(64, 64, 32)`, 1.04-1.08x at M=128 on
`(128, 128, 32)` across all four shapes. The view is still kept: layout is
chosen once per layer at load time, decode is the case that repeats, and
the view wins decode by 25-55% against a 2-11% prefill loss. See "Tile
ladder" below for the same finding in full context.

## Tile ladder, keyed on (M, N) — three tiles, fitting the grid exactly

Inherited the W4A16 kernel's ladder shape as a starting point (per the
kernel's own "not independently retuned" note at first landing), then swept
properly against fp8 decode cost across an 11 N-value x 10 M-value grid, all
three tiles float64-oracle-verified before being timed (4.5e-07 max rel
err each). Final commit: `a28f57dea8` (amends `96be2e3c58`, itself a
correction of `4ee0252bf8` — the branch tip is `a28f57dea8`, still 6
commits). This section went through two rewrites before landing here; the
second rewrite is as load-bearing as the ladder numbers themselves, see
"The sampling artifact" below.

Three tiles are in play:

    narrow  (16, 32, 64)  num_warps=2
    bm32    (32, 32, 64)  num_warps=2
    wide    (64, 64, 32)  default warps

**Unambiguous, unchanged across every version of this sweep**: M=9..16 fell
to `BLOCK_M=64`, running a 64-row tile over at most 16 real rows. narrow
beats that at every N measured, by 1.3-1.8x.

**The corrected ladder, final.** Above M=16 the boundary depends on N too,
and — once sampled correctly (see below) — the surface this ladder is fit
to is **monotonic in both M and N**, so simple thresholds reproduce the
per-cell best tile on **all 110 cells of the grid: 0.00% mean loss, 0.00%
worst loss.** The bands:

- **narrow** `(16, 32, 64)`, `num_warps=2` — M<=16 everywhere, **and**
  M=17..32 at N<=6144. Also subsumes the old `N >= 16384` split inside the
  narrow rung itself: the same sweep found `(16, 32, 64)` beats
  `(16, 16, 128)` by 7-9% on o_proj and q_proj, the small-N shapes that
  split existed to serve, so there is one narrow tile at all N.
- **bm32** `(32, 32, 64)`, `num_warps=2` — survives only at N=14336 and
  N=16384 for M=17..32, where it genuinely wins (M=24: 332.4 GB/s against
  narrow's 320.2 at N=14336; 365.7 against 338.7 at N=16384). Keeps
  `grid_m` at 1 through M=32, so weights are read once instead of twice —
  the mechanism behind the M=32 double-decode finding (see "Documented
  backlog," item 1, which this tile mostly closes for the N it wins).
- **wide** — everywhere else, including N=34816 at M=17..32 (M=24: wide
  400.3 GB/s against bm32's 392.0 and narrow's 355.7 — wide wins here, not
  bm32, contrary to an earlier version of this ladder).
- The one threshold not resting directly on a measured crossover: bm32's
  upper N bound, placed at 20000 — bm32 measured best at 16384, wide at
  34816, nothing was sampled between them.
- **Documented gap, not silently assumed away**: this grid covers M<=64.
  The `(128, 128, 32)` rung above M=64 is inherited assumption, not
  measured by this sweep — stated here rather than left implicit.

### The sampling artifact — the biggest lesson in this section

An earlier version of this ladder (the one `96be2e3c58` shipped) reported
the best tile **oscillating** with N at fixed M — bm32, bm32, narrow,
narrow, narrow, narrow, narrow, bm32, narrow, bm32 across N=5120..16384 —
and treated that as a real, if awkward, property of the hardware: no
threshold schedule can be optimal against a genuinely ragged surface, so
that version scored candidate rules against per-cell best instead of
reading off crossovers (0.08% mean / 5.02% worst was its own conclusion).
**That oscillation was the sampling method, not the hardware.**

The grid that produced it rotated the three tiles call-to-call, on the
theory that rotation stops serving drift from favoring whichever arm runs
first. It does that, but it introduced a worse effect: narrow at M>16 has
`grid_m=2` and re-reads its weights, so its throughput depends on those
weights still being resident in L2 — and the wide tile's own access
pattern evicts them between samples. bm32 (`grid_m=1`) was unaffected, so
the handicap landed asymmetrically on exactly the tile the ladder was
trying to place correctly. Measured at N=5120, K=5120, M=24, median of 41:

| configuration | narrow | bm32 | wide |
|---|---:|---:|---:|
| isolated (alone) | 99.9us | 109.8us | 137.6us |
| narrow+bm32 (rotated) | 101.5us | 111.1us | — |
| narrow+wide (rotated) | 123.7us | — | 137.9us |
| narrow+bm32+wide (rotated) | 123.6us | 110.0us | 138.5us |

Production runs one tile repeatedly for a layer and never alternates
mid-stream, so the **isolated** column is the one that matches what
actually happens at inference — narrow's ~22% handicap under rotation
(99.9us isolated vs 123.6-123.7us rotated) is a measurement artifact with
no production analog. Corrected sampling runs each arm in blocks long
enough for cache state to be that arm's own, still alternates which block
goes first so serving drift reaches every arm, and discards 25 warmup
iterations per block — bm32 specifically needs a deeper warmup than that
number implies, its own first-touch call measuring 151us before settling
to 110us.

**Correcting the sampling removed a phantom rather than shifting a
number.** The corrected grid is monotonic; simple thresholds fit it
exactly (0.00%/0.00%, see above). Scored against the *corrected* grid, the
interleaved-sampling-fit ladder from `96be2e3c58` would have been 0.83%
mean / 10.59% worst — that pair of numbers is the honest before/after of
the sampling fix itself, not of a further tuning pass. Two concrete
mistakes it made, both now fixed: it picked bm32 where narrow actually
wins by ~10% (N=5120, N=6144 — e.g. N=5120 M=24: narrow 256.3 GB/s against
bm32's 232.4), and it picked bm32 at N=34816 M=17..32, where wide is best
(M=24: wide 400.3 against bm32's 392.0).

**Ruled out, each measured, before landing on sampling as the cause**:
`num_warps` on bm32 (1/2/4/default — 2 is correct and is what the wrapper
passes), K (5120 vs 17408, same ordering), scale distribution (ones vs
constant vs random, identical), and warmup depth (1/5/50/200, identical).
Isolating the actual variable took ruling out four plausible-looking ones
first.

**What this means for "first version was wrong" as a recurring story in
this section**: `4ee0252bf8` (first version) read a crossover off three
shapes and mis-set a boundary; `96be2e3c58` (second version) caught that
by scoring against per-cell best instead of eyeballing crossovers, but the
grid it scored against was itself corrupted by the sampling artifact
above, so it fit a phantom oscillation with real-looking honesty; this
version (`a28f57dea8`, final) fixed the sampling, found the surface
monotonic, and the exact-fit result is not a coincidence of a smoother
surface — it is what a correctly-measured surface looks like here.
Two independent methodology lessons this session, worth stating together:
wall-clock comparisons across code paths with different overhead profiles
are invalid (see "Why bf16-native shipped as the default" above), and
interleaved A/B sampling is invalid across arms whose performance depends
on cache residency (this section) — isolated or block-alternating sampling
is the protocol that matches how production actually runs a kernel.

**Weight-layout correction, same commit.** The MEASUREMENT BASIS section's
claim that the `[N,K]`-backed view beats a forced-contiguous `[K,N]` copy
"for every rung" was false, not just unmeasured for prefill as an earlier
draft hedged — it holds on the narrow rung (where the comparison was
actually run, at M=1) and **fails** at `(64, 64, 32)` and `(128, 128, 32)`:
measured, the contiguous copy wins there — gate_up 1.02x at M=32, 1.11x at
M=64, and 1.04-1.08x at M=128 across all four shapes, exactly as that
argument's own kill condition (wider `BLOCK_N` than `BLOCK_K` inverts which
layout coalesces) predicted. This comparison is unaffected by the sampling
artifact above — both arms used the same tile, so the interleaving effect
cancels between them and the result stands as measured. The view is kept
anyway: layout is chosen once per layer at load time, decode is the case
that repeats, and the view wins decode by 25-55% against a 2-11% prefill
loss — a trade now stated with numbers instead of an overclaim.

**The honest follow-up is still autotuning, but for a different reason
than an earlier draft of this section gave.** Not because the surface is
ragged — it is not, it is monotonic and fits exactly — but because these
boundaries are fitted to K=5120 on one card, and nothing here claims they
generalize past that.

**Coupling resolved.** An earlier draft of this document flagged a
companion commit extending the *W4A16* kernel's own narrow rung to `M<=16`
by analogy, sitting on this same FP8 branch in violation of the repo's
one-branch-per-patch rule. Resolved: that change is `3e17cd4213`, now on
its own branch (`rocm-w4a16-gfx90a-v0.27.2rc0`), no longer part of this
branch's 6 commits, and no longer a hypothesis — it has its own validating
run (1.37-1.45x GB/s recovered on M=9..16, correctness against the float64
oracle at 1.1e-06/1.6e-06). It deliberately does *not* take fp8's
(M,N)-keyed boundaries or the bm32 tile: int4 decode costs ~19 VALU
ops/weight against fp8's ~4.5, and the occupancy-vs-reuse balance that sets
a boundary moves with decode cost — confirmed by that commit's own M=24
boundary check, where the W4A16 kernel *regresses* 21% on gate_up at
exactly the M the fp8 ladder is still widening through with bm32/wide. The
two ladders are not expected to share boundaries, and now measurably
don't. See `patches/registry.yaml`'s new `w4a16-narrow-m16-gfx90a` entry.

## Tested and rejected

Mirrors the W4A16 doc's own section — recording negative results so nobody
re-spends the time.

- **fp16 bit-trick decode as the default.** Bit-exact, fully verified, never
  faster — see the variant-race table above. Kept as an explicit
  `decode="fp16"` override rather than deleted, since it's the fallback if a
  Triton lowering change ever regresses the native cast.

## Documented backlog

Ranked by the Phase 2 (ultrasearch) pass, prize sized where measured. Every
item below is a real backlog item — nothing here gates ship; the two items
that formerly did (the M=32 tile search and the ISA attribution addendum)
are resolved or retracted, see items 1 and the removed item below.

1. **M=32 double-decode — closed for the measured grid (M<=64), not for
   M>64.** The #17 table above confirms the original finding directly:
   every shape on both models jumps from ~6.8 ops/weight at M∈{1,8} to
   ~13.4-13.7 at M=32 — a clean doubling, not a gradual rise.
   `a28f57dea8`'s corrected ladder addresses M=17..32 with a mix of narrow
   (`16, 32, 64`, at N<=6144) and bm32 (`32, 32, 64`, keeping `grid_m=1` so
   weights are read once instead of twice, but surviving only at N=14336
   and N=16384 — wide wins everywhere else in this M range, including
   N=34816) — fitting the per-cell best exactly (0.00% mean / 0.00% worst)
   over the measured grid, see "Tile ladder" above. **What remains**: the
   grid this fit was measured on covers M<=64 only — the `(128, 128, 32)`
   rung above M=64 is inherited assumption, not measured, and the
   narrow-N shapes the double-decode regime affects most (`kv_proj`-class)
   haven't been extended past this grid's N range. The prize here is
   correspondingly smaller than originally scoped — extending the same
   measured, monotonic methodology to M>64 and to `kv_proj`-shaped narrow
   N, not a from-scratch M=32 fix, and not (per the ladder section's own
   sampling-artifact finding) a case for autotuning over hand-fitting on
   raggedness grounds — the surface fits exactly once sampled correctly.
   **Bonus finding, same ultrasearch pass**: the wide rung's `32x32x8`
   MFMA instruction measures 8.5% slower per MAC than `16x16x16` on this
   card (165.5 vs 152.6 TFLOP/s, `16x16x16` faster) — a mild, independent
   argument for preferring 16x16x16-shaped tiles wherever the ladder's
   occupancy/reuse tradeoff allows it, on top of whatever tile-selection
   wins the search above finds.
2. **split-K for kv_proj-shaped layers.** The (M,N)-keyed tile-ladder work
   is entirely about occupancy: a narrow-enough N under-fills 104 CUs no
   matter which tile is chosen, once N is small enough that even the
   narrowest tile in the ladder can't launch enough workgroups. #17
   confirms `kv_proj` (8B, GQA-narrow N) is the shape that actually hits
   this floor in practice — 5.6-12.3% VALUBusy, the "starved" end of the
   three-regime split above, an order of magnitude below gate_up's
   VALU-bound 46-72%. Untested as a fix here, flagged as the natural next
   occupancy lever once the finer M>32 search in item 1 lands.
3. **bf16 transposed-layout observation (1.26x-1.54x).** Confirmed
   directly from the #17 wall-clock data: `bf16_us` (vLLM's real
   `[N,K].t()` baseline, the layout a plain unquantized linear layer
   actually uses at inference) is consistently 1.05x-1.68x slower than
   `bf16kn_us` (an idealized, freshly-contiguous `[K,N]` copy) across all
   27 rows. That gap is real and is *not* about this kernel — it's about
   vLLM's own bf16 baseline paying a non-contiguous-access tax it doesn't
   have to. **Caveat, stated deliberately before this goes further**:
   verify this against the actual production bf16 code path first (is
   `bf16_us`'s harness faithfully reproducing what a real unquantized
   `LinearBase` layer does, or is it a simplified stand-in?) before
   treating the 1.26-1.54x figure as an actionable win to chase — an
   apples-to-apples confirmation against the real path is the
   prerequisite, not optional polish.
4. **uint32 repack — REJECTED.** Previously listed here as "pending Phase
   2": bringing bf16-native's ~4.00 measured ops/weight down to ~2.00 via
   a repack. Superseded by item 1's own math: recovering the M=32 floor
   was already the larger, cheaper win (a tile-selection fix, not a new
   packed-weight layout requiring its own load-time conversion step and
   its own correctness surface), and dominated enough of the same VALU
   budget that the repack's incremental gain on top of it doesn't clear
   the bar the repack alone once did — the ladder rewrite above confirms
   this rather than reopening it. Recorded as REJECTED rather than
   silently dropped, per this doc's own "tested and rejected" convention
   above.
5. **Fp8Config fallback.** The native (non-compressed-tensors) `Fp8Config` /
   `Fp8LinearMethod` path (`quant_method: "fp8"`, e.g. official Qwen FP8
   releases) does not route to this kernel even when the checkpoint is
   effectively weight-only-shaped. Per the FP8 entry-points recon: on ROCm,
   `Fp8LinearMethod.__init__`'s intended non-fp8-hardware fallback is Marlin
   (CUDA-only), so `use_marlin` is always `False` here and the checkpoint
   falls through to the real A8W8 kernel lists instead — meaning it does not
   hard-fail the way the pre-`d74ce3db1d` compressed-tensors path did, but it
   also never reaches `TritonW8A16Fp8LinearKernel`. Whether that matters in
   practice (does any real `quant_method:"fp8"` checkpoint end up
   effectively weight-only?) is unmeasured; flagged rather than fixed here.
6. **Capability-threshold audit.** `d74ce3db1d` fixed one CUDA-shaped
   capability check (`get_device_capability().to_int() >= 89`) that
   misfires on gfx90a's `(9,0) -> 90` numbering. The NVFP4 dispatcher fix on
   the sibling branch found the identical bug *class* in a different
   function (`init_nvfp4_linear_kernel`'s unconditional Marlin force). Worth
   a deliberate sweep for other CUDA-shaped capability/backend checks in the
   FP8 and adjacent quantization dispatch paths before assuming this class
   of bug is fully closed out — not done here, flagged for the audit task
   (#22, since completed — worth checking whether it already covered this).
7. **(M,N) grid search follow-up — largely superseded.** An earlier draft
   of this backlog flagged an interpolated (unmeasured) N=8192 row in the
   tile-ladder boundary table. `a28f57dea8`'s final 11x10 grid fits the
   per-cell best exactly (0.00% mean / 0.00% worst) across every cell it
   measured, with no interpolated cells remaining in the current ladder —
   the gap this item was scoped around is closed for M<=64. What's left is
   the documented gap stated in "Tile ladder" above: this grid does not
   cover M>64, where `(128, 128, 32)` is inherited assumption, not
   measured — that is the actual remaining scope for a fuller sweep, not
   the original N=8192 interpolation.
8. **`num_stages` sweep — not yet run.** Genuinely open, not resolved by
   #17 or the ladder work: this kernel's `num_stages` has never been swept
   against real hardware, unlike `num_warps` (fixed per-tile, measured) and
   the tile shapes themselves (the whole point of "Tile ladder" above). The
   W4A16 doc's own experience is the guidance to carry forward here: test
   against the FINAL kernel, not an intermediate one — the sampling
   artifact this document's own ladder section found (see "The sampling
   artifact") is a direct illustration of what testing against a
   moving/uncorrected target can produce, and a `num_stages` sweep run
   before `a28f57dea8` landed would have been sweeping the wrong ladder.

**Retracted, not landed — gate_up `BLOCK_K=32` "ISA attribution" item.**
An earlier draft of this backlog carried a pending item attributing
gate_up's VALU-bound regime to a `BLOCK_K=32`-specific instruction cost,
based on an apparent ops/weight jump at M=32 with `num_warps=4`.
Ultrasearch's static ISA dump (`triton.compile()` against an explicit
gfx90a target, no GPU launch) retracts this: the jump is a **counter
normalization artifact**, not a real cost. `SQ_INSTS_VALU` counts per-wave
instruction issues, and comparing ops/weight ratios across configurations
with different `num_warps` silently rescales the count — predicted ratio
was 2.000, measured 1.961, consistent with a wave-count effect rather than
extra instructions. The AGPR hypothesis floated alongside it is also dead:
zero `v_accvgpr` instructions in the dump. **Ops/weight comparisons in this
document are only valid at equal `num_warps`** — noted here once, applies
to every ops/weight table in this document, including the one in "#17
formal benchmark" below.

## #17 formal benchmark

Real `rocprofv3` run against `Qwen3.8-27B` (dense, 4 shapes) and
`Qwen3-8B` (5 shapes, adds `kv_proj`), M ∈ {1, 8, 32}, artifacts at
`/tmp/prof17/*.json` on `qwen38`. Independently re-derived here (not
reusing fp8-decode's own report numbers) via `analyze17.py` against the
raw counter CSVs and a wall-clock aggregator against
`wall_qwen3_{27b,8b}.json` — every number below was recomputed from the
primary artifacts, cross-checked against the pre-labeled JSON fields by
direct recomputation from raw microsecond values where it mattered (see
the gate_up correction below).

### Three-regime VALUBusy split

Replaces any earlier "issue-bound everywhere" framing — the formal run
shows the kernel's bottleneck regime is shape-dependent, not uniform:

| regime | shape | VALUBusy (bf16) | VALUBusy (fp16) |
|---|---|---|---|
| VALU-bound | gate_up | 46.29% – 68.75% | up to 72.05% |
| mixed | q_proj, o_proj, down_proj | roughly 18% – 57%, shape- and M-dependent | not separately tabulated here — see `analyze17.py`'s full 81-row output |
| starved (occupancy-limited) | kv_proj (8B only, GQA-narrow N) | 5.63% – 10.65% | up to 12.25% |

(fp16-variant rows split into their own column here, rather than folded into
the bf16 range as an earlier draft did — cleaner, and matches the split
applied to the ops/weight table below.)

`kv_proj` only exists in the 8B (GQA) manifest — Qwen3.8-27B's dense
attention has no separately-narrow KV projection, so there's no 27B row
to compare against. MemUnitStalled is 0.002%–0.228% across every row in
both models — confirms, at the corrected tile ladder and at formal scale,
the W4A16 doc's own finding that this kernel family is never memory-stalled;
gate_up and kv_proj sit at the two ends of an occupancy spectrum, not a
memory-vs-compute one.

### ops/weight, VALUBusy, kernel time — full table

Median-of-9 dispatches per manifest entry, from the real counter CSVs.
**Counter-normalization caveat, stated once here and applying to every
ops/weight figure in this document**: `SQ_INSTS_VALU` counts per-wave
instruction issues, so an ops/weight ratio is only valid as a comparison
between rows measured at equal `num_warps` — comparing across different
`num_warps` configurations silently rescales the ratio and can look like a
real cost difference when it is not (see the retracted gate_up
`BLOCK_K=32` backlog item, where exactly this artifact produced an
apparent-but-fake VALU cost).

| shape | M | ops/wt (bf16) | ops/wt (fp16) | VALUBusy (bf16) | kernel us (bf16) |
|---|---:|---:|---:|---:|---:|
| q_proj (27B) | 1 | 6.84 | 7.84 | 29.5% | 61.3 |
| q_proj (27B) | 8 | 6.84 | 7.84 | 33.6% | 53.9 |
| q_proj (27B) | 32 | 13.67 | 15.67 | 55.7% | 69.3 |
| gate_up (27B) | 1 | 6.84 | 7.84 | 47.1% | 240.5 |
| gate_up (27B) | 8 | 6.84 | 7.84 | 46.3% | 245.9 |
| gate_up (27B) | 32 | 13.41 | 15.41 | 68.8% | 382.7 |
| down_proj (27B) | 32 | 13.55 | 15.55 | 54.0% | 211.7 |
| kv_proj (8B) | 1 | 6.86 | 7.86 | 5.9% | 37.8 |
| kv_proj (8B) | 8 | 6.86 | 7.86 | 5.6% | 39.5 |
| kv_proj (8B) | 32 | 13.71 | 15.71 | 10.7% | 42.9 |

(Full 81-row table — both models, all 5 shapes, all 3 M, both decode
variants — in `analyze17.py`'s own output; not reproduced in full here to
keep this doc readable. Every row confirms **fp16 costs exactly bf16 + 1.00
ops/weight at M∈{1,8}, and + 2.00 at M=32** — the M=32 doubling is real
and consistent across every shape and both models, not an artifact of one
measurement. See the M=32 double-decode backlog item below for what causes
it and what to do about it.)

### W4A16 comparison, and the gate_up co-residency correction

`fp8_over_w4` (wall-clock ratio, <1 means fp8 faster) across all 27
shape/M/model rows, fp8 mostly wins or ties W4A16 at M=1/8 and loses more
at M=32 (both kernels' decode cost grows with M, but W4A16's grows faster
off a higher base) — **except 27B gate_up, which needed a correction**:

The raw `w4_us` for 27B gate_up at M=1 and M=8 (395.9us, 413.4us) reads
16-18% slower than those same configurations' own `w4_ref_us` reference
values (341.9us, 353.4us) — a co-residency artifact, not a real
regression: those two rows were captured in a separate profiling session
from the rest of the sweep (disclosed here rather than silently using the
contaminated number). M=32's `w4_us` (545.3) already matches its
`w4_ref_us` (544.4, delta +0.16%) — only M=1 and M=8 needed the swap.
Using `fp8_us / w4_ref_us` instead of the contaminated `fp8_over_w4`:

| M | raw fp8/w4 (contaminated) | corrected fp8/w4ref |
|---:|---:|---:|
| 1 | 0.875 | **1.013** |
| 8 | 0.856 | **1.001** |
| 32 | 0.889 | 0.890 (unaffected, already clean) |

**27B gate_up is a TIE against W4A16 at M=1/M=8** (within 1.3%/0.1%), not
the ~12-15% fp8 win the contaminated numbers suggested. M=32 remains a
genuine ~11% fp8 win, unaffected by the correction.

## E2E serving results (#18)

Three runs, in order of increasing checkpoint complexity: two synthetic
tiny weight-only checkpoints to prove the dispatch path resolves correctly
across both compressed-tensors strategies, then a real production-shaped
checkpoint at tip (all 4 overlay files applied) to prove the end-to-end
crash fix and measure serving throughput on live production infrastructure.

**RUN1 — tiny-CHANNEL (fp8-model).** Battery 6/6 coherent. Kernel confirmed
in-container against the exact bind-mounted code the live server loaded,
not a separate process (same technique `d74ce3db1d`'s own verification
used): weight `(5632, 2048)` `e4m3fn`, scale `(5632, 1)` `fp32`. **80.4
tok/s** at M=1, **TTFT 407.5ms**.

**RUN2 — tiny-TENSOR (run by team-lead).** Battery passed (Paris / 105 /
correct square fn). Scheme resolves `CompressedTensorsW8A16Fp8` ->
`TritonW8A16Fp8LinearKernel`, scale `(1,)` `fp32` pre-promotion — the
TENSOR strategy path specifically exercised (promoted to CHANNEL before
the kernel sees it, confirmed by the shape at the kernel boundary, not
assumed from the strategy name alone; see the recon report).

**RUN3 — 8B-FP8-dynamic (run by team-lead, at tip with all 4 overlay
files).** `RedHatAI/Qwen3-8B-FP8-dynamic` — the same checkpoint
`d74ce3db1d`'s own commit message reproduces the crash against. Resolved
`CompressedTensorsW8A16Fp8` from `input_activations {dynamic: true,
strategy: token}` in the live container. Battery correct/coherent
(raw-completions self-continuation noted explicitly as expected sampling
behavior, not corruption). Decode **45.9 tok/s** (prefill-subtracted),
TTFT **484ms** at ~1k-token prompt. Teardown clean, production healthy
throughout.

| | before `d74ce3db1d` (RUN3 checkpoint) | after (RUN3) |
|---|---|---|
| load / first forward pass | **crash** (`torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+`) | serves |
| decode throughput | n/a | **45.9 tok/s** |
| TTFT | n/a | **484 ms** (~1k-token prompt) |

This is the headline result for the dispatcher fix specifically: an
FP8-native checkpoint that could not load on gfx90a at all now serves.
Independent of whatever the formal #17 benchmark says about wall-clock
competitiveness against bf16/W4A16 — this row is why "compatibility" is
listed as a value proposition in its own right below, not just "capacity."

**ISA-attribution slot**: left as a pending addendum, deliberately not
run — see the retracted `BLOCK_K=32` backlog item above for why there is
no longer an open attribution question to address here. M=32
double-decode's own numbers are in the r27/r8b counter CSVs (ops/weight
6.84 -> 13.5 columns) plus `a28f57dea8`'s `grid_m=2` mechanism explanation
in "Tile ladder" above — not restated here.

## Capacity, not latency — the framing this kernel exists under

**Confirmed against the formal #17 run, not a working hypothesis anymore.**
Re-derived directly from `wall_qwen3_{27b,8b}.json` (27 shape/M/model
rows): **fp8 W8A16 never beats contiguous bf16 on any row** (`fp8_over_bf16kn`
— the memory-contiguous `[K,N]` bf16 baseline — exceeds 1.0 on all 27/27
rows) — **fp8 is not being shipped because it's the fastest weight format
on this card.** It does beat the *transposed* bf16 baseline
(`fp8_over_bf16` — `bf16_us`, vLLM's real `[N,K].t()` layout, i.e. what a
plain unquantized linear layer actually does at inference, not an
idealized contiguous copy) on exactly **5/27 rows: both gate_up rows at
M≤8 on each model (27B and 8B), plus 27B down_proj at M=8.** Every other
row loses to both bf16 baselines. (The transposed-vs-contiguous bf16 gap
itself is 1.05x-1.68x across the 27 rows — see the backlog item below;
that's a separate, real observation about vLLM's own baseline layout, not
about this kernel.)

The value case is capacity and compatibility, not speed:

- **Capacity**: fp8 weights are half the size of bf16, so a model that
  doesn't fit in VRAM as bf16 may fit as fp8 — a capability question, not a
  tok/s-per-shape one.
- **Compatibility**: before `d74ce3db1d`, an fp8-native checkpoint didn't
  load on gfx90a *at all* (hard crash, see "The dispatcher mis-route" and
  the E2E before/after table above). Making FP8 checkpoints load and run
  correctly is valuable independent of whether the resulting kernel beats
  bf16 or int4 on wall clock for any particular shape.

This is the plainest, most defensible framing for both PRs: **fp8 W8A16 on
gfx90a is a capacity-and-compatibility feature, not a speed feature.** Say
it exactly this way, not softened into "competitive in some cases."

## Reproducing

Correctness: task #16's float64 oracle harness, `/home/dave/fp8-oracle/` on
`big` (`oracle.py`, `build_fixtures.py`, `verify_kernel_gpu.py`) — 18
fixtures across 5 case families, both decode variants, NaN load-time guard,
fp16-activation-forcing check. All passed for the default (bf16-native)
variant; the fp16 variant's known limitation (activation-subnormal flush at
large K with small-magnitude activation elements) was independently
reproduced there too, consistent with this document's subnormal-flush
section.

Benchmarking: `fp8_counters.py` (session scratchpad) — joins a
`rocprofv3 --pmc SQ_INSTS_VALU VALUBusy MemUnitStalled` counter CSV against
a dispatch manifest and an optional wall-clock JSON. Not yet run against
real hardware output for this document; every number above that says
"measured" comes from the kernel module's own header or its commit
messages, not from this tool's output.
