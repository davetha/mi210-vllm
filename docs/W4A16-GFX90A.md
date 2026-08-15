# W4A16 on gfx90a

Where the dense int4 GEMM actually spends its time on MI210, what was fixed,
and what was ruled out. Measured 2026-08-15 on 2x MI210, vllm 0.27.2rc0+mi210.1.

## The numbers everything else is compared against

| | GB/s | note |
|---|---|---|
| HBM2e spec | 1600 | per card |
| practical read ceiling | **1386** | 87% of spec; use THIS as 100% |
| bf16 GEMM control | 776-830 | 56-60% of ceiling |
| int8 weights, read only | 1259 | 91%, gate_up shape |
| shipped W4A16 GEMM | 80-165 | **5.7-11.9%** |

The gap between the last two rows is the whole subject of this document.

## Which kernel actually runs

`TritonW4A16LinearKernel`, confirmed at runtime three ways rather than inferred
from gating logic: `choose_mp_linear_kernel()` on gfx90a; a real
`CompressedTensorsWNA16` scheme driven through `create_weights ->
process_weights_after_loading -> apply_weights`; and a rocprofv3 trace showing
`triton_w4a16_gemm_kernel` at 99.3% of GPU time.

`_POSSIBLE_KERNELS[ROCM]` is `[RDNA3W4A16, RDNAHybridW4A16, TritonW4A16, Conch,
Exllama]`. Both RDNA entries gate on `on_gfx1x()`, which is False here, so
selection falls through to the Triton kernel. Unchanged by
`VLLM_ROCM_USE_AITER=1` -- there is no AITER entry in this family at all.

Two adjacent facts worth knowing before anyone re-quantizes:

- A **W8A8** checkpoint does NOT use this kernel. It routes through
  `TritonInt8ScaledMMLinearKernel`. Decode figures measured on a W8A8 model say
  nothing about W4A16.
- A **W8A16** checkpoint does not either: `TritonW4A16LinearKernel` accepts only
  uint4b8/uint4, so 8-bit weight-only selects `ConchLinearKernel`.

## It is NOT memory-bound, and there is no round-trip to remove

The obvious hypothesis -- that the kernel dequantizes to memory and reads it
back -- is wrong. rocprofv3 `--pmc FETCH_SIZE WRITE_SIZE`:

| shape | w4 bytes | fetched | fetch/w4 |
|---|---|---|---|
| q_proj | 15,728,640 | 16,235,776 | 1.03x |
| gate_up | 89,128,960 | 91,965,152 | 1.03x |
| down_proj | 44,564,480 | 45,997,312 | 1.03x |

Fetched matches weights+scales+activations to within 0.01%, writes are ~0. The
kernel already moves the minimum possible bytes. **No rewrite can win on
traffic.**

`--pmc VALUBusy MemUnitBusy MemUnitStalled`: `MemUnitStalled` is **0.0% on every
shape**. Scratch_Size 0 (no spilling), LDS 0, VGPR 92.

The wall is instruction count: **~19.1 VALU lane-ops per int4 weight**
(53.2M wave-instructions x 64 lanes / 178.3M weights for gate_up), VALUBusy 63%,
`SQ_INSTS_VALU:SQ_INSTS_MFMA` about **19:1**.

**MFMA is ~5% of instructions.** A hand-written MFMA kernel would optimise the
one unit that is neither saturated nor stalled. This is the single most
important line in this document.

## What was fixed: tile selection (patch `w4a16-tiles-gfx90a`)

gfx90a had no branch in the kernel's tile table and inherited MI300's, which
assumes 304 CUs against MI210's 104. With `BLOCK_N=64` a narrow-N layer launches
only 80-96 workgroups onto 104 CUs -- under one per CU. Shrinking `BLOCK_N`
raises that to 320-384.

    q_proj/o_proj/down_proj  BLOCK_M=16 BLOCK_N=16 BLOCK_K=128 num_warps=2
    gate_up                  BLOCK_M=16 BLOCK_N=32 BLOCK_K=64  num_warps=2

1.43-1.82x per shape by rocprofv3 kernel time, 1.63x per decoder layer
(1453us -> 889us). Output bit-identical, worst relative difference 0.000e+00
across 36 tensors.

Two things that bit during this work and are cheap to repeat:

- The configs were searched at M=1. Applied to all `M <= 32` they **regress**
  gate_up 125 -> 99 GB/s, because once M fills the machine the small BLOCK_N
  stops buying occupancy and only costs reuse. Hence the `M <= 8` bound.
  (The bit-trick patch has its own bound on the same shapes, keyed on BLOCK_M
  rather than M -- see below.)

The magic-bias bound is keyed on **BLOCK_M, not M**, and is structural rather
than a tuning artifact. The rank-1 correction costs O(BLOCK_M x BLOCK_N) per
K-tile, while the decode work it removes is O(BLOCK_K x BLOCK_N) and
independent of BLOCK_M. Saving fixed per K-tile, cost linear in BLOCK_M, so the
benefit/cost ratio degrades monotonically and a crossover is guaranteed: worse
at larger tiles, never better, and not recoverable by tuning. M and BLOCK_M
agree only because of how the current ladder is written; they diverge as soon
as M is not padded 1:1 to the tile, at which point an M-keyed bound silently
either disables a profitable path or enables an unprofitable one. Prefill takes
the plain path for free under this rule.
- `BLOCK_K=128` only survives because the existing
  `if group_size < BLOCK_K: BLOCK_K = group_size` clamp runs AFTER tile
  selection. Verify group sizes 64 and 32, not just 128; without that clamp the
  tail of each tile dequantizes against the wrong scale group and corrupts
  output silently.

This caps near **17% of ceiling**. No tile choice changes ops-per-weight.

## Magic-bias dequant: only works WITH the scale hoist

An earlier revision of this document called magic-bias a dead end. That was
measured with the per-weight scale multiply still in place, which makes it
misleading on its own. Corrected here.

**Magic-bias alone changes nothing. Magic-bias plus a scale hoist cuts
ops/weight by 2.4-3.2x.** The hoist is the essential partner, for an
architectural reason: **gfx90a has no bf16 VALU arithmetic** -- bf16 is
MFMA-operand only -- so a per-weight `* scales` forces an fp32 round-trip,
which is exactly the cost the bit-trick is meant to remove. The trick without
the hoist just moves work around.

The hoist is legal because `BLOCK_K <= group_size` is already enforced by the
existing clamp:

    sum_k a_k (n_k - 8) s  ==  s * (dot(a, v) - 136 * rowsum(a)),  v = 128 + n

If that clamp ever moves, the hoist becomes silently wrong.

Measured (rocprofv3, M=1, at the tuned tiles):

| shape | shipped | V1 (bit-trick + hoist) | V2 (+ offline pre-interleave) | cut |
|---|---|---|---|---|
| gate_up | 15.57 | 6.19 | 4.94 | 3.2x |
| q_proj | 14.93 | 7.18 | 6.31 | 2.4x |
| down_proj | 14.80 | 7.05 | 6.18 | 2.4x |

Time improves only 1.32x per decoder layer (896 -> 681us), because the wall
moves rather than disappearing -- see the next section. Total against shipped,
with the tile patch, is 2.13x (1453 -> 681us).

The 19.1 vs 14.8-15.6 ops/weight discrepancy is RESOLVED: both were right, at
different tile configs. Measured all three states in one run:

| shape | shipped MI300 tiles | tuned tiles | + bit-trick |
|---|---|---|---|
| q_proj | 19.10 | 15.74 | 8.51 |
| gate_up | 19.10 | 16.20 | 7.33 |

19.10 at the shipped tiles reproduces the original profiling exactly. So the
tile patch itself already cuts ops/weight -- a larger BLOCK_K amortises
per-tile setup over more weights -- and the bit-trick then roughly halves it
again. Full chain from shipped on gate_up is 19.10 -> 7.33, 2.6x.

A corollary worth remembering when reading any ops/weight number in this
document: it is only meaningful alongside the tile config it was measured at.

V1 is also strictly MORE accurate than the shipped kernel (1.95e-03 vs
2.49e-03 relative), because the hoist does one multiply per output instead of
one per weight. Exactness was shown with a one-hot probe: with `a = e_k` the
output is exactly row k of the dequantized matrix, which removes accumulation
error and isolates the dequant. A GEMM-level comparison cannot show this
because it mixes dequant error with fp32-vs-fp64 accumulation.

No inline asm was needed; Triton already emits `v_perm_b32` and fused and-or
forms for this on its own.

## The wall after the dequant fix: MFMA operand staging

Cutting the ALU cost did not reach the memory floor. It moved the bottleneck:

    VALUBusy       70.7% -> 35.9%   ALU freed, as designed
    MemUnitStalled  0.0% ->  0.0%   still not HBM-bound
    MemUnitBusy    73.4% -> 88.6%   the new wall

The loop is now bound by `tl.dot` staging the B tile through LDS for the MFMA
operand layout: 16x `ds_read_u16` behind 6 `s_barrier` per iteration. At M=1 a
16x16x16 MFMA computes 16 output rows in order to use one. The memory unit
saturates issuing narrow transactions while moving only 388 GB/s.

**The obvious fix was tested and fails.** Dropping `tl.dot` for plain FMAs --
no LDS, no barriers -- is 1.4-2.4x SLOWER, because Triton's cross-lane
reduction codegen costs more than the MFMA path it replaces. Reaching the
memory floor would need a GEMV operand layout with wide `ds_read_b128`-class
access, which could not be expressed in Triton.

Triton also has no `int32 -> 2x bf16` reinterpret, which costs back roughly
1.5 ops/weight of the pre-interleave saving. That is why V2 beats V1 by only
3% in time despite 1.3 fewer ops/weight, and why V2 is not worth an on-disk
layout change.

## Superseded: magic-bias measured WITHOUT the hoist

The obvious next lever -- replace shift/mask/sub/convert with the Marlin-style
mantissa trick (`0x4300 | n` read as bf16 is exactly `128+n`, since bf16 has 8
explicit mantissa bits and 143 needs 8) -- **is bit-exact but not faster**.
Measured at `max_abs_diff = 0` versus the current decode, and marginally slower
in time. The conversion step is not where the ~19 ops go; the
three chained `tl.interleave` broadcasts and the shift/mask are.

So if the remaining win exists it is in **removing the interleave chain** via an
offline nibble pre-interleave, not in the int-to-float conversion.

## Benchmarking traps found the hard way

**A standalone decode benchmark cannot answer this question.** Decode in
isolation is memory-bound (~538 GB/s, 38% of ceiling, and identical whether or
not the unpack is present); decode inside the GEMM is ALU-bound. Different
regimes. Measure in-GEMM or not at all.

**A decode benchmark that stores its output measures the store.** A first
attempt wrote 67MB of bf16 and every variant looked the same. Reduce instead.

**Diffing a new dequant against the old one proves nothing about either.**
`verify_dequant.py` builds an independent float64 reference straight from the
packed bits on CPU; the shipped kernel agrees with it to 3.3e-03, consistent
with bf16 inputs and fp32-vs-fp64 accumulation. Use that, not a kernel-to-kernel
diff. Note `set_default_device('cuda')` will silently put a "CPU reference" on
the GPU -- pin it and assert.

## Tested and rejected: num_stages on the wide-N config

The tile patch sets `num_warps` but leaves `num_stages` at Triton's default of
2. Against the TILES-ONLY kernel, forcing `num_stages=1` on the wide-N
(gate_up) config measured 445.4us -> 386.1us, about 1.15%.

That win does not survive the bit-trick. Re-measured on the merged
tiles+bit-trick kernel, `num_stages=1` gives gate_up 257 -> 252 GB/s at M=1 and
249 -> 242 at M=8, i.e. slightly WORSE. Removing the dequant cost changes the
kernel's character enough to invert the tradeoff -- VALUBusy falls 70.7 ->
35.9% while MemUnitBusy rises to 88.6%, so there is no longer idle issue
capacity for deeper pipelining to fill.

Recorded because it is a plausible-looking knob that someone will otherwise
re-try. The general lesson: a tuning result measured against one kernel does
not transfer to a kernel whose bottleneck has moved.

## Reproducing

Harnesses live at `/home/dave/` on the MI210 box: `wna16_profile.py` (modes
select/prod/sweep/sweep2/trace/trace2/counters), `verify_tiles.py` (bit-exact
vs the current kernel, right gate for tile changes), `verify_dequant.py`
(independent float64 oracle, right gate for arithmetic changes), `decode_ops.py`
(decode variants -- see the trap above before trusting it). Analyzers
`analyze.py`, `t3.py`; CSVs in `prof-wna16/`.

    rocprofv3 --pmc FETCH_SIZE WRITE_SIZE --kernel-include-regex triton_w4a16 \
      --output-format csv -d /work/prof-wna16 -o pmc -- python /work/wna16_profile.py counters

## Open

Whether an offline nibble pre-interleave gets ops/weight from ~19 to ~4. At ~4,
gate_up's ALU time falls from ~301us to ~63us, meeting its 64us memory floor --
i.e. it would finally become memory-bound. Rough end-to-end for a 64-layer dense
model on one card, linears only: shipped ~10.7 tok/s, tuned ~17.6, memory-bound
ceiling ~131.

Whatever is attempted, it stays in Triton. There is no precedent anywhere in
these repos for hand-authoring gfx90a MFMA -- the existing ASM work is binary
repatching of AMD's precompiled gfx942 code objects, and the CK work is C++
template instantiation. Neither starts from a blank asm file.
