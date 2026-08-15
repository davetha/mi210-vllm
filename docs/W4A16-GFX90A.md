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
- `BLOCK_K=128` only survives because the existing
  `if group_size < BLOCK_K: BLOCK_K = group_size` clamp runs AFTER tile
  selection. Verify group sizes 64 and 32, not just 128; without that clamp the
  tail of each tile dequantizes against the wrong scale group and corrupts
  output silently.

This caps near **17% of ceiling**. No tile choice changes ops-per-weight.

## What was ruled out: magic-bias dequant alone

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
