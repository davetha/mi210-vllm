#!/usr/bin/env bash
# Build gates. Runs INSIDE the image, during the build. Any failure fails the
# build, so a half-patched image cannot be produced.
#
# This exists because of a specific failure mode this project has already hit
# twice: a patched-looking image that is simply slow. Round 73 published a
# throughput ratio it had not earned, because the benchmark client image was
# patched and the SERVER image was not. Nothing errored. The number looked fine.
#
# Every check below is either "this marker is present" or "this test passes".
# Neither is a proxy for the other, so both are here.
#
# Tiers, after lemonade-sdk/vllm-rocm's qualification suite:
#
#   0 static    no GPU   markers, imports, native ext present     gating
#   1 gate      GPU       does the gate SELECT the patched path    gating
#   2 numeric   GPU       agreement with reference implementations gating
#
# Only tier 0 runs during `docker build`. Tier 1 looked GPU-free -- it calls a
# pure Python predicate -- but that predicate branches on the detected
# architecture, and vllm.platforms.rocm cannot resolve _GCN_ARCH without
# /dev/kfd. In a GPU-less build every gate returns False and reads as a
# regression. Tiers 1-2 therefore run from build.sh against the real cards.
#
# Claims we have NOT run on the relevant silicon are recorded
# hardware_validated=false rather than asserted. The RDNA4 head_size
# consequence is derived from a constexpr, not measured; gfx942 block_size
# likewise. Shipping that distinction is the point.
set -uo pipefail

MAX_TIER=2
[ "${1:-}" = "--max-tier" ] && MAX_TIER="${2:-2}"
RECORD=${RECORD:-/tmp/qualification.json}

PY=/opt/python/bin/python
SP=$($PY -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
fail=0

note() { printf '  %-58s %s\n' "$1" "$2"; }
gate() { # name, command
    if eval "$2" >/dev/null 2>&1; then note "$1" "OK"; else note "$1" "FAIL"; fail=1; fi
}

echo "=== tier 0: static markers (could this image possibly do X) ==="
gate "vllm imports and _rocm_C loads" \
     "$PY -c 'import vllm, vllm._rocm_C'"
gate "paged-attention reduction extended past npar_loops=8" \
     "grep -q 'reduction_extended=[1-9]' /usr/local/share/build-manifest.txt"
gate "built for gfx90a" \
     "grep -q 'pytorch_rocm_arch=gfx90a' /usr/local/share/build-manifest.txt"
gate "free-kernel head_size rule present" \
     "grep -q '_FREE_KERNEL_HEAD_MULTIPLE' $SP/platforms/rocm.py"
gate "int4 interleave enabled on GFX9" \
     "grep -q 'on_gfx1x() or on_gfx9()' $SP/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
gate "wvSplitK stride guard present" \
     "grep -q 'contiguous_format' $SP/model_executor/layers/utils.py"

# The mi210.5 wave (patches/registry.yaml: w8a16-fp8-gfx90a,
# w4a16-tiles-gfx90a, w4a16-bittrick-gfx90a, w4a16-narrow-m16-gfx90a). Until
# these existed, an image could be missing the entire wave and still pass tier
# 0 green, which is the same "patched-looking but slow" failure the top of this
# file describes -- none of these fail loudly at run time, they just route to a
# slower or wrong-shaped kernel. Confirmed discriminating: all six pass on an
# image built from v0.27.2rc0+mi210.5 and all six fail on one built from
# v0.27.2rc0 with the wave absent.
gate "fp8 W8A16 Triton kernel source present" \
     "test -f $SP/model_executor/kernels/linear/scaled_mm/triton_fp8_w8a16.py"
# The file existing does not mean the dispatcher will ever reach it: upstream
# ships _POSSIBLE_WFP8A16_KERNELS[ROCM] as an empty '# To be added' list, so
# check membership rather than grepping for the class name (which also matches
# the import and the __all__ entry).
gate "fp8 W8A16 kernel registered for ROCm" \
     "$PY -c \"from vllm.model_executor.kernels.linear import _POSSIBLE_WFP8A16_KERNELS as K;from vllm.platforms.interface import PlatformEnum as P;from vllm.model_executor.kernels.linear.scaled_mm.triton_fp8_w8a16 import TritonW8A16Fp8LinearKernel as T;assert T in K[P.ROCM]\""
# Without this, a CUDA-shaped get_device_capability() threshold clears on
# gfx90a's (9,0) numbering and routes fp8 checkpoints into a torch._scaled_mm
# crash instead of the weight-only scheme.
gate "CDNA2 fp8 W8A8->W8A16 dispatcher fallback present" \
     "grep -q 'get_cdna_version() <= 2 and not on_rdna4()' $SP/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
# Absent, gfx90a silently inherits the MI300 tiles (304 CUs assumed vs 104).
gate "W4A16 gfx90a tile table present" \
     "grep -q 'elif on_gfx90a():' $SP/model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
gate "W4A16 magic-bias dequant path present" \
     "grep -q USE_MAGIC_BIAS $SP/model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
gate "W4A16 narrow rung widened to M<=16" \
     "grep -q 'if M <= 16:' $SP/model_executor/kernels/linear/mixed_precision/triton_w4a16.py"
# nvfp4-w4a16-gfx90a is POST-TAG: its branch is ahead of VLLM_REF, so a correct
# image built from the current pin does NOT have it. Reported, never gated --
# gating it would fail every legitimate build. Flip this to a gate() when
# VLLM_REF moves past a23149ed0f.
if [ -f "$SP/model_executor/kernels/linear/nvfp4/triton_gfx90a.py" ]; then
    note "nvfp4 gfx90a kernel present" "OK"
else
    note "nvfp4 gfx90a kernel" "ABSENT (post-tag; not in VLLM_REF)"
fi

# AITER is optional and added later by build/add-aiter.sh, so this reports
# rather than gates. An image without it is fully functional; it just does not
# reach the ASM attention paths.
if [ -s /usr/local/share/repatch-report.txt ]; then
    note "aiter gfx90a code objects present" "OK"
else
    note "aiter gfx90a code objects" "ABSENT (optional; see build/add-aiter.sh)"
fi

echo
HAVE_GPU=no
$PY -c 'import torch;assert torch.cuda.is_available()' 2>/dev/null && HAVE_GPU=yes

if [ "$MAX_TIER" -ge 1 ] && [ "$HAVE_GPU" = no ]; then
    echo "=== tier 1: gate behaviour ==="
    note "no GPU visible" "SKIPPED - arch detection needs /dev/kfd"
fi
if [ "$MAX_TIER" -ge 1 ] && [ "$HAVE_GPU" = yes ]; then
echo "=== tier 1: gate behaviour (does it actually select it) ==="
# A marker says the source is present. Only a run says the gate selects it.
# docs/55 records an image that carried the CK carve-outs whose workers still
# chose the Triton kernel.
gate "gate ACCEPTS head_size 256 (Qwen3-Next needs this)" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert g(torch.bfloat16,256,16,8,65536,0,\"auto\",None,None)'"
gate "gate DECLINES block_size 544 on gfx90a (Qwen3-Next safety)" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert not g(torch.bfloat16,128,544,8,65536,0,\"auto\",None,None)'"
gate "gate ACCEPTS 1M context" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert g(torch.bfloat16,128,16,8,1048576,0,\"auto\",None,None)'"
fi

echo
[ "$MAX_TIER" -ge 2 ] && echo "=== tier 2: numeric acceptance (needs a GPU) ==="
if [ "$MAX_TIER" -ge 2 ] && [ "$HAVE_GPU" = yes ]; then
    # No `set -e` in this script: an unguarded cd would run the tests
    # from the wrong directory and report whatever it found there.
    cd /opt/vllm || { echo "cannot cd /opt/vllm"; return 1; }
    gate "long-context paged attention vs Triton (11 tests)" \
         "$PY -m pytest tests/kernels/attention/test_rocm_paged_attention_long_context.py -q -p no:warnings"
    gate "wvSplitK strided activations (4 tests)" \
         "$PY -m pytest tests/model_executor/layers/test_rocm_unquantized_gemm.py -q -p no:warnings"
    gate "int4 w4a16 MoE quant config (8 tests)" \
         "$PY -m pytest tests/benchmarks/test_benchmark_moe_quant_config.py -q -p no:warnings"
    HW_VALIDATED=true
elif [ "$MAX_TIER" -ge 2 ]; then
    note "no GPU visible" "SKIPPED - run 'docker run --device=/dev/kfd ... verify-image'"
    HW_VALIDATED=false
else
    HW_VALIDATED=false
fi

# NOTE on scope. Tiers 0-2 are unit-level and are necessary but not sufficient:
# they all passed once on an image whose engine could not start, because AITER
# was requested and absent. build.sh adds a tier 3 serve smoke test for exactly
# that, gated on SMOKE_MODEL. Do not treat a green run here as "it works".

# Claims carried by this image that hardware here cannot check. Recorded, never
# asserted -- an untested claim stated as fact is how a project loses trust.
cat >> /tmp/unvalidated.txt <<'UNVAL'
gfx12/RDNA4 free-kernel head_size rule : derived from constexpr, no RDNA4 tested
gfx942 free-kernel block_size behaviour: inferred from upstream test cases only
UNVAL

echo
if [ "$fail" -ne 0 ]; then
    echo "VERIFY FAILED - refusing to produce an image."
    echo "A half-patched image fails by being SLOW rather than by erroring,"
    echo "which is exactly what this gate exists to prevent."
    exit 1
fi
arch=$($PY -c 'import torch;print(torch.cuda.get_device_properties(0).gcnArchName)' 2>/dev/null || echo unknown)
cat > "$RECORD" <<JSON
{
  "result": "pass",
  "max_tier": $MAX_TIER,
  "arch": "$arch",
  "hardware_validated": ${HW_VALIDATED:-false},
  "unvalidated_claims": [
    "gfx12/RDNA4 head_size rule: derived from constexpr, not measured",
    "gfx942 block_size behaviour: inferred from upstream test cases, not measured"
  ]
}
JSON
echo "VERIFY OK  (record: $RECORD, hardware_validated=${HW_VALIDATED:-false})"
