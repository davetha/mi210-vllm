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
