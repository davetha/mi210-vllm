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
set -uo pipefail

PY=/opt/python/bin/python
SP=$($PY -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
fail=0

note() { printf '  %-58s %s\n' "$1" "$2"; }
gate() { # name, command
    if eval "$2" >/dev/null 2>&1; then note "$1" "OK"; else note "$1" "FAIL"; fail=1; fi
}

echo "=== 1. static markers (could this image possibly do X) ==="
gate "vllm imports and _rocm_C loads" \
     "$PY -c 'import vllm, vllm._rocm_C'"
gate "paged-attention reduction extended past npar_loops=8" \
     "grep -q 'npar_loops > 16' /src/vllm/csrc/rocm/attention.cu"
gate "free-kernel head_size rule present" \
     "grep -q '_FREE_KERNEL_HEAD_MULTIPLE' $SP/platforms/rocm.py"
gate "int4 interleave enabled on GFX9" \
     "grep -q 'on_gfx1x() or on_gfx9()' $SP/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py"
gate "wvSplitK stride guard present" \
     "grep -q 'contiguous_format' $SP/model_executor/layers/utils.py"
gate "aiter gfx90a code objects were produced" \
     "test -s /tmp/repatch-report.txt"

echo
echo "=== 2. runtime behaviour (does it actually do X) ==="
# A marker says the source is present. Only a run says the gate selects it.
# docs/55 records an image that carried the CK carve-outs whose workers still
# chose the Triton kernel.
gate "gate ACCEPTS head_size 256 (Qwen3-Next needs this)" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert g(torch.bfloat16,256,16,8,65536,0,\"auto\",None,None)'"
gate "gate DECLINES block_size 544 on gfx90a (Qwen3-Next safety)" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert not g(torch.bfloat16,128,544,8,65536,0,\"auto\",None,None)'"
gate "gate ACCEPTS 1M context" \
     "$PY -c 'import torch;from vllm.platforms.rocm import use_rocm_custom_paged_attention as g;assert g(torch.bfloat16,128,16,8,1048576,0,\"auto\",None,None)'"

echo
echo "=== 3. numeric acceptance (needs a GPU; skipped if none visible) ==="
if $PY -c 'import torch;assert torch.cuda.is_available()' 2>/dev/null; then
    cd /src/vllm
    gate "long-context paged attention vs Triton (11 tests)" \
         "$PY -m pytest tests/kernels/attention/test_rocm_paged_attention_long_context.py -q -p no:warnings"
    gate "wvSplitK strided activations (4 tests)" \
         "$PY -m pytest tests/model_executor/layers/test_rocm_unquantized_gemm.py -q -p no:warnings"
    gate "int4 w4a16 MoE quant config (8 tests)" \
         "$PY -m pytest tests/kernels/moe/test_moe.py -k 'wn16 and 4' -q -p no:warnings"
else
    note "no GPU visible at build time" "SKIPPED - run 'docker run ... verify-image' before deploying"
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "VERIFY FAILED - refusing to produce an image."
    echo "A half-patched image fails by being SLOW rather than by erroring,"
    echo "which is exactly what this gate exists to prevent."
    exit 1
fi
echo "VERIFY OK"
