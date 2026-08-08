#!/usr/bin/env bash
# OPTIONAL. Adds AITER with gfx90a ASM kernels to an image built by build.sh.
#
# This is separate by choice, not necessity. `import aiter` probes the GPU
# through rocminfo, and plain `docker build` exposes no /dev/kfd -- but BuildKit
# CDI can pass the cards into a build, and that was verified working (see
# docs/GPU-IN-BUILD.md). It is kept out of the Dockerfile so the core image
# builds on any machine with plain docker, which most people rebuilding this
# will have and a spare MI210 is not.
#
#   ./build/add-aiter.sh <input-image> [output-image]
#
# Requires: this machine has the gfx90a cards.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./VERSIONS; set +a

IN="${1:?usage: add-aiter.sh <input-image> [output-image]}"
OUT="${2:-${IN}-aiter}"
C=aiter-add-$$
SITE=/opt/python/lib/python3.14/site-packages

cleanup() { docker rm -f "$C" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "=== aiter  : $AITER_REPO @ $AITER_REF"
echo "=== repatch: $AITER_CDNA2 @ $AITER_CDNA2_REF"

docker run -d --name "$C" --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
  -v /var/cache/mi210-ccache:/ccache -e CCACHE_DIR=/ccache \
  --entrypoint sleep "$IN" infinity >/dev/null

# 1. AITER from source. The base image ships an older amd-aiter whose gates are
#    spelled differently, so the patches below would refuse to apply to it.
docker exec "$C" bash -lc "
  set -e
  git clone --recursive --depth 1 --branch '$AITER_REF' '$AITER_REPO' /src/aiter
  python3 -m pip uninstall -y amd-aiter aiter 2>/dev/null || true
  cd /src/aiter && python3 -m pip install --no-build-isolation . 2>&1 | tail -5
  python3 -m pip install --quiet 'triton==$TRITON_PIN'
  python3 -c \"import triton; assert triton.__version__ == '$TRITON_PIN', triton.__version__\"
  python3 -c 'import aiter; print(\"aiter imports OK\")'
"

# 2. Generate gfx90a code objects from the gfx942 ones AITER ships. Portability
#    is proven by re-assembling each instruction, never assumed; kernels that do
#    not translate are reported, not silently skipped.
docker exec "$C" bash -lc "
  set -e
  git clone --depth 1 --branch '$AITER_CDNA2_REF' '$AITER_CDNA2' /src/aiter-cdna2
  HSA=\$(ls -d $SITE/aiter_meta/hsa 2>/dev/null || ls -d $SITE/aiter/hsa 2>/dev/null || ls -d /src/aiter/hsa)
  python3 /src/aiter-cdna2/repatch_gfx942_to_gfx90a.py \"\$HSA/gfx942\" /tmp/gfx90a | tee /tmp/repatch-report.txt
  grep -q TALLY /tmp/repatch-report.txt
  cp -r /tmp/gfx90a \"\$HSA/gfx90a\"
  ls \"\$HSA/gfx90a\" | wc -l | xargs echo 'gfx90a code objects written:' | tee -a /tmp/repatch-report.txt
  cp /tmp/repatch-report.txt /usr/local/share/repatch-report.txt
"

# 3. Open AITER's gfx90a dispatch, then let vLLM route attention to it. --check
#    re-reads the file: a patch that silently no-ops leaves an image that is
#    merely slow, which is the failure this whole build guards against.
#
#    enable_vllm_aiter_gfx90a.py opens ATTENTION ONLY -- deliberately, because
#    is_enabled() has ~22 consumers and widening it changes branches inside a
#    working optimization. So LINEAR needs its own patch, and without it an int8
#    (W8A8) checkpoint silently serves from vLLM's generic Triton kernel: correct
#    output, no warning, ~1/3 the decode rate. That is exactly the failure this
#    build guards against, and it shipped undetected until 2026-08-07.
docker exec "$C" bash -lc "
  set -e
  python3 /src/aiter-cdna2/patches/enable_gfx90a_asm_paths.py
  python3 /src/aiter-cdna2/patches/enable_vllm_aiter_gfx90a.py
  python3 /src/aiter-cdna2/patches/enable_vllm_aiter_gfx90a.py --check
  python3 /src/aiter-cdna2/patches/enable_aiter_ck_gemm_gfx90a.py
  python3 /src/aiter-cdna2/patches/enable_aiter_ck_gemm_gfx90a.py --check
"

docker exec "$C" bash -lc "rm -rf /src/aiter /src/aiter-cdna2"
# --change restores the entrypoint. `docker commit` snapshots the CONTAINER's
# config, and this container was started with `--entrypoint sleep` to keep it
# alive -- without this the committed image runs `sleep "$@"` and every
# invocation dies with `sleep: unrecognized option ...`.
ENTRY=$(docker inspect "$IN" --format '{{json .Config.Entrypoint}}')
docker commit --change "ENTRYPOINT ${ENTRY}" "$C" "$OUT" >/dev/null
GOT=$(docker inspect "$OUT" --format '{{json .Config.Entrypoint}}')
[ "$GOT" = "$ENTRY" ] || { echo "entrypoint not preserved: want $ENTRY, got $GOT" >&2; exit 1; }
echo "=== built $OUT"
echo "=== verifying on the cards ==="
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  --entrypoint verify-image "$OUT" --max-tier 2
