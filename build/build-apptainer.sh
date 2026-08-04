#!/usr/bin/env bash
# Build the Apptainer image, with every pin taken from VERSIONS so the SIF and
# the Docker image are built from the same sources.
#
#   ./build/build-apptainer.sh [output.sif]
#
# UNVERIFIED: never run to completion here -- no Apptainer on the development
# machines. See docs/APPTAINER.md.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./VERSIONS; set +a

OUT="${1:-vllm-mi210.sif}"

command -v apptainer >/dev/null || { echo "apptainer not found"; exit 1; }

# {{ }} templating and --build-arg landed in Apptainer 1.3. On older versions
# the placeholders are passed through literally and the build fails somewhere
# confusing, so say so here instead.
V=$(apptainer --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
case "$V" in
  1.[012]|0.*) echo "apptainer $V is too old: --build-arg needs 1.3+"; exit 1 ;;
esac

echo "=== base : $BASE_IMAGE"
echo "=== vllm : $VLLM_FORK @ $VLLM_REF"
echo "=== out  : $OUT"
echo
echo "Needs no GPU. Expect tens of GB of scratch: set APPTAINER_TMPDIR to"
echo "somewhere with room, not \$HOME, if your site quotas it."
echo

# --build-arg feeds the {{ }} placeholders in the def file, so VERSIONS stays
# the single source of truth rather than the values being copied into two files.
apptainer build \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_FORK="$VLLM_FORK" \
  --build-arg VLLM_REF="$VLLM_REF" \
  --build-arg MAX_JOBS="${MAX_JOBS:-24}" \
  "$OUT" build/vllm-mi210.def

echo
echo "=== built $OUT"
echo "Verify on real cards before trusting it:"
echo "  apptainer exec --rocm $OUT cat /usr/local/share/build-manifest.txt"
echo "  apptainer exec --rocm $OUT python3 -c 'import torch; print(torch.cuda.get_device_properties(0))'"
