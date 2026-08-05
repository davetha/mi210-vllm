#!/usr/bin/env bash
# OPTIONAL. Adds llm-compressor to an image, for `model-convert --run`.
#
#   ./build/add-convert.sh <input-image> [output-image]
#
# Separate from the serving image on purpose. llm-compressor depends on torch
# and transformers, so installing it can move the versions vLLM was built
# against -- and a serving image with a drifted torch is exactly the failure
# this project guards against. vllm-radiance shipped multi-GPU hangs for five
# releases from precisely that cause.
#
# So: install, then ASSERT that torch and transformers did not move. If they
# did, the image is rejected rather than published.
set -euo pipefail
cd "$(dirname "$0")/.."

IN="${1:?usage: add-convert.sh <input-image> [output-image]}"
OUT="${2:-${IN%%:*}:convert}"
C=convert-add-$$

cleanup() { docker rm -f "$C" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "=== base   : $IN"
echo "=== output : $OUT"

VERSIONS_PY='import torch,transformers,compressed_tensors as ct; \
print(torch.__version__, transformers.__version__, ct.__version__)'

BEFORE=$(docker run --rm --entrypoint python3 "$IN" -c "$VERSIONS_PY")
echo "=== before : torch/transformers/compressed-tensors $BEFORE"

docker run -d --name "$C" --entrypoint sleep "$IN" infinity >/dev/null

# No --no-deps: llm-compressor genuinely needs its dependency tree. The two
# assertions below are what make that safe, rather than hoping.
docker exec "$C" bash -lc "python3 -m pip install --quiet llmcompressor 2>&1 | tail -3" || true

# Assertion 1: it has to actually import. Installing cleanly is not the same
# as working -- on Python 3.14 it installs and then dies in pydantic's
# forward-reference evaluation.
if ! docker exec "$C" python3 -c \
    'import llmcompressor; print("llmcompressor", llmcompressor.__version__)'; then
  echo >&2
  echo "REJECTED: llmcompressor installed but does not import in this image." >&2
  echo "  Measured on 2026-08-04 against the rocm/vllm 0.23 base (Python 3.14):" >&2
  echo "  pydantic 2.13.4 (latest) fails to evaluate 'dict[str, Any]' under 3.14's" >&2
  echo "  annotationlib, so the import raises TypeError before anything runs." >&2
  echo >&2
  echo "  Nothing was committed. Run the generated quantize.py somewhere with a" >&2
  echo "  Python that llm-compressor supports -- it is a standalone script and" >&2
  echo "  does not depend on this image." >&2
  exit 1
fi

AFTER=$(docker exec "$C" python3 -c "$VERSIONS_PY")
echo "=== after  : torch/transformers/compressed-tensors $AFTER"

# Assertion 2: nothing vLLM was built against may move.
if [ "$BEFORE" != "$AFTER" ]; then
  echo >&2
  echo "REJECTED: installing llmcompressor moved the stack." >&2
  echo "  before: $BEFORE" >&2
  echo "  after : $AFTER" >&2
  echo >&2
  echo "vLLM in this image was built against the first set. Measured here, the" >&2
  echo "install downgraded transformers 5.14.0 -> 5.10.1 and bumped" >&2
  echo "compressed-tensors past vllm's ==0.17.0 pin. A serving image with a" >&2
  echo "drifted stack is the failure this project guards against: vllm-radiance" >&2
  echo "shipped multi-GPU hangs across five releases from exactly this." >&2
  exit 1
fi

docker commit "$C" "$OUT" >/dev/null
echo
echo "=== built $OUT  (torch/transformers unchanged)"
echo "Convert with:"
echo "  docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \\"
echo "    -v /path/to/models:/models $OUT \\"
echo "    model-convert /models/source --to W4A16 --out /models/target --run"
