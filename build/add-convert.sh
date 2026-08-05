#!/usr/bin/env bash
# OPTIONAL. Adds llm-compressor to an image, for `model-convert --run`.
#
#   ./build/add-convert.sh <input-image> [output-image]
#
# Kept out of the serving image by default because a quantizer is not something
# a serving image needs, and because a plain `pip install llmcompressor` moves
# torch/transformers -- a serving image with a drifted stack is the failure this
# project guards against, and it cost vllm-radiance five releases of multi-GPU
# hangs. Two measures make adding it safe rather than hopeful:
#
#   --no-deps        llm-compressor's requirements are already satisfied by the
#                    image, so resolving them again only risks moving them.
#                    Verified: torch/transformers/compressed-tensors unchanged.
#   a 3-line patch   llmcompressor 0.12.0.1 does not import on Python 3.14; see
#                    build/patch_llmcompressor_py314.py for the why.
#
# Both are then ASSERTED, so this fails loudly if either stops holding.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./VERSIONS; set +a

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
docker cp build/patch_llmcompressor_py314.py "$C:/tmp/patch_lc.py" >/dev/null

docker exec "$C" bash -lc "
  set -e
  python3 -m pip install --quiet --no-deps llmcompressor==${LLMCOMPRESSOR_REF} 2>&1 | tail -3
  python3 /tmp/patch_lc.py
  rm -f /tmp/patch_lc.py
"

# Assertion 1: it imports. Installing cleanly is not the same as working -- on
# Python 3.14 it installs and then dies in pydantic's annotation evaluation.
if ! docker exec "$C" python3 -c \
    'import llmcompressor; print("llmcompressor", llmcompressor.__version__)'; then
  echo >&2
  echo "REJECTED: llmcompressor still does not import after the 3.14 patch." >&2
  echo "  Nothing was committed. The generated quantize.py is standalone -- run" >&2
  echo "  it wherever llm-compressor works." >&2
  exit 1
fi

# Assertion 2: everything the generated recipe imports actually resolves.
# --no-deps means a genuinely missing dependency would otherwise surface hours
# into a calibration run rather than here.
if ! docker exec "$C" python3 -c '
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
print("recipe imports OK")'; then
  echo >&2
  echo "REJECTED: --no-deps left something the generated recipe needs missing." >&2
  exit 1
fi

# Assertion 3: nothing vLLM was built against may move, and vLLM still loads.
AFTER=$(docker exec "$C" python3 -c "$VERSIONS_PY")
echo "=== after  : torch/transformers/compressed-tensors $AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo >&2
  echo "REJECTED: installing llmcompressor moved the stack." >&2
  echo "  before: $BEFORE" >&2
  echo "  after : $AFTER" >&2
  echo "vLLM in this image was built against the first set." >&2
  exit 1
fi
docker exec "$C" python3 -c 'import vllm; print("vllm still imports:", vllm.__version__)'

# --change restores the entrypoint. `docker commit` snapshots the CONTAINER's
# config, and this container was started with `--entrypoint sleep` to keep it
# alive -- without this the committed image runs `sleep "$@"` and every
# invocation dies with `sleep: unrecognized option '--to'`.
ENTRY=$(docker inspect "$IN" --format '{{json .Config.Entrypoint}}')
docker commit --change "ENTRYPOINT ${ENTRY}" "$C" "$OUT" >/dev/null

# Assertion 4: the committed image dispatches like its parent.
GOT=$(docker inspect "$OUT" --format '{{json .Config.Entrypoint}}')
if [ "$GOT" != "$ENTRY" ]; then
  echo "REJECTED: entrypoint not preserved (want $ENTRY, got $GOT)" >&2
  docker rmi "$OUT" >/dev/null 2>&1 || true
  exit 1
fi

echo
echo "=== built $OUT  (stack unchanged, quantizer working, entrypoint $GOT)"
echo "Convert with:"
echo "  docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \\"
echo "    -v /path/to/models:/models $OUT \\"
echo "    model-convert /models/source --to W4A16 --out /models/target --run"
