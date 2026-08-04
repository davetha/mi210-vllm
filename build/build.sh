#!/usr/bin/env bash
# Build the one compiled layer, verify it, and record its digest.
#
# Consumers never run this. They run `docker compose up` against the digest
# this script writes into VERSIONS.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./VERSIONS; set +a

# bake builds ${REGISTRY}/vllm-mi210:${TAG}. Derive both from one argument so
# the name used after the build cannot drift from the name bake produced.
FULL="${1:-ghcr.io/davetha/vllm-mi210:$(date +%Y%m%d)}"
REGISTRY="${FULL%/*}"          # ghcr.io/davetha   (or the bare name if no slash)
TAG="${FULL##*:}"              # the tag only
[ "$REGISTRY" = "$FULL" ] && REGISTRY="local"
IMAGE="${REGISTRY}/vllm-mi210:${TAG}"

echo "=== image  : $IMAGE"
echo "=== base   : $BASE_IMAGE"
echo "=== vllm   : $VLLM_FORK @ $VLLM_REF"
echo "=== aiter  : $AITER_REPO @ $AITER_REF"
echo "=== repatch: $AITER_CDNA2 @ $AITER_CDNA2_REF"
echo

# bake reads VERSIONS through the environment; the gated `verified` stage is
# the default target, so the safe thing happens without a flag.
BASE_IMAGE="$BASE_IMAGE" VLLM_FORK="$VLLM_FORK" VLLM_REF="$VLLM_REF" \
AITER_REPO="$AITER_REPO" AITER_REF="$AITER_REF" \
AITER_CDNA2="$AITER_CDNA2" AITER_CDNA2_REF="$AITER_CDNA2_REF" \
REGISTRY="$REGISTRY" TAG="$TAG" TRITON_PIN="$TRITON_PIN" \
  docker buildx bake -f build/docker-bake.hcl gfx90a --load

# The in-build gate runs without a GPU, so the numeric suites were skipped.
# Run them now, with the cards visible, before anyone deploys this.
echo
# Tiers 0-1 ran inside the build. Tier 2 needs cards.
echo "=== tier 2: numeric acceptance on real hardware ==="
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /tmp/qualification:/out -e RECORD=/out/qualification.json \
  --entrypoint verify-image "$IMAGE" --max-tier 2
echo "qualification record: /tmp/qualification/qualification.json"

# Tier 3: does it actually serve? The unit tiers are necessary and not
# sufficient -- they all passed once on an image whose engine could not start,
# because AITER was requested and absent. Only an end-to-end start catches that.
# Skipped unless SMOKE_MODEL points at a small local checkpoint.
if [ -n "${SMOKE_MODEL:-}" ]; then
  echo
  echo "=== tier 3: serve smoke test ($SMOKE_MODEL) ==="
  C=smoke-$$
  docker rm -f "$C" >/dev/null 2>&1 || true
  docker run -d --name "$C" --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
    -v "$(dirname "$SMOKE_MODEL")":/smoke:ro \
    -e GPU_PINNED_MIN_XFER_SIZE=67108864 -e VLLM_ROCM_USE_AITER=0 \
    -p 8399:8000 --entrypoint vllm "$IMAGE" serve "/smoke/$(basename "$SMOKE_MODEL")" \
      --served-model-name smoke --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size "${SMOKE_TP:-2}" --max-model-len 8192 >/dev/null

  ok=0
  for _ in $(seq 1 120); do
    if docker logs "$C" 2>&1 | grep -q "Application startup complete"; then ok=1; break; fi
    if docker logs "$C" 2>&1 | grep -qE "Engine core initialization failed|Traceback"; then break; fi
    sleep 10
  done
  if [ "$ok" -ne 1 ]; then
    echo "  serve smoke test        FAILED - engine did not start"
    docker logs "$C" 2>&1 | grep -oE "RuntimeError:.*|ValueError:.*" | head -3
    docker rm -f "$C" >/dev/null 2>&1; exit 1
  fi
  OUTPUT=$(curl -s --max-time 60 http://localhost:8399/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"smoke","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | python3 -c 'import json,sys;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null || echo "")
  docker rm -f "$C" >/dev/null 2>&1
  case "$OUTPUT" in
    *Paris*) echo "  serve smoke test        OK  (completion: $(echo "$OUTPUT" | head -c 40))" ;;
    "")      echo "  serve smoke test        FAILED - no completion"; exit 1 ;;
    *)       echo "  serve smoke test        WARN - served, unexpected text: $(echo "$OUTPUT" | head -c 60)" ;;
  esac
fi

DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo "")
if [ -n "$DIGEST" ]; then
  sed -i "s|^DERIVED_IMAGE=.*|DERIVED_IMAGE=$DIGEST|" VERSIONS
  echo "recorded DERIVED_IMAGE=$DIGEST"
else
  echo "NOTE: no RepoDigest yet - push the image, then re-run to record it."
  echo "      compose.yaml must reference a digest, not '$IMAGE'."
fi
