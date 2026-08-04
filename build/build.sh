#!/usr/bin/env bash
# Build the one compiled layer, verify it, and record its digest.
#
# Consumers never run this. They run `docker compose up` against the digest
# this script writes into VERSIONS.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./VERSIONS; set +a

TAG="${1:-ghcr.io/davetha/vllm-mi210:$(date +%Y%m%d)}"

echo "=== base   : $BASE_IMAGE"
echo "=== vllm   : $VLLM_FORK @ $VLLM_REF"
echo "=== aiter  : $AITER_FORK @ $AITER_REF"
echo "=== repatch: $AITER_CDNA2 @ $AITER_CDNA2_REF"
echo

docker build -f build/Dockerfile -t "$TAG" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_FORK="$VLLM_FORK"   --build-arg VLLM_REF="$VLLM_REF" \
  --build-arg AITER_FORK="$AITER_FORK" --build-arg AITER_REF="$AITER_REF" \
  --build-arg AITER_CDNA2="$AITER_CDNA2" --build-arg AITER_CDNA2_REF="$AITER_CDNA2_REF" \
  .

# The in-build gate runs without a GPU, so the numeric suites were skipped.
# Run them now, with the cards visible, before anyone deploys this.
echo
echo "=== numeric acceptance on real hardware ==="
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  "$TAG" verify-image

DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' "$TAG" 2>/dev/null || echo "")
if [ -n "$DIGEST" ]; then
  sed -i "s|^DERIVED_IMAGE=.*|DERIVED_IMAGE=$DIGEST|" VERSIONS
  echo "recorded DERIVED_IMAGE=$DIGEST"
else
  echo "NOTE: no RepoDigest yet - push the image, then re-run to record it."
  echo "      compose.yaml must reference a digest, not '$TAG'."
fi
