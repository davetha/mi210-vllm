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
echo "=== aiter  : $AITER_REPO @ $AITER_REF"
echo "=== repatch: $AITER_CDNA2 @ $AITER_CDNA2_REF"
echo

# bake reads VERSIONS through the environment; the gated `verified` stage is
# the default target, so the safe thing happens without a flag.
BASE_IMAGE="$BASE_IMAGE" VLLM_FORK="$VLLM_FORK" VLLM_REF="$VLLM_REF" \
AITER_REPO="$AITER_REPO" AITER_REF="$AITER_REF" \
AITER_CDNA2="$AITER_CDNA2" AITER_CDNA2_REF="$AITER_CDNA2_REF" \
TAG="${TAG##*:}" TRITON_PIN="$TRITON_PIN" \
  docker buildx bake -f build/docker-bake.hcl gfx90a --load

# The in-build gate runs without a GPU, so the numeric suites were skipped.
# Run them now, with the cards visible, before anyone deploys this.
echo
# Tiers 0-1 ran inside the build. Tier 2 needs cards.
echo "=== tier 2: numeric acceptance on real hardware ==="
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -v /tmp/qualification:/out -e RECORD=/out/qualification.json \
  --entrypoint verify-image "$TAG" --max-tier 2
echo "qualification record: /tmp/qualification/qualification.json"

DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' "$TAG" 2>/dev/null || echo "")
if [ -n "$DIGEST" ]; then
  sed -i "s|^DERIVED_IMAGE=.*|DERIVED_IMAGE=$DIGEST|" VERSIONS
  echo "recorded DERIVED_IMAGE=$DIGEST"
else
  echo "NOTE: no RepoDigest yet - push the image, then re-run to record it."
  echo "      compose.yaml must reference a digest, not '$TAG'."
fi
