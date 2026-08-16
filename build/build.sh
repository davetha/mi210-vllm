#!/usr/bin/env bash
# Build the one compiled layer, verify it, and record its digest.
#
# Consumers never run this. They run `docker compose up` against the digest
# this script writes into VERSIONS.
set -euo pipefail
cd "$(dirname "$0")/.."
# VERSIONS is the default, not the law: BUILDING FROM UPSTREAM tells you to
# override VLLM_FORK/VLLM_REF/VLLM_IS_FORK on the command line. Sourcing the
# file would clobber those, so capture them first and put them back after.
_ovr_fork="${VLLM_FORK:-}"; _ovr_ref="${VLLM_REF:-}"; _ovr_isfork="${VLLM_IS_FORK:-}"
set -a; . ./VERSIONS; set +a
if [ -n "$_ovr_fork" ];   then VLLM_FORK="$_ovr_fork";     echo "note: VLLM_FORK overridden from the environment"; fi
if [ -n "$_ovr_ref" ];    then VLLM_REF="$_ovr_ref";       echo "note: VLLM_REF overridden from the environment"; fi
if [ -n "$_ovr_isfork" ]; then VLLM_IS_FORK="$_ovr_isfork"; echo "note: VLLM_IS_FORK overridden from the environment"; fi

# Flags before the image name. Defaults are the safe/honest ones: hardware
# tiers run, and :latest is NOT moved.
NO_GPU=0        # --no-gpu     : build + tier 0 only, no cards required
STAMP_LATEST=0  # --latest     : also tag ${REGISTRY}/vllm-mi210:latest
while [ $# -gt 0 ]; do
  case "$1" in
    --no-gpu|--build-only) NO_GPU=1; shift ;;
    --latest)              STAMP_LATEST=1; shift ;;
    -h|--help)
      echo "usage: build/build.sh [--no-gpu] [--latest] [registry/name:tag]"
      echo "  --no-gpu   build and run tier 0 only; skips tiers 2-3, which need /dev/kfd"
      echo "  --latest   also move ${REGISTRY:-ghcr.io/davetha}/vllm-mi210:latest to this build"
      exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *)  break ;;
  esac
done

# bake builds ${REGISTRY}/vllm-mi210:${TAG}. Derive both from one argument so
# the name used after the build cannot drift from the name bake produced.
FULL="${1:-ghcr.io/davetha/vllm-mi210:$(date +%Y%m%d)}"
REGISTRY="${FULL%/*}"          # ghcr.io/davetha   (or the bare name if no slash)
TAG="${FULL##*:}"              # the tag only
[ "$REGISTRY" = "$FULL" ] && REGISTRY="local"
IMAGE="${REGISTRY}/vllm-mi210:${TAG}"

echo "=== image  : $IMAGE"
echo "=== base   : $BASE_IMAGE"
echo "=== vllm   : $VLLM_FORK @ $VLLM_REF (is_fork=${VLLM_IS_FORK:-1})"
echo "=== aiter  : $AITER_REPO @ $AITER_REF"
echo "=== repatch: $AITER_CDNA2 @ $AITER_CDNA2_REF"
echo

# bake reads VERSIONS through the environment; the gated `verified` stage is
# the default target, so the safe thing happens without a flag.
BASE_IMAGE="$BASE_IMAGE" VLLM_FORK="$VLLM_FORK" VLLM_REF="$VLLM_REF" \
VLLM_IS_FORK="${VLLM_IS_FORK:-1}" \
AITER_REPO="$AITER_REPO" AITER_REF="$AITER_REF" \
AITER_CDNA2="$AITER_CDNA2" AITER_CDNA2_REF="$AITER_CDNA2_REF" \
# docker-bake.hcl's gfx90a target tags both :${TAG} and :latest. Moving :latest
# is a publishing decision, not a build step -- an iteration build should not
# silently repoint the tag other people pull -- so override the tag list unless
# --latest was asked for.
BAKE_SET=()
[ "$STAMP_LATEST" -eq 1 ] || BAKE_SET=(--set "gfx90a.tags=$IMAGE")

REGISTRY="$REGISTRY" TAG="$TAG" TRITON_PIN="$TRITON_PIN" \
  docker buildx bake -f build/docker-bake.hcl gfx90a --load "${BAKE_SET[@]}"

# The in-build gate runs without a GPU, so the numeric suites were skipped.
# Run them now, with the cards visible, before anyone deploys this.
echo
# Tiers 0-1 ran inside the build. Tier 2 needs cards -- which is why this is
# skippable: without --no-gpu, the canonical entrypoint cannot complete on a
# machine that has no MI210s, even though the image itself builds fine there.
if [ "$NO_GPU" -eq 1 ]; then
  echo "=== tier 2: SKIPPED (--no-gpu) ==="
  echo "  This image is NOT hardware-qualified. Run it on the cards before deploying:"
  echo "    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \\"
  echo "      --entrypoint verify-image $IMAGE --max-tier 2"
else
  echo "=== tier 2: numeric acceptance on real hardware ==="
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -v /tmp/qualification:/out -e RECORD=/out/qualification.json \
    --entrypoint verify-image "$IMAGE" --max-tier 2
  echo "qualification record: /tmp/qualification/qualification.json"
fi

# Tier 3: does it actually serve? The unit tiers are necessary and not
# sufficient -- they all passed once on an image whose engine could not start,
# because AITER was requested and absent. Only an end-to-end start catches that.
# Skipped unless SMOKE_MODEL points at a small local checkpoint.
if [ -n "${SMOKE_MODEL:-}" ] && [ "$NO_GPU" -eq 1 ]; then
  echo
  echo "=== tier 3: SKIPPED (--no-gpu); SMOKE_MODEL set but serving needs cards ==="
elif [ -n "${SMOKE_MODEL:-}" ]; then
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

# DERIVED_IMAGE has to mean "fetchable by digest from the registry compose.yaml
# pulls from". A non-empty .RepoDigests does NOT establish that: under the
# containerd snapshotter buildx populates RepoDigests on a purely local --load
# build that was never pushed (verified on this host), so the old
# `[ -n "$DIGEST" ]` test happily recorded a digest nobody else could resolve --
# and the failure lands on the consumer, at pull time, not here. Ask the
# registry instead of trusting the local record.
LOCAL_DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo "")
REG_DIGEST=$(docker buildx imagetools inspect --format '{{.Manifest.Digest}}' "$IMAGE" 2>/dev/null || echo "")

if [ -n "$REG_DIGEST" ]; then
  DIGEST="${IMAGE%:*}@$REG_DIGEST"
  sed -i "s|^DERIVED_IMAGE=.*|DERIVED_IMAGE=$DIGEST|" VERSIONS
  echo "recorded DERIVED_IMAGE=$DIGEST  (confirmed present in the registry)"
else
  echo "NOTE: $IMAGE is not resolvable in its registry - nothing recorded."
  echo "      docker push $IMAGE, then re-run this script to record it."
  echo "      compose.yaml must reference a digest, not '$IMAGE'."
  if [ -n "$LOCAL_DIGEST" ]; then
    echo "      (A local RepoDigest exists -- $LOCAL_DIGEST -- but that is a"
    echo "       local artifact of the containerd snapshotter, not proof of a"
    echo "       push. If you DID push, this is a registry auth failure:"
    echo "       docker login ${REGISTRY%%/*}.)"
  fi
fi
