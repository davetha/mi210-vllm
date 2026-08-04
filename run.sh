#!/usr/bin/env bash
# Serve any model on the MI210s.
#
#   ./run.sh /mnt/models/my-model                 a directory on this machine
#   ./run.sh Qwen/Qwen3-8B                        an HF repo id (downloads)
#   ./run.sh /mnt/models/big --tensor-parallel-size 2 --max-model-len 200000
#
# Anything after the model is passed to vLLM untouched, so the full server CLI
# is available without this script having to know about it.
#
# compose.yaml is the other way in. It is for a deployment you run repeatedly
# and have tuned; this is for a model you just want to serve. See docs/RUNNING.md.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: ./run.sh <model-path-or-hf-id> [vllm args...]}"
shift || true

set -a; . ./VERSIONS; set +a
IMAGE="${IMAGE:-${DERIVED_IMAGE:-}}"
[ -n "$IMAGE" ] || {
  echo "No image. Run build/build.sh (it records DERIVED_IMAGE in VERSIONS)," >&2
  echo "or set IMAGE=<tag> to use one you already have." >&2
  exit 1
}

NAME="${NAME:-mi210-vllm}"
PORT="${PORT:-8000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

MOUNTS=(-v "$HF_CACHE:/root/.cache/huggingface")

# A local directory is mounted at its OWN path inside the container. Remapping
# to /models would make every error message name a path that does not exist on
# the host, which is a bad trade for a shorter command line.
if [ -d "$MODEL" ]; then
  MODEL="$(cd "$MODEL" && pwd)"
  [ -f "$MODEL/config.json" ] || echo "warning: no config.json in $MODEL" >&2
  MOUNTS+=(-v "$MODEL:$MODEL:ro")
fi

# Tuned fused_moe configs are per-deployment and must never be pooled -- the
# filename does not encode K, so two models can collide. Opt in explicitly;
# unset means vLLM uses its own shipped configs, which is the right default for
# a model nobody here has tuned. See tuning/README.md.
TUNING=()
if [ -n "${DEPLOYMENT:-}" ]; then
  D="$PWD/tuning/by-deployment/$DEPLOYMENT"
  [ -d "$D" ] || { echo "no tuning folder: $D" >&2; exit 1; }
  TUNING=(-v "$D:/opt/tuning:ro" -e VLLM_TUNED_CONFIG_FOLDER=/opt/tuning)
fi

# NOT optional on ROCm, and not a vLLM setting -- a HIP runtime tunable. Above
# HIP's ~1 MiB default, .to(device) page-locks the caller's buffer and
# hsa_amd_memory_lock_to_pool costs ~1 s per tensor while the DMA is 14 ms. On a
# MoE checkpoint with thousands of expert tensors that is hours. docs/LOAD-TIME.md.
: "${GPU_PINNED_MIN_XFER_SIZE:=67108864}"

# AITER off unless the image has it. The core image does not; forcing it on
# makes AITER's JIT build a module that fails to compile and takes engine
# startup down with it. build/add-aiter.sh produces an image where 1 is correct.
: "${VLLM_ROCM_USE_AITER:=0}"

# -it only when there IS a terminal. Unconditional -it dies with "the input
# device is not a TTY" under ssh, cron, nohup and CI -- i.e. exactly the
# non-interactive cases where a server is most likely to be started.
TTY=()
[ -t 0 ] && TTY=(-it)

echo "=== image  : $IMAGE"
echo "=== model  : $MODEL"
echo "=== serving: http://0.0.0.0:$PORT   (container: $NAME)"

exec docker run --rm "${TTY[@]}" --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 16G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  -p "$PORT:8000" \
  "${MOUNTS[@]}" "${TUNING[@]}" \
  -e GPU_PINNED_MIN_XFER_SIZE="$GPU_PINNED_MIN_XFER_SIZE" \
  -e VLLM_ROCM_USE_AITER="$VLLM_ROCM_USE_AITER" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$IMAGE" \
  vllm serve "$MODEL" --host 0.0.0.0 "$@"
