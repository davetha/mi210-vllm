#!/usr/bin/env bash
# Run anything in the MI210 image, with the GPUs, mounts and ROCm environment
# already correct.
#
#   ./run.sh /mnt/models/my-model            serve it (the common case)
#   ./run.sh Qwen/Qwen3-8B                   an HF repo id; downloads
#   ./run.sh serve /mnt/models/big --tensor-parallel-size 2
#
#   ./run.sh bench latency --model /mnt/models/my-model
#   ./run.sh chat --url http://localhost:8000/v1
#   ./run.sh complete --url http://localhost:8000/v1
#   ./run.sh collect-env
#   ./run.sh run-batch -i prompts.jsonl -o out.jsonl --model ...
#
#   ./run.sh shell                           interactive bash in the image
#   ./run.sh exec probe-image-patches        any command at all
#   ./run.sh exec python3 -c 'import torch; print(torch.cuda.device_count())'
#
# Every vLLM subcommand is passed through untouched, so the full CLI is
# available without this script having to know what the flags mean.
#
# compose.yaml is the other way in. It is for a deployment you run repeatedly
# and have tuned; this is for everything else. See docs/RUNNING.md.
set -euo pipefail
cd "$(dirname "$0")"

# Prints the header block above, stopping at the first line that is not a
# comment -- so editing the header cannot desync a hardcoded line range.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; exit "${1:-0}"; }
[ $# -gt 0 ] || usage 1
case "$1" in -h|--help|help) usage 0 ;; esac

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

# ---------------------------------------------------------------- dispatch ---
# The vLLM CLI as of v0.26.1rc0. A bare first argument that is not one of these
# is taken to be a model, because serving is what nearly every invocation wants
# and `./run.sh <model>` should stay the short form.
LISTENS=0          # does this command bind a port, or connect out?
ENTRY=(--entrypoint vllm)
case "$1" in
  serve|launch)
    LISTENS=1; ARGS=("$@"); shift $# ;;
  chat|complete|bench|collect-env|run-batch)
    ARGS=("$@"); shift $# ;;
  shell)
    shift; ARGS=(); ENTRY=(--entrypoint bash) ;;
  exec)
    shift
    [ $# -gt 0 ] || { echo "exec needs a command" >&2; exit 1; }
    ARGS=("$@"); ENTRY=(--entrypoint "$1"); shift; ARGS=("$@") ;;
  -*)
    echo "unknown option: $1 (a model must not start with -)" >&2; exit 1 ;;
  *)
    LISTENS=1; MODEL="$1"; shift; ARGS=(serve "$MODEL" "$@") ;;
esac

# --host 0.0.0.0 or the server is unreachable from outside the container. Only
# added when absent, so an explicit --host still wins.
if [ "$LISTENS" = 1 ] && [[ " ${ARGS[*]} " != *" --host "* ]]; then
  ARGS+=(--host 0.0.0.0)
fi

# ------------------------------------------------------------------ mounts ---
# Any argument that names a path on this machine is bind-mounted read-only at
# the SAME path inside. Remapping to /models would make every log line and
# error name a path that does not exist on the host, which is a bad trade for a
# shorter command line. Doing it for every argument rather than just the model
# means `bench --model X --dataset-path Y` works without special cases.
MOUNTS=(-v "$HF_CACHE:/root/.cache/huggingface")
declare -A SEEN=()
for a in "${ARGS[@]}"; do
  [ -e "$a" ] || continue
  p=$(cd "$(dirname "$a")" 2>/dev/null && printf '%s/%s' "$(pwd)" "$(basename "$a")") || continue
  case "$p" in
    /|/dev/*|/proc/*|/sys/*|"$HF_CACHE"/*|"$HF_CACHE") continue ;;
  esac
  [ -n "${SEEN[$p]:-}" ] && continue
  SEEN[$p]=1
  MOUNTS+=(-v "$p:$p:ro")
done

# Tuned fused_moe configs are per-deployment and must never be pooled -- the
# filename does not encode K, so two models can collide. Opt in explicitly;
# unset means vLLM uses its own shipped configs, which is right for a model
# nobody here has tuned. See tuning/README.md.
TUNING=()
if [ -n "${DEPLOYMENT:-}" ]; then
  D="$PWD/tuning/by-deployment/$DEPLOYMENT"
  [ -d "$D" ] || { echo "no tuning folder: $D" >&2; exit 1; }
  TUNING=(-v "$D:/opt/tuning:ro" -e VLLM_TUNED_CONFIG_FOLDER=/opt/tuning)
fi

# ----------------------------------------------------------------- network ---
# A command that listens publishes a port. A command that CONNECTS -- chat,
# complete, bench serve -- gets host networking instead, so http://localhost
# reaches a server running on this machine rather than resolving inside an
# empty container namespace.
if [ "$LISTENS" = 1 ]; then
  NET=(-p "$PORT:8000")
else
  NET=(--network host)
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

# -i always, -t only when there IS a terminal. Unconditional -it dies with "the
# input device is not a TTY" under ssh, cron, nohup and CI -- exactly the
# non-interactive cases where a server is most likely to be started. But
# dropping -i as well silently discards piped stdin, so `echo cmd | run.sh
# shell` produced no output at all rather than an error.
TTY=(-i)
[ -t 0 ] && TTY+=(-t)

echo "=== image  : $IMAGE"
[ "$LISTENS" = 1 ] && echo "=== serving: http://0.0.0.0:$PORT   (container: $NAME)" \
                   || echo "=== running: ${ENTRY[1]} ${ARGS[*]}   (container: $NAME)"

exec docker run --rm "${TTY[@]}" --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 16G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  "${NET[@]}" \
  "${MOUNTS[@]}" "${TUNING[@]}" \
  -e GPU_PINNED_MIN_XFER_SIZE="$GPU_PINNED_MIN_XFER_SIZE" \
  -e VLLM_ROCM_USE_AITER="$VLLM_ROCM_USE_AITER" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "${ENTRY[@]}" \
  "$IMAGE" \
  ${ARGS[@]+"${ARGS[@]}"}
