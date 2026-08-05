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
# The image's own entrypoint does the dispatch -- model to serve, subcommand to
# that subcommand, executable to exec -- so it behaves identically under plain
# docker run, Kubernetes or Slurm. All this script decides is networking, which
# is a property of the container rather than of the command.
LISTENS=0          # does this command bind a port, or connect out?
case "$1" in
  serve|launch)                            LISTENS=1; ARGS=("$@") ;;
  chat|complete|bench|collect-env|run-batch)          ARGS=("$@") ;;
  shell)                                   shift;     ARGS=(bash "$@") ;;
  exec)
    shift
    [ $# -gt 0 ] || { echo "exec needs a command" >&2; exit 1; }
    ARGS=("$@") ;;
  -*)
    echo "unknown option: $1 (a model must not start with -)" >&2; exit 1 ;;
  *)
    LISTENS=1; ARGS=("$@") ;;
esac

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

# GPU_PINNED_MIN_XFER_SIZE and VLLM_ROCM_USE_AITER are baked into the image, so
# they are correct for anyone running it without this script. Only forward them
# when the caller has actually set one, which keeps the image the single place
# a default is defined instead of two places that can disagree.
ENVS=()
for v in GPU_PINNED_MIN_XFER_SIZE VLLM_ROCM_USE_AITER HF_TOKEN MI210_QUIET; do
  [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
done

# Persist the Triton / inductor / AITER compile caches. Without a mount they
# still work, they just die with the container and every start recompiles.
CACHE_DIR="${CACHE_DIR:-$PWD/cache}"
mkdir -p "$CACHE_DIR"
MOUNTS+=(-v "$CACHE_DIR:/cache")

# -i always, -t only when there IS a terminal. Unconditional -it dies with "the
# input device is not a TTY" under ssh, cron, nohup and CI -- exactly the
# non-interactive cases where a server is most likely to be started. But
# dropping -i as well silently discards piped stdin, so `echo cmd | run.sh
# shell` produced no output at all rather than an error.
TTY=(-i)
[ -t 0 ] && TTY+=(-t)

echo "=== image  : $IMAGE"
[ "$LISTENS" = 1 ] && echo "=== serving: http://0.0.0.0:$PORT   (container: $NAME)" \
                   || echo "=== running: ${ARGS[*]}   (container: $NAME)"

exec docker run --rm "${TTY[@]}" --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 16G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  "${NET[@]}" \
  "${MOUNTS[@]}" "${TUNING[@]}" \
  ${ENVS[@]+"${ENVS[@]}"} \
  "$IMAGE" \
  ${ARGS[@]+"${ARGS[@]}"}
