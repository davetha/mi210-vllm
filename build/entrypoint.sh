#!/usr/bin/env bash
# The image's ENTRYPOINT. Makes the image usable on its own -- docker run,
# Kubernetes, Slurm, Apptainer -- rather than only through this repo's run.sh.
#
#   docker run ... <image> /models/my-model --tensor-parallel-size 2
#   docker run ... <image> bench latency --model /models/my-model
#   docker run ... <image> bash
#
# Before this existed the image had no entrypoint at all, so a model path was
# exec'd as a program:
#   exec: "/models/my-model": is a directory: permission denied
# and the settings that are not optional on ROCm lived in run.sh, which someone
# pulling the published image does not have.
set -euo pipefail

# ------------------------------------------------------------------ preamble -
# Cheap by design: no torch import, no GPU allocation. This project's whole
# claim is that a half-patched image fails by being SLOW rather than by
# erroring, so the one thing worth saying out loud at every start is what this
# image actually is. MI210_QUIET=1 silences it.
preamble() {
  [ "${MI210_QUIET:-0}" = 1 ] && return 0
  local m=/usr/local/share/build-manifest.txt
  {
    echo "=== vllm-mi210 (gfx90a / CDNA2)"
    [ -f "$m" ] && sed 's/^/    /' "$m"
    echo "    GPU_PINNED_MIN_XFER_SIZE=${GPU_PINNED_MIN_XFER_SIZE:-<unset>}  VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-<unset>}"

    # The check worth making: this image contains gfx90a code objects only.
    # On another architecture that is a slow, confusing failure downstream, so
    # say it here where it is one line.
    if command -v rocminfo >/dev/null 2>&1; then
      local arch
      arch=$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-z]*' || true)
      if [ -z "$arch" ]; then
        echo "    arch: no GPU visible -- pass --device /dev/kfd --device /dev/dri"
      elif [ "$arch" = gfx90a ]; then
        echo "    arch: $arch  OK"
      else
        echo "    arch: $arch  WARNING: this image is built for gfx90a only"
      fi
    fi

    if [ "${VLLM_ROCM_USE_AITER:-0}" = 1 ] && [ ! -f /usr/local/share/repatch-report.txt ]; then
      echo "    WARNING: VLLM_ROCM_USE_AITER=1 but this image has no AITER."
      echo "             AITER's JIT will fail to build and take startup with it."
      echo "             Run build/add-aiter.sh, or set VLLM_ROCM_USE_AITER=0."
    fi
  } >&2
}

# ------------------------------------------------------------------ dispatch -
# A bare model means serve, because that is what nearly every invocation wants.
# Order matters: subcommand, then real executable, then model. A model
# directory is not an executable file, so `command -v` does not claim it, and an
# HF repo id like Qwen/Qwen3-8B is not on PATH either -- both fall through to
# serve. Verified for directories, HF ids, bash and /bin/bash.

# vLLM's --host defaults to None. A server in a container that does not bind
# 0.0.0.0 is unreachable from the host, which presents as "it started fine but
# nothing answers". Default it here; an explicit --host still wins.
add_host() {
  case " $* " in *" --host "*|*" --host="*) return 1 ;; esac
  return 0
}

case "${1:-}" in
  serve|launch)
    preamble
    if add_host "$@"; then exec vllm "$@" --host 0.0.0.0; else exec vllm "$@"; fi ;;
  chat|complete|bench|collect-env|run-batch)
    preamble; exec vllm "$@" ;;
  "")
    exec vllm --help ;;
  -h|--help)
    exec vllm --help ;;
esac

if command -v "$1" >/dev/null 2>&1; then
  preamble; exec "$@"
fi

preamble
if add_host "$@"; then exec vllm serve "$@" --host 0.0.0.0; else exec vllm serve "$@"; fi
