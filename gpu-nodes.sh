# Which GPU device nodes a container should be given. Sourced by run.sh and by
# build/add-aiter.sh; it defines gpu_devices() and sets no state of its own.
#
# Which GPUs the container sees is chosen HERE, by device node, and not left to
# HIP_VISIBLE_DEVICES.
#
# vLLM resolves the GPU architecture once, at import, from amdsmi, asking for
# PHYSICAL device 0 (`_GCN_ARCH = _get_gcn_arch()` in vllm/platforms/rocm.py).
# amdsmi enumerates real hardware, so it ignores HIP_VISIBLE_DEVICES *and*
# ROCR_VISIBLE_DEVICES -- both were tested, neither helps. On a host that also
# holds a non-CDNA2 card, physical 0 can be that other card; vLLM then decides
# the box is gfx12, `_ON_GFX9` goes False, and every gfx9-gated path silently
# turns itself off. Measured on a mixed MI210 + R9700 host: vLLM skips the AITER
# int8 GEMM for its generic Triton fallback and Qwen3.8-27B-W8A8 decode drops
# from 48.0 to 19.9 tok/s. Nothing errors. The only tell is the line
# `Selected TritonInt8ScaledMMLinearKernel` where it should say `AiterInt8...`.
#
# Handing the container only the gfx90a render nodes fixes it at the source. A
# host whose cards are all gfx90a is unaffected and still gets /dev/dri whole.
#
#   GPU_NODES='/dev/dri/renderD128 /dev/dri/renderD131' ./run.sh ...   pin explicitly
#   GFX90A_PCI_IDS='0x740f' ./run.sh ...                               narrow the match
GFX90A_PCI_IDS="${GFX90A_PCI_IDS:-0x7408 0x740c 0x740f}"   # Aldebaran: MI210, MI250(X)

gpu_devices() {
  local node id total=0 ours=()

  if [ -n "${GPU_NODES:-}" ]; then
    for node in $GPU_NODES; do printf '%s\n%s\n' --device "$node"; done
    return
  fi

  for node in /dev/dri/renderD*; do
    [ -e "$node" ] || continue
    total=$((total + 1))
    id=$(cat "/sys/class/drm/${node##*/}/device/device" 2>/dev/null) || continue
    case " $GFX90A_PCI_IDS " in *" $id "*) ours+=("$node") ;; esac
  done

  # No match (hardware this list does not know) or every card matches (a uniform
  # box): hand over /dev/dri exactly as before. Selecting nothing, or second
  # guessing an unfamiliar host, would be worse than the bug this avoids.
  if [ "${#ours[@]}" -eq 0 ] || [ "${#ours[@]}" -eq "$total" ]; then
    printf '%s\n%s\n' --device /dev/dri
    return
  fi

  echo "=== gpus   : ${#ours[@]}/$total render nodes (gfx90a only -- mixed-arch host;" \
       "see docs/RUNNING.md 'Which GPUs the container sees')" >&2
  for node in "${ours[@]}"; do printf '%s\n%s\n' --device "$node"; done
}
