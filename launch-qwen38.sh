#!/bin/bash
# Canonical qwen38 launch. 2026-08-22: moved to the v0.28.0rc2 base.
# Measured 192.0 / 99.8 / 54.8 tok/s decode at 2K / 41K / 101K.
#
# Image local/vllm-mi210:mi210.6-aiter = tag v0.28.0rc2+mi210.6 with AITER
# layered by build/add-aiter.sh. The two Triton attention partitioning patches
# are now COMPILED INTO THE IMAGE, so the bind-mounts the previous launcher
# needed are gone -- do not re-add them. Mounting the old dflash2-int-based
# copies over a v0.28.0rc2 image would be actively wrong, not merely redundant.
# Confirm the patches are live by grepping the log for "partitioning ON".
#
# DFlash2 is now UPSTREAM's implementation, not the fork's port; "dflash" is a
# supported method in vllm/config/speculative.py at this tag.
#
# CRITICAL: cudagraph_mode FULL_DECODE_ONLY is load-bearing -- vLLM silently
# downgrades to PIECEWISE under spec decoding, costing 60pc decode.
# CRITICAL: before relaunching, wait for ROCm to release VRAM. Check with
# `docker exec qwen35 rocm-smi --showpids` and expect only qwen35's process.
# NOTE: `pgrep -f VLLM::` and `pgrep -f add-aiter` SELF-MATCH their own ssh
# command line and always report a hit; bracket the pattern (VLLM[:]:) instead.
# NOTE: first boot after ANY restart pays ~6-9 min of AITER JIT compilation.
# Do NOT probe decode speed by counting SSE chunks -- vLLM coalesces ~2.7
# tokens/chunk under spec decode; use completion_tokens / wall time.
#
# Rollback: launch-qwen38.sh.mi210.5 is the previous (bind-mounted) launcher.
docker rm -f qwen38 2>/dev/null
docker run -d --name qwen38 --restart unless-stopped \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --ipc host \
  -p 8063:8000 -v /mnt/llm-storage:/models -v /mnt/llm-storage/cache:/cache \
  -e HIP_VISIBLE_DEVICES=0,1 -e VLLM_USE_V1=1 -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e VLLM_ROCM_USE_AITER=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /usr/local/bin/mi210-entrypoint local/vllm-mi210:mi210.6-aiter \
  --model /models/qwen38-27b-ablit-w8a8 --served-model-name qwen38-27b \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.72 --max-model-len 262144 \
  --max-num-batched-tokens 8192 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method": "dflash", "model": "/models/qwen38-dflash2", "num_speculative_tokens": 8}'
echo "launched; wait ~6-7 min for JIT compile, then verify:"
echo "  curl -s http://127.0.0.1:8063/v1/models"
echo "  exact-rate probe: non-streaming request, completion_tokens / wall"
