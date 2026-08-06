#!/usr/bin/env bash
# Spec-decode variant of start-glm52.sh: clean config (no probe) PLUS n-gram
# speculative decoding, to test whether speculative verification can beat the
# PCIe offload wall. Revert with the original start-glm52.sh.
set -euo pipefail
R=/opt/python/lib/python3.14/site-packages/vllm/model_executor/layers/quantization/utils/moe_wna16_utils.py
docker rm -f bench-glm52 >/dev/null 2>&1 || true
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
sleep 2
docker run -d --name bench-glm52 --restart unless-stopped \
  --memory=465g --memory-swap=465g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 440g --security-opt seccomp=unconfined --ulimit memlock=-1 \
  -p 8021:8000 -e VLLM_ROCM_USE_AITER=1 \
  -v /tmp/chunked.py:"$R":ro \
  -v /mnt/llm-storage/glm52-int4int8:/mnt/llm-storage/glm52-int4int8:ro \
  -v /mnt/llm-storage/cache:/cache \
  local/vllm-mi210:dsa7 \
  /mnt/llm-storage/glm52-int4int8 \
  --served-model-name glm-5.2 \
  --tensor-parallel-size 2 \
  --cpu-offload-params experts \
  --cpu-offload-gb 162 \
  --gpu-memory-utilization 0.97 \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 4 \
  --enforce-eager \
  --attention-config '{"sparse_mla_force_mqa": true}' \
  --speculative_config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm47
echo "started (n-gram spec decode, K=3)"
