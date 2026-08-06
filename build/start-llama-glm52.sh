#!/usr/bin/env bash
# llama.cpp server for GLM-5.2 (unsloth UD-Q4_K_M GGUF, 438 GB) on 2x MI210 (gfx90a).
#
# Image: llama-rocm714-rpc:full  (ROCm 7.14, AMDGPU_TARGETS=gfx90a,
#   GGML_HIP_MMQ_MFMA=ON = CDNA quantized-GEMM fastpath, GGML_HIP_ROCWMMA_FATTN=OFF).
#
# Win mechanism on this box: -cmoe keeps the MoE EXPERT weights on CPU (read over the
# ~120 GB/s DDR4 bus) while -ngl 999 places dense/attention/shared on the GPUs. At short
# context this beats vLLM, which ships every expert over PCIe (~25 GB/s) to the GPU.
# Long context flips back to vLLM (its DSA sparse attention), see docs/.
#
# RAM NOTE: the 438 GB model barely fits the 499 GB box, so NO --mlock -- we rely on the
# kernel page cache. Run this ONLY with bench-glm52 stopped (it holds ~160 GB offload).
# llama.cpp auto-loads the 11 split parts when -m points at the -00001- file.
set -euo pipefail

MODEL_DIR=/mnt/llm-storage/glm52-gguf-q4km
MODEL=GLM-5.2-UD-Q4_K_M-00001-of-00011.gguf
PORT=${1:-8031}

docker rm -f llama-glm52 >/dev/null 2>&1 || true

docker run -d --name llama-glm52 --restart unless-stopped \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --security-opt seccomp=unconfined --ulimit memlock=-1 \
  --memory=480g --memory-swap=480g \
  -p "${PORT}":8080 \
  -e HIP_VISIBLE_DEVICES=0,1 -e ROCBLAS_USE_HIPBLASLT=1 -e MODELFILE="${MODEL}" \
  -v "${MODEL_DIR}":/models:ro \
  --entrypoint /bin/bash llama-rocm714-rpc:full \
  -lc 'export LD_LIBRARY_PATH=/src/build/bin:$LD_LIBRARY_PATH; cd /src/build && exec bin/llama-server \
    -m /models/$MODELFILE \
    --host 0.0.0.0 --port 8080 \
    -a glm-5.2 \
    -ngl 999 -cmoe -fa on -sm layer \
    -t 24 -tb 24 -c 8192 \
    --temp 0 --top-k 0 --top-p 1.0'

echo "started llama-server (gfx90a: experts-CPU / dense-GPU) on :${PORT}  (model load ~2-3 min)"
