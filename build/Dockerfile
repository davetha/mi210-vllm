# syntax=docker/dockerfile:1.7
#
# The ONLY compiled layer in this project. Staged after vLLM's own
# docker/Dockerfile.rocm: fetch and compile in throwaway stages, export just the
# artifacts, assemble a final image that carries no build toolchain.
#
# One thing forces a build: csrc/rocm/attention.cu carries the multi-pass
# paged-attention reduction and the gfx90a guards, so _rocm_C must be rebuilt.
# Everything else this project carries is pure Python.
#
# THIS STAGE NEEDS NO GPU. It builds anywhere with docker, which matters because
# most people who want the image do not have a spare MI210 to build it on.
#
# AITER is deliberately NOT here, but not because it is impossible. `import
# aiter` probes the GPU through rocminfo, and plain `docker build` exposes no
# /dev/kfd -- however BuildKit CDI can pass the cards in, and it was verified
# working here (see docs/GPU-IN-BUILD.md). It is left out because AITER is an
# enhancement rather than a requirement -- every patch this project carries
# works without it -- and keeping it out means this stage builds on any machine
# with plain docker, no CDI setup and no GPU. Add it with build/add-aiter.sh.
#
# Budget ~45-90 min cold; minutes with a warm ccache.

ARG BASE_IMAGE

# ---------------------------------------------------------------- base -------
FROM ${BASE_IMAGE} AS base
ENV PYTORCH_ROCM_ARCH=gfx90a \
    VLLM_TARGET_DEVICE=rocm \
    CCACHE_DIR=/root/.cache/ccache \
    CCACHE_COMPILERCHECK=content \
    CCACHE_MAXSIZE=100G \
    CCACHE_DEPEND=1
ARG MAX_JOBS=24
ENV MAX_JOBS=${MAX_JOBS}

# --------------------------------------------------------------- fetch -------
# Separate so a ref bump invalidates the clone but not the toolchain layers.
FROM base AS fetch
ARG VLLM_FORK
ARG VLLM_REF
ARG AITER_REPO
ARG AITER_REF
ARG AITER_CDNA2
ARG AITER_CDNA2_REF
RUN git clone --depth 1 --branch "${VLLM_REF}"             "${VLLM_FORK}"   /src/vllm \
 && git clone --recursive --depth 1 --branch "${AITER_REF}" "${AITER_REPO}" /src/aiter \
 && git clone --depth 1 --branch "${AITER_CDNA2_REF}"       "${AITER_CDNA2}" /src/aiter-cdna2 \
 && for d in /src/vllm /src/aiter /src/aiter-cdna2; do \
        git -C "$d" rev-parse HEAD > "$d.sha"; done

# ----------------------------------------------------------- csrc-build ------
# Rebuild vLLM for gfx90a only and emit a wheel. Nothing from this stage reaches
# the final image except the wheel.
FROM base AS csrc-build
COPY --from=fetch /src/vllm /src/vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    cd /src/vllm && python3 -m pip wheel --no-deps --no-build-isolation -w /out .

# Attest what was compiled. The final image has no source tree, so a static
# marker cannot grep attention.cu -- this records the answer at the moment it
# was true, from the tree that produced the wheel.
RUN cd /src/vllm && { \
      echo "vllm_sha=$(git rev-parse HEAD)"; \
      echo "reduction_extended=$(grep -c 'npar_loops > 16' csrc/rocm/attention.cu)"; \
      echo "reduction_cases_9_16=$(grep -c 'LAUNCH_CUSTOM_REDUCTION(1[0-6])\|LAUNCH_CUSTOM_REDUCTION(9)' csrc/rocm/attention.cu)"; \
      echo "pytorch_rocm_arch=${PYTORCH_ROCM_ARCH}"; \
    } > /out/build-manifest.txt && cat /out/build-manifest.txt

# ---------------------------------------------------------------- final ------
FROM base AS final
ARG BASE_IMAGE
ARG VLLM_FORK
ARG VLLM_REF
ARG AITER_REF
ARG AITER_CDNA2_REF

# Provenance on the image itself, so nobody has to guess what built it.
LABEL org.opencontainers.image.title="vllm-mi210" \
      org.opencontainers.image.description="vLLM for AMD MI210 (gfx90a/CDNA2), patched and gate-verified" \
      org.opencontainers.image.source="https://github.com/davetha/mi210-vllm" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      dev.mi210.vllm.ref="${VLLM_REF}" \
      dev.mi210.vllm.repo="${VLLM_FORK}" \
      dev.mi210.aiter.ref="${AITER_REF}" \
      dev.mi210.aiter-cdna2.ref="${AITER_CDNA2_REF}"

COPY --from=csrc-build  /out/build-manifest.txt  /usr/local/share/build-manifest.txt
COPY --from=csrc-build  /out/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-deps /tmp/*.whl && rm -f /tmp/*.whl

# The acceptance tests ship WITH the image. This project distributes to people
# who have the hardware we do not, so the image has to be able to prove itself
# on their cards, not only on ours.
# Layout mirrors the repo, because test_benchmark_moe_quant_config.py locates
# benchmark_moe.py relative to itself. Flattening these into one directory is
# what broke it the first time: upstream's test_moe.py also imports the `tests`
# package, which a single copied file cannot satisfy.
COPY --from=fetch /src/vllm/tests/kernels/attention/test_rocm_paged_attention_long_context.py /opt/vllm/tests/kernels/attention/
COPY --from=fetch /src/vllm/tests/model_executor/layers/test_rocm_unquantized_gemm.py         /opt/vllm/tests/model_executor/layers/
COPY --from=fetch /src/vllm/tests/benchmarks/test_benchmark_moe_quant_config.py               /opt/vllm/tests/benchmarks/
COPY --from=fetch /src/vllm/benchmarks/kernels/benchmark_moe.py                               /opt/vllm/benchmarks/kernels/
COPY probe/probe_image_patches.sh /usr/local/bin/probe-image-patches
COPY build/verify.sh              /usr/local/bin/verify-image
RUN chmod +x /usr/local/bin/probe-image-patches /usr/local/bin/verify-image

# ------------------------------------------------------------- verified ------
# Gate. Tiers 0-1 need no GPU and run here, so a half-patched image cannot be
# built. Tiers 2-3 need cards and run from build.sh against the real hardware.
FROM final AS verified
# Only tier 0 can run here: tiers 1-2 need /dev/kfd, which docker build has not.
RUN /usr/local/bin/verify-image --max-tier 0
