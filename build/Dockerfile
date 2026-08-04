# syntax=docker/dockerfile:1.7
#
# The ONLY compiled layer in this project.
#
# Everything that can be a read-only file overlay is one (see ../compose.yaml).
# Two things cannot, and they are why this file exists:
#
#   1. csrc/rocm/attention.cu carries the multi-pass paged-attention reduction
#      and the gfx90a fixes, so vLLM's _rocm_C extension has to be rebuilt.
#   2. AITER ships no gfx90a code objects. aiter-cdna2 disassembles the gfx942
#      objects, substitutes CDNA2 mnemonics and re-assembles for gfx90a. That is
#      a build-time transform of shipped binaries, not a source patch.
#
# Build time is dominated by (1). Budget ~45-90 min cold, minutes with a warm
# ccache mounted at /ccache.

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS build

ARG VLLM_FORK
ARG VLLM_REF
ARG AITER_FORK
ARG AITER_REF
ARG AITER_CDNA2
ARG AITER_CDNA2_REF

ENV PYTORCH_ROCM_ARCH=gfx90a \
    VLLM_TARGET_DEVICE=rocm \
    MAX_JOBS=24 \
    CCACHE_DIR=/ccache \
    CCACHE_MAXSIZE=100G \
    CCACHE_DEPEND=1

# --- 1. AITER: source patches from our fork, then the gfx90a repatch --------
RUN --mount=type=cache,target=/ccache \
    git clone --recursive --depth 1 --branch "${AITER_REF}" "${AITER_FORK}" /src/aiter \
 && cd /src/aiter \
 && python3 -m pip install --no-build-isolation .

RUN git clone --depth 1 --branch "${AITER_CDNA2_REF}" "${AITER_CDNA2}" /src/aiter-cdna2 \
 && python3 /src/aiter-cdna2/repatch_gfx942_to_gfx90a.py --report /tmp/repatch-report.txt
# The repatcher REPORTS kernels that do not translate rather than skipping them
# silently. Keep the report in the image; it is part of the manifest.

# --- 2. vLLM: our fork, rebuilt for gfx90a ---------------------------------
RUN --mount=type=cache,target=/ccache \
    git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_FORK}" /src/vllm \
 && cd /src/vllm \
 && python3 -m pip install --no-build-isolation -e .

# --- 3. Ship the probe and the verification suite --------------------------
COPY probe/probe_image_patches.sh /usr/local/bin/probe-image-patches
COPY build/verify.sh              /usr/local/bin/verify-image
RUN chmod +x /usr/local/bin/probe-image-patches /usr/local/bin/verify-image \
 && cp /tmp/repatch-report.txt /usr/local/share/repatch-report.txt

# --- 4. Gate: a half-patched image fails by being SLOW, not by erroring -----
# So it must not be possible to produce one. If any check fails, no image.
RUN /usr/local/bin/verify-image
