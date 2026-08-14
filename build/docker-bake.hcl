# Declarative build targets. Replaces a bash build.sh for the build itself;
# build.sh now only orchestrates bake -> hardware verification -> digest record.
#
#   docker buildx bake -f build/docker-bake.hcl gfx90a
#
# Every variable defaults from VERSIONS via build.sh, so `bake` alone is
# reproducible and nothing floats.

variable "BASE_IMAGE" {}
variable "VLLM_FORK" {}
variable "VLLM_REF" {}
variable "VLLM_IS_FORK" { default = "1" }
variable "AITER_REPO" {}
variable "AITER_REF" {}
variable "AITER_CDNA2" {}
variable "AITER_CDNA2_REF" {}
variable "MAX_JOBS"   { default = "24" }
variable "TRITON_PIN" { default = "3.7.1" }
variable "REGISTRY" { default = "ghcr.io/davetha" }
variable "TAG"      { default = "dev" }

target "_common" {
  context    = "."
  dockerfile = "build/Dockerfile"
  args = {
    BASE_IMAGE      = BASE_IMAGE
    VLLM_FORK       = VLLM_FORK
    VLLM_REF        = VLLM_REF
    VLLM_IS_FORK    = VLLM_IS_FORK
    AITER_REPO      = AITER_REPO
    AITER_REF       = AITER_REF
    AITER_CDNA2     = AITER_CDNA2
    AITER_CDNA2_REF = AITER_CDNA2_REF
    MAX_JOBS        = MAX_JOBS
    TRITON_PIN      = TRITON_PIN
  }
  # Attestations, so a consumer can audit what went in without trusting a tag.
  attest = [
    "type=provenance,mode=max",
    "type=sbom",
  ]
}

# The gate is the default target: `bake` with no argument builds the VERIFIED
# stage, not the unverified one. Making the safe thing the default is the whole
# point -- a half-patched image fails by being slow, not by erroring.
target "gfx90a" {
  inherits = ["_common"]
  target   = "verified"
  tags     = ["${REGISTRY}/vllm-mi210:${TAG}", "${REGISTRY}/vllm-mi210:latest"]
}

# Escape hatch for iterating on the build without paying for the gate. Never
# publish this: it has not proven anything.
target "gfx90a-unverified" {
  inherits = ["_common"]
  target   = "final"
  tags     = ["${REGISTRY}/vllm-mi210:${TAG}-unverified"]
}

group "default" { targets = ["gfx90a"] }
