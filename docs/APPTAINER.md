# Apptainer / HPC

**Yes, and the fit is better than usual: Frontier's MI250X is `gfx90a` — the
same architecture this image is compiled for.**

Every patch here is gated on `gfx90a`, not on a device ID, so a compute node on
Frontier gets the same code path an MI210 does. The long-context paged-attention
work in particular targets exactly the regime an HPC user is likely to want.

> **Status: UNVERIFIED.** No Apptainer is installed on the machines this project
> is developed on, so `build/vllm-mi210.def` has never been built or run. It is
> derived from `VERSIONS`, the working Dockerfile, and OLCF's
> [Frontier vLLM example](https://github.com/olcf/olcf_containers_examples/tree/main/frontier/sample_apps/vllm).
> Treat it as a starting point that still needs a first successful build.
> A PR replacing this notice with a build log is the most useful contribution
> anyone with allocation time could make.

---

## Three routes, easiest first

**1. Convert an existing Docker image.** If you have one already built:

```bash
apptainer build vllm-mi210.sif docker-daemon://local/vllm-mi210:latest
```

Needs Docker on the machine doing the conversion, which is usually a workstation
rather than the cluster. Move the `.sif` afterwards.

**2. Pull from a registry.** Once an image is published, this is what OLCF's own
example does:

```bash
apptainer pull --disable-cache vllm-mi210.sif docker://ghcr.io/davetha/vllm-mi210:latest
```

Nothing is published yet — `DERIVED_IMAGE` in `VERSIONS` is a local digest.

**3. Build from the definition file.** The route for a site with no Docker
anywhere:

```bash
./build/build-apptainer.sh vllm-mi210.sif
```

This works on a **login node**, because the compile needs no GPU — the same
property that keeps AITER out of the Dockerfile. See
[GPU-IN-BUILD.md](GPU-IN-BUILD.md).

## Running it

```bash
apptainer run --rocm vllm-mi210.sif /path/to/model --host 0.0.0.0 --port 8000
```

`--rocm` binds the GPU devices and host ROCm libraries. On sites that configure
device binds globally, OLCF's example among them, it may already be handled —
check your site's docs before assuming either way.

## Four things that bite on HPC specifically

**A SIF is read-only.** Anything that writes at runtime needs redirecting.
Triton compiles on first use and will fail without a writable cache; the def
file defaults `TRITON_CACHE_DIR` to `/tmp/triton-cache`, and OLCF sets the same
variable for the same reason. Use `--writable-tmpfs` if something else needs to
write.

**Environment variables need the `APPTAINERENV_` prefix** to cross into the
container from a batch script:

```bash
export APPTAINERENV_GPU_PINNED_MIN_XFER_SIZE=67108864
export APPTAINERENV_TRITON_CACHE_DIR=/tmp/triton-cache
```

The def file sets both in `%environment` so they are correct by default, but a
batch script overriding them without the prefix will silently not apply — and
missing `GPU_PINNED_MIN_XFER_SIZE` costs *hours* on a MoE checkpoint, not
seconds. See [LOAD-TIME.md](LOAD-TIME.md). That one is worth checking rather
than assuming, because the failure looks like slow storage.

**Size.** The Docker image is ~69 GB uncompressed. The SIF will be smaller
(squashfs) but still large enough to matter against a home quota. Point
`APPTAINER_TMPDIR` at scratch before building, and keep the SIF on a project
filesystem.

**Model staging.** Reading a large checkpoint from a shared parallel filesystem
across many nodes is its own bottleneck. OLCF's example stages to node-local
NVMe (`-C nvme`, `HF_HOME=/mnt/bb/$USER`) and that pattern applies here
unchanged. It is orthogonal to the pin-threshold problem — fixing the storage
path does not fix `hsa_amd_memory_lock_to_pool`, and vice versa. Both were
measured separately; only the second is fixed by an environment variable.

## AITER on a cluster

`build/add-aiter.sh` needs the GPU (it repatches gfx942 code objects and imports
AITER, which probes the device) **and** a writable filesystem. Neither holds for
a plain SIF on a login node.

If you want it, the shape is: build a `--sandbox` directory, run the AITER steps
inside it in a job that has GPUs, then `apptainer build` the sandbox back into a
SIF. Untested. AITER is an enhancement — every patch this project carries works
without it — so skip it for a first run.

## Multi-node

Nothing here is multi-node aware. vLLM's own Ray path is what OLCF uses:

```bash
vllm serve --tensor-parallel-size 8 --pipeline-parallel-size $SLURM_NNODES \
           --distributed-executor-backend ray ...
```

with `ray start --head` on one node and workers elsewhere. Their sbatch scripts
are a good template. TP=8 per node matches Frontier's 8 GCDs; the guards in this
image are per-shape and per-arch, so they behave the same at any TP size.

## What is actually known versus assumed

| claim | basis |
|---|---|
| Frontier's MI250X is gfx90a, the arch this image targets | AMD documentation; the same `PYTORCH_ROCM_ARCH` this project builds |
| The patches are arch-gated, not device-gated | the source: `_ON_GFX90A` in `vllm/platforms/rocm.py` |
| The core build needs no GPU | verified — the Docker build runs GPU-free |
| `GPU_PINNED_MIN_XFER_SIZE` matters | verified on MI210, 2637x on a single copy |
| The def file builds | **not verified** |
| The SIF runs on Frontier | **not verified** |
| MI250X performance | **not measured** — no access to that hardware |

Anything in the second group needs someone with allocation time. Please open an
issue either way; a failed build report is as useful as a successful one.
