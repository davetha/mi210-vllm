# Architecture

## The one fact that drives this repo's shape

Of everything this stack carries, exactly one file requires compilation:
`csrc/rocm/attention.cu` in the davetha/vllm fork, which carries the
multi-pass paged-attention reduction and the gfx90a guards. Everything else
this project carries is pure Python (davetha/vllm's other patches) or a
prebuilt binary artifact produced once at image-build time (AITER's gfx90a
ASM, produced by aiter-cdna2's repatcher).

That is the whole reason `build/Dockerfile` has exactly one stage that
compiles anything (`csrc-build`, plus `aiter-build` for the ASM repatch),
and the whole reason `compose.yaml` separates `image` (what had to be
compiled) from `volumes` (what did not):

```
image    -> everything that had to be compiled (attention.cu, AITER ASM)
volumes  -> everything that did not (Python overlays, tuned configs)
```

The volume-mounted half — `patches/*.py` overlays and `tuning/*.json`
configs — can be edited and the container restarted, no rebuild. The
image half cannot: change one line of `attention.cu` and the wheel has to
be rebuilt from `csrc-build`. `patches/` is empty today specifically
because every patch this project carries has been merged into the
davetha/vllm fork rather than kept as a runtime overlay — the overlay
mechanism exists for things not yet promoted to the fork, not as the normal
path.

This is also why AITER gets a build stage at all, even though its source
patches are just as compilable-or-not as vLLM's: AITER ships zero gfx90a
code objects. There is no upstream binary to overlay. aiter-cdna2 has to
disassemble the gfx942 ones, substitute CDNA2 mnemonics, and re-assemble,
every time the AITER ref changes. That repatch step is why `aiter-build` is
a build stage and not something done once and vendored — the ASM is
architecture-specific output, not portable source.

## Repo flow into one image

```
                          VERSIONS
              (pins every ref/tag/digest below)
                              |
       +----------------------+----------------------+
       |                      |                       |
davetha/vllm           davetha/aiter          davetha/aiter-cdna2
one branch per patch    gfx90a source          gfx942 -> gfx90a
merged to main           patches                ASM repatcher
       |                      |                       |
       | git clone            | git clone              | git clone
       | (fetch stage)        | (fetch stage)           | (fetch stage)
       v                      v                       v
  +---------------------------------------------------------------+
  |                      build/Dockerfile                          |
  |                                                                 |
  |  csrc-build stage:              aiter-build stage:              |
  |    compile _rocm_C for            pip install aiter (source     |
  |    gfx90a from vLLM fork          patches applied)              |
  |    -> wheel + build-manifest      run aiter-cdna2's repatcher   |
  |                                   on the gfx942 ASM objects     |
  |                                   -> gfx90a code objects +      |
  |                                      repatch-report.txt         |
  |                                                                 |
  |  final stage: install wheel + patched AITER, copy in the       |
  |    acceptance tests, verify-image, probe-image-patches         |
  |                                                                 |
  |  verified stage: RUN verify-image --max-tier 1 (gates the      |
  |    build itself; fails the docker build on any FAIL)           |
  +---------------------------------------------------------------+
                              |
                     build/build.sh runs
                     tier 2 against real cards,
                     records digest into VERSIONS
                              |
                              v
                     DERIVED_IMAGE=<digest>
                    (compose.yaml consumes this;
                     volumes mount patches/ and
                     tuning/ on top at runtime)
```

## Why "could be a read-only overlay" matters

Because only `attention.cu` forces a rebuild, every other patch this
project carries is, in principle, deployable as a bind-mounted `.py` file
over the installed package — no image rebuild, no new digest, restart the
container. `compose.yaml` reserves exactly this mechanism (see the
commented `patches/<name>.py:...` line) for patches not yet merged into the
davetha/vllm fork. It is unused today because every carried patch has
already been promoted into the fork and therefore compiled into the wheel —
but the mechanism is why `patches/` exists as a top-level directory at all,
separate from the fork.

The same split explains `tuning/`: tuned `fused_moe` JSON configs are read
by `VLLM_TUNED_CONFIG_FOLDER` at process start, not compiled in, so they are
a volume mount (`tuning/by-deployment/<name>` -> `/opt/tuning`) rather than
something baked into the image.

## Why the build gates run where they do

`build/verify.sh` has three tiers (see `README.md` for the full rationale —
a half-patched image fails by being slow, not by erroring):

- Tier 0 (static) and tier 1 (gate behaviour) need no GPU, so they run
  inside `docker build` itself, in the `verified` stage. A build that fails
  either tier never produces an image.
- Tier 2 (numeric acceptance) needs real cards, which `docker build`
  sandboxes do not have. `build/build.sh` runs it separately, against real
  hardware, after the image is built — and only then records
  `DERIVED_IMAGE` into `VERSIONS`. An image without a recorded digest has
  not been verified on real hardware, by construction.

This mirrors the compile/no-compile split one level up: what can be checked
without hardware happens at build time; what needs hardware happens after,
against real MI210 cards, before the digest anyone deploys is ever written
down.
