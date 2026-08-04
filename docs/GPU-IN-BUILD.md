# Passing a GPU into `docker build`

An earlier version of this repo claimed AITER "cannot be a Dockerfile stage"
because `import aiter` probes the GPU and `docker build` exposes no `/dev/kfd`.
The first half is true. The conclusion was wrong: BuildKit supports CDI devices,
and it works.

Verified on this hardware:

```
RUN --device=amd.com/gpu=all python -c "import aiter, torch; ..."
  -> AITER_IMPORT_OK arch=gfx90a:sramecc+:xnack-
```

## What it takes

**1. A CDI spec on the host.** Normally from AMD's container toolkit
(`amd-ctk cdi generate`); it is also just a device list, so it can be written by
hand:

```yaml
# /etc/cdi/amd-gfx90a.yaml
cdiVersion: "0.6.0"
kind: "amd.com/gpu"
devices:
  - name: all
    containerEdits:
      deviceNodes:
        - path: /dev/kfd
        - path: /dev/dri/card0
        - path: /dev/dri/renderD128
```

Confirm with `docker info | grep -A2 "Discovered Devices"`.

**2. The spec inside buildkitd.** This is the step that is easy to miss and the
reason a first attempt fails with the devices still absent. With the
`docker-container` driver, buildkitd runs in its own container and reads CDI
specs from *its own* `/etc/cdi`, not the host's:

```bash
docker cp /etc/cdi/amd-gfx90a.yaml buildx_buildkit_<builder>0:/etc/cdi/
docker restart buildx_buildkit_<builder>0
```

**3. The labs frontend.** `RUN --device` is not in the stable Dockerfile syntax;
`# syntax=docker/dockerfile:1-labs` is required. Using `1.7` fails with
`unknown flag: security` or `unknown flag: device`.

**4. The entitlement, at both ends:**

```bash
docker buildx create --name gpubuilder --driver docker-container --use
docker buildx build --allow device=amd.com/gpu=all ...
```

## Why this repo still does not use it

Not because it does not work. Because the core image is worth more if it builds
anywhere:

| | Dockerfile + CDI | build.sh then add-aiter.sh |
|---|---|---|
| GPU needed to build | yes | no, for the core image |
| CDI spec + buildkitd surgery | yes | no |
| labs Dockerfile frontend | yes | no |
| commands | one | two |

Most people who want this image do not have a spare MI210 to build it on. Every
patch this project carries works without AITER, so the core image is useful on
its own, and `add-aiter.sh` adds the ASM paths for those who do have the cards.

## What does not work

`--allow security.insecure` with `RUN --security=insecure` is the other obvious
route and it does not help. The entitlement is granted and the step runs, but
the device nodes are still absent:

```
ls: cannot access '/dev/kfd': No such file or directory
```

It grants capabilities, not host devices. CDI is the mechanism that passes
devices.
