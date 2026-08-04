---
name: Bug report
about: Report a problem with a built image or a running server
title: ""
labels: bug
---

<!--
A half-patched image fails by being SLOW rather than by erroring. This
project has already published a throughput ratio it had not earned, because
one image in a comparison was patched and the other was not, and nothing
errored — the number just looked reasonable. "It's slow" reports are
therefore not useful without knowing exactly what is in the image. Fill in
everything below before describing the problem.
-->

## Image

- Image digest (`docker inspect --format '{{.Id}}' <image>`):
- `DERIVED_IMAGE` from `VERSIONS`, if this is a build from this repo:

## `probe-image-patches` output

Run from a shell where the image(s) in question are visible to `docker
images`, or run `probe-image-patches` directly inside a container from the
image:

```
<paste output here>
```

## `verify-image --max-tier 1` output

Run from inside the image (add `--device=/dev/kfd --device=/dev/dri
--group-add video` and `--max-tier 2` if you have the cards and want the
numeric tier too):

```
<paste output here>
```

## GPU arch

```
rocminfo | grep gfx
```

## What happened

<!-- What you ran, what you expected, what you got. Numbers if you have them. -->

## What you expected
