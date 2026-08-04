# patches/

**There are no patch files here, and that is deliberate.** If you came looking
for the diffs listed in `registry.yaml`, they are branches on the vLLM fork.

## Where the code actually is

Every patch is a branch on [davetha/vllm](https://github.com/davetha/vllm),
merged into the tag `VERSIONS` pins:

    VLLM_REF=v0.26.1rc0+mi210.1

`build/Dockerfile` clones that tag and builds it. So the patches are *in the
image* without ever being *in this repo* — nothing here has to be applied,
rebased, or kept in sync with a diff.

Verify any of them yourself:

```bash
git clone https://github.com/davetha/vllm && cd vllm
git log --oneline v0.26.1rc0..v0.26.1rc0+mi210.1        # what the tag adds
git diff v0.26.1rc0..v0.26.1rc0+mi210.1 -- vllm/platforms/rocm.py
```

## Then what is registry.yaml?

An index, plus the two things a branch name cannot tell you: **when the patch
becomes unnecessary**, and **which test proves it still works**.

`../upgrade.sh <new-tag>` runs every `obsolete_when` predicate against a fresh
upstream worktree and prints UPSTREAMED or STILL NEEDED per patch. That is the
whole reason the registry exists — without it, every upgrade is archaeology.

Five branches, seven changes: `rocm-paged-attention-256k` carries the multi-pass
reduction plus three local gfx90a commits, which the README table lists
separately because they are separately justified.

## What this directory is *for*

Runtime Python overlays, bind-mounted by `compose.yaml` over the installed
package:

```yaml
# - ./patches/<name>.py:/opt/python/lib/python3.14/site-packages/vllm/<path>:ro
```

It is empty because every patch has been promoted into the fork, which is the
better place for anything permanent — it gets compiled, tested and tagged with
the rest. Overlays are for changes not yet promoted: things you are still
iterating on, where a rebuild per edit is too slow.

If you add one, add a `registry.yaml` entry too, and open an issue to promote
it. An overlay that outlives the iteration it was for is a patch nobody can
find.

## One branch is not in the tag

`offer/gfx90a-blocksize-guard` exists on the fork but is deliberately excluded
from `v0.26.1rc0+mi210.1`. It is not a patch — it is the same block_size guard
rebased onto upstream PR #39001's commits, packaged to offer to that PR. Its
*content* is already in the tag via `rocm-paged-attention-256k`; verify with:

```bash
git show v0.26.1rc0+mi210.1:vllm/platforms/rocm.py | grep 'block_size <= 64'
```

It has no registry entry because it carries nothing the tag does not already
have. Upstream-offering branches are packaging, not patches.
