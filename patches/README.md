# patches/

**There are no patch files here, and that is deliberate.** If you came looking
for the diffs listed in `registry.yaml`, they are branches on the vLLM fork.

## Where the code actually is

Almost every patch is a branch on
[davetha/vllm](https://github.com/davetha/vllm) whose content is in the tag
`VERSIONS` pins:

    VLLM_REF=v0.27.2rc0+mi210.5

`build/Dockerfile` clones that tag and builds it. So the patches are *in the
image* without ever being *in this repo* — nothing here has to be applied,
rebased, or kept in sync with a diff.

**One exception, and it is load-bearing.** `nvfp4-w4a16-gfx90a` is ahead of that
tag and is marked `status: post-tag` in `registry.yaml`. It is indexed here but
is **not** in any image built from the current pin. Read that field before
assuming a listed patch ships.

Verify any of them yourself:

```bash
git clone https://github.com/davetha/vllm && cd vllm
git log --oneline v0.27.2rc0..v0.27.2rc0+mi210.5        # what the tag adds
git diff v0.27.2rc0..v0.27.2rc0+mi210.5 -- vllm/platforms/rocm.py
```

Ask the **tag**, not the branch. Assembling a wave re-applies commits under new
SHAs, so a branch tip can be "ahead" of the tag while its content shipped —
`git merge-base --is-ancestor <branch> $VLLM_REF` answers a question about
commit identity, not about what is in the image.

## Then what is registry.yaml?

An index, plus the two things a branch name cannot tell you: **when the patch
becomes unnecessary**, and **which test proves it still works**.

`../upgrade.sh <new-tag>` runs every `obsolete_when` predicate against a fresh
upstream worktree and prints UPSTREAMED or STILL NEEDED per patch. That is the
whole reason the registry exists — without it, every upgrade is archaeology.

Eleven branches, thirteen changes: `rocm-paged-attention-256k` carries the multi-pass
reduction plus three local gfx90a commits, which the README table lists
separately because they are separately justified. (Five branches / seven changes
was the mi210.1 shape; the W4A16, FP8 and NVFP4 entries are the mi210.5 wave.)

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

## Branches that are not in the tag, and why they differ

Two, for opposite reasons. Do not treat them as the same case.

**`offer/gfx90a-blocksize-guard` — packaging.** Deliberately excluded from the
tag. It is not a patch: it is the same block_size guard rebased onto upstream PR
#39001's commits, packaged to offer to that PR. Its *content* is already in the
tag via `rocm-paged-attention-256k`; verify with:

```bash
git show v0.27.2rc0+mi210.5:vllm/platforms/rocm.py | grep 'block_size <= 64'
```

It has no registry entry because it carries nothing the tag does not already
have. Upstream-offering branches are packaging, not patches.

**`rocm-nvfp4-a16-gfx90a-v0.27.2rc0` — not yet shipped.** The opposite: real work
that the current pin does not carry. It *does* have a registry entry, marked
`status: post-tag`, because losing track of it between pins is the failure this
index exists to prevent. Verify the difference — the file simply is not there:

```bash
git cat-file -e v0.27.2rc0+mi210.5:vllm/model_executor/kernels/linear/nvfp4/triton_gfx90a.py 2>/dev/null \
  && echo shipped || echo "post-tag: not in this pin"
```
