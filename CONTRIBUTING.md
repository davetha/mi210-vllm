# Contributing

This project pins three repos into one image:

- **davetha/vllm** — fork of vllm-project/vllm, one branch per patch, merged
  into `main`, based on the tag in `VLLM_UPSTREAM_BASE` (see `VERSIONS`).
- **davetha/aiter** — fork of ROCm/aiter, source patches for gfx90a.
- **davetha/aiter-cdna2** — binary-repatches AITER's gfx942 ASM code objects
  to gfx90a. Not a source patch: AITER ships no gfx90a code objects at all,
  so this tool disassembles the gfx942 ones, substitutes CDNA2 mnemonics, and
  re-assembles.
- **this repo** — pins all three by ref/digest, builds the one compiled
  layer, and runs the gates in `build/verify.sh`.

Work out which repo your change belongs in before writing any code. The
wrong repo means the fix cannot be dropped independently later, or cannot be
sent upstream at all.

## Which repo

| change | repo |
|---|---|
| vLLM Python or C++ behaviour (`csrc/`, `vllm/`) | davetha/vllm, own branch |
| AITER kernel source | davetha/aiter |
| gfx942 -> gfx90a ASM translation itself | davetha/aiter-cdna2 |
| Dockerfile, compose.yaml, VERSIONS, tuning configs, verify.sh | this repo |

If you are not sure, ask which layer the bug lives in: does it need a
compile (`csrc/rocm/attention.cu` or AITER ASM), or is it pure Python /
packaging? See `docs/ARCHITECTURE.md` for why that distinction is the whole
reason the build is shaped the way it is.

## One branch per patch

Every patch to davetha/vllm lives on its own branch, merged to `main`.
`patches/registry.yaml` records what each branch carries and when it can be
dropped.

The reason is upgrade cost. `upgrade.sh` runs each patch's `obsolete_when`
predicate against a new upstream tag and reports UPSTREAMED or STILL NEEDED
per patch, one at a time. That only works if a patch is a single droppable
unit. A patch folded into a pile of unrelated changes cannot be triaged or
dropped without re-deriving what it touches by hand — which is exactly what
determining `obsolete_when`/`verify` for the existing six patches cost:
hours, done once, so it never has to be redone from scratch.

If your change is unrelated to an existing patch, open a new branch. Do not
add it to an existing patch's branch even if the diff is small.

## Every vLLM patch needs a registry entry

A new branch on davetha/vllm is not done until `patches/registry.yaml` has
an entry for it:

```yaml
- name: short-slug
  branch: exact-branch-name
  upstream: [<PR number>, ...]   # empty list if nothing sent upstream yet
  carries: "one line: what this patch does and who wrote the original, if not you"
  obsolete_when: "shell predicate; exit 0 means upstream already fixed this, drop the branch"
  still_needed_when: "optional; the converse condition, when it clarifies obsolete_when"
  verify: "the test command that proves this patch still works"
```

`obsolete_when` is not documentation, it is a script `upgrade.sh` runs
against a checked-out worktree of the new upstream tag. Write it as a
`grep`/test invocation that returns 0 exactly when the upstream fix has
landed, matching the style of the existing entries (e.g. `grep -q
'npar_loops > 16' csrc/rocm/attention.cu`). If you cannot express the
condition as a predicate, that is a sign it needs more thought, not that it
can be skipped — without it, upgrades degenerate into re-reading every diff
by hand for every patch, every time.

`verify` must be a real, runnable test command, not a description of manual
steps.

## Sending fixes upstream

If a fix belongs in vllm-project/vllm proper — not something specific to
this fork's packaging — send it there too, and put the PR number in the
entry's `upstream:` list once it exists. Carrying a patch is an ongoing
cost: every upstream rebase has to re-check it, every reader has to
understand why it exists. Deleting a patch because it landed upstream is the
goal, not carrying it forever. `benchmark-moe-int8`'s `upstream: [31011]` is
marked stale (auto-closed) precisely so the next person does not assume it
is still open — check the PR before assuming `upstream:` means "in flight."

## Claims you cannot test

If you cannot run something on the hardware you have — RDNA, gfx942, or any
architecture you do not own a card for — do not assert it as measured.
Record it instead. `build/verify.sh` writes a `hardware_validated` field
into its JSON record (`RECORD`, default `/tmp/qualification.json`): `true`
only when tier 2 actually ran against a real GPU, `false` when it was
skipped. The script also appends a list of `unvalidated_claims` for things
this repo carries but cannot check on its own cards — for example the
RDNA4 head_size consequence, which is derived from the same constexpr as the
gfx90a rule but has never run on RDNA4 silicon.

If your patch carries a similar untested claim, follow that pattern: state
the derivation, say plainly that it has not been run, and do not write "this
works on X" unless a test actually ran on X.

## Running verification locally

`build/verify.sh` runs inside the image and gates the build. Three tiers,
not substitutes for one another — a static marker proves source is present,
not that the gate selects it; a passing gate proves selection, not numeric
correctness:

```
build/verify.sh --max-tier 0   # static markers only, no GPU needed
build/verify.sh --max-tier 1   # + gate behaviour, no GPU needed
build/verify.sh --max-tier 2   # + numeric tests against reference impls, needs a GPU
```

Tiers 0-1 run automatically during `docker build` (see the `verified` stage
in `build/Dockerfile`) and fail the build on any FAIL line. Tier 2 needs
real cards, so it does not run in `docker build` — `build/build.sh` runs it
against real hardware before recording a digest into `VERSIONS`.

Before opening a PR that touches a patched code path, run at least the tier
your change can reach: tier 0/1 if you have no card, tier 2 if you do.
State in the PR which tier you ran and on what hardware — see the PR
template's checklist.

## Packaging, compose, and tuning changes

Changes to `compose.yaml`, `VERSIONS`, `build/Dockerfile`, or
`tuning/` do not touch either fork; they belong in this repo directly. If
you are adding a tuned `fused_moe` config, read `tuning/README.md` first —
three separate mistakes have already cost real GPU-hours there (missing
`dtype` tag, missing prefill coverage, folder collisions from `K` not being
part of the filename key), and the same rules apply to new configs.
