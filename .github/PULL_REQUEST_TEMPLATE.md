## What repo does this belong in

- [ ] davetha/vllm (link the branch/PR)
- [ ] davetha/aiter (link the branch/PR)
- [ ] davetha/aiter-cdna2 (link the branch/PR)
- [ ] this repo (packaging / compose / tuning / docs)

## If this is a vLLM patch

- [ ] `patches/registry.yaml` entry added, with `obsolete_when` and `verify`
- [ ] `upstream:` list updated (PR number, or `[]` if not sent upstream)

## Verification

- [ ] Ran `build/verify.sh --max-tier 0`
- [ ] Ran `build/verify.sh --max-tier 1`
- [ ] Ran `build/verify.sh --max-tier 2` (needs real cards)

Hardware tested on:
- [ ] gfx90a (MI210)
- [ ] gfx942 (MI300-class)
- [ ] RDNA
- [ ] none — could not test on available hardware

## Claims

List anything in this PR stated as fact but not actually measured on the
hardware above (derived from a constexpr, inferred from another arch's test
cases, etc.). See `build/verify.sh`'s `hardware_validated` field and
`unvalidated_claims` for the existing pattern. Write "none" if everything
here was measured.
