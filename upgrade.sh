#!/usr/bin/env bash
# ./upgrade.sh v0.27.0
#
# Triage every patch against a new upstream tag BEFORE touching anything.
# Prints one line per patch: UPSTREAMED (drop it) or STILL NEEDED (rebase it).
#
# The patches themselves are NOT in this repo -- they are branches on the vLLM
# fork, merged into the tag VERSIONS pins. patches/registry.yaml is the index of
# those branches. See patches/README.md.
set -uo pipefail

# Resolve the registry BEFORE cd'ing into the fork. Reading it relative to the
# working directory is what broke this script: every path resolved inside
# $VLLM_SRC, which has no patches/registry.yaml, so it died on FileNotFoundError
# before triaging anything.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="$REPO/patches/registry.yaml"
[ -f "$REGISTRY" ] || { echo "no registry at $REGISTRY"; exit 1; }

NEW="${1:?usage: ./upgrade.sh <upstream-tag>}"
VLLM_SRC="${VLLM_SRC:-$HOME/eypc/vllm-fork}"
die() { echo "$*" >&2; exit 1; }
[ -d "$VLLM_SRC/.git" ] || die "no vLLM clone at $VLLM_SRC (set VLLM_SRC)"

cd "$VLLM_SRC" || die "cannot cd to $VLLM_SRC"
git fetch upstream --tags --quiet
git rev-parse "$NEW" >/dev/null 2>&1 || { echo "unknown tag: $NEW"; exit 1; }

WT=$(mktemp -d)
git worktree add --detach "$WT" "$NEW" --quiet
trap 'git worktree remove --force "$WT" 2>/dev/null' EXIT

printf '%-24s %-14s %s\n' PATCH STATUS NOTE
printf '%-24s %-14s %s\n' ----- ------ ----
REGISTRY="$REGISTRY" NEW="$NEW" python3 - "$WT" <<'PY'
import os, subprocess, sys

import yaml

wt = sys.argv[1]
new = os.environ["NEW"]
text = open(os.environ["REGISTRY"]).read()

# Parse the registry as YAML, not with a regex over the raw file.
#
# The regex version -- re.search(r'obsolete_when:\s*"?(.*?)"?\s*$') -- read the
# file text verbatim, so YAML escapes never got decoded: a double-quoted scalar
# containing \" or \\[ reached the shell with its backslashes intact, and a
# single-quoted scalar kept its leading quote. The resulting greps matched
# nothing. Because predicates are conventionally written as `! grep ...`, a
# predicate that fails to match -- or errors outright -- INVERTS TO ZERO, which
# this script reports as UPSTREAMED. Combined with stderr going to DEVNULL, a
# broken predicate silently recommended deleting a patch that was still needed.
# Measured against v0.28.0rc2: four of fourteen patches were wrongly reported
# UPSTREAMED, including two whose PRs had been opened the same day.
#
# Predicate stderr is now surfaced rather than swallowed, so a predicate that
# errors is visible instead of masquerading as a verdict.
for entry in yaml.safe_load(text):
    name = entry.get("name", "<name?>")
    branch = entry.get("branch", "<branch?>")
    pred = entry.get("obsolete_when")
    if not pred:
        print(f"{name:<24} {'NO PREDICATE':<14} add obsolete_when to registry.yaml")
        continue
    r = subprocess.run(pred, shell=True, cwd=wt, capture_output=True, text=True)
    err = (r.stderr or "").strip().splitlines()
    if err:
        print(f"{name:<24} {'PREDICATE ERR':<14} {err[0][:60]}")
        continue
    if r.returncode == 0:
        print(f"{name:<24} {'UPSTREAMED':<14} drop {branch}, close the PR")
    else:
        print(f"{name:<24} {'STILL NEEDED':<14} git rebase --onto {new} <old-base> {branch}")
PY

echo
echo "Next: rebase survivors, reset main to $NEW, re-merge, rebuild, run build/verify.sh."
echo "Re-measure or mark stale any performance claim in README.md - it is tied to a build."
