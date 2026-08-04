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
[ -d "$VLLM_SRC/.git" ] || { echo "no vLLM clone at $VLLM_SRC (set VLLM_SRC)"; exit 1; }

cd "$VLLM_SRC"
git fetch upstream --tags --quiet
git rev-parse "$NEW" >/dev/null 2>&1 || { echo "unknown tag: $NEW"; exit 1; }

WT=$(mktemp -d)
git worktree add --detach "$WT" "$NEW" --quiet
trap 'git worktree remove --force "$WT" 2>/dev/null' EXIT

printf '%-24s %-14s %s\n' PATCH STATUS NOTE
printf '%-24s %-14s %s\n' ----- ------ ----
REGISTRY="$REGISTRY" NEW="$NEW" python3 - "$WT" <<'PY'
import os, re, subprocess, sys

wt = sys.argv[1]
new = os.environ["NEW"]
text = open(os.environ["REGISTRY"]).read()

for block in text.split("\n- name:")[1:]:
    name = block.splitlines()[0].strip()
    bm = re.search(r'^\s*branch:\s*(\S+)', block, re.M)
    branch = bm.group(1) if bm else "<branch?>"
    m = re.search(r'obsolete_when:\s*"?(.*?)"?\s*$', block, re.M)
    if not m:
        print(f"{name:<24} {'NO PREDICATE':<14} add obsolete_when to registry.yaml")
        continue
    ok = subprocess.run(m.group(1), shell=True, cwd=wt,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if ok:
        print(f"{name:<24} {'UPSTREAMED':<14} drop {branch}, close the PR")
    else:
        print(f"{name:<24} {'STILL NEEDED':<14} git rebase --onto {new} <old-base> {branch}")
PY

echo
echo "Next: rebase survivors, reset main to $NEW, re-merge, rebuild, run build/verify.sh."
echo "Re-measure or mark stale any performance claim in README.md - it is tied to a build."
