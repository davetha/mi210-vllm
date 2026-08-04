#!/usr/bin/env bash
# ./upgrade.sh v0.27.0
#
# Triage every patch against a new upstream tag BEFORE touching anything.
# Prints one line per patch: UPSTREAMED (drop it) or STILL NEEDED (rebase it).
set -uo pipefail
NEW="${1:?usage: ./upgrade.sh <upstream-tag>}"
VLLM_SRC="${VLLM_SRC:-$HOME/eypc/vllm-fork}"

cd "$VLLM_SRC"
git fetch upstream --tags --quiet
git rev-parse "$NEW" >/dev/null 2>&1 || { echo "unknown tag: $NEW"; exit 1; }

WT=$(mktemp -d)
git worktree add --detach "$WT" "$NEW" --quiet
trap 'git worktree remove --force "$WT" 2>/dev/null' EXIT

printf '%-24s %-14s %s\n' PATCH STATUS NOTE
printf '%-24s %-14s %s\n' ----- ------ ----
python3 - "$WT" <<'PY'
import re, subprocess, sys, pathlib
wt = sys.argv[1]
reg = pathlib.Path(__file__).parent  # not used; registry read below
text = open("patches/registry.yaml").read() if pathlib.Path("patches/registry.yaml").exists() \
       else open(str(pathlib.Path(__file__).resolve().parent / "patches/registry.yaml")).read()
for block in text.split("\n- name:")[1:]:
    name = block.splitlines()[0].strip()
    m = re.search(r'obsolete_when:\s*"?(.*?)"?\s*$', block, re.M)
    if not m:
        print(f"{name:<24} {'NO PREDICATE':<14} add obsolete_when to registry.yaml"); continue
    ok = subprocess.run(m.group(1), shell=True, cwd=wt,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if ok:
        print(f"{name:<24} {'UPSTREAMED':<14} drop the branch, close the PR")
    else:
        print(f"{name:<24} {'STILL NEEDED':<14} git rebase --onto {'$NEW'} <old> <branch>")
PY
echo
echo "Next: rebase survivors, reset main to $NEW, re-merge, rebuild, run build/verify.sh."
echo "Re-measure or mark stale any performance claim in README.md - it is tied to a build."
