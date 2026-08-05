#!/usr/bin/env python3
"""Make llm-compressor work on Python 3.14.

Two independent 3.14 incompatibilities in llmcompressor 0.12.0.1. Both are
latent bugs that older Pythons simply never exercised, and both fixes are the
smallest edit that removes the incompatibility.


BUG 1 -- import fails: a builtin shadowed by a method

llmcompressor 0.12.0.1 fails at import on 3.14:

    File ".../llmcompressor/recipe/recipe.py", line 27, in <module>
    ...
    TypeError: 'function' object is not subscriptable
    Unable to evaluate type annotation 'dict[str, Any]'

The cause is a builtin shadowed by a method on the same pydantic model:

    class Recipe(RecipeBase):
        args: dict[str, Any] = Field(default_factory=dict)   # line 25
        ...
        def dict(self, *args, **kwargs) -> dict[str, Any]:   # line 232

Through Python 3.13 the annotation was evaluated eagerly while the class body
ran, at which point `dict` was still the builtin -- `def dict` had not been
reached. Python 3.14 defers annotation evaluation (PEP 649 / annotationlib), so
pydantic resolves the string later, with the completed class namespace in
scope. `dict` is then the method, and `function[str, Any]` raises.

Nothing about the code is wrong on its own terms; it is a latent shadowing that
only a lazy-annotation Python exposes. So the fix is the smallest possible one:
spell the annotations `Dict[...]`, which nothing shadows. No behaviour changes
and the public `.dict()` method keeps its name.


BUG 2 -- oneshot() fails: an unescaped % in an argparse help string

Once it imports, the first real call dies inside transformers' HfArgumentParser:

    File ".../llmcompressor/args/utils.py", line 47, in parse_args
    File ".../transformers/hf_argparser.py", line 235, in _parse_dataclass_field
    ValueError: badly formed help string

Python 3.14 added argparse._check_help, which %-expands every help string at
add_argument time. The `--splits` field's help documents a dataset slice:

    "... Passing a string like 'train' or 'train[:50%]' is strongly recommended"

Those `%` characters are literal text, but argparse now reads them as format
specifiers. Doubling them to `%%` is exactly what argparse expects and renders
identically.

Idempotent: safe to run twice, and reports honestly if there is nothing to do.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def locate(*parts: str) -> Path:
    # Deliberately do NOT import llmcompressor to locate it: bug 1 raises
    # TypeError at import, so import-based discovery dies before it can find
    # the file it needs to fix.
    for base in sys.path:
        p = Path(base).joinpath(*parts)
        if p.is_file():
            return p
    print(f"error: {'/'.join(parts)} not found on sys.path", file=sys.stderr)
    raise SystemExit(2)


def fix_shadowed_annotations() -> int:
    f = locate("llmcompressor", "recipe", "recipe.py")
    src = f.read_text()

    # Only annotations, never `Field(default_factory=dict)` -- that one is
    # evaluated while the class body runs, where `dict` is still the builtin,
    # and rewriting it would change the default factory.
    hits = re.findall(r"(?::|->)\s*dict\[", src)
    if not hits:
        print(f"  bug 1: nothing to do ({f.name})")
        return 0

    out = re.sub(r"(:\s*)dict\[", r"\1Dict[", src)
    out = re.sub(r"(->\s*)dict\[", r"\1Dict[", out)

    if re.search(r"^from typing import .*\bDict\b", out, re.M) is None:
        if re.search(r"^from typing import ", out, re.M):
            out = re.sub(r"^from typing import ", "from typing import Dict, ", out, count=1, flags=re.M)
        else:
            out = "from typing import Dict\n" + out

    f.write_text(out)
    print(f"  bug 1: rewrote {len(hits)} annotation(s) in {f.name}")
    return len(hits)


def fix_help_percent() -> int:
    f = locate("llmcompressor", "args", "dataset_arguments.py")
    src = f.read_text()

    # Only bare `%` -- an already-doubled `%%` is correct and must be left
    # alone, or running this twice would turn it into `%%%%`.
    pat = re.compile(r"(?<!%)%(?!%)")
    lines, changed = src.splitlines(keepends=True), 0
    for i, ln in enumerate(lines):
        if "%" in ln and pat.search(ln):
            lines[i] = pat.sub("%%", ln)
            changed += 1
    if not changed:
        print(f"  bug 2: nothing to do ({f.name})")
        return 0

    f.write_text("".join(lines))
    print(f"  bug 2: escaped % on {changed} line(s) in {f.name}")
    return changed


TOLERANT = '''        # PATCHED: these are side-effect registrations for OTHER architectures.
        # granitemoe imports GraniteMoeParallelExperts, which transformers 5.14
        # no longer exports, so an unguarded import here kills quantization for
        # every model -- including ones needing neither. Skip what will not load,
        # and say so rather than failing silently.
        import importlib as _il, warnings as _w
        for _m in ("granitemoe", "llama4"):
            try:
                _il.import_module("." + _m, __package__)
            except ImportError as _e:
                _w.warn(
                    f"llmcompressor: {_m} experts registration unavailable ({_e}); "
                    f"quantizing that architecture is unsupported with this "
                    f"transformers version",
                    RuntimeWarning,
                    stacklevel=2,
                )
'''


def fix_eager_arch_imports() -> int:
    """BUG 3 -- not a 3.14 issue: a transformers-version incompatibility.

    LinearExperts2D.get_registration() imports every architecture-specific
    registration module eagerly. transformers 5.14 (what vLLM is built against)
    removed GraniteMoeParallelExperts, so that import raises and takes down
    quantization for unrelated models. Upstream's own metadata says
    transformers<=5.14.1 is supported, so this is a real bug rather than us
    running an unsupported combination.
    """
    f = locate("llmcompressor", "modeling", "moe", "linear_experts.py")
    src = f.read_text()
    if "PATCHED: these are side-effect registrations" in src:
        print(f"  bug 3: nothing to do ({f.name})")
        return 0

    pat = re.compile(
        r"^[ \t]*from \.granitemoe import [^\n]*\n[ \t]*from \.llama4 import [^\n]*\n",
        re.M,
    )
    if not pat.search(src):
        print(f"  bug 3: import block not found in {f.name} -- upstream may have "
              "restructured it; skipping rather than guessing")
        return 0

    f.write_text(pat.sub(TOLERANT, src, count=1))
    print(f"  bug 3: made architecture registrations tolerant in {f.name}")
    return 1


def main() -> int:
    print("patching llmcompressor for Python 3.14 + transformers 5.14:")
    total = fix_shadowed_annotations() + fix_help_percent() + fix_eager_arch_imports()
    if total == 0:
        print("  already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
