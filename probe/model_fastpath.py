#!/usr/bin/env python3
"""Which gfx90a fast paths will this checkpoint hit, and what would be better.

    model-fastpath /models/my-model
    model-fastpath Qwen/Qwen3-8B
    model-fastpath /models/my-model --tp 2 --json

Reads config.json only -- no weights, no GPU allocation, seconds not minutes.

The verdicts come from calling vLLM's OWN predicates in this image, not from a
copy of their logic. That is the whole point: a hand-written summary of the
rules drifts from the code the moment either changes, and a confident wrong
answer here is worse than no tool. Anything this cannot ask the code directly
is labelled as such.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NoReturn

# Block sizes worth reporting: the two the template-specialised kernel takes,
# the largest the gfx90a guard admits, and two that hybrid models pick for
# themselves (Qwen3-Next lands on 544 by aligning attention to the mamba page).
BLOCK_SIZES = (16, 32, 64, 128, 544)

RESET, BOLD, RED, GREEN, YELLOW, DIM = (
    ("", "", "", "", "", "")
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty()
    else ("\033[0m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[2m")
)


def die(msg: str) -> NoReturn:
    print(f"{RED}error:{RESET} {msg}", file=sys.stderr)
    raise SystemExit(2)


def normalise_ref(ref: str) -> tuple[str, str | None]:
    """Accept a Hub URL as readily as a repo id, and return (repo_id, revision).

    People paste the address bar. Rejecting that and demanding they retype
    'Org/Model' is a pointless papercut, and /tree/<rev> carries a revision
    worth honouring rather than discarding.
    """
    s = ref.strip().rstrip("/")
    for prefix in (
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "huggingface.co/",
        "hf.co/",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    else:
        return ref, None

    # Strip the Hub's own routes; /tree/<rev> and /blob/<rev> name a revision.
    rev = None
    for route in ("/tree/", "/blob/", "/resolve/"):
        if route in s:
            s, _, tail = s.partition(route)
            rev = tail.split("/", 1)[0] or None
            break
    for suffix in ("/discussions", "/commits", "/settings"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s, rev


def load_config(ref: str) -> tuple[dict, str]:
    """Local directory, an HF repo id, or a Hub URL (config.json only)."""
    local = os.path.join(ref, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f), local
    if os.path.isdir(ref):
        die(f"{ref} is a directory but has no config.json")

    repo_id, rev = normalise_ref(ref)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die(f"{ref} is not a local model and huggingface_hub is unavailable")
    try:
        p = hf_hub_download(repo_id=repo_id, filename="config.json", revision=rev)
    except Exception as e:  # noqa: BLE001 - report the real reason, do not guess
        die(f"could not fetch config.json for {repo_id!r}: {e}")
    with open(p) as f:
        return json.load(f), p


def text_config(cfg: dict) -> dict:
    """Multimodal checkpoints nest the language model's shape one level down."""
    for k in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(k), dict):
            return cfg[k]
    return cfg


def facts(cfg: dict) -> dict:
    t = text_config(cfg)
    heads = t.get("num_attention_heads")
    kv_heads = t.get("num_key_value_heads", heads)
    hidden = t.get("hidden_size")

    head_size = t.get("head_dim")
    if not head_size and heads and hidden:
        head_size = hidden // heads

    q = cfg.get("quantization_config") or t.get("quantization_config") or {}

    # compressed-tensors does not carry a top-level `bits`. Weight and
    # activation widths live per config group, and the difference between
    # W4A16 and W4A8 decides which kernel runs -- reading only `bits` reported
    # "no verdict" for every compressed-tensors checkpoint.
    w_bits = q.get("bits") or q.get("weight_bits")
    a_bits = None
    for grp in (q.get("config_groups") or {}).values():
        if not isinstance(grp, dict):
            continue
        w = grp.get("weights") or {}
        a = grp.get("input_activations") or {}
        w_bits = w.get("num_bits", w_bits)
        # A null input_activations block means activations are left at 16-bit.
        a_bits = a.get("num_bits", 16) if grp.get("input_activations") else 16
    experts = (
        t.get("num_experts")
        or t.get("n_routed_experts")
        or t.get("num_local_experts")
        or t.get("num_experts_per_tok") and t.get("n_routed_experts")
    )

    return {
        "model_type": cfg.get("model_type") or t.get("model_type"),
        "dtype": str(t.get("torch_dtype") or cfg.get("torch_dtype") or "unknown"),
        "heads": heads,
        "kv_heads": kv_heads,
        "head_size": head_size,
        "hidden": hidden,
        "layers": t.get("num_hidden_layers"),
        "max_pos": t.get("max_position_embeddings"),
        "experts": experts,
        "moe_intermediate": t.get("moe_intermediate_size"),
        "quant_method": (q.get("quant_method") or "").lower() or None,
        "quant_bits": w_bits,
        "quant_act_bits": a_bits,
        "quant_scheme": f"W{w_bits}A{a_bits}" if w_bits and a_bits else None,
        "quant_fmt": (q.get("format") or "").lower() or None,
        "quant_raw": q,
    }


def load_gates():
    """vLLM's real predicates, or a loud failure. Never a guess."""
    try:
        from vllm.platforms import rocm
    except Exception as e:  # noqa: BLE001
        die(
            "cannot import vllm.platforms.rocm -- run this inside the image, "
            f"with --device /dev/kfd --device /dev/dri. ({e})"
        )
    missing = [
        n
        for n in ("use_rocm_custom_paged_attention", "_rocm_free_paged_attention_ok")
        if not hasattr(rocm, n)
    ]
    if missing:
        die(
            f"this vllm build has no {', '.join(missing)} -- the gfx90a guards "
            "are not in this image, so no verdict can be trusted"
        )
    return rocm


def report(ref: str, f: dict, rocm, tp: int) -> dict:
    import torch

    arch = getattr(rocm, "_GCN_ARCH", None) or "unknown"
    on_gfx90a = getattr(rocm, "_ON_GFX90A", False)
    out: dict = {"model": ref, "arch": arch, "facts": f, "attention": {}, "notes": []}

    print(f"{BOLD}{ref}{RESET}")
    print(
        f"  {f['model_type']}  {f['layers']} layers  hidden {f['hidden']}  "
        f"heads {f['heads']}/{f['kv_heads']}  head_size {f['head_size']}  "
        f"dtype {f['dtype']}"
    )
    print(f"  running on {arch}")
    if not on_gfx90a:
        print(
            f"  {YELLOW}note{RESET} this report is for gfx90a; the gates below "
            "are being evaluated on a different architecture"
        )
    print()

    # ---------------------------------------------------------- attention ----
    print(f"{BOLD}attention{RESET}")
    hs, heads, kvh = f["head_size"], f["heads"], f["kv_heads"]
    if not (hs and heads and kvh):
        print(f"  {YELLOW}cannot evaluate{RESET}: config lacks head/hidden sizes")
        out["notes"].append("attention not evaluated: incomplete config")
    else:
        gqa = heads // kvh
        if not 1 <= gqa <= 16:
            print(
                f"  {YELLOW}gqa_ratio {gqa}{RESET} is outside the custom kernel's "
                "1..16 range; it will use Triton at every block size"
            )
        for bs in BLOCK_SIZES:
            ok = rocm.use_rocm_custom_paged_attention(
                torch.bfloat16, hs, bs, gqa, f["max_pos"] or 4096, 0, "auto"
            )
            free = rocm._uses_rocm_free_paged_attention(hs, bs)
            why = ""
            if not ok:
                if not rocm._rocm_free_paged_attention_ok(hs, bs):
                    mult = rocm._FREE_KERNEL_HEAD_MULTIPLE
                    why = f"head_size not a multiple of {mult}: free kernel would drop part of V"
                elif on_gfx90a and bs > 64:
                    why = "block_size > 64 is wrong on gfx90a; routed to Triton"
                elif hs not in (64, 128, 192, 256):
                    why = f"head_size {hs} not supported by the custom kernel"
                else:
                    why = "declined by the gate"
            kind = "free kernel" if free else "specialised kernel"
            mark = f"{GREEN}custom PA{RESET}" if ok else f"{DIM}Triton{RESET}   "
            print(f"  block_size {bs:>4}  {mark}  {kind if ok else why}")
            out["attention"][bs] = {"custom_paged_attention": ok, "reason": why or kind}

    # ------------------------------------------------------------ context ----
    mp = f["max_pos"]
    if mp:
        print()
        print(f"{BOLD}context{RESET}")
        if mp > 131072:
            print(
                f"  {GREEN}{mp} tokens{RESET} needs the multi-pass reduction in "
                "this image; stock vLLM caps at 131072 on this path"
            )
        else:
            print(f"  {mp} tokens, within the single-pass range")

    # -------------------------------------------------------- quantization ---
    print()
    print(f"{BOLD}quantization{RESET}")
    method, bits, fmt = f["quant_method"], f["quant_bits"], f["quant_fmt"]
    is_moe = bool(f["experts"])
    recs: list[str] = []

    if not method:
        print(f"  none ({f['dtype']}) -- runs natively, no quantization fast path")
    elif "fp8" in (method or "") or "fp8" in (fmt or ""):
        print(
            f"  {RED}fp8{RESET} -- MI210 is CDNA2 and has {BOLD}no FP8 datapath{RESET}. "
            "CDNA3 (MI300) introduced it."
        )
        print("  The checkpoint may load, but there is no hardware acceleration to hit.")
        recs.append(
            "Prefer an int4 (AWQ / GPTQ / compressed-tensors W4A16) or int8 W8A8 "
            "export of the same model. Both have hardware support on gfx90a; fp8 does not."
        )
    elif bits == 4:
        scheme = f["quant_scheme"] or "W4A?"
        if method == "compressed-tensors" and f["quant_act_bits"] == 16:
            print(f"  {GREEN}compressed-tensors {scheme}{RESET}")
            if is_moe:
                print(
                    "  MoE W4A16 hits the int4 interleave path this image widens to "
                    "GFX9 (measured 1.45-4.8x on the int4 MoE kernel, bit-identical)."
                )
            else:
                print("  Dense int4. The interleave patch is MoE-only, so it does not apply.")
        elif method == "compressed-tensors":
            print(f"  {YELLOW}compressed-tensors {scheme}{RESET}")
            print(
                "  4-bit weights with quantised activations take the int-quantized "
                "path, not the wNa16 path the interleave patch widens."
            )
            if is_moe:
                recs.append(
                    "W4A16 (16-bit activations) would reach the int4 interleave kernel; "
                    f"{scheme} does not. Worth comparing if this model is MoE-bound."
                )
        else:
            print(f"  {YELLOW}{method} 4-bit{RESET}")
            if is_moe:
                print(
                    f"  {YELLOW}This misses the int4 interleave fast path.{RESET} That patch "
                    "lives in the compressed-tensors MoE path; the awq/gptq path "
                    "(moe_wna16.py) has not been ported."
                )
                recs.append(
                    "A compressed-tensors W4A16 export of this model would hit the int4 "
                    "interleave path (1.45-4.8x on the int4 MoE kernel, bit-identical). "
                    "Same weights, different packing -- `model-convert --to W4A16`."
                )
    elif bits == 8:
        scheme = f["quant_scheme"] or "8-bit"
        print(f"  {GREEN}{method} {scheme}{RESET} -- int8 has hardware support on gfx90a")
    else:
        print(f"  {method} (bits={bits}) -- no specific verdict for this combination")
        out["notes"].append(f"unrecognised quantization: {method}/{bits}")

    # ---------------------------------------------------------------- MoE ----
    if is_moe:
        print()
        print(f"{BOLD}MoE{RESET}")
        E, inter = f["experts"], f["moe_intermediate"]
        print(f"  {E} experts, moe_intermediate_size {inter}")
        if inter and tp:
            N = inter // tp
            print(f"  at TP={tp} the tuned-config key is E={E},N={N}")
            print(
                f"  {DIM}the filename does not encode K (hidden={f['hidden']}), so a "
                f"folder holding another model's E={E},N={N} config would be "
                f"silently misapplied -- see tuning/README.md{RESET}"
            )
        recs.append(
            "Tuned fused_moe configs measured neutral or worse on this hardware "
            "across ~13 GPU-hours. Start untuned; see tuning/manifest.json."
        )

    # ------------------------------------------------------ recommendations --
    print()
    if recs:
        print(f"{BOLD}recommendations{RESET}")
        for r in recs:
            print(f"  - {r}")
    else:
        print(f"{GREEN}No changes recommended{RESET} -- this checkpoint hits the "
              "paths this image optimises.")

    out["recommendations"] = recs
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="model-fastpath",
        description="Which gfx90a fast paths will this checkpoint hit?",
    )
    ap.add_argument("model", help="local model directory or HF repo id")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (default 1)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    cfg, src = load_config(a.model)
    f = facts(cfg)
    rocm = load_gates()

    if a.json:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = report(a.model, f, rocm, a.tp)
        out["config_source"] = src
        out.pop("facts", None)
        print(json.dumps({**out, "facts": f}, indent=2, default=str))
    else:
        report(a.model, f, rocm, a.tp)
        print(f"\n{DIM}config: {src}{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
