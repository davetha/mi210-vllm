#!/usr/bin/env python3
"""Enable the DSA sparse-attention indexer on CDNA2 (gfx90a).

GLM-5.2 and other DSA models need per-token top-k over cached keys. Both of
vLLM's implementations compute the logits in FP8 and neither runs here:

    forward_hip   -> AITER, gated `_ON_GFX942`, deepgemm_fp8_paged_mqa_logits
    forward_cuda  -> DeepGEMM's fp8_fp4_mqa_logits, CUDA-only

so forward_hip raises "Sparse attention indexer ROCm path is only supported on
AITER" on gfx90a even with AITER installed and VLLM_ROCM_USE_AITER=1 -- the
gate is an architecture check, not the env var.

FP8 here is a storage format, not an arithmetic requirement. gfx90a stores and
converts all four FP8 dtypes exactly (measured), so the logits can be computed
in fp32 instead -- more accurate than the path it replaces, at the cost of
reading keys as fp32.

This patch makes two edits:

  1. forward_hip delegates to forward_cuda instead of raising, so all of
     vLLM's paging, chunking, masking and top-k logic is reused and only the
     one kernel is replaced.
  2. The two DeepGEMM entry points are rebound in the module namespace to the
     portable implementations in mqa_logits_gfx9.

Only on ROCm and only when the AITER path is unavailable; gfx942 is untouched.

Idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# PATCHED-gfx9-sparse-indexer"

DELEGATE = '''        # PATCHED-gfx9-sparse-indexer: CDNA2 has no FP8 datapath, so the AITER
        # path above is unreachable here. Fall through to the generic
        # implementation, whose DeepGEMM logits calls are rebound below to
        # fp32 equivalents, rather than refusing to run at all.
        return self.forward_cuda(hidden_states, q_quant, k, weights)
'''

SHIM = '''

# PATCHED-gfx9-sparse-indexer
# Rebind the two DeepGEMM logits entry points to portable fp32 versions when
# the FP8 kernels cannot run. Rebinding in THIS module's namespace (rather than
# editing vllm.utils.deep_gemm) keeps the change local to the indexer.
def _gfx9_install_portable_mqa_logits() -> None:
    try:
        from vllm.platforms import current_platform

        if not current_platform.is_rocm():
            return
        import torch

        arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx942" in arch or "gfx950" in arch:
            return  # native FP8 available; leave the fast path alone
    except Exception:
        return

    from vllm.model_executor.layers import mqa_logits_gfx9 as _p

    def _mqa(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False):
        q_vals, q_scale = q
        k_vals, k_scale = kv
        return _p.mqa_logits(
            q_vals, q_scale, k_vals, k_scale, weights,
            cu_seqlen_ks, cu_seqlen_ke, clean_logits,
        )

    def _paged(q, kv_cache, weights, context_lens, block_tables,
               schedule_metadata, max_model_len, clean_logits=False):
        q_vals, q_scale = q
        return _p.paged_mqa_logits(
            q_vals, q_scale, kv_cache, weights, context_lens, block_tables,
            max_model_len, clean_logits,
        )

    g = globals()
    g["fp8_fp4_mqa_logits"] = _mqa
    g["fp8_fp4_paged_mqa_logits"] = _paged


_gfx9_install_portable_mqa_logits()
'''


def locate(*parts: str) -> Path:
    for base in sys.path:
        p = Path(base).joinpath(*parts)
        if p.is_file():
            return p
    print(f"error: {'/'.join(parts)} not found on sys.path", file=sys.stderr)
    raise SystemExit(2)


def patch_sparse_metadata() -> int:
    """Give ROCMAiterMLASparseMetadata the decode/prefill split fields.

    unified_mla_attention asserts and then USES all three:

        assert (num_decodes is not None and num_prefills is not None
                and num_decode_tokens is not None)
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

    so they decide how the batch is split between the MQA (decode) and MHA
    (prefill) paths. A wrong value here does not raise -- it routes tokens
    through the wrong kernel and produces plausible, wrong output. So these
    are COMPUTED, never defaulted: build() already receives the
    CommonAttentionMetadata that vLLM's own split_decodes_and_prefills()
    takes, and every other backend derives them the same way.

    The AITER sparse builder simply never populated them, because the AITER
    attention path it was written for does not read them. Routing the indexer
    through the generic path is what makes them load-bearing.
    """
    f = locate("vllm", "v1", "attention", "backends", "mla", "rocm_aiter_mla_sparse.py")
    src = f.read_text()
    if MARKER in src:
        print("  metadata: already patched")
        return 0

    # 1. Declare the fields. Anchor on the trailing defaulted field: inserting
    #    a defaulted field before undefaulted ones makes the dataclass invalid.
    anchor = "    block_size: int = 1\n"
    if anchor not in src:
        print("error: trailing `block_size: int = 1` not found in the sparse "
              "metadata dataclass; refusing to guess", file=sys.stderr)
        return 2
    src = src.replace(anchor, anchor + (
        "\n    " + MARKER + "\n"
        "    # Decode/prefill split, read by unified_mla_attention to size the\n"
        "    # MQA vs MHA halves of the batch. Computed in build(); None only\n"
        "    # ever means 'builder did not run', which the assert there catches.\n"
        "    num_decodes: int | None = None\n"
        "    num_prefills: int | None = None\n"
        "    num_decode_tokens: int | None = None\n"
        "    # Max sequence length over the PREFILL slice only. Sibling sparse\n"
        "    # backends (flashmla_sparse, flashattn_mla_sparse) declare it the\n"
        "    # same way and default to 0, which is what 'no prefill' means.\n"
        "    prefill_max_seq_len: int = 0\n"
    ), 1)

    # 2. Compute them from the same helper every other backend uses.
    ctor = "        metadata = ROCMAiterMLASparseMetadata(\n"
    if ctor not in src:
        print("error: metadata constructor call not found; refusing to guess",
              file=sys.stderr)
        return 2
    src = src.replace(ctor, (
        "        " + MARKER + "\n"
        "        from vllm.v1.attention.backends.utils import (\n"
        "            split_decodes_and_prefills as _split,\n"
        "        )\n\n"
        "        _num_decodes, _num_prefills, _num_decode_tokens, _ = _split(\n"
        "            common_attn_metadata\n"
        "        )\n\n"
        "        # Same expression sparse_mla_attention.py uses: max over the\n"
        "        # prefill slice of the batch, 0 when there is no prefill.\n"
        "        _prefill_max_seq_len = 0\n"
        "        if _num_prefills > 0:\n"
        "            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound\n"
        "            if _seq_lens_cpu is None:\n"
        "                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu\n"
        "            _prefill_max_seq_len = int(\n"
        "                _seq_lens_cpu[\n"
        "                    _num_decodes : _num_decodes + _num_prefills\n"
        "                ].max().item()\n"
        "            )\n\n"
        + ctor +
        "            num_decodes=_num_decodes,\n"
        "            num_prefills=_num_prefills,\n"
        "            num_decode_tokens=_num_decode_tokens,\n"
        "            prefill_max_seq_len=_prefill_max_seq_len,\n"
    ), 1)

    f.write_text(src)
    print(f"  metadata: added + computed decode/prefill split in {f.name}")
    return 0


def patch_enable_mla() -> int:
    """Route is_mla_enabled through the gfx90a attention carve-out.

    aiter-cdna2 added is_aiter_attention_supported() -- deliberately broader
    than is_aiter_found_and_supported(), and deliberately only for attention --
    because AITER's hand-written gfx9 ASM attention kernels do run on gfx90a
    once the code objects exist. is_mha_enabled already uses it; is_mla_enabled
    still uses the narrow predicate and so returns None, leaving
    torch.ops.vllm.rocm_aiter_mla_decode_fwd unregistered.

    The repatcher does emit gfx90a mla code objects. But aiter-cdna2's own
    port-matrix.md is explicit about the state:

        mla: 24 kernels, 11 portable, 13 blocked (fp8)
        "mla (11 portable kernels) is not yet enabled or validated."

    So this flips a switch on kernels that have been PORTED but never CHECKED.
    A wrong MLA decode does not raise -- it returns plausible, wrong
    attention. Treat any output from this as unvalidated until it has been
    compared against a non-AITER reference.
    """
    f = locate("vllm", "_aiter_ops.py")
    src = f.read_text()
    if "MARKER-mla" in src:
        print("  mla: already enabled")
        return 0

    needle = (
        "    @classmethod\n"
        "    @if_aiter_supported\n"
        "    def is_mla_enabled(cls) -> bool:\n"
    )
    if needle not in src:
        print("error: is_mla_enabled's decorator not found as expected; "
              "refusing to guess", file=sys.stderr)
        return 2

    src = src.replace(needle, (
        "    @classmethod\n"
        "    # MARKER-mla " + MARKER + ": use the gfx90a attention carve-out, the\n"
        "    # same predicate is_mha_enabled uses. UNVALIDATED -- see\n"
        "    # aiter-cdna2 port-matrix.md: 11 of 24 mla kernels are portable and\n"
        "    # none have been checked on this arch.\n"
        "    @if_aiter_attention_supported\n"
        "    def is_mla_enabled(cls) -> bool:\n"
    ), 1)
    # register_ops_once() carries the narrow predicate too, so on gfx90a NO
    # aiter ops are registered and torch.ops.vllm.rocm_aiter_mla_decode_fwd
    # never exists regardless of is_mla_enabled. Registration only makes the
    # symbol available -- the is_*_enabled predicates still decide what is
    # actually called -- so broadening it does not enable any unvalidated path
    # by itself.
    reg = (
        "    @staticmethod\n"
        "    @if_aiter_supported\n"
        "    def register_ops_once() -> None:\n"
    )
    if reg not in src:
        print("error: register_ops_once decorator not found as expected; "
              "refusing to guess", file=sys.stderr)
        return 2
    src = src.replace(reg, (
        "    @staticmethod\n"
        "    # " + MARKER + ": register the op symbols on gfx90a. Which of them\n"
        "    # may actually run is still decided by the is_*_enabled predicates.\n"
        "    @if_aiter_attention_supported\n"
        "    def register_ops_once() -> None:\n"
    ), 1)

    f.write_text(src)
    print(f"  mla: enabled + ops registered in {f.name} (UNVALIDATED)")
    return 0


def main() -> int:
    rc = patch_sparse_metadata()
    if rc:
        return rc
    rc = patch_enable_mla()
    if rc:
        return rc
    f = locate("vllm", "model_executor", "layers", "sparse_attn_indexer.py")
    src = f.read_text()
    if MARKER in src:
        print("  already patched")
        return 0

    needle = (
        '        raise RuntimeError(\n'
        '            "Sparse attention indexer ROCm path is only supported on AITER. "\n'
        '            "Please enable aiter with VLLM_ROCM_USE_AITER=1"\n'
        '        )\n'
    )
    if needle not in src:
        print("error: the forward_hip raise site was not found -- upstream may "
              "have restructured it; refusing to guess", file=sys.stderr)
        return 2

    src = src.replace(needle, DELEGATE, 1) + SHIM
    f.write_text(src)
    print(f"  patched {f.name}: forward_hip delegates, logits calls rebound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
