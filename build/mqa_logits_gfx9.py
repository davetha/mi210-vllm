# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable MQA logits for the sparse-attention indexer.

The DSA indexer needs, per query row i and cached key j:

    logits[i, j] = sum_h weights[i, h] * dot(q[i, h, :], k[j, :] * k_scale[j])

masked to j in [cu_seqlen_ks[i], cu_seqlen_ke[i]).

Both existing implementations compute this in FP8 and neither runs on CDNA2:

  * forward_hip -> AITER, gated `_ON_GFX942` and calling
    deepgemm_fp8_paged_mqa_logits
  * forward_cuda -> DeepGEMM's fp8_fp4_mqa_logits, CUDA-only

CDNA2 has no FP8 datapath, so an FP8 kernel is the wrong shape for it -- but
FP8 here is only a storage format. Measured on gfx90a: float8_e4m3fn,
e4m3fnuz, e5m2 and e5m2fnuz all store and convert to f32/bf16 exactly. So
dequantise and do the arithmetic in fp32, which the hardware is happy with and
which is strictly more accurate than the FP8 path it replaces.

The cost is bandwidth, not correctness: k is read as fp32 rather than fp8. That
is the honest trade for making the model run at all.

Chunked over keys because the natural expression materialises [M, H, N] --
128x64x32768 fp32 is 1.07 GiB, and N grows with context. Chunking bounds it
without changing the result: every key is independent, and the weighted sum
over h is computed within a chunk.
"""

from __future__ import annotations

import torch

# Keys per chunk. 4096 keeps the [M, H, chunk] intermediate near 100 MB at
# M=128, H=64 while staying large enough that the GEMM is not launch-bound.
DEFAULT_KEY_CHUNK = 4096


def _dequant(x: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """FP8 (or any low-precision) storage -> fp32 values."""
    out = x.to(torch.float32)
    if scale is not None:
        s = scale.to(torch.float32)
        # scale is per-token/per-row; broadcast over the trailing feature dim
        while s.dim() < out.dim():
            s = s.unsqueeze(-1)
        out = out * s
    return out


def mqa_logits(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    k_scale: torch.Tensor | None,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = False,
    key_chunk: int = DEFAULT_KEY_CHUNK,
) -> torch.Tensor:
    """Reference MQA logits in fp32.

    q:       [M, H, D]  low-precision values
    q_scale: [M] or None (None means the scale is already folded into weights)
    k:       [N, D]  low-precision values
    k_scale: [N] or None
    weights: [M, H]
    returns: [M, N] float32, masked outside each row's [ks, ke)
    """
    M, H, D = q.shape
    N = k.shape[0]
    assert weights.shape == (M, H), f"weights {tuple(weights.shape)} != {(M, H)}"
    assert k.shape[1] == D, f"k dim {k.shape[1]} != q dim {D}"

    qf = _dequant(q, q_scale)              # [M, H, D]
    wf = weights.to(torch.float32)         # [M, H]

    # The head dimension collapses. This is multi-QUERY attention: every head
    # dots against the SAME k, so
    #
    #   sum_h w[m,h] * sum_d q[m,h,d]*k[n,d]
    #     == sum_d ( sum_h w[m,h]*q[m,h,d] ) * k[n,d]
    #
    # Folding the weights into q first turns the whole thing into one
    # [M, D] @ [D, chunk] GEMM. No H in the inner loop, so both the FLOPs and
    # the intermediate shrink by a factor of H (64x for GLM-5.2) versus
    # computing per-head logits and reducing afterwards.
    qw = torch.einsum("mhd,mh->md", qf, wf)   # [M, D]

    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)
    for start in range(0, N, key_chunk):
        stop = min(start + key_chunk, N)
        kf = _dequant(k[start:stop], None if k_scale is None else k_scale[start:stop])
        logits[:, start:stop] = qw @ kf.transpose(0, 1)

    # Mask outside the per-row valid key range. -inf so a downstream top-k
    # cannot select an invalid key even if it ignores the range.
    idx = torch.arange(N, device=q.device).unsqueeze(0)
    valid = (idx >= cu_seqlen_ks.unsqueeze(1)) & (idx < cu_seqlen_ke.unsqueeze(1))
    logits.masked_fill_(~valid, float("-inf"))
    if clean_logits:
        logits = torch.nan_to_num(logits, neginf=float("-inf"))
    return logits


def mqa_logits_naive(
    q, q_scale, k, k_scale, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=False
) -> torch.Tensor:
    """Unchunked, written straight from the definition. Test oracle only."""
    qf = _dequant(q, q_scale)
    kf = _dequant(k, k_scale)
    per_head = torch.einsum("mhd,nd->mhn", qf, kf)
    logits = (per_head * weights.to(torch.float32).unsqueeze(-1)).sum(dim=1)
    idx = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (idx >= cu_seqlen_ks.unsqueeze(1)) & (idx < cu_seqlen_ke.unsqueeze(1))
    return logits.masked_fill(~valid, float("-inf"))


def paged_mqa_logits(
    q: torch.Tensor,
    q_scale: torch.Tensor | None,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Paged-KV variant, same math over a block table.

    q:         [B, next_n, H, D] low-precision
    kv_cache:  [num_blocks, block_size, 1, D+4] uint8 -- the trailing 4 bytes
               of each (block, position) hold the float32 dequant scale
    weights:   [B * next_n, H]
    returns:   [B * next_n, max_model_len] float32
    """
    B, next_n, H, D = q.shape
    M = B * next_n
    num_blocks, block_size, _, stride = kv_cache.shape
    assert stride == D + 4, f"kv_cache last dim {stride} != D+4 ({D + 4})"

    qf = _dequant(q.reshape(M, H, D), q_scale)
    qw = torch.einsum("mhd,mh->md", qf, weights.to(torch.float32))

    logits = torch.full(
        (M, max_model_len), float("-inf"), dtype=torch.float32, device=q.device
    )

    for b in range(B):
        ctx = int(context_lens[b].item())
        if ctx <= 0:
            continue
        n_blk = (ctx + block_size - 1) // block_size
        blocks = kv_cache[block_tables[b, :n_blk].long()]      # [n_blk, bs, 1, D+4]
        flat = blocks.reshape(n_blk * block_size, D + 4)[:ctx]

        # Split the packed row: D bytes of fp8 key, then a float32 scale.
        k_bits = flat[:, :D]
        k_scale = flat[:, D:].contiguous().view(torch.float32).reshape(-1)
        kf = k_bits.view(torch.float8_e4m3fn).to(torch.float32) * k_scale.unsqueeze(-1)

        rows = slice(b * next_n, (b + 1) * next_n)
        out = qw[rows] @ kf.transpose(0, 1)                    # [next_n, ctx]

        # Within a speculative group the i-th token may not see keys belonging
        # to later tokens of the same group.
        if next_n > 1:
            pos = ctx - next_n + torch.arange(next_n, device=q.device)
            key_idx = torch.arange(ctx, device=q.device).unsqueeze(0)
            out = out.masked_fill(key_idx > pos.unsqueeze(1), float("-inf"))

        logits[rows, :ctx] = out

    return logits
