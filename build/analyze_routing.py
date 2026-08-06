#!/usr/bin/env python3
"""Analyze captured MoE routing -> expert VRAM cache hit-rate curve.

Reads probe_*.pkl dumped by route_probe_sitecustomize.py. The decisive number
is the per-layer top-K coverage: if caching each layer's K hottest experts
covers a high fraction of token-expert accesses, a VRAM expert cache wins.

Counts may be recorded by >1 TP rank (each rank routes the same tokens), so
counts are SUMMED across dumps -- and since hit-rate is a ratio, the rank
duplication factor cancels. The LRU replay uses a single worker's stream.
"""
import glob
import os
import pickle
import sys

import numpy as np

# Expert footprint (W4A16): 3 mats x 6144 x 2048 params x 0.5 byte ~= 18.9 MiB.
# With TP=2 an expert is sharded across both ranks (~9.45 MiB/rank). The cache
# budget in "experts" is approximate; we report coverage at K fractions.
EXPERT_MIB = 18.9
K_POINTS = [1, 4, 8, 16, 26, 39, 52, 78, 104, 128, 192, 256]
K_LABEL = {1: "0.4%", 8: "3% (=active/tok)", 26: "10%", 39: "15%",
           52: "20%", 78: "30%", 128: "50%"}


def load(dirpath):
    counts = {}      # layer -> int64[num_experts]  (summed across dumps)
    files = sorted(glob.glob(os.path.join(dirpath, "probe_*.pkl")))
    assert files, f"no probe_*.pkl in {dirpath}"
    print(f"loading {len(files)} dump file(s):")
    per_file = []    # (total, basename, data)
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        data = d["data"]
        tot = sum(int(v["counts"].sum()) for v in data.values())
        print(f"  {os.path.basename(f)}: {len(data)} layers, {tot} routings")
        per_file.append((tot, os.path.basename(f), data))
        for li, v in data.items():
            c = v["counts"].astype(np.int64)
            cur = counts.get(li)
            if cur is None or cur.size < c.size:
                nc = np.zeros(max(c.size, cur.size if cur is not None else 0),
                              dtype=np.int64)
                if cur is not None:
                    nc[:cur.size] = cur
                cur = nc
            cur[:c.size] += c
            counts[li] = cur
    # LRU stream: take a single worker's dump (max data) so the access order is
    # not duplicated across TP ranks.
    primary_total, primary_name, primary_data = max(per_file, key=lambda t: t[0])
    del primary_total  # only used to select; stream comes from primary_data
    print(f"primary stream source: {primary_name}")
    # pkl stores two PARALLEL 1-D arrays: stream[i]=expert id,
    # step_ntok[i]=batch size when it was accessed (order preserved across
    # forwards). Keep both so the LRU replay can isolate decode steps.
    streams = {li: (np.asarray(v["step_ntok"], dtype=np.int32),
                   np.asarray(v["stream"], dtype=np.int64))
               for li, v in primary_data.items()}
    return counts, streams


def trim(c):
    nz = np.nonzero(c)[0]
    return c[: (nz.max() + 1)] if nz.size else c


def topk_coverage(c, K):
    c = trim(c)
    K = min(K, c.size)
    total = c.sum()
    if total == 0:
        return 0.0
    return float(np.sort(c)[-K:].sum()) / float(total)


def gini(c):
    c = trim(c).astype(np.float64)
    if c.sum() == 0:
        return 0.0
    cs = np.sort(c)
    n = cs.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * cs).sum()) / (n * cs.sum()) - (n + 1) / n)


def norm_entropy(c):
    c = trim(c).astype(np.float64)
    s = c.sum()
    if s == 0:
        return 1.0
    p = c[c > 0] / s
    H = float(-(p * np.log(p)).sum())
    return H / np.log(c.size)


def lru_hitrate(accesses, K):
    """Replay an ordered expert-id stream through an LRU cache of size K."""
    import collections
    cache = collections.OrderedDict()
    hits = 0
    for e in accesses:
        if e in cache:
            hits += 1
            cache.move_to_end(e)
        else:
            if len(cache) >= K:
                cache.popitem(last=False)
            cache[e] = True
    return hits / len(accesses) if accesses else 0.0


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "/route"
    counts, streams = load(dirpath)

    layers = sorted(counts)
    nL = len(layers)
    print(f"\n== {nL} layers recorded ==\n")

    # Per-layer skew
    ginis = [gini(counts[l]) for l in layers]
    ents = [norm_entropy(counts[l]) for l in layers]
    print(f"per-layer Gini   : mean={np.mean(ginis):.3f} "
          f"min={np.min(ginis):.3f} max={np.max(ginis):.3f}  (1=perfect skew)")
    print(f"per-layer entropy: mean={np.mean(ents):.3f} "
          f"(0=one expert, 1=uniform over {counts[layers[0]].size})\n")

    # coverage curve: per-layer top-K, averaged (the real cache metric)
    print("== Per-layer static top-K cache coverage (mean over layers) ==")
    print(f"{'K':>5} {'frac':>8} {'hit-rate':>9}")
    curve = {}
    for K in K_POINTS:
        hr = np.mean([topk_coverage(counts[l], K) for l in layers])
        curve[K] = hr
        lbl = K_LABEL.get(K, "")
        print(f"{K:>5} {K/256*100:>7.1f}% {hr*100:>8.1f}%  {lbl}")
    pooled = sum(counts[l] for l in layers)
    print(f"\npooled top-8 global coverage: {topk_coverage(pooled,8)*100:.1f}% "
          f"(8/256 uniform would be {8/256*100:.1f}%)")

    # Sample layer top-10
    print("\n== sample layers: hottest experts ==")
    for l in layers[:3] + layers[nL // 2:nL // 2 + 1] + layers[-1:]:
        c = trim(counts[l])
        top = np.argsort(c)[-10:][::-1]
        cov = float(np.sort(c)[-10:].sum()) / float(c.sum())
        print(f"  layer {l:>2}: top10={top.tolist()} "
              f"covers {cov*100:.1f}% (freq {[int(c[i]) for i in top]})")

    # LRU over decode stream (small step_ntok == decode steps). Each entry of
    # streams[l] is (step_ntok_array, id_array); select ids whose batch was a
    # decode step, preserving order, then replay through an LRU cache.
    print("\n== LRU replay over DECODE stream (temporal locality) ==")
    dec_layers = []
    for l in layers:
        pair = streams.get(l)
        if pair is None:
            continue
        ntok_arr, id_arr = pair
        dec = id_arr[ntok_arr <= 4]   # decode-batch accesses, in order
        if dec.size:
            dec_layers.append((l, dec))
    if dec_layers:
        print(f"{'K':>5} {'static':>8} {'LRU':>8}")
        for K in [8, 16, 26, 39, 52, 78]:
            st = np.mean([topk_coverage(counts[l], K) for l, _ in dec_layers])
            lr = np.mean([lru_hitrate(d.tolist(), K) for _, d in dec_layers])
            print(f"{K:>5} {st*100:>7.1f}% {lr*100:>7.1f}%")
    else:
        print("  (no decode steps captured; FREQ-only run)")

    print("\n== VERDICT ==")
    c10 = curve.get(26, 0.0)
    c20 = curve.get(52, 0.0)
    print(f"top-10%-experts-per-layer covers {c10*100:.1f}% of accesses; "
          f"top-20% covers {c20*100:.1f}%.")
    if c10 >= 0.6:
        print("=> STRONG skew: a VRAM expert cache is worth building.")
    elif c10 >= 0.4:
        print("=> MODERATE skew: caching helps but less than hoped; "
              "weigh engineering cost.")
    else:
        print("=> WEAK skew (~uniform): caching will not help; the bottleneck "
              "is bandwidth, not locality.")


if __name__ == "__main__":
    main()
