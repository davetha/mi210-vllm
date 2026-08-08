#!/usr/bin/env python3
"""Prefill (prompt-eval) throughput benchmark for llama.cpp server.

Isolates prefill by requesting n_predict=1 and reading the server's own
timings.prompt_per_second. Runs several prompt lengths, 3 reps each,
reports the best (hot) result per length.
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8032"
LENGTHS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2048", "8192", "16384", "32000"])]
REPS = 3

# Deterministic filler: varied technical prose so tokens aren't trivially repetitive.
SENTENCES = [
    "The memory controller schedules bank activations to hide row precharge latency.",
    "Gradient checkpointing trades recomputation for reduced activation storage.",
    "A wavefront executes in lockstep across sixty-four lanes on this architecture.",
    "Speculative decoding drafts several tokens and verifies them in one pass.",
    "The allocator coalesces adjacent free blocks to limit external fragmentation.",
    "Rotary embeddings encode relative position directly into query and key vectors.",
    "Tail latency dominates perceived responsiveness in interactive serving systems.",
    "The scheduler preempts long requests to bound head-of-line blocking.",
]

def build_prompt(target_tokens):
    # ~14 tokens/sentence heuristic; server reports actual count.
    n = max(1, int(target_tokens / 14))
    return " ".join(SENTENCES[i % len(SENTENCES)] for i in range(n))

def run(prompt):
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": False,   # force real prefill every time
    }).encode()
    req = urllib.request.Request(f"{BASE}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.loads(r.read())
    wall = time.time() - t0
    t = body.get("timings", {})
    return {
        "n_prompt": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_per_second": t.get("prompt_per_second"),
        "wall_s": wall,
    }

print(f"{'target':>8} {'n_tok':>7} {'prefill_ms':>11} {'tok/s':>10} {'wall_s':>8}")
print("-" * 50)
results = []
for L in LENGTHS:
    p = build_prompt(L)
    best = None
    for _ in range(REPS):
        try:
            r = run(p)
        except Exception as e:
            print(f"{L:>8} ERROR: {type(e).__name__}: {str(e)[:60]}")
            best = None
            break
        if r["prompt_per_second"] and (best is None or r["prompt_per_second"] > best["prompt_per_second"]):
            best = r
    if best:
        results.append((L, best))
        print(f"{L:>8} {best['n_prompt']:>7} {best['prompt_ms']:>11.1f} "
              f"{best['prompt_per_second']:>10.1f} {best['wall_s']:>8.2f}")

if results:
    peak = max(results, key=lambda x: x[1]["prompt_per_second"])
    print("-" * 50)
    print(f"PEAK PREFILL: {peak[1]['prompt_per_second']:.1f} tok/s at {peak[1]['n_prompt']} tokens")
