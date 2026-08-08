#!/usr/bin/env python3
"""Depth sweep for vLLM matching llama-bench's -d semantics.

For each depth d: send a prompt of ~d tokens, then measure
  - prefill: time to first token for a 2048-token chunk at that depth
  - decode:  tok/s generating 256 tokens with d tokens already in context

Mirrors: llama-bench -p 2048 -n 256 -d <depths>
"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8033"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "q36-awq"
DEPTHS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
          else ["0", "8192", "16384", "32768", "65536", "98304", "131072", "262144"])]
PP, TG = 2048, 256

FILLER = ("The memory controller schedules bank activations to hide row precharge latency. "
          "Gradient checkpointing trades recomputation for reduced activation storage. "
          "A wavefront executes in lockstep across sixty-four lanes on this architecture. "
          "Speculative decoding drafts several tokens and verifies them in one pass. ")

def post(path, payload, timeout=3600):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def make_prompt(n_tokens):
    # ~4 chars/token for this filler; oversize then let server truncate via max_tokens accounting
    reps = max(1, int(n_tokens / 60))   # ~60 tok per FILLER block
    return FILLER * reps

def bench(depth):
    """Return (prefill_tok_s, decode_tok_s, n_prompt)."""
    prompt = make_prompt(depth + PP) if depth else make_prompt(PP)
    # --- prefill: 1 token out, so nearly all latency is prompt processing ---
    t0 = time.time()
    r = post("/v1/completions", {"model": MODEL, "prompt": prompt,
                                 "max_tokens": 1, "temperature": 0})
    t_pf = time.time() - t0
    n_prompt = r.get("usage", {}).get("prompt_tokens", 0)
    pf_tok_s = n_prompt / t_pf if t_pf > 0 else 0

    # --- decode: generate TG tokens from same prompt; subtract prefill time ---
    t0 = time.time()
    r2 = post("/v1/completions", {"model": MODEL, "prompt": prompt,
                                  "max_tokens": TG, "temperature": 0,
                                  "ignore_eos": True})
    t_total = time.time() - t0
    n_gen = r2.get("usage", {}).get("completion_tokens", 0)
    t_dec = max(t_total - t_pf, 1e-6)
    dec_tok_s = n_gen / t_dec if n_gen else 0
    return pf_tok_s, dec_tok_s, n_prompt

print(f"{'depth':>8} {'n_prompt':>9} {'pp2048 t/s':>12} {'tg256 t/s':>11}")
print("-" * 45)
for d in DEPTHS:
    try:
        pf, dec, n = bench(d)
        print(f"{d:>8} {n:>9} {pf:>12.1f} {dec:>11.2f}", flush=True)
    except Exception as e:
        print(f"{d:>8} ERROR {type(e).__name__}: {str(e)[:70]}", flush=True)
