#!/usr/bin/env python3
"""Benchmark GLM-5.2 decode throughput with vs without speculative decoding.

Reports per-prompt tok/s (accepted-tokens / wall-time). n-gram spec decoding
only accepts when recent n-grams repeat, so the prompt set deliberately mixes
repetitive (spec-favourable) and open-ended (spec-neutral) content. The signal:

  - repetitive tok/s clearly > baseline (~1.0) AND open-ended not much worse
      => speculative verification beats the PCIe wall here -> native MTP worth
         building (it accepts broadly, not just on repetition).
  - nothing beats baseline
      => the bandwidth wall holds; MTP won't help -> door closed by data.
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.252:8021/v1/completions"
MODEL = "glm-5.2"
N = 80   # tokens to generate per prompt

# (label, kind, prompt)
PROMPTS = [
    ("rep1", "repetitive",
     "Count from 1 to 60, printing each number and its square on its own line, "
     "like: 1 -> 1, 2 -> 4, 3 -> 9, "),
    ("rep2", "repetitive",
     "Write the word 'hello' followed by a number, ten times, incrementing the "
     "number: hello1 hello2 hello3 "),
    ("code", "structured",
     "Write a Python class Stack with push, pop, peek, is_empty, and size "
     "methods. Include a full docstring and a short example. "),
    ("open1", "open-ended",
     "Explain the causes of the French Revolution in three paragraphs, covering "
     "social, economic, and political factors. "),
    ("open2", "open-ended",
     "Describe how photosynthesis works, step by step, including the light "
     "reactions and the Calvin cycle. "),
    ("story", "open-ended",
     "Write a short mystery story about a detective who finds a strange letter "
     "in an old library. "),
]


def post(label, kind, prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt,
                       "max_tokens": N, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ntok = out["usage"]["completion_tokens"]
    tps = ntok / dt
    flag = "  <-- beats baseline" if tps > 1.15 else ""
    print(f"  [{label:>5} {kind:>11}] {ntok:>3} toks in {dt:5.1f}s = "
          f"{tps:4.2f} tok/s{flag}", flush=True)
    return tps, kind


def main():
    print(f"== spec-decode benchmark: {len(PROMPTS)} prompts, max_tokens={N} ==", flush=True)
    print(f"   baseline to beat: ~1.0 tok/s\n", flush=True)
    by_kind = {}
    for label, kind, p in PROMPTS:
        try:
            tps, _ = post(label, kind, p)
            by_kind.setdefault(kind, []).append(tps)
        except Exception as e:
            print(f"  [{label}] ERROR: {e}", flush=True)
    print("\n== summary by kind ==", flush=True)
    for k, v in by_kind.items():
        import statistics
        print(f"  {k:>11}: mean {statistics.mean(v):4.2f} tok/s  "
              f"(min {min(v):.2f} / max {max(v):.2f})", flush=True)
    allv = [x for v in by_kind.values() for x in v]
    if allv:
        import statistics
        print(f"  {'overall':>11}: mean {statistics.mean(allv):4.2f} tok/s", flush=True)


if __name__ == "__main__":
    main()
