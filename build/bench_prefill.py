#!/usr/bin/env python3
"""Measure llama.cpp prefill (TTFT) and decode tok/s at several prompt lengths.

With --cpu-moe the prefill's expert compute runs on CPU too, so prefill (time to
first token) is the sensitive metric on this box. Streams each request to separate
prefill (TTFT) from decode, and reports prefill tok/s = prompt_tokens / TTFT.

Compare the 8k row to the cold-mmap figure in docs/LLAMACPP-GFX90A.md (~170 s,
~51 tok/s) -- with --no-mmap + -tb 48 it should be far faster.
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.252:8031/v1/completions"
SENT = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "travelers debate the merits of regional trade routes and the historical "
        "legacy of the surrounding valley, pausing occasionally to note the weather. ")
TARGETS = [256, 1024, 4096]


def approx_tokens(s):
    return len(s) // 4


def run(prompt, gen=64):
    body = json.dumps({"model": "glm-5.2", "prompt": prompt, "max_tokens": gen,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            s = line.decode(errors="replace").strip()
            if s.startswith("data: "):
                d = s[6:]
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:
                    continue
                ch = o.get("choices", [{}])[0].get("text", "")
                if ch:
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
    dt = time.time() - t0
    dec = (dt - ttft) if ttft else dt
    return ttft, n, (n / dec if dec > 0 else 0.0)


def main():
    run("Hello", 8)  # warm any first-use path
    print(f"== prefill/decode bench: {URL} ==")
    print(f"{'prompt~tok':>10} {'TTFT(s)':>9} {'prefill t/s':>12} {'decode t/s':>11}")
    for tgt in TARGETS:
        prompt = ("Read this passage and then answer:\n"
                  + SENT * ((tgt * 4) // len(SENT) + 1)
                  + "\n\nIn one sentence, summarize the passage. Answer: ")
        pt = approx_tokens(prompt)
        try:
            ttft, _, dec = run(prompt)
            if ttft is None:
                print(f"{pt:>10}   no tokens streamed (rejected / context? see server logs)")
                continue
            pre = pt / ttft if ttft else 0.0
            print(f"{pt:>10} {ttft:>9.2f} {pre:>12.1f} {dec:>11.2f}")
        except Exception as e:
            print(f"{pt:>10}  ERROR: {e}")


if __name__ == "__main__":
    main()
