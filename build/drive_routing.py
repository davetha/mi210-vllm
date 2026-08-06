#!/usr/bin/env python3
"""Drive a diverse workload through a vLLM server to capture MoE routing.

Two phases:
  FREQ   : long, varied prompts, max_tokens=1  -> prefill routing (frequency)
  DECODE : short prompts, max_tokens=N         -> ordered decode stream (LRU)

Diversity matters: GLM is multilingual, so Chinese/translated text is included
alongside code, math, prose, and instruction-following -- routing can be
domain- and language-specific, and the cache hit-rate must hold across them.
"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.252:8021/v1/completions"
MODEL = "glm-5.2"

# ---- FREQ phase: long, varied prompts (prefill is where most routing samples
#      come from; max_tokens=1 means we pay only the prefill). ----
_BASE = [
    # English prose / knowledge
    ("Explain the history of the Byzantine Empire from the founding of "
     "Constantinople to its fall in 1453. Cover Justinian's reconquests, the "
     "iconoclasm controversy, the Great Schism of 1054, the Fourth Crusade, "
     "and the final Ottoman siege. Be detailed and structured. ") ,
    # Code: Python
    ("Write a Python implementation of an LRU cache using an ordered dictionary. "
     "Include get, put, eviction, and a thread-safe variant. Add type hints, "
     "docstrings, unit tests with assert statements, and a usage example. "
     "Explain the time complexity of each operation. ") ,
    # Code: systems / C
    ("Write a C program that implements a simple HTTP/1.1 server using epoll. "
     "It should serve static files from a directory, handle multiple concurrent "
     "connections, parse the request line and headers, and return appropriate "
     "status codes. Add comments explaining the event loop. ") ,
    # Math
    ("Prove that the square root of 2 is irrational using a proof by "
     "contradiction. Then derive the quadratic formula by completing the "
     "square. Finally, explain how to compute the determinant of an n x n "
     "matrix using cofactor expansion and LU decomposition. ") ,
    # Reasoning / logic
    ("Five pirates must divide 100 gold coins. The most senior pirate proposes "
     "a split, and all pirates vote. If at least half accept, it passes; "
     "otherwise the proposer is thrown overboard and the next senior pirate "
     "proposes. Pirates are perfectly rational, prefer survival, then gold, "
     "then throwing others overboard. What is the outcome? Reason step by step. ") ,
    # Instruction following / structured
    ("Create a JSON schema for a blog platform. Include users, posts, comments, "
     "tags, and categories. Each post has a title, body, author, timestamps, "
     "and a list of tags. Comments are nested one level deep. Provide example "
     "data and a few SQL queries to find the most-commented posts. ") ,
    # Technical / ML
    ("Explain the transformer architecture in detail: multi-head self-attention, "
     "positional encoding, layer normalization, and the feed-forward network. "
     "Derive the attention formula, discuss scaling, and compare encoder-only, "
     "decoder-only, and encoder-decoder variants. Then explain mixture-of-experts. ") ,
    # Creative writing
    ("Write a short story about a lighthouse keeper who discovers a message in "
     "a bottle that appears to be from the future. The story should have a clear "
     "narrative arc, vivid imagery, and an ambiguous ending. Make it atmospheric. ") ,
]
ZH = [
    # Chinese: history, science, code-comment style, literature
    "请详细介绍中国历史上的四大发明，包括造纸术、印刷术、火药和指南针。"
    "说明每项发明的起源、发展过程、对社会的影响，以及传播到西方的路径。"
    "请分点论述，结构清晰。 ",
    "用 Python 实现一个二叉搜索树，包含插入、删除、查找、中序遍历和层序遍历。"
    "给出完整的类型注解、注释和测试用例，并分析每种操作的时间复杂度。 ",
    "解释量子力学中的薛定谔方程，包括它的物理意义、推导过程，以及如何用它"
    "求解一维无限深势阱中的粒子。讨论波函数的统计诠释和不确定性原理。 ",
    "写一篇关于人工智能对未来教育影响的文章。讨论个性化学习、教师角色的转变、"
    "教育公平、隐私问题，以及可能的解决方案。观点要平衡，论证要充分。 ",
]
MULTI = [
    "Traduzca al español: The mixture-of-experts architecture routes each token "
    "to a small number of expert networks, reducing compute while keeping model "
    "capacity high. Explique las ventajas y desventajas en detalle. ",
    "Übersetzen Sie ins Deutsche und erklären Sie: Gradient descent is an "
    "optimization algorithm that iteratively moves toward the minimum of a loss "
    "function by following the negative gradient. ",
]
FREQ = []
for p in _BASE:
    FREQ.append(p * 2)          # ~double length for more prefill tokens
for p in ZH:
    FREQ.append(p * 3)
for p in MULTI:
    FREQ.append(p * 4)

# ---- DECODE phase: short prompts, generate a chunk -> ordered access stream ----
DECODE = [
    "Write a Python function that sorts a list of dictionaries by a given key.",
    "Explain how TCP establishes a connection using the three-way handshake.",
    "写一段关于秋天的散文，描述落叶、秋风和夕阳。",
    "What are the trade-offs between BFS and DFS for graph traversal?",
    "Implement a function to detect cycles in a linked list.",
]
DECODE_TOKENS = 96


def post(prompt, max_tokens, label):
    body = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ntok = out["usage"]["completion_tokens"]
    print(f"  [{label}] {ntok:>4} toks in {dt:5.1f}s "
          f"({ntok/dt:4.1f} tok/s) :: {prompt[:48]!r}", flush=True)
    return out


def main():
    print(f"== FREQ phase: {len(FREQ)} long prompts, max_tokens=1 ==", flush=True)
    tot = 0
    for i, p in enumerate(FREQ):
        post(p, 1, f"freq{i}")
        tot += 1
    print(f"== FREQ done ({tot} prompts) ==\n", flush=True)

    print(f"== DECODE phase: {len(DECODE)} prompts, max_tokens={DECODE_TOKENS} =="
          , flush=True)
    for i, p in enumerate(DECODE):
        post(p, DECODE_TOKENS, f"dec{i}")
    print("== DECODE done ==", flush=True)
    print("ALL DONE -- flush the probe dump now", flush=True)


if __name__ == "__main__":
    main()
