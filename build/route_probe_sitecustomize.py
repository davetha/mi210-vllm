# SPDX-License-Identifier: Apache-2.0
"""Routing-distribution probe for vLLM MoE (CDNA2 / GLM-5.2).

Installed by bind-mounting this file as sitecustomize.py on PYTHONPATH. It
patches BaseRouter._select_experts -- the single template-method choke point
every router subclass flows through -- to record, per layer:

  * a frequency count over expert ids (for the static top-K cache curve), and
  * the ordered access stream with per-step token counts (for LRU replay).

The frequency counts answer the real question: is routing skewed enough that a
small VRAM-resident expert set has a high hit rate?

It is deliberately robust to a hard kill: counts are flushed to
/route/probe_<pid>.pkl every FLUSH_EVERY recorded calls, so the host-visible
bind mount always holds near-complete data. atexit is a best-effort bonus.

Zero correctness impact on serving: it only reads topk_ids after they are
computed, never mutates them, and every recorder op is wrapped in try/except.
"""

import sys
import threading
import time

# ---------------------------------------------------------------- import hook
_TARGET = "vllm.model_executor.layers.fused_moe.router.base_router"


def _install(mod):
    BaseRouter = getattr(mod, "BaseRouter", None)
    if BaseRouter is None:
        return
    if getattr(BaseRouter, "_route_probe_patched", False):
        sys.stderr.write("route_probe: already patched\n")
        return

    import atexit
    import os
    import numpy as np

    try:
        import torch
    except Exception:  # no torch yet -> nothing to do
        return

    LOCK = threading.Lock()
    COUNTS = {}    # layer_idx -> np.int64[num_experts]
    STREAM = {}    # layer_idx -> list[(ntokens:int, np.ndarray int32)]
    LIDX = {}      # id(router) -> layer_idx
    CALLS = {"n": 0}

    FLUSH_EVERY = 50
    OUTDIR = "/route"

    def _record(self, topk_ids):
        try:
            tid = topk_ids.detach().to(torch.int32).cpu().numpy()
            ntokens = int(tid.shape[0]) if tid.ndim >= 1 else 1
            flat = tid.reshape(-1)
            with LOCK:
                key = id(self)
                li = LIDX.get(key)
                if li is None:
                    li = len(LIDX)
                    LIDX[key] = li
                ne = int(flat.max()) + 1 if flat.size else 0
                c = COUNTS.get(li)
                cap = max(ne, c.size if c is not None else 0, 256)
                if c is None or c.size < cap:
                    nc = np.zeros(cap, dtype=np.int64)
                    if c is not None:
                        nc[: c.size] = c
                    c = nc
                    COUNTS[li] = c
                if flat.size:
                    np.add.at(c, flat, 1)
                STREAM.setdefault(li, []).append((ntokens, flat.copy()))
                CALLS["n"] += 1
                if CALLS["n"] % FLUSH_EVERY == 0:
                    _flush_locked()
        except Exception as e:  # never break a forward
            sys.stderr.write(f"route_probe rec err: {e}\n")

    def _flush_locked():
        try:
            import pickle
            os.makedirs(OUTDIR, exist_ok=True)
            data = {}
            total = 0
            for li, c in COUNTS.items():
                st = STREAM.get(li, [])
                flat = (np.concatenate([s for _, s in st])
                        if st else np.empty(0, dtype=np.int32))
                # token-count per step, parallel to flat via cumcount
                steps = np.concatenate(
                    [np.full(s.shape[0], n, dtype=np.int32) for n, s in st]
                ) if st else np.empty(0, dtype=np.int32)
                data[li] = {"counts": c, "stream": flat, "step_ntok": steps}
                total += int(c.sum())
            with open(os.path.join(OUTDIR, f"probe_{os.getpid()}.pkl"), "wb") as f:
                pickle.dump({"layers": dict(LIDX), "data": data}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            sys.stderr.write(
                f"route_probe: flushed {total} routings, "
                f"{len(COUNTS)} layers -> {OUTDIR}/probe_{os.getpid()}.pkl\n")
        except Exception as e:
            sys.stderr.write(f"route_probe flush err: {e}\n")

    orig = BaseRouter._select_experts

    def wrapped(self, hidden_states, router_logits, topk_indices_dtype=None,
                *, input_ids=None):
        out = orig(self, hidden_states, router_logits, topk_indices_dtype,
                   input_ids=input_ids)
        # out == (topk_weights, topk_ids)
        _record(self, out[1])
        return out

    BaseRouter._select_experts = wrapped
    BaseRouter._route_probe_patched = True
    atexit.register(_flush_locked)
    sys.stderr.write("route_probe: installed on BaseRouter._select_experts\n")


def _worker():
    # The module is imported lazily during model build; poll until it lands,
    # well before the first forward (weights load for minutes).
    for _ in range(2400):  # ~20 min ceiling
        m = sys.modules.get(_TARGET)
        if m is not None:
            try:
                _install(m)
            except Exception as e:
                sys.stderr.write(f"route_probe install err: {e}\n")
            return
        time.sleep(0.5)
    sys.stderr.write("route_probe: target module never imported; no data\n")


threading.Thread(target=_worker, daemon=True).start()
sys.stderr.write("route_probe: sitecustomize loaded\n")
