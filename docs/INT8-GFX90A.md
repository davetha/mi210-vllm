# INT8 W8A8 on gfx90a

CDNA2 has native INT8 MFMA and no FP8 datapath, so INT8 W8A8 is the quantization that
actually pays here. Two things decide whether it does: whether vLLM selects AITER's CK
GEMM, and how much of the model is really quantized.

## First: confirm the AITER kernel is selected

```bash
docker logs <container> 2>&1 | grep 'Selected.*ScaledMMLinearKernel'
# want: Selected AiterInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

`TritonInt8ScaledMMLinearKernel` means the generic fallback, at roughly a third of the
decode rate. Two causes, both silent:

- the image has no AITER — run `build/add-aiter.sh`;
- the host has GPUs of more than one architecture and vLLM read the arch off the wrong
  card — see [RUNNING.md](RUNNING.md#which-gpus-the-container-sees). Worth **2.4x**.

Do **not** try to fix this by tuning the AITER GEMM. Tuned configs measured *slower* at
decode here (41.25 vs 42.31 tok/s, n=9); `not found tuned config in a8w8_tuned_gemm.csv,
will use default config!` is correct behaviour on this part, not a warning.

## Then: quantize what the stock recipe leaves in BF16

A typical `llm-compressor` W8A8 pass on a hybrid Qwen3.8-27B quantizes the MLPs and the
full-attention projections and protects everything else. On a bandwidth-bound decode the
protected tensors dominate what is left. Measured on one MI210, TP=1, MTP N=1:

| | decode | ms/step | on disk |
|---|---|---|---|
| stock W8A8 | 40.5 tok/s | 43.68 | 35 GB |
| + GDN projections INT8 | 50.0 | 35.68 | 30 GB |
| + `lm_head` INT8 | **57.2** | **32.10** | **28 GB** |

Both steps are **data-free** — symmetric per-output-channel weight quantization, no
calibration set. Match the checkpoint's own convention exactly:

```python
scale = (w.float().abs().amax(dim=1, keepdim=True) / 127.0).to(torch.bfloat16)  # bf16 FIRST
q     = torch.round(w.float() / scale.float()).clamp_(-128, 127).to(torch.int8)
```

Casting the scale to BF16 *before* dividing is load-bearing: it is what reproduces the
parent's statistics (row maxima usually 127 but often 120-126, and `-128` present).

### Getting vLLM to honour a quantized `lm_head`

`lm_head` is a `ParallelLMHead`, not a `LinearBase`. compressed-tensors *does* support it —
`get_quant_method` returns a `CompressedTensorsLinearMethod` whenever `get_scheme` matches,
and the `embedding()` requirement does not apply — but:

- `targets: ["Linear"]` never matches, because the class is not `Linear`;
- adding the literal `"lm_head"` is **still not enough** on a `ForConditionalGeneration`
  wrapper, where the prefix is nested and only an exact layer-name match resolves.

Use a regex, and drop `lm_head` from `ignore`:

```json
"targets": ["Linear", "re:.*lm_head$"]
```

No vLLM change is needed. When debugging this, pass a real `ParallelLMHead` to
`find_matched_target` — testing with a `torch.nn.Linear` makes nested names appear to match
via the `Linear` *class-name* target and sends you the wrong way.

### What to leave alone

| tensor | why |
|---|---|
| `linear_attn.in_proj_a`, `in_proj_b` | 48x5120 each, **0.42%** of the GDN block's bytes, but they drive the recurrent gating (`a`→softplus→exp→decay, `b`→sigmoid→beta) where error compounds along the sequence |
| `mtp.*` / the dflash draft | quantizing a draft costs acceptance rate — see [SPEC-DECODE.md](SPEC-DECODE.md) |
| `embed_tokens` | a row lookup, not a scan; quantizing it saves no bandwidth |
| norms, `conv1d` | small and sensitive |

INT4 `lm_head` is **not** worth it: it saves a further 0.63 GB (~1% of a step) while max
relative weight error goes from 0.39% to ~6%, on the layer that directly produces logits
over a 248k vocab — and it would leave the CK INT8 GEMM for the W4A16 Triton path.

## Quality

The GDN step costs ~2.3% perplexity; `lm_head` costs ~0.01% (per-output-channel scales, and
it sits outside the recurrent state). Long-context recall was unchanged: 9/9 needle at
107,337 tokens, matching a BF16-GDN control run on the same hardware.

That control matters — this box declines the fast paged-attention kernel (the hybrid model
forces `block_size` ~800, over the gfx90a `block_size>64` veto), so a long-context failure
could easily be the attention path rather than the weights. Run both arms.

A published checkpoint with both steps applied:
[davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8](https://huggingface.co/davetha/Qwen3.8-27B-ABLITERATED-W8A8-gdnint8).
