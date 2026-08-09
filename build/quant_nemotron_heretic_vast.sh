#!/usr/bin/env bash
# Quantise Nemotron-3-Super-120B heretic to W4A16 on a rented GPU box.
#
# Run this ON THE VAST INSTANCE, not on the MI210 box. It needs ~300 GB VRAM:
# device_map="auto" holds the 247 GB BF16 model plus GPTQ Hessians (~30 GB) and
# activations (~20 GB). 4x80GB or 8x48GB clears it; 2xH200 (282 GB) does not.
#
#   bash quant_nemotron_heretic_vast.sh [OUTDIR]
#
# WHY THE IGNORE LIST MATTERS. Nemotron-H is a Mamba/attention hybrid: ~50 of
# its 88 layers are SSM. Quantising the SSM projections does not error -- it
# produces fluent, confident, WRONG output, which is the hardest failure mode to
# catch. The list below is not guesswork; it is read back out of NVIDIA's own
# INT4 release (241 ignore entries collapsing to 7 families), so this reproduces
# their recipe rather than inventing one. Do not "simplify" it.
set -euo pipefail

SRC=${SRC:-trohrbaugh/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic-v2}
OUT=${1:-/workspace/nemotron-heretic-w4a16}
SAMPLES=${SAMPLES:-512}
SEQLEN=${SEQLEN:-2048}

# Rented images do not agree on where Python lives. vast's own image ships torch
# in /venv/main and has no `python` on PATH at all, so a bare `python3` here
# fails with ModuleNotFoundError: No module named 'torch' -- which reads like a
# broken image rather than the wrong interpreter. Pick whichever one can
# actually import torch, and fail loudly if none can.
PY=""
for c in /venv/main/bin/python /opt/conda/bin/python python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    if "$c" -c "import torch" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "FATAL: no interpreter with torch found"; exit 1; }
PIP="$PY -m pip"
echo "  interpreter: $PY ($("$PY" -c 'import torch;print("torch",torch.__version__)'))"

echo "=== 0. preflight ==========================================="
"$PY" - <<'PY'
import torch, shutil
n = torch.cuda.device_count()
vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1024**3
free = shutil.disk_usage("/workspace").free / 1024**3
print(f"  GPUs: {n}   total VRAM: {vram:.0f} GB   free disk: {free:.0f} GB")
# 247 GB source + ~76 GB output, plus HF cache churn.
assert vram >= 290, f"need >=290 GB VRAM for the BF16 model + Hessians, have {vram:.0f}"
assert free >= 380, f"need >=380 GB free disk (247 src + 76 out + slack), have {free:.0f}"
print("  preflight OK")
PY

echo "=== 1. deps ================================================"
# TORCH IS PINNED TO 2.10.0 AND THAT PIN IS LOAD-BEARING. Three constraints
# intersect at exactly one version:
#
#   llmcompressor 0.12.0.1  requires torch >=2.10.0,<=2.12.0
#   mamba-ssm 2.3.2         prebuilt wheels: torch 2.6-2.10 only
#   causal-conv1d 1.6.2     prebuilt wheels: torch 2.6-2.10 only
#
# Nemotron-H's modeling code hard-requires mamba-ssm ("mamba-ssm is required by
# the Mamba model but cannot be imported"), so it is not optional. Letting pip
# resolve freely pulls torch 2.12, for which NO mamba wheel exists; it then
# tries to build from source and dies on a 404. Downgrading torch afterwards
# breaks llmcompressor instead. 2.10.0 is the only value that satisfies all three.
# ONE RESOLVE, NOT A SEQUENCE. llmcompressor declares a bare `torch`, so any
# separate `pip install torch==2.10.0` either gets overridden by a later install
# or overrides an earlier one -- sequencing them just moves which constraint
# loses. Two rounds of that ended with torch 2.13.0 (mamba wheels ABI-fail) and
# then a venv whose torch would not import at all
# (libtorch_cuda.so: undefined symbol: ncclCommResume). Give pip every
# constraint at once and let it solve, then assert the answer.
$PIP install -q "torch==2.10.0" llmcompressor datasets accelerate "huggingface_hub[hf_xet]"

# Install the CUDA kernels from the release wheels rather than PyPI: pip's sdist
# path compiles from source (slow, and fails without CUDA_HOME), while these are
# prebuilt for this exact torch/python/abi tuple. Only cxx11abiTRUE is published
# for torch2.10/cp310/x86_64, so there is no variant to choose.
MAMBA_TAG=v2.3.2.post1
CONV_TAG=v1.6.2.post1
ABI=cu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
$PIP install -q --no-build-isolation \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/${CONV_TAG}/causal_conv1d-${CONV_TAG#v}+${ABI}" \
  "https://github.com/state-spaces/mamba/releases/download/${MAMBA_TAG}/mamba_ssm-${MAMBA_TAG#v}+${ABI}"

# Nemotron-H imports these at model-load time, i.e. AFTER the 247 GB download.
# Failing here costs seconds; failing there costs the download and the rental.
"$PY" -c "import mamba_ssm, causal_conv1d; print('  mamba_ssm + causal_conv1d OK')"
"$PY" -c "import torch; assert torch.__version__.startswith('2.10'), torch.__version__; print('  torch', torch.__version__)"

# torchvision is the actual landmine. Rented images ship torch+torchvision as a
# matched pair, then the line above upgrades torch and orphans torchvision. The
# resulting failure is maximally misleading: transformers' lazy importer reports
#   ModuleNotFoundError: Could not import module 'PreTrainedModel'
# which looks like a transformers/llmcompressor version fight and sends you
# downgrading the wrong package. The real error, only visible by importing
# transformers.modeling_utils directly, is
#   RuntimeError: operator torchvision::nms does not exist
# Nothing in this script needs torchvision, so remove it rather than chase a
# matching build.
"$PY" - <<'PY' || $PIP uninstall -y -q torchvision torchaudio
import transformers.modeling_utils  # noqa: F401
PY
"$PY" -c "import transformers.modeling_utils" 2>/dev/null || {
    echo "  transformers still broken after dropping torchvision"; exit 1; }

"$PY" -c "import llmcompressor, transformers; print('  llmcompressor', llmcompressor.__version__, '| transformers', transformers.__version__)"
"$PY" -c "from llmcompressor import oneshot; from llmcompressor.modifiers.quantization import GPTQModifier; print('  GPTQ imports OK')"

echo "=== 2. fetch source (247 GB) ==============================="
# hf_transfer is NOT optional at this size. Measured against this exact repo:
# a single HTTPS stream from HF sustains ~3.5 MB/s (28 Mbps), which is 19 HOURS
# for 247 GB. Eight parallel ranged streams hit 112 MB/s -- a 32x difference, so
# HF throttles per-connection, not per-client. Without this env var
# huggingface_hub may quietly use the slow path and the rental burns out
# mid-download.
# HF_HUB_ENABLE_HF_TRANSFER is DEPRECATED and now silently ignored -- setting it
# gets you the single-stream path (measured: 6 MB/s, i.e. 11+ hours for 247 GB,
# which on a $3.32/hr box costs more than the whole job budget). Xet is the
# current mechanism: 155 MB/s measured on the same host, a 26x difference.
export HF_XET_HIGH_PERFORMANCE=1
"$PY" - "$SRC" <<'PY'
import os, sys, time
assert os.environ.get("HF_XET_HIGH_PERFORMANCE") == "1", "Xet fast path not enabled"
import hf_xet  # noqa: F401  -- fail here, not 11 hours in
from huggingface_hub import snapshot_download
t0 = time.time()
p = snapshot_download(sys.argv[1], local_dir="/workspace/src", max_workers=32)
dt = time.time() - t0
gb = sum(os.path.getsize(os.path.join(r, f))
         for r, _, fs in os.walk(p) for f in fs) / 1024**3
print(f"  downloaded {gb:.0f} GB -> {p} in {dt/60:.1f} min ({gb*1024/dt:.0f} MB/s)")
# A rented box bills by the hour; a silent fallback to the slow path is a budget
# failure, not a performance note. Fail loudly instead.
assert gb / (dt / 3600) > 100, (
    f"only {gb/(dt/3600):.0f} GB/h -- Xet did not engage, aborting before GPTQ")
PY

echo "=== 3. quantise ============================================"
cat > /workspace/run_quant.py <<PY
import json, os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

SOURCE = "/workspace/src"
OUTPUT = "${OUT}"

# Derived from NVIDIA's own AWQ-INT4 release: lm_head plus every Mamba mixer
# projection and the shared-expert MLPs stay in 16-bit. Quantising these is the
# difference between a working model and a fluent liar.
IGNORE = [
    "lm_head",
    r"re:.*mixer\.fc1_latent_proj\$",
    r"re:.*mixer\.fc2_latent_proj\$",
    r"re:.*mixer\.in_proj\$",
    r"re:.*mixer\.out_proj\$",
    r"re:.*mixer\.shared_experts\.down_proj\$",
    r"re:.*mixer\.shared_experts\.up_proj\$",
]

tok = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    SOURCE, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds = ds.shuffle(seed=42).select(range(${SAMPLES}))

def prep(ex):
    msgs = ex.get("messages") or ex.get("conversation")
    text = tok.apply_chat_template(msgs, tokenize=False) if msgs else ex.get("text", "")
    return tok(text, truncation=True, max_length=${SEQLEN}, padding=False)

ds = ds.map(prep, remove_columns=ds.column_names)

oneshot(
    model=model,
    dataset=ds,
    # W4A16 symmetric group-32 == what the MI210 box already serves, and the
    # only 4-bit path that reaches the int4 interleave kernel on gfx90a.
    recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE),
    output_dir=OUTPUT,
    max_seq_length=${SEQLEN},
    num_calibration_samples=${SAMPLES},
)
tok.save_pretrained(OUTPUT)

# Assert the recipe actually landed: symmetric, 4-bit, and the mixer layers
# excluded. A quant that silently came out asymmetric will not load in vLLM's
# WNA16 MoE kernel (it asserts symmetric), and one that quantised the SSM
# projections will load fine and be wrong.
cfg = json.load(open(os.path.join(OUTPUT, "config.json")))["quantization_config"]
grp = next(iter(cfg["config_groups"].values()))["weights"]
assert grp["num_bits"] == 4, grp
assert grp["symmetric"] is True, "ASYMMETRIC -- vLLM WNA16 MoE will refuse this"
ign = cfg.get("ignore", [])
assert any("mixer" in i for i in ign), "mixer layers were NOT ignored -- model will be subtly broken"
print(f"  quant OK: {grp['num_bits']}-bit sym group={grp.get('group_size')}, {len(ign)} ignored")
PY
"$PY" /workspace/run_quant.py

echo "=== 4. correctness gate ===================================="
# Speed proves nothing. This repo has already shipped one benchmark of a kernel
# that was emitting garbage at 3,820 t/s. Read the tokens.
"$PY" - "$OUT" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained(out)
m = AutoModelForCausalLM.from_pretrained(out, torch_dtype="auto", device_map="auto",
                                         trust_remote_code=True)
ids = tok("def reverse_linked_list(head):", return_tensors="pt").to(m.device)
txt = tok.decode(m.generate(**ids, max_new_tokens=80, do_sample=False)[0])
print(txt)
low = txt.lower()
assert "prev" in low and "next" in low, "output does not look like a reversal"
# crude degeneration check: no 12-word span repeated three times
w = low.split()
assert not any(w[i:i+12] == w[i+12:i+24] == w[i+24:i+36] for i in range(max(0, len(w)-36))), \
    "REPETITION LOOP -- quant is degenerate, do not ship"
print("  correctness gate PASSED")
PY

echo "=== 5. publish ============================================="
# HF, not rsync. Measured: this link is asymmetric (901 Mbps down on 8 parallel
# streams, 60 Mbps up) and HF throttles PER CONNECTION -- a single stream gets
# ~28 Mbps. rsync is single-threaded, so a 76 GB pull could run for hours with
# the instance still billing. Pushing to HF lets the box be destroyed as soon as
# the upload verifies, and the box then pulls at ~112 MB/s (~11 min) for free.
# It also means a failed pull is a free retry instead of a re-rental.
REPO=${REPO:-davetha/Nemotron-3-Super-120B-heretic-W4A16}
if [ -n "${HF_TOKEN:-}" ]; then
  "$PY" - "$OUT" "$REPO" <<'PY'
import sys, os
from huggingface_hub import HfApi
out, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=repo, repo_type="model")
# Verify the push landed before anyone destroys the instance -- an upload that
# half-finished looks identical to one that worked until you try to load it.
files = {f.rfilename for f in api.model_info(repo, files_metadata=False).siblings}
need = {f for f in os.listdir(out) if f.endswith((".safetensors", ".json"))}
missing = need - files
assert not missing, f"upload INCOMPLETE, missing: {sorted(missing)[:5]}"
print(f"  pushed {len(need)} files -> https://huggingface.co/{repo} (private)")
PY
else
  echo "  HF_TOKEN unset -- skipping upload. Set it, or rsync manually:"
  echo "    rsync -av --progress $OUT/ dave@23.121.56.115:/mnt/llm-storage/nemotron-heretic-w4a16/"
fi

echo
echo "=== DONE ==================================================="
du -sh "$OUT"
echo
echo "On the MI210 box, pull it back (parallel, ~11 min):"
echo "  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $REPO \\"
echo "    --local-dir /mnt/llm-storage/nemotron-heretic-w4a16"
echo
echo "!!! DESTROY THE INSTANCE ONCE THE UPLOAD VERIFIED ABOVE !!!"
