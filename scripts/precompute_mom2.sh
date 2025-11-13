#!/usr/bin/env bash
set -euo pipefail

# Precompute and cache mom2 (second moment) for Llama/Qwen-style models.
# Defaults target DeepSeek-R1-Distill-Qwen-7B and layer template:
#   layer_name = model.layers.{L}.mlp.down_proj
#
# Usage examples:
#   # Precompute for layer 26 with 100k samples
#   LAYERS=26 SAMPLE_SIZE=100000 bash scripts/precompute_mom2.sh
#
#   # Precompute for multiple layers
#   LAYERS=26,27,28 SAMPLE_SIZE=100000 bash scripts/precompute_mom2.sh
#
#   # Clean old cached stats for selected layers before recompute
#   CLEAN=1 LAYERS=26,27 bash scripts/precompute_mom2.sh
#
# Tunables via env vars:
#   MODEL_ID   (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   LAYERS     (default: 26)            # comma-separated list
#   SAMPLE_SIZE(default: 100000)
#   PRECISION  (default: float32)
#   DATASET    (default: wikipedia)     # uses WIKI_DATASET/WIKI_CONFIG when DATASET=wikipedia
#   STATS_DIR  (default: ./data/stats)
#   DEVICE     (default: first CUDA index or 0)
#

MODEL_ID=${MODEL_ID:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
LAYERS=${LAYERS:-"26"}
SAMPLE_SIZE=${SAMPLE_SIZE:-100000}
PRECISION=${PRECISION:-float32}
DATASET=${DATASET:-wikipedia}
STATS_DIR=${STATS_DIR:-"./data/stats"}

# Pick a device index: first of CUDA_VISIBLE_DEVICES if present
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE="$DEVICE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DEVICE="${CUDA_VISIBLE_DEVICES%%,*}"
else
  DEVICE=0
fi

# Cache roots (match repo defaults; override if already set)
export HF_HOME="${HF_HOME:-$PWD/models/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

# Wikipedia dataset config (used by layer_stats when DATASET=wikipedia)
export WIKI_DATASET="${WIKI_DATASET:-wikipedia}"
export WIKI_CONFIG="${WIKI_CONFIG:-20231101.en}"

echo "[CONF] MODEL_ID=$MODEL_ID"
echo "[CONF] LAYERS=$LAYERS SAMPLE_SIZE=$SAMPLE_SIZE PRECISION=$PRECISION DATASET=$DATASET"
echo "[CONF] STATS_DIR=$STATS_DIR DEVICE=$DEVICE"

# Optional: clean old cached files for the specified layers
if [[ "${CLEAN:-}" != "" ]]; then
  IFS=',' read -r -a _layers <<< "$LAYERS"
  for L in "${_layers[@]}"; do
    echo "[CLEAN] removing cached stats for layer $L under $STATS_DIR"
    find "$STATS_DIR" -type f -name "model.layers.${L}.mlp.down_proj*_mom2_*.npz" -print -delete || true
  done
fi

# Run Python to compute mom2 using the correct layer_name template for Llama/Qwen
python - << PY
import os, sys
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor.models.rome.layer_stats import layer_stats

model_id = os.environ.get("MODEL_ID")
stats_dir = os.environ.get("STATS_DIR")
dataset   = os.environ.get("DATASET")
layers_str= os.environ.get("LAYERS","26")
layers    = [int(x) for x in layers_str.split(',') if x.strip()]
sample_sz = int(os.environ.get("SAMPLE_SIZE","100000"))
precision = os.environ.get("PRECISION","float32")
device    = int(os.environ.get("DEVICE","0"))

print(f"[LOAD] tokenizer: {model_id}")
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

dtype = torch.float16
if torch.cuda.is_available():
    try:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    except Exception:
        pass
else:
    dtype = torch.float32

print(f"[LOAD] model: {model_id} (dtype={dtype}, device={device})")
mdl = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map={"": device} if torch.cuda.is_available() else None,
)
if not hasattr(mdl, "device") and torch.cuda.is_available():
    mdl = mdl.to(f"cuda:{device}")

hp = SimpleNamespace(device=device)

for L in layers:
    layer_name = f"model.layers.{L}.mlp.down_proj"
    print(f"[MOM2] computing: {layer_name} dataset={dataset} sample_size={sample_sz} precision={precision}")
    _ = layer_stats(
        mdl, tok, layer_name,
        stats_dir,
        dataset,
        to_collect=["mom2"],
        sample_size=sample_sz,
        precision=precision,
        hparams=hp,
        force_recompute=True,
    )
    print(f"[MOM2] done: {layer_name}")

print("[DONE] mom2 precompute finished.")
PY

echo "[VERIFY] listing generated mom2 files:"
find "$STATS_DIR" -type f -name 'model.layers.*.mlp.down_proj*_mom2_*.npz' -maxdepth 6 -print || true

