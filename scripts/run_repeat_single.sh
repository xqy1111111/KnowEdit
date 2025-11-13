#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_repeat_single.sh [START_IDX] [COUNT]
#
# Single-GPU edit + single-GPU infer (no DeepSpeed, no torchrun).
# You can choose the GPU via GPUS env (first id taken).
#
# Env overrides (examples provided below):
#   GPUS       (default: 0)
#   ALG        (default: FT)
#   HPARAMS    (default: hparams/FT/llama-r1d.yaml)
#   DATA       (default: data/cot.jsonl)
#   OUT_JSONL  (default: outputs/cot_llama_single.jsonl)
#   SAVE_POOL  (default: outputs/edited_model_pool)
#   GEN_MODE   (default: noprompt)
#   MAX_NEW_TOKENS (default: 512)
#   TEMPERATURE    (default: 0)
#   TOP_P          (default: 1)
#   SKIP_SECS      (default: 1)

START_IDX=${1:-0}
COUNT=${2:-15}

GPUS=${GPUS:-"0"}
ALG=${ALG:-"FT"}
HPARAMS=${HPARAMS:-"hparams/FT/llama-r1d.yaml"}
DATA=${DATA:-"data/cot.jsonl"}
OUT_JSONL=${OUT_JSONL:-"outputs/cot_llama_single.jsonl"}
SAVE_POOL=${SAVE_POOL:-"outputs/edited_model_pool"}

GEN_MODE=${GEN_MODE:-noprompt}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
SKIP_SECS=${SKIP_SECS:-1}

export TOKENIZERS_PARALLELISM=false

# Ensure Hugging Face caches default to a writable repo-local location if not set
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

mkdir -p "$SAVE_POOL" || true

EDIT_GPU="${GPUS%%,*}"

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  ALG_LOWER=$(echo "$ALG" | tr '[:upper:]' '[:lower:]')
  CASE_DIR="$SAVE_POOL/${ALG_LOWER}-${IDX}"

  echo "[EDIT-single] case_index=$IDX using GPU $EDIT_GPU"
  CUDA_VISIBLE_DEVICES="$EDIT_GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --save_edited_to "$SAVE_POOL" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage edit

  echo "[INFER-single] case_index=$IDX using GPU $EDIT_GPU"
  CUDA_VISIBLE_DEVICES="$EDIT_GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage infer \
    --load_edited_from "$CASE_DIR" \
    --delete_saved_after

  sleep "$SKIP_SECS"
done

# Example:
#   ALG=FT HPARAMS=hparams/FT/llama-r1d.yaml GPUS=0 \
#   DATA=data/cot.jsonl OUT_JSONL=outputs/cot_llama_single.jsonl \
#   bash scripts/run_repeat_single.sh 0 1

