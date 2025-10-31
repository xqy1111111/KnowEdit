#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_repeat_memit_anchors.sh [START_IDX] [COUNT]
# Runs MEMIT with CoT-guided 6-anchor editing per case using anchors JSONL.
#
# Env overrides:
#   GPUS (default: 0,1)
#   HPARAMS (default: hparams/MEMIT/llama3.2-3b.yaml)
#   ANCHORS (default: data/memit_anchors_reph.jsonl)
#   OUT_JSONL (default: outputs/MEMIT/memit_eval_ds_infer.jsonl)
#   SAVE_POOL (default: outputs/edited_model_pool)
#   DS_MP_SIZE (default: 2)
#   DS_DTYPE (default: auto)
#   GEN_MODE (default: noprompt)
#   MAX_NEW_TOKENS (default: 256)
#   TEMPERATURE (default: 0)
#   TOP_P (default: 1)
#   SKIP_SECS (default: 1)
#   CASE_ID_OFFSET (default: 1)  # anchors case_id is 1-based by default

START_IDX=${1:-0}
COUNT=${2:-15}

GPUS=${GPUS:-"0,1,2,3"}
ANCHORS=${ANCHORS:-"data/memit_anchors_reph.jsonl"}
OUT_JSONL=${OUT_JSONL:-"outputs/MEMIT/memit_eval_ds_infer.jsonl"}
SAVE_POOL=${SAVE_POOL:-"outputs/edited_model_pool"}

DS_MP_SIZE=${DS_MP_SIZE:-4}
DS_DTYPE=${DS_DTYPE:-auto}

GEN_MODE=${GEN_MODE:-noprompt}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
SKIP_SECS=${SKIP_SECS:-1}
CASE_ID_OFFSET=${CASE_ID_OFFSET:-1}

export TOKENIZERS_PARALLELISM=false
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# Default HF caches under repo (writable)
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

mkdir -p "$SAVE_POOL" || true

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  CASE_ID=$((IDX + CASE_ID_OFFSET))
  ALG="MEMIT"
  ALG_LOWER="memit"
  CASE_DIR="$SAVE_POOL/${ALG_LOWER}-${IDX}"

  EDIT_GPU="${GPUS%%,*}"
  echo "[EDIT][MEMIT-anchors] case_index=$IDX case_id=$CASE_ID using GPU $EDIT_GPU"
  CUDA_VISIBLE_DEVICES="$EDIT_GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --anchors_jsonl "$ANCHORS" \
    --anchors_case_id "$CASE_ID" \
    --anchors_take 6 \
    --no_judge \
    --save_edited_to "$SAVE_POOL" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage edit

  export MASTER_ADDR=127.0.0.1
  BASE_PORT=${BASE_PORT:-29501}
  export MASTER_PORT=$((BASE_PORT + (IDX % 256)))

  echo "[INFER][MEMIT-anchors] case_index=$IDX case_id=$CASE_ID  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun --nproc_per_node="${DS_MP_SIZE}" \
    --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --anchors_jsonl "$ANCHORS" \
    --anchors_case_id "$CASE_ID" \
    --no_judge \
    --use_ds_infer --ds_mp_size "$DS_MP_SIZE" --ds_dtype "$DS_DTYPE" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage infer \
    --load_edited_from "$CASE_DIR" \
    --delete_saved_after

  sleep "$SKIP_SECS"
done

