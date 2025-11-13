#!/usr/bin/env bash
set -euo pipefail

# Run end-to-end (stage=full) so pred_before/final_before are recorded.
# Usage: ./scripts/run_repeat_full.sh [START_IDX] [COUNT]
# Env overrides:
#   GPUS          visible GPUs for torchrun (default: 0)
#   ALG           editing algorithm (default: FT)
#   HPARAMS       path to YAML (default: hparams/FT/deepseek-r1d-qwen-7b-cot.yaml)
#   DATA          zsre-like JSON/JSONL (default: data/cot.jsonl)
#   OUT_JSONL     results path (default: outputs/ft_eval_full.jsonl)
#   SAVE_POOL     where edited checkpoints are saved (required for TP) (default: outputs/edited_model_pool)
#   DS_MP_SIZE    tensor parallel degree for deepspeed (default: 1)
#   DS_DTYPE      auto|bf16|fp16 for DS inference (default: auto)
#   GEN_MODE      concise|reason|noprompt|r1d (default: r1d)
#   MAX_NEW_TOKENS  (default: 512)
#   TEMPERATURE     (default: 0)
#   TOP_P           (default: 1)
#   SKIP_SECS       sleep between cases (default: 1)
#   OFFLINE       if set, pass --offline to avoid network
#   JUDGE_MODEL   if set, use this judge; else --no_judge

START_IDX=${1:-0}
COUNT=${2:-15}

GPUS=${GPUS:-"0"}
ALG=${ALG:-"FT"}
HPARAMS=${HPARAMS:-"hparams/FT/deepseek-r1d-qwen-7b-cot.yaml"}
DATA=${DATA:-"data/cot.jsonl"}
OUT_JSONL=${OUT_JSONL:-"outputs/ft_eval_full.jsonl"}
SAVE_POOL=${SAVE_POOL:-"outputs/edited_model_pool"}

DS_MP_SIZE=${DS_MP_SIZE:-1}
DS_DTYPE=${DS_DTYPE:-auto}

GEN_MODE=${GEN_MODE:-r1d}
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

mkdir -p "$(dirname "$OUT_JSONL")" "$SAVE_POOL" || true

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))

  export MASTER_ADDR=127.0.0.1
  BASE_PORT=${BASE_PORT:-29501}
  export MASTER_PORT=$((BASE_PORT + (IDX % 256)))

  echo "========================================"
  echo "[FULL] case_index=$IDX using GPUs $GPUS port=$MASTER_PORT"
  echo "========================================"

  # Build judge args: default to --no_judge unless JUDGE_MODEL is provided
  JUDGE_ARGS=(--no_judge)
  if [[ -n "${JUDGE_MODEL:-}" ]]; then
    JUDGE_ARGS=(--judge_model "$JUDGE_MODEL")
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun --nproc_per_node="${DS_MP_SIZE}" \
    --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    "${JUDGE_ARGS[@]}" \
    --use_ds_infer --ds_mp_size "$DS_MP_SIZE" --ds_dtype "$DS_DTYPE" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    --save_edited_to "$SAVE_POOL" \
    --eval_before \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage full

  sleep "$SKIP_SECS"
done

