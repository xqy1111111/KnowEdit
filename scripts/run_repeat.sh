#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_repeat.sh [START_IDX] [COUNT]
# Env overrides:
#   GPUS (default: 0,1,2,3,4,5,6,7)
#   ALG (default: FT)
#   HPARAMS (default: hparams/FT/deepseek-r1d-qwen-7b-cot.yaml)
#   DATA (default: data/cot.jsonl)
#   OUT_JSONL (default: outputs/ft_eval_ds_infer.jsonl)
#   SAVE_POOL (default: outputs/edited_model_pool)
#   DS_MP_SIZE (default: 8)
#   DS_DTYPE (default: auto)
#   GEN_MODE (default: noprompt)
#   MAX_NEW_TOKENS (default: 256)
#   TEMPERATURE (default: 0)
#   TOP_P (default: 1)
#   SKIP_SECS (default: 1)

START_IDX=${1:-0}
COUNT=${2:-15}

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
ALG=${ALG:-"FT"}
HPARAMS=${HPARAMS:-"hparams/FT/deepseek-r1d-qwen-7b-cot.yaml"}
DATA=${DATA:-"data/cot.jsonl"}
OUT_JSONL=${OUT_JSONL:-"outputs/ft_eval_ds_infer.jsonl"}
SAVE_POOL=${SAVE_POOL:-"outputs/edited_model_pool"}

DS_MP_SIZE=${DS_MP_SIZE:-8}
DS_DTYPE=${DS_DTYPE:-auto}

GEN_MODE=${GEN_MODE:-noprompt}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
SKIP_SECS=${SKIP_SECS:-1}

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$((29501 + (IDX % 256)))

  echo "[RUN] case_index=$IDX  MASTER_PORT=$MASTER_PORT"
  torchrun --nproc_per_node=${DS_MP_SIZE} --module scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --use_ds_infer --ds_mp_size "$DS_MP_SIZE" --ds_dtype "$DS_DTYPE" \
    --save_edited_to "$SAVE_POOL" \
    --delete_saved_after \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    --print_every 1

  sleep "$SKIP_SECS"
done

