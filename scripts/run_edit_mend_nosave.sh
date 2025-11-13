#!/usr/bin/env bash
set -euo pipefail

# 单卡：MEND 就地编辑后直接生成（不落盘保存/重载），并在每条结束后释放显存
# 用法：
#   ./scripts/run_edit_mend_nosave.sh [START_IDX] [COUNT]
# 环境变量：
#   GPU=0
#   HPARAMS=hparams/MEND/deepseek-r1d-llama-8b.yaml
#   DATA=data/mend/zsre_mend_eval.json
#   OUT_JSONL=output/mend_eval.jsonl
#   GEN_MODE=r1d
#   MAX_NEW_TOKENS=512
#   TEMPERATURE=0
#   TOP_P=1
#   PRINT_EVERY=1

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-0}
ALG="MEND"
HPARAMS=${HPARAMS:-"hparams/MEND/deepseek-r1d-llama-8b.yaml"}
DATA=${DATA:-"data/mend/zsre_mend_eval.json"}
OUT_JSONL=${OUT_JSONL:-"output/mend_eval.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
PRINT_EVERY=${PRINT_EVERY:-1}

export TOKENIZERS_PARALLELISM=false

# 本地缓存（若外界未设置）
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true
mkdir -p "$(dirname "$OUT_JSONL")" || true

# 说明：
# - 使用 stage=full 在同一进程内完成编辑+生成；不提供 --save_edited_to。
# - 默认 reset_each=True，脚本会在每条结束时删除编辑器/模型并 empty_cache，确保显存及时回收。

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  echo "[FULL] case_index=$IDX GPU=$GPU (no save/reload)"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --print_every "$PRINT_EVERY" \
    --stage full \
    --save_jsonl "$OUT_JSONL"
done

