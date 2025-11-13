#!/usr/bin/env bash
set -euo pipefail

# 单卡运行 MEND 编辑+生成（两阶段串行），默认生成长度 512。
# 用法：
#   ./scripts/run_edit_mend_single.sh [START_IDX] [COUNT]
# 变量：
#   GPU=0                      使用的 GPU
#   HPARAMS=hparams/MEND/deepseek-r1d-llama-8b.yaml
#   DATA=data/mend/zsre_mend_eval.json
#   SAVE_POOL=output/edited_model_pool   编辑后模型保存目录
#   OUT_JSONL=output/mend_eval.jsonl     结果追加写入
#   GEN_MODE=r1d               生成模式标签（透传给脚本以便记录）
#   MAX_NEW_TOKENS=512         生成最大新 tokens 数
#   TEMPERATURE=0              采样温度
#   TOP_P=1                    nucleus sampling p
#   SKIP_SECS=1                每个 case 间隔秒数

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-0}
ALG="MEND"
HPARAMS=${HPARAMS:-"hparams/MEND/deepseek-r1d-llama-8b.yaml"}
DATA=${DATA:-"data/mend/zsre_mend_eval.json"}
SAVE_POOL=${SAVE_POOL:-"output/edited_model_pool"}
OUT_JSONL=${OUT_JSONL:-"output/mend_eval.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
SKIP_SECS=${SKIP_SECS:-1}

export TOKENIZERS_PARALLELISM=false

# 本地缓存目录（若外界未设置则回落到仓库内）
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

mkdir -p "$SAVE_POOL" || true
mkdir -p "$(dirname "$OUT_JSONL")" || true

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  CASE_DIR="$SAVE_POOL/mend-${IDX}"

  echo "[EDIT] case_index=$IDX GPU=$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --save_edited_to "$SAVE_POOL" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --print_every 1 \
    --stage edit

  echo "[INFER] case_index=$IDX GPU=$GPU -> $OUT_JSONL"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    --print_every 1 \
    --stage infer \
    --load_edited_from "$CASE_DIR"

  sleep "$SKIP_SECS"
done

