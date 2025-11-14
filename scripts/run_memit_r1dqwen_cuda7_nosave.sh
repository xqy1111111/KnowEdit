#!/usr/bin/env bash
set -euo pipefail

# 单卡 MEMIT（DeepSeek-R1D-Qwen-7B），使用 CUDA 7，就地编辑后直接生成（不保存/不重载）。
# 用法：
#   ./scripts/run_memit_r1dqwen_cuda7_nosave.sh [START_IDX] [COUNT]
# 可覆盖环境变量：
#   GPU=7
#   HPARAMS=hparams/MEMIT/deepseek-r1d-qwen-7b.yaml
#   DATA=data/noncot.json
#   OUT_JSONL=output/memit_r1dqwen_cuda7.jsonl
#   GEN_MODE=r1d
#   MAX_NEW_TOKENS=1024
#   USE_SYNTHETIC_STATS=0   # 设为 1 可避免下载 Wikipedia（用于无网/小磁盘场景）

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-7}
ALG="MEMIT"
HPARAMS=${HPARAMS:-"hparams/MEMIT/deepseek-r1d-qwen-7b.yaml"}
DATA=${DATA:-"data/noncot.json"}
OUT_JSONL=${OUT_JSONL:-"output/memit_r1dqwen.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
PRINT_EVERY=${PRINT_EVERY:-1}

# 将缓存放到 ~/autodl-tmp，减少系统盘占用
HF_HOME_DEFAULT="/data1/hf_cache"
export HF_HOME=${HF_HOME:-$HF_HOME_DEFAULT}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

# 可选：避免下载大型 Wikipedia，使用小型合成语料来估计 mom2
# export USE_SYNTHETIC_STATS=1

export TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "$OUT_JSONL")" || true

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  echo "[FULL] alg=$ALG case_index=$IDX CUDA=$GPU (no save/reload) -> $OUT_JSONL"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m scripts.edit_once_with_judge \
    --alg "$ALG" \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    --print_every "$PRINT_EVERY" \
    --stage full
done
