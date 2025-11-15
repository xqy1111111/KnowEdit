#!/usr/bin/env bash
set -euo pipefail

# 单卡 ROME（Qwen2.5-Math-7B-SimpleRL），使用 CUDA 6，就地编辑后直接生成（不保存/不重载）。
# 用法：
#   ./scripts/run_rome_qwen25math_cuda6_nosave.sh [START_IDX] [COUNT]
# 可覆盖的环境变量：
#   GPU=6
#   HPARAMS=hparams/ROME/qwen2.5-math-7b-simplerl.yaml
#   DATA=data/noncot.json
#   OUT_JSONL=output/rome_qwen25math_cuda6.jsonl
#   GEN_MODE=r1d
#   MAX_NEW_TOKENS=1024

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-6}
ALG="ROME"
HPARAMS=${HPARAMS:-"hparams/ROME/qwen2.5-math-7b-simplerl.yaml"}
DATA=${DATA:-"data/noncot.json"}
OUT_JSONL=${OUT_JSONL:-"output/rome_qwen25math_cuda6.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
PRINT_EVERY=${PRINT_EVERY:-1}

# HF 缓存：默认指向 /data1，可按需覆写
export HF_HOME=${HF_HOME:-/data1/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

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
