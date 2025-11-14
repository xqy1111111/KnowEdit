#!/usr/bin/env bash
set -euo pipefail

# 单卡 RLEdit（DeepSeek-R1D-LLaMA-8B），接口与 run_ultraedit_* 统一：
# - 以 scripts/edit_once_with_judge.py 为驱动，输出 JSONL 结构一致；
# - 每次调用只编辑一个样本，不保存模型；
# - 临时文件与缓存优先放在 $LOCAL_SSD_ROOT (默认 ~/autodl-tmp)。
#
# 用法：
#   ./scripts/run_rledit_standalone.sh [START_IDX] [COUNT]
# 支持覆盖的环境变量：
#   GPU=0
#   HPARAMS=hparams/RLEdit/deepseek-r1d-llama-8b.yaml
#   DATA=znoncot.json
#   OUT_JSONL=output/rledit_llama8b.jsonl
#   GEN_MODE=r1d
#   MAX_NEW_TOKENS=1024
#   LOCAL_SSD_ROOT=/root/autodl-tmp

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-0}
HPARAMS=${HPARAMS:-"hparams/RLEdit/deepseek-r1d-llama-8b.yaml"}
DATA=${DATA:-"znoncot.json"}
OUT_JSONL=${OUT_JSONL:-"output/rledit_llama8b.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
PRINT_EVERY=${PRINT_EVERY:-1}

LOCAL_SSD_ROOT=${LOCAL_SSD_ROOT:-"$HOME/autodl-tmp"}
mkdir -p "$LOCAL_SSD_ROOT"
export LOCAL_SSD_ROOT

if [ -w "$LOCAL_SSD_ROOT" ]; then
  export HF_HOME=${HF_HOME:-"$LOCAL_SSD_ROOT/hf_cache"}
else
  HF_HOME_FALLBACK="$PWD/models/hf_cache"
  export HF_HOME=${HF_HOME:-$HF_HOME_FALLBACK}
fi
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"$HF_HOME/hub"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=${WANDB_DISABLED:-true}
mkdir -p "$(dirname "$OUT_JSONL")" || true

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  echo "[RLEDIT] case_index=$IDX GPU=$GPU -> $OUT_JSONL"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m scripts.edit_once_with_judge \
    --alg RLEDIT \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    --print_every "$PRINT_EVERY" \
    --stage full
  echo
  sleep 1
done
