#!/usr/bin/env bash
set -euo pipefail

# 单卡 RLEdit（DeepSeek-R1D-LLaMA-8B），使用 CUDA 1，就地编辑后直接生成（不保存/不重载）。
# 用法：
#   ./scripts/run_rledit_llama_cuda1_nosave.sh [START_IDX] [COUNT]
# 可覆盖环境变量：
#   GPU=1
#   HPARAMS=hparams/RLEdit/deepseek-r1d-llama-8b.yaml
#   DATA=znoncot.json       # 也可改成 data/noncot.json
#   OUT_JSONL=output/rledit_llama8b_cuda1.jsonl
#   GEN_MODE=r1d
#   MAX_NEW_TOKENS=1024

START_IDX=${1:-0}
COUNT=${2:-1}

GPU=${GPU:-1}
ALG="RLEDIT"
HPARAMS=${HPARAMS:-"hparams/RLEdit/deepseek-r1d-llama-8b.yaml"}
DATA=${DATA:-"znoncot.json"}
OUT_JSONL=${OUT_JSONL:-"output/rledit_llama8b_cuda1.jsonl"}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
PRINT_EVERY=${PRINT_EVERY:-1}

# 轻量缓存：优先 /dev/shm；若不可写再退回到 ./models/hf_cache
if [ -w /dev/shm ]; then
  export HF_HOME=${HF_HOME:-/dev/shm/hf_cache}
else
  HF_HOME_DEFAULT="$PWD/models/hf_cache"
  export HF_HOME=${HF_HOME:-$HF_HOME_DEFAULT}
fi
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

# 说明：
# - 不保存任何已编辑模型权重；编辑在内存中完成，生成后进程结束即释放显存/内存。
# - 缓存默认走 /dev/shm，磁盘占用最小；如不可写退回到仓库本地 models/hf_cache。

