#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_train_mend.sh [qwen7b|llama8b]
# Default: llama8b

TARGET=${1:-llama8b}

if [[ "$TARGET" == "qwen7b" ]]; then
  HPARAMS=${HPARAMS:-hparams/TRAINING/MEND/deepseek-r1d-qwen-7b.yaml}
else
  HPARAMS=${HPARAMS:-hparams/TRAINING/MEND/deepseek-r1d-llama-8b.yaml}
fi

TRAIN_DATA=${TRAIN_DATA:-data/zsre/zsre_mend_train.json}
EVAL_DATA=${EVAL_DATA:-data/zsre/zsre_mend_eval.json}
ARCHIVE=${ARCHIVE:-}
GPU=${GPU:-0}

export TOKENIZERS_PARALLELISM=false

# Set local HF caches if not provided
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

CMD=(python -m scripts.train_mend --hparams "$HPARAMS" --train_data "$TRAIN_DATA" --eval_data "$EVAL_DATA")
if [[ -n "${ARCHIVE}" ]]; then
  CMD+=(--archive "$ARCHIVE")
fi

echo "Running: CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

