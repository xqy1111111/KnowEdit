#!/usr/bin/env bash
set -euo pipefail

# 单卡训练 MEND（最简版）
# 使用方式：
#   ./scripts/run_train_mend_single.sh
# 可用环境变量：
#   HPARAMS   - 训练超参文件 (默认: hparams/TRAINING/MEND/deepseek-r1d-llama-8b.yaml)
#   TRAIN_DATA- 训练集 (默认: data/mend/zsre_mend_train.json)
#   EVAL_DATA - 验证集 (默认: data/mend/zsre_mend_eval.json)
#   ARCHIVE   - 断点/评估-only 的权重路径 (默认: 空)
#   GPU       - 使用的 GPU 编号 (默认: 0)

HPARAMS=${HPARAMS:-hparams/TRAINING/MEND/deepseek-r1d-llama-8b.yaml}
TRAIN_DATA=${TRAIN_DATA:-data/mend/zsre_mend_train.json}
EVAL_DATA=${EVAL_DATA:-data/mend/zsre_mend_eval.json}
ARCHIVE=${ARCHIVE:-}
GPU=${GPU:-0}

export TOKENIZERS_PARALLELISM=false

# 本地 Hugging Face 缓存（若外界未设置则使用仓库内目录）
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

echo "[TRAIN:MEND] CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"

