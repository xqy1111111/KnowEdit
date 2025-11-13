#!/usr/bin/env bash
set -euo pipefail

# Simple IKE (In-Context Knowledge Editing) Runner
# IKE不需要训练，不需要统计计算，只通过上下文学习进行知识编辑
#
# Usage: ./scripts/run_ike_simple.sh [START_IDX] [COUNT]
#
# 环境变量:
#   MODEL         - 模型选择: qwen (默认) | llama
#   GPUS          - 使用的GPU (默认: 0)
#   DATA          - 数据文件 (默认: data/zsre_mend_eval.json)
#   OUT_JSONL     - 输出文件 (默认: outputs/ike_eval_{model}.jsonl)
#   JUDGE_MODEL   - 评判模型 (默认: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   GEN_MODE      - 生成模式 (默认: r1d)
#   MAX_NEW_TOKENS - 最大生成token数 (默认: 256)

START_IDX=${1:-0}
COUNT=${2:-10}

MODEL=${MODEL:-qwen}
GPUS=${GPUS:-"0"}
DATA=${DATA:-"data/noncot.jsonl"}
TRAIN_DATA=${TRAIN_DATA:-"data/ike_train_examples.json"}  # IKE示例库

# 根据模型选择配置文件
if [[ "$MODEL" == "qwen" ]]; then
    HPARAMS="hparams/IKE/deepseek-r1d-qwen-7b.yaml"
    OUT_JSONL=${OUT_JSONL:-"outputs/ike_eval_qwen.jsonl"}
elif [[ "$MODEL" == "llama" ]]; then
    HPARAMS="hparams/IKE/deepseek-r1d-llama-8b.yaml"
    OUT_JSONL=${OUT_JSONL:-"outputs/ike_eval_llama.jsonl"}
else
    echo "错误: MODEL 必须是 'qwen' 或 'llama'"
    exit 1
fi

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}

export TOKENIZERS_PARALLELISM=false

# 设置 Hugging Face 缓存
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

echo "========================================"
echo "IKE (In-Context Knowledge Editing)"
echo "========================================"
echo "模型:          $MODEL"
echo "配置文件:      $HPARAMS"
echo "数据:          $DATA"
echo "示例库:        $TRAIN_DATA"
echo "输出:          $OUT_JSONL"
echo "起始索引:      $START_IDX"
echo "数量:          $COUNT"
echo "GPU:           $GPUS"
echo "========================================"

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))

  echo ""
  echo "========================================"
  echo "[IKE] 处理样本 $IDX"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="$GPUS" \
  python -m scripts.edit_once_with_judge \
    --alg IKE \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --train_data_path "$TRAIN_DATA" \
    --case_index "$IDX" --repeat 1 \
    --no_judge \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    ${OFFLINE:+--offline} \
    --print_every 1

  echo "[完成] 样本 $IDX 处理完毕"
done

echo ""
echo "========================================"
echo "全部 $COUNT 个样本处理完成！"
echo "结果保存至: $OUT_JSONL"
echo "========================================"
