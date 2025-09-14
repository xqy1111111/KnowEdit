#!/usr/bin/env bash
set -e


MQ="../MQuAKE/datasets/MQuAKE-CF-3k-v2.json"
ZS="../data/zsre/zsre_mend_eval.json"
OUT="../runs"

JUDGE="Qwen/Qwen2.5-14B-Instruct"

# -------- Qwen2.5-7B-Instruct --------
# python eval_mquake_multihop.py \
#   --model "Qwen/Qwen2.5-7B-Instruct" \
#   --data_path "$MQ" --num_cases 200 \
#   --metric judge --judge_model "$JUDGE" \
#   --gold_mode orig\
#   --debug_judge \
#   --save_jsonl "$OUT/mq_baseline_qwen7b_judge.jsonl"

# python eval_zsre_singlehop.py \
#   --model "meta-llama/Llama-2-7b-chat-hf" \
#   --data_path "$ZS" --num_cases 200 \
#   --metric judge --judge_model "$JUDGE" \
#   --gold_mode orig \
#   --debug_judge \
#   --save_jsonl "$OUT/zsre_baseline_llama2_7b_chat_judge.jsonl"

# python eval_zsre_singlehop.py \
#   --model "Qwen/Qwen2.5-7B-Instruct" \
#   --data_path "$ZS" --num_cases 200 \
#   --metric judge --judge_model "$JUDGE" \
#   --gold_mode orig\
#   --debug_judge \
#   --save_jsonl "$OUT/zsre_baseline_qwen7b_judge.jsonl"

# python eval_zsre_singlehop.py \
#   --model "Qwen/Qwen2.5-7B-Instruct" \
#   --data_path "$ZS" --num_cases 200 \
#   --metric judge --judge_model "$JUDGE" \
#   --gold_mode new\
#   --debug_judge \
#   --save_jsonl "$OUT/zsre_baseline_qwen7b_judge_new.jsonl"


# python eval_zsre_singlehop.py \
#   --model "Qwen/Qwen2.5-7B-Instruct" \
#   --data_path "$ZS" --num_cases 200 \
#   --metric judge --judge_model "$JUDGE" \
#   --two_pass_reasoning --reveal_reasoning \
#   --gold_mode orig\
#   --debug_judge \
#   --save_jsonl "$OUT/zsre_baseline_qwen7b_judge_cot.jsonl"


# -------- DeepSeek R1 Distill Qwen-7B --------
python eval_mquake_multihop.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --data_path "$MQ" --num_cases 200 \
  --metric judge --judge_model "$JUDGE" \
  --two_pass_reasoning --reveal_reasoning \
  --gold_mode orig\
  --debug_judge \
  --save_jsonl "$OUT/mq_baseline_r1dq7b_judge.jsonl"

python eval_zsre_singlehop.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  --data_path "$ZS" --num_cases 200 \
  --metric judge --judge_model "$JUDGE" \
  --two_pass_reasoning --reveal_reasoning \
  --gold_mode orig\
  --debug_judge \
  --save_jsonl "$OUT/zsre_baseline_r1dq7b_judge.jsonl"

# 汇总
python summarize_jsonl.py "$OUT/*.jsonl"
