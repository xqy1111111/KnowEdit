run_combo() {
  ALG="$1"                 # ROME / WISE
  HP="$2"                  # hparams 路径
  BASE_MODEL="$3"          # 基础模型 HF 名称或本地路径
  TAG="$4"                 # 标识（用于文件名）
  OUT="$5"                 # 输出目录（包含 edited_model）

  echo "=== ${ALG} × ${TAG} | 编辑 on ZSRE ==="
  python scripts/run_easyedit_on_zsre.py \
    --alg "$ALG" --hparams "$HP" \
    --data_path "$ZSRE" --edit_num $N_EDIT \
    --out_dir "$OUT"

  EDITED="${OUT}/edited_model"

  echo "=== ZSRE 单跳评测（编辑前/后） ==="
  python scripts/eval_zsre_singlehop.py --model "$BASE_MODEL" \
    --data_path "$ZSRE" --num_cases $N_EVAL --gold_mode orig \
    --temperature $TEMP --top_p $TOPP \
    --save_jsonl "${OUT}/${TAG}_zsre_base_orig.jsonl"
  python scripts/eval_zsre_singlehop.py --model "$BASE_MODEL" \
    --data_path "$ZSRE" --num_cases $N_EVAL --gold_mode new \
    --temperature $TEMP --top_p $TOPP \
    --save_jsonl "${OUT}/${TAG}_zsre_base_new.jsonl"
  python scripts/eval_zsre_singlehop.py --model "$EDITED" \
    --data_path "$ZSRE" --num_cases $N_EVAL --gold_mode new \
    --temperature $TEMP --top_p $TOPP \
    --save_jsonl "${OUT}/${TAG}_zsre_edit_new.jsonl"
  python scripts/eval_zsre_singlehop.py --model "$EDITED" \
    --data_path "$ZSRE" --num_cases $N_EVAL --gold_mode orig \
    --temperature $TEMP --top_p $TOPP \
    --save_jsonl "${OUT}/${TAG}_zsre_edit_orig.jsonl"

  python scripts/compare_pre_post.py \
    --base_vs_orig "${OUT}/${TAG}_zsre_base_orig.jsonl" \
    --base_vs_new  "${OUT}/${TAG}_zsre_base_new.jsonl"  \
    --edit_vs_new  "${OUT}/${TAG}_zsre_edit_new.jsonl"  \
    --edit_vs_orig "${OUT}/${TAG}_zsre_edit_orig.jsonl" \
    --judge_key hit --tag "${ALG}×${TAG} on ZSRE"

  echo "=== MQuAKE 多跳评测（编辑前/后） ==="
  python scripts/eval_mquake_multihop.py --model "$BASE_MODEL" \
    --data_path "$MQ" --num_cases $N_EVAL --gold_mode orig \
    --temperature $TEMP --top_p $TOPP --require_k 1 \
    --save_jsonl "${OUT}/${TAG}_mq_base_orig.jsonl"
  python scripts/eval_mquake_multihop.py --model "$BASE_MODEL" \
    --data_path "$MQ" --num_cases $N_EVAL --gold_mode new \
    --temperature $TEMP --top_p $TOPP --require_k 1 \
    --save_jsonl "${OUT}/${TAG}_mq_base_new.jsonl"
  python scripts/eval_mquake_multihop.py --model "$EDITED" \
    --data_path "$MQ" --num_cases $N_EVAL --gold_mode new \
    --temperature $TEMP --top_p $TOPP --require_k 1 \
    --save_jsonl "${OUT}/${TAG}_mq_edit_new.jsonl"
  python scripts/eval_mquake_multihop.py --model "$EDITED" \
    --data_path "$MQ" --num_cases $N_EVAL --gold_mode orig \
    --temperature $TEMP --top_p $TOPP --require_k 1 \
    --save_jsonl "${OUT}/${TAG}_mq_edit_orig.jsonl"

  python scripts/compare_pre_post.py \
    --base_vs_orig "${OUT}/${TAG}_mq_base_orig.jsonl" \
    --base_vs_new  "${OUT}/${TAG}_mq_base_new.jsonl"  \
    --edit_vs_new  "${OUT}/${TAG}_mq_edit_new.jsonl"  \
    --edit_vs_orig "${OUT}/${TAG}_mq_edit_orig.jsonl" \
    --judge_key hit --tag "${ALG}×${TAG} on MQuAKE"
}
