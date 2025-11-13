#!/usr/bin/env bash
set -euo pipefail

# AlphaEdit Runner with Multi-GPU Inference Support
# Usage: ./scripts/run_repeat_alphaedit.sh [START_IDX] [COUNT]
#
# Env overrides:
#   GPUS           (default: 0,1,2,3,4,5,6,7) - GPUs to use
#   MODEL          (default: auto-detect from HPARAMS) - Model to edit
#   HPARAMS        (default: hparams/AlphaEdit/deepseek-r1d-qwen-7b.yaml)
#   DATA           (default: data/zsre_mend_eval.json)
#   OUT_JSONL      (default: outputs/alphaedit_eval.jsonl)
#   SAVE_POOL      (default: outputs/edited_model_pool)
#   JUDGE_MODEL    (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   DS_MP_SIZE     (default: 8) - DeepSpeed tensor parallel size for inference
#   DS_DTYPE       (default: auto) - DeepSpeed dtype: auto|bf16|fp16
#   GEN_MODE       (default: r1d) - Generation mode: r1d|concise|reason|noprompt
#   MAX_NEW_TOKENS (default: 256)
#   TEMPERATURE    (default: 0)
#   TOP_P          (default: 1)
#   SKIP_SECS      (default: 2) - Delay between cases
#   BASE_PORT      (default: 29501)
#   EVAL_BEFORE    (default: no) - Set to "yes" to evaluate before editing

START_IDX=${1:-0}
COUNT=${2:-10}

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
HPARAMS=${HPARAMS:-"hparams/AlphaEdit/deepseek-r1d-qwen-7b.yaml"}
DATA=${DATA:-"data/zsre_mend_eval.json"}
OUT_JSONL=${OUT_JSONL:-"outputs/alphaedit_eval.jsonl"}
SAVE_POOL=${SAVE_POOL:-"outputs/edited_model_pool"}
JUDGE_MODEL=${JUDGE_MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}

BASE_PORT=${BASE_PORT:-29501}
DS_MP_SIZE=${DS_MP_SIZE:-8}
DS_DTYPE=${DS_DTYPE:-auto}

GEN_MODE=${GEN_MODE:-r1d}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-1}
SKIP_SECS=${SKIP_SECS:-2}
EVAL_BEFORE=${EVAL_BEFORE:-no}

export TOKENIZERS_PARALLELISM=false

# Set Wikipedia dataset version (change to 20200501.en if needed)
export WIKI_CONFIG="${WIKI_CONFIG:-20220301.en}"
export WIKI_DATASET="${WIKI_DATASET:-wikipedia}"

# Ensure Hugging Face caches default to a writable repo-local location
HF_HOME_DEFAULT="$PWD/models/hf_cache"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

# Optional: use a Hugging Face Hub mirror (cannot direct connect)
# Set USE_MIRROR=yes or specify HF_MIRROR URL explicitly.
if [[ "${USE_MIRROR:-}" == "yes" || -n "${HF_MIRROR:-}" ]]; then
  export HF_ENDPOINT="${HF_MIRROR:-https://hf-mirror.com}"
  echo "[Mirror] Using HF mirror endpoint: $HF_ENDPOINT"
fi

# Optional: precompute mom2 (second moment) for layers in HPARAMS before the loop
# Enable by: PRECOMPUTE_MOM2=yes
if [[ "${PRECOMPUTE_MOM2:-}" == "yes" || "${PRECOMPUTE_MOM2:-}" == "true" || "${PRECOMPUTE_MOM2:-}" == "1" ]]; then
  echo "========================================"
  echo "[AlphaEdit MOM2] Precomputing second moment (mom2) using $HPARAMS"
  echo "========================================"
  python - << 'PY'
import os, torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor.models.rome.layer_stats import layer_stats
from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams as HP

hp_path = os.environ.get("HPARAMS", "hparams/AlphaEdit/deepseek-r1d-qwen-7b.yaml")
hp = HP.from_hparams(hp_path)

tok = AutoTokenizer.from_pretrained(hp.model_name, trust_remote_code=True)

# pick safe dtype
if torch.cuda.is_available():
    dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
else:
    dtype = torch.float32

# place on the first visible GPU (or CPU)
if torch.cuda.is_available():
    dev = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    devmap = {"": dev}
else:
    dev, devmap = 0, None

mdl = AutoModelForCausalLM.from_pretrained(
    hp.model_name,
    torch_dtype=dtype,
    trust_remote_code=True,
    device_map=devmap,
)
hps = SimpleNamespace(device=dev)

print(f"[MOM2] model={hp.model_name}")
print(f"[MOM2] layers={hp.layers}")
print(f"[MOM2] dataset={hp.mom2_dataset}  n={hp.mom2_n_samples}  precision={hp.mom2_dtype}")
print(f"[MOM2] stats_dir={hp.stats_dir}")

for L in hp.layers:
    layer_name = hp.rewrite_module_tmp.format(L)
    print(f"[MOM2] computing: {layer_name}")
    _ = layer_stats(
        mdl, tok, layer_name,
        hp.stats_dir,
        hp.mom2_dataset,
        to_collect=["mom2"],
        sample_size=hp.mom2_n_samples,
        precision=hp.mom2_dtype,
        hparams=hps,
        force_recompute=False,  # set True to rebuild cache
    )
    print(f"[MOM2] done: {layer_name}")
print("[MOM2] all done.")
PY
fi

# Build eval_before flag
EVAL_BEFORE_FLAG=""
if [[ "$EVAL_BEFORE" == "yes" ]] || [[ "$EVAL_BEFORE" == "true" ]] || [[ "$EVAL_BEFORE" == "1" ]]; then
  EVAL_BEFORE_FLAG="--eval_before"
fi

for ((i=0; i<COUNT; i++)); do
  IDX=$((START_IDX + i))
  CASE_DIR="$SAVE_POOL/alphaedit-${IDX}"

  EDIT_PORT=$((BASE_PORT + 512 + (IDX % 256)))

  echo "========================================"
  echo "[AlphaEdit EDIT] case_index=$IDX using GPUs $GPUS port=$EDIT_PORT"
  echo "========================================"

  CUDA_VISIBLE_DEVICES="$GPUS" \
  accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --main_process_port="$EDIT_PORT" \
    --module scripts.edit_once_with_judge \
    --alg AlphaEdit \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --judge_model "$JUDGE_MODEL" \
    $EVAL_BEFORE_FLAG \
    --save_edited_to "$SAVE_POOL" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage edit

  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=$((BASE_PORT + (IDX % 256)))

  echo "========================================"
  echo "[AlphaEdit INFER] case_index=$IDX MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  echo "========================================"

  # If edit stage failed to produce a saved model, skip inference for this case
  if [[ ! -d "$CASE_DIR" ]]; then
    echo "[SKIP] No edited model directory found: $CASE_DIR (edit stage likely failed)."
    echo "[DONE] case $IDX completed"
    sleep "$SKIP_SECS"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun --nproc_per_node="${DS_MP_SIZE}" \
    --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --module scripts.edit_once_with_judge \
    --alg AlphaEdit \
    --hparams "$HPARAMS" \
    --data_path "$DATA" \
    --case_index "$IDX" --repeat 1 \
    --judge_model "$JUDGE_MODEL" \
    --use_ds_infer --ds_mp_size "$DS_MP_SIZE" --ds_dtype "$DS_DTYPE" \
    --gen_mode "$GEN_MODE" --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" \
    --save_jsonl "$OUT_JSONL" \
    ${OFFLINE:+--offline} \
    --print_every 1 \
    --stage infer \
    --load_edited_from "$CASE_DIR" \
    --delete_saved_after

  echo "[DONE] case $IDX completed"
  sleep "$SKIP_SECS"
done

echo "========================================"
echo "All $COUNT cases completed!"
echo "Results saved to: $OUT_JSONL"
echo "========================================"
