#!/usr/bin/env bash
set -euo pipefail

# 说明（零集成，完全复用 RLEdit）：
# - 从 data/noncot.json 里取单条样本，转换为 RLEdit 的 ZSRE 格式（把 ans=alt），
# - 调 RLEdit/main.py 运行 rledit，一次性训练/编辑/评测（RLEdit 内部完成），
# - 全程不保存模型权重，缓存与 Hydra 输出尽量放 /dev/shm，并在结束后清理。
#
# 用法：
#   bash scripts/run_rledit_standalone.sh [CASE_INDEX]
# 环境可覆盖：
#   GPU=0
#   DATA=data/noncot.json
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
#   RLEDIT_CACHE_DIR=$HOME/autodl-tmp/rledit_cache
#   HYDRA_RUN_DIR=$HOME/autodl-tmp/rledit_runs
#   EDIT_MODULES="[model.layers.29.mlp.gate_proj,model.layers.29.mlp.up_proj,model.layers.29.mlp.down_proj]"

CASE_INDEX=${1:-0}

GPU=${GPU:-0}
DATA=${DATA:-"data/noncot.json"}
MODEL=${MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
RLEDIT_CACHE_DIR=${RLEDIT_CACHE_DIR:-"$HOME/autodl-tmp/rledit_cache"}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-"$HOME/autodl-tmp/rledit_runs"}
# 顶层 MLP（示例，可按需改）
EDIT_MODULES=${EDIT_MODULES:-"[model.layers.29.mlp.gate_proj,model.layers.29.mlp.up_proj,model.layers.29.mlp.down_proj,model.layers.30.mlp.gate_proj,model.layers.30.mlp.up_proj,model.layers.30.mlp.down_proj,model.layers.31.mlp.gate_proj,model.layers.31.mlp.up_proj,model.layers.31.mlp.down_proj]"}

# 轻量 HF 缓存，尽量走内存盘
HF_HOME_DEFAULT="$HOME/autodl-tmp/hf_cache"
export HF_HOME=${HF_HOME:-$HF_HOME_DEFAULT}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" || true

export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=${WANDB_DISABLED:-true}

TMPDIR_BASE=${TMPDIR:-"$HOME/autodl-tmp"}
WORK_DIR=$(mktemp -d "$TMPDIR_BASE/rledit_case.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

ONE_JSON="$WORK_DIR/one.json"
mkdir -p "$RLEDIT_CACHE_DIR" "$HYDRA_RUN_DIR" || true

# 取一条样本，转成 RLEdit ZSRE 所需字段，并把 ans 设置为 alt（目标新答案）
python - <<'PY'
import json, os
DATA=os.environ.get('DATA')
CASE=int(os.environ.get('CASE_INDEX','0'))
OUT=os.environ.get('ONE_JSON')
with open(DATA,'r',encoding='utf-8') as f:
    arr=json.load(f)
if CASE<0 or CASE>=len(arr):
    raise SystemExit(f"CASE_INDEX {CASE} out of range (0..{len(arr)-1})")
row=arr[CASE]
src = row.get('src') or row.get('prompt') or ''
rephrase = row.get('rephrase') or row.get('rephrase_prompt') or src
alt = row.get('alt') or row.get('target_new') or ''
loc = row.get('loc') or ''
loc_ans = row.get('loc_ans') or ''
sample={
  'src': src,
  'rephrase': rephrase,
  'ans': alt,
  'loc': loc,
  'loc_ans': loc_ans,
}
with open(OUT,'w',encoding='utf-8') as f:
    json.dump([sample], f, ensure_ascii=False, indent=2)
print(f"[prep] wrote one-sample JSON: {OUT}")
PY

# 注意：RLEdit/model.py 会设置 CUDA_VISIBLE_DEVICES=0，建议使用 GPU=0；
# 如需切换 GPU，请在容器/节点层做映射。
echo "[RLEdit] Running single edit on case $CASE_INDEX (GPU=$GPU)"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m RLEdit.main \
  model=mistral-7b \
  editor=rledit \
  dataset=zsre \
  model.name_or_path="$MODEL" \
  model.name="$(basename "$MODEL")" \
  model.edit_modules="$EDIT_MODULES" \
  dataset.n_edits=1 dataset.batch_size=1 \
  dataset.train_path="$ONE_JSON" dataset.valid_path="$ONE_JSON" \
  editor.n_epochs=1 editor.batch_size=1 \
  editor.cache_dir="$RLEDIT_CACHE_DIR" \
  editor.save_checkpoint=False editor.load_checkpoint=False \
  glue_step=0 \
  hydra.run.dir="$HYDRA_RUN_DIR" || true

# 结束后尽量清理缓存与 Hydra 输出（保守删除，仅删除我们本次目录内容）
subdir_name="$(basename "$MODEL")_rledit_1"
if [ -d "$RLEDIT_CACHE_DIR/$subdir_name" ]; then
  rm -rf "$RLEDIT_CACHE_DIR/$subdir_name" || true
fi
if [ -d "$HYDRA_RUN_DIR" ]; then
  find "$HYDRA_RUN_DIR" -maxdepth 1 -type d -mmin +10 -exec rm -rf {} + 2>/dev/null || true
fi

echo "[RLEdit] Done. (artifacts cleaned; model not saved)"
