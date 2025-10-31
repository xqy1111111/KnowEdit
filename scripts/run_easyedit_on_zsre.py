# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union

from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# ---------------- 读 ZsRE-like 数据 ----------------
def _get_first(x, default=""):
    if x is None:
        return default
    if isinstance(x, list):
        return x[0] if x else default
    return x

def read_zsre_like(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    requests = []
    for r in data:
        prompt = r.get("prompt") or r.get("src") or ""
        if not prompt:
            continue

        target_new = r.get("target_new") or r.get("alt") or ""
        if not target_new:
            continue  # 没有新答案无法编辑

        ground_truth = r.get("ground_truth")
        if ground_truth is None:
            ground_truth = _get_first(r.get("answers"), "")

        subject = r.get("subject") or r.get("sub_label") or r.get("subject_entity") or ""

        # rephrase：允许 str 或 list，统一成 list
        rephrase = r.get("rephrase")
        if rephrase is None:
            rephrase_list = []
        elif isinstance(rephrase, list):
            rephrase_list = [x for x in rephrase if isinstance(x, str) and x.strip()]
        elif isinstance(rephrase, str) and rephrase.strip():
            rephrase_list = [rephrase.strip()]
        else:
            rephrase_list = []

        # locality：loc/loc_ans，允许 str 或 list（list 取首个）
        lp = _get_first(r.get("loc"), "")
        la = _get_first(r.get("loc_ans"), "")

        req = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "rephrase": rephrase_list,
            "locality": {"nq": {"prompt": lp, "ground_truth": la}} if (lp and la) else {},
            "portability": {},
        }
        if subject:
            req["subject"] = subject

        requests.append(req)
    return requests

# ---------------- 主语兜底抽取（ROME 需要 subject） ----------------
_SUBJ_PATTERNS = [
    re.compile(r'(?:did|does|was|is|were|has|have)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
    re.compile(r'^[Ww]ho (?:is|was)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
    re.compile(r'\babout\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
]

def heuristic_extract_subject(prompt: str) -> str:
    for pat in _SUBJ_PATTERNS:
        m = pat.search(prompt)
        if m:
            return m.group(1).strip()
    caps = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})', prompt)
    if caps:
        caps.sort(key=len, reverse=True)
        return caps[0].strip()
    return ""

# ---------------- 稳健保存：模型 + tokenizer ----------------
def _pick_tokenizer_obj(tok_like: Any) -> Optional[PreTrainedTokenizerBase]:
    """从各种形态里把真正的 tokenizer 拿出来。没有就返回 None。"""
    if tok_like is None:
        return None
    if isinstance(tok_like, PreTrainedTokenizerBase):
        return tok_like
    if hasattr(tok_like, "save_pretrained"):  # 兼容子类
        return tok_like
    if isinstance(tok_like, dict):
        for k in ("tokenizer", "tok", "fast_tokenizer", "hf_tokenizer"):
            t = tok_like.get(k)
            if isinstance(t, PreTrainedTokenizerBase) or hasattr(t, "save_pretrained"):
                return t
    return None

def save_edited_model(
    model,
    tokenizer_like: Any,
    out_dir: str,
    src_tokenizer_id_or_path: Optional[str] = None,
    trust_remote_code: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) 保存模型（safetensors）
    model.save_pretrained(out_dir, safe_serialization=True)

    # 2) 保存 tokenizer（优先用传入对象；不行就从基座加载）
    tok_obj = _pick_tokenizer_obj(tokenizer_like)
    if tok_obj is not None:
        tok_obj.save_pretrained(out_dir)
        print(f"[SAVE] Tokenizer saved to: {out_dir}")
        return

    if src_tokenizer_id_or_path:
        try:
            base_tok = AutoTokenizer.from_pretrained(
                src_tokenizer_id_or_path, trust_remote_code=trust_remote_code
            )
            base_tok.save_pretrained(out_dir)
            print(f"[SAVE] Tokenizer copied from '{src_tokenizer_id_or_path}' to: {out_dir}")
            return
        except Exception as e:
            warnings.warn(f"[WARN] Fallback load tokenizer from '{src_tokenizer_id_or_path}' failed: {e}")

    # 3) 兜底：不给 tokenizer 也不报错中止（避免像你之前那样把流程卡死）
    warnings.warn(
        "[WARN] Tokenizer not saved (no tokenizer object and no --src_tokenizer provided). "
        "Make sure to load with the SAME tokenizer as the base model when using the edited weights."
    )

# ---------------- 构建 Editor ----------------
def build_editor(alg: str, hparams_path: str, model_name: str) -> BaseEditor:
    if alg.upper() == "ROME":
        hp = ROMEHyperParams.from_hparams(hparams_path)
        if model_name:
            hp.model_name = model_name
        return BaseEditor.from_hparams(hp)
    if alg.upper() == "WISE":
        hp = WISEHyperParams.from_hparams(hparams_path)
        if model_name:
            hp.model_name = model_name
        return BaseEditor.from_hparams(hp)
    raise ValueError("alg must be one of {ROME, WISE}")

# ---------------- 小型自检（可选） ----------------
def quick_sanity_generate(model_dir: str, base_tokenizer: Optional[str], prompt: str) -> str:
    """
    用保存到 model_dir 的 tokenizer（若不存在则用 base_tokenizer）做一次短生成；
    对 Qwen Instruct 模型自动套 chat_template。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        if not base_tokenizer:
            raise
        tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)

    # 若 tokenizer 有 chat 模板（Instruct 模型），自动走 chat 格式
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    else:
        input_ids = tok(prompt, return_tensors="pt").to(model.device)

    gen = model.generate(
        **input_ids,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=getattr(tok, "eos_token_id", None),
        pad_token_id=getattr(tok, "pad_token_id", None),
    )
    return tok.decode(gen[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=["ROME", "WISE"], help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=True, help="Path to ZsRE/CounterFact-style JSON file")
    ap.add_argument("--model", default="", help="HF model id or local path; overrides YAML")
    ap.add_argument("--edit_num", type=int, default=100, help="Number of edits to apply")
    ap.add_argument("--out_dir", default="runs/edited_model", help="Directory to save the edited model")
    ap.add_argument("--sequential", action="store_true", help="Apply edits sequentially (lifelong)")
    ap.add_argument("--loc_key", default="nq", help="Key name for locality sub-metric")
    ap.add_argument("--src_tokenizer", default="", help="(强烈推荐) 基座 tokenizer 的 HF 名称或本地路径，用于保存/兜底加载")
    ap.add_argument("--sanity_prompt", default="", help="保存完成后做一次自检生成的提示语（可选）")
    args = ap.parse_args()

    # 1) 读数据
    requests = read_zsre_like(args.data_path, limit=args.edit_num)
    if not requests:
        raise RuntimeError(f"No valid edits parsed from {args.data_path}")

    # 2) ROME 需要 subject
    if args.alg.upper() == "ROME":
        fixed = []
        for i, req in enumerate(requests):
            p = req["prompt"]
            s = req.get("subject", "")
            if not s:
                s = heuristic_extract_subject(p)
                if s:
                    warnings.warn(f"[ROME] subject missing; heuristically extracted '{s}' from: {p}")
                else:
                    raise ValueError(
                        f"[ROME] Failed to find subject for sample #{i}. "
                        f"Prompt: {p}\n"
                        f"Tip: 使用带 subject 的数据(如 CounterFact)，或提前把 subject 写入 JSON。"
                    )
            if s not in p:
                # 避免模板依赖，直接把 subject 注到提示里（不影响问法含义）
                p = f"{p} (Subject: {s})"
            req["prompt"] = p
            req["subject"] = s
            fixed.append(req)
        requests = fixed

    # 3) 构建编辑器
    editor = build_editor(args.alg, args.hparams, args.model)

    # 4) 执行编辑
    metrics = None
    edited_model = None
    tok_like = None
    try:
        metrics, edited_model, tok_like = editor.edit_requests(
            requests=requests,
            sequential_edit=args.sequential,
        )
    finally:
        # 5) 落盘（模型 + tokenizer）
        if edited_model is not None:
            out_model_dir = os.path.join(args.out_dir, "edited_model")
            print(f"[SAFE SAVE] Saving edited model to: {out_model_dir}")
            save_edited_model(
                edited_model,
                tok_like,
                out_model_dir,
                src_tokenizer_id_or_path=(args.src_tokenizer or args.model),
                trust_remote_code=True,
            )

    # 6) 保存指标
    os.makedirs(args.out_dir, exist_ok=True)
    if metrics is not None:
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Metrics saved to: {os.path.join(args.out_dir, 'metrics.json')}")

    # 7) 可选自检：生成一段看看是不是正常文字
    if args.sanity_prompt:
        model_dir = os.path.join(args.out_dir, "edited_model")
        try:
            out = quick_sanity_generate(model_dir, args.src_tokenizer or args.model, args.sanity_prompt)
            print("[SANITY OUTPUT]", out)
        except Exception as e:
            warnings.warn(f"[WARN] sanity generate failed: {e}")

    print("[DONE] Edited model and metrics saved at:", args.out_dir)

if __name__ == "__main__":
    main()
