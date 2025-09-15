# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoTokenizer
from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams


# -------------------------
# 小工具
# -------------------------
def _get_first(x, default: str = "") -> str:
    if x is None:
        return default
    if isinstance(x, list):
        return x[0] if x else default
    return x


# -------------------------
# 读取 ZsRE-like 数据
# 支持字段：src/prompt, answers/ground_truth, alt/target_new, rephrase, loc/loc_ans, subject/*
# 输出：EasyEdit 的 requests 列表（每条一个 dict），内含 locality 与 rephrase
# -------------------------
def read_zsre_like(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    reqs: List[Dict[str, Any]] = []
    for r in data:
        prompt = r.get("prompt") or r.get("src") or ""
        if not prompt:
            continue

        target_new = r.get("target_new") or r.get("alt") or ""
        if not target_new:
            # 没有新答案，无从编辑，跳过
            continue

        ground_truth = r.get("ground_truth")
        if ground_truth is None:
            ground_truth = _get_first(r.get("answers"), "")

        # rephrase: 统一成 list[str]
        rp_raw = r.get("rephrase")
        if rp_raw is None:
            rephrase_list: List[str] = []
        elif isinstance(rp_raw, list):
            rephrase_list = [x.strip() for x in rp_raw if isinstance(x, str) and x.strip()]
        elif isinstance(rp_raw, str) and rp_raw.strip():
            rephrase_list = [rp_raw.strip()]
        else:
            rephrase_list = []

        # locality: 你的 zrse 里有 loc / loc_ans
        loc_prompt = _get_first(r.get("loc"), "")
        loc_answer = _get_first(r.get("loc_ans"), "")

        one = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "rephrase": rephrase_list,
            "locality": {"nq": {"prompt": loc_prompt, "ground_truth": loc_answer}} if (loc_prompt and loc_answer) else {},
            "portability": {},  # 你的数据没有 portability，就留空
        }

        # 主语（ROME 必需）
        subj = r.get("subject") or r.get("sub_label") or r.get("subject_entity") or ""
        if subj:
            one["subject"] = subj

        reqs.append(one)

    return reqs


# -------------------------
# 主语兜底抽取（ROME 用）
# -------------------------
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


# -------------------------
# 保存模型（容错版 tokenizer）
# -------------------------
def save_edited_model(model, tokenizer_like, out_dir: str, fallback_name: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    # 1) 先存模型
    model.save_pretrained(out_dir)

    # 2) tokenizer 可能是对象 / dict / None
    tok_obj = None
    if hasattr(tokenizer_like, "save_pretrained"):
        tok_obj = tokenizer_like
    elif isinstance(tokenizer_like, dict):
        # EasyEdit 有时返回 {'tokenizer': tok, ...}
        tok_obj = tokenizer_like.get("tokenizer") or tokenizer_like.get("tok")
        if tok_obj is not None and not hasattr(tok_obj, "save_pretrained"):
            tok_obj = None

    if tok_obj is not None:
        tok_obj.save_pretrained(out_dir)
        return

    # 3) 回退重建 tokenizer
    cand_names: List[str] = []
    if fallback_name:
        cand_names.append(fallback_name)
    if getattr(model, "config", None) is not None:
        n = getattr(model.config, "_name_or_path", None)
        if n:
            cand_names.append(n)
    cand_names.append(out_dir)  # 最后试一下刚保存的模型目录（有些模型能从config里恢复）

    last_err = None
    for name in cand_names:
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            tok.save_pretrained(out_dir)
            return
        except Exception as e:
            last_err = e

    print(f"[WARN] Tokenizer save failed; model weights are saved at {out_dir}. Last error: {last_err}")


# -------------------------
# 构建 Editor，并保留 hparams 供 fallback
# -------------------------
def build_editor(alg: str, hparams_path: str, model_name: str):
    if alg.upper() == "ROME":
        hp = ROMEHyperParams.from_hparams(hparams_path)
    elif alg.upper() == "WISE":
        hp = WISEHyperParams.from_hparams(hparams_path)
    else:
        raise ValueError("alg must be one of {ROME, WISE}")

    if model_name:
        hp.model_name = model_name
    editor = BaseEditor.from_hparams(hp)
    return editor, hp


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=["ROME", "WISE"], help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=True, help="Path to ZsRE/CounterFact-style JSON file")
    ap.add_argument("--model", default="", help="HF model id or local path; overrides YAML")
    ap.add_argument("--edit_num", type=int, default=100, help="Number of edits to apply")
    ap.add_argument("--out_dir", default="runs/edited_model", help="Directory to save the edited model")
    ap.add_argument("--sequential", action="store_true", help="Apply edits sequentially (lifelong)")
    ap.add_argument("--tokenizer_fallback", default="", help="Tokenizer name for fallback, e.g., Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    # 1) 读数据为 requests
    requests = read_zsre_like(args.data_path, limit=args.edit_num)
    if not requests:
        raise RuntimeError(f"No valid edits parsed from {args.data_path}")

    # 2) ROME 需要 subject 且出现在 prompt 中
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
                        f"Tip: 使用带 subject 的数据(如 CounterFact)，或将 subject 写回 JSON。"
                    )
            if s not in p:
                warnings.warn(f"[ROME] subject '{s}' not in prompt; appending for safety.")
                p = f"{p} (Subject: {s})"
            req["prompt"] = p
            req["subject"] = s
            fixed.append(req)
        requests = fixed

    # 3) 构建编辑器
    editor, hp = build_editor(args.alg, args.hparams, args.model)

    # 4) 统一用 edit_requests（会启用 rephrase/locality 的评测）
    metrics = None
    edited_model = tok = None
    try:
        metrics, edited_model, tok = editor.edit_requests(
            requests=requests,
            sequential_edit=args.sequential,
        )
    finally:
        # 即使后续出错，也尽量把已编辑模型落盘
        if edited_model is not None:
            out_model_dir = os.path.join(args.out_dir, "edited_model")
            print(f"[SAFE SAVE] Saving edited model to: {out_model_dir}")
            fb = args.tokenizer_fallback or getattr(hp, "tokenizer_name", None)
            save_edited_model(edited_model, tok, out_model_dir, fallback_name=fb)

    # 5) 保存 metrics
    os.makedirs(args.out_dir, exist_ok=True)
    if metrics is not None:
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] Edited model and metrics saved at:", args.out_dir)


if __name__ == "__main__":
    main()
