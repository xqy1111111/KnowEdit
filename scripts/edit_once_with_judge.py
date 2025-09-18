# scripts/edit_once_and_judge.py
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from easyeditor import (
    BaseEditor,
    ROMEHyperParams,
    WISEHyperParams,
    FTHyperParams,
    LoRAHyperParams,
    QLoRAHyperParams,
)

# ===================== 工具 =====================
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def ensure_pad_token(tokenizer: AutoTokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

def _unwrap_tokenizer(tok_like: Any) -> Optional[PreTrainedTokenizerBase]:
    if tok_like is None:
        return None
    if isinstance(tok_like, PreTrainedTokenizerBase):
        return tok_like
    if hasattr(tok_like, "save_pretrained"):
        return tok_like
    if isinstance(tok_like, dict):
        for k in ("tokenizer", "tok", "fast_tokenizer", "hf_tokenizer"):
            v = tok_like.get(k)
            if isinstance(v, PreTrainedTokenizerBase) or hasattr(v, "save_pretrained"):
                return v
    return None

# ===================== 读取 ZsRE-like =====================
def _get_first(x, default=""):
    if x is None:
        return default
    if isinstance(x, list):
        return x[0] if x else default
    return x

def read_zsre_like(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reqs = []
    for r in data:
        prompt = r.get("prompt") or r.get("src") or ""
        if not prompt:
            continue
        target_new = r.get("target_new") or r.get("alt") or ""
        if not target_new:
            continue
        ground_truth = r.get("ground_truth")
        if ground_truth is None:
            ground_truth = _get_first(r.get("answers"), "")

        subject = r.get("subject") or r.get("sub_label") or r.get("subject_entity") or ""

        rephrase = r.get("rephrase")
        if isinstance(rephrase, str):
            rephrase_list = [rephrase] if rephrase.strip() else []
        elif isinstance(rephrase, list):
            rephrase_list = [x for x in rephrase if isinstance(x, str) and x.strip()]
        else:
            rephrase_list = []

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
        reqs.append(req)
    return reqs

# ===================== 主语 =====================
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

# ===================== 构建 Editor =====================
ALG_HP_MAP = {
    "ROME": (ROMEHyperParams, "ROME"),
    "WISE": (WISEHyperParams, "WISE"),
    "FT": (FTHyperParams, "FT"),
    "LORA": (LoRAHyperParams, "LoRA"),
    "QLORA": (QLoRAHyperParams, "QLoRA"),
}

SUPPORTED_ALG_NAMES = [display for _, display in ALG_HP_MAP.values()]


def build_editor(alg: str, hparams_path: str, model_name: str) -> BaseEditor:
    key = alg.upper()
    if key not in ALG_HP_MAP:
        supported = ", ".join(SUPPORTED_ALG_NAMES)
        raise ValueError(f"alg must be one of {{{supported}}}")

    hp_cls, _ = ALG_HP_MAP[key]
    hp = hp_cls.from_hparams(hparams_path)
    if model_name:
        hp.model_name = model_name
    return BaseEditor.from_hparams(hp)

# ===================== 生成（两种模式） =====================
def _build_messages(prompt: str, mode: str) -> list:
    if mode == "reason":
        sys_msg = (
            "You are a helpful assistant. "
            "Think step by step INSIDE <reasoning>...</reasoning>. "
            "Then output ONLY the final short answer INSIDE <final>...</final>. "
            "Do not include anything else outside these tags."
        )
    else:
        sys_msg = "Answer with a short noun phrase only. Do NOT explain."
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": prompt}
    ]

def generate_answer(model, tokenizer, prompt: str,
                    max_new_tokens=64, temperature=0.6, top_p=0.95,
                    mode: str = "concise") -> str:
    messages = _build_messages(prompt, mode=mode)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if mode == "reason":
            text = (
                "You are a helpful assistant. Think step by step INSIDE <reasoning>...</reasoning>. less but useful steps are appriciated"
                "Then output ONLY the final short answer INSIDE <final>...</final>.\n"
                f"Q: {prompt}\nA:"
            )
        else:
            text = f"Answer with a short noun phrase only. Do NOT explain.\nQ: {prompt}\nA:"

    ensure_pad_token(tokenizer)
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    eos_ids = []
    for tok in ["<|im_end|>", "<|end|>", tokenizer.eos_token]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok) if tok else None
        except Exception:
            tid = None
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)
    if not eos_ids and tokenizer.eos_token_id is not None:
        eos_ids = [tokenizer.eos_token_id]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=3,
    )
    ans = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip()

# ===================== 解析 <final> =====================
_FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
def extract_final(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    m_f = _FINAL_RE.search(text)
    m_r = _REASON_RE.search(text)
    final = (m_f.group(1).strip() if m_f else "").strip("：:").strip()
    reason = (m_r.group(1).strip() if m_r else "").strip()
    return final, reason

# ===================== LLM-Judge =====================
_JUDGE_CACHE = {}
def load_judge(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if model_id in _JUDGE_CACHE:
        return _JUDGE_CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ensure_pad_token(tok)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=pick_dtype(),
        trust_remote_code=True, device_map="auto"
    )
    _JUDGE_CACHE[model_id] = (tok, mdl)
    return tok, mdl

JUDGE_PROMPT = """You are a precise grader.
Only judge the FINAL_ANSWER string; ignore any hidden thoughts/reasoning if present.
Normalize: lowercase, strip accents/punctuation; minor typos are acceptable.
If FINAL_ANSWER semantically matches ANY of the GOLD items, answer YES; otherwise NO.

Return exactly one token: YES or NO.

GOLD: {golds}
FINAL_ANSWER: {final}

Decision (YES or NO):"""

def judge_hit(judge_model_id: str, question: str, final: str, golds: List[str],
              max_new_tokens: int = 8, debug: bool = False) -> int:
    tok, mdl = load_judge(judge_model_id)
    prompt = JUDGE_PROMPT.format(question=question, golds=", ".join(golds), final=final)
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    if debug:
        print("[LLM-JUDGE RAW]:", txt)
    return int(("YES" in txt) and ("NO" not in txt))

# ===================== 单条：可选“编辑前” + 编辑 + 评测 =====================
def run_one_case(
    req: Dict[str, Any],
    alg: str,
    hparams: str,
    model_override: str,
    judge_model: str,
    gen_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eval_before: bool = False,
    show_eemetrics: bool = False,
) -> Dict[str, Any]:

    # 0) ROME subject 准备
    if alg.upper() == "ROME":
        s = req.get("subject", "")
        if not s:
            s = heuristic_extract_subject(req["prompt"])
            if s:
                warnings.warn(f"[ROME] subject missing; heuristically extracted '{s}'")
            else:
                raise ValueError(f"[ROME] cannot find subject for prompt: {req['prompt']}")
        if s not in req["prompt"]:
            req = dict(req)
            req["prompt"] = f"{req['prompt']} (Subject: {s})"
            req["subject"] = s

    # 构建编辑器（优先一次性加载，避免重复占显存）
    base_model_id = model_override  # 若为空，后面会从 editor.hparams 取
    editor = build_editor(alg, hparams, model_override)
    if not base_model_id:
        base_model_id = getattr(editor.hparams, "model_name", "")

    #（可选）编辑前评测（尽量直接用已加载的 editor.model 与 editor.tok）
    pred_before, a_for_score_before, rewrite_hit_before = "", "", None
    if eval_before:
        mdl0 = getattr(editor, "model", None)
        tok0 = _unwrap_tokenizer(getattr(editor, "tok", None))
        if mdl0 is None or tok0 is None:
            # 回退到单独加载一次基座模型
            bm_id = base_model_id
            tok0 = AutoTokenizer.from_pretrained(bm_id, trust_remote_code=True)
            ensure_pad_token(tok0)
            mdl0 = AutoModelForCausalLM.from_pretrained(bm_id, torch_dtype=pick_dtype(), trust_remote_code=True, device_map="auto")
            need_release = True
        else:
            ensure_pad_token(tok0)
            need_release = False

        pred_before = generate_answer(
            mdl0, tok0, req["prompt"],
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            mode=gen_mode
        )
        if gen_mode == "reason":
            final_b, _ = extract_final(pred_before)
            a_for_score_before = final_b if final_b else pred_before
        else:
            a_for_score_before = pred_before
        rewrite_hit_before = judge_hit(judge_model, question=req["prompt"], final=a_for_score_before, golds=[req.get("target_new","")])

        # 如有必要释放回退加载的模型
        if need_release:
            try:
                del mdl0; del tok0
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass

    # 执行编辑（单条）。关键：sequential_edit=True 以保留编辑后的权重供外部生成。
    metrics, edited_model, _ = editor.edit_requests(requests=[req], sequential_edit=True)
    if show_eemetrics:
        print("[EASYEDIT METRICS]", json.dumps(metrics, ensure_ascii=False))

    # 用编辑后模型生成（直接复用 editor.tok，保持与训练一致）
    tok = _unwrap_tokenizer(getattr(editor, "tok", None)) or AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    ensure_pad_token(tok)
    pred_after = generate_answer(
        edited_model, tok, req["prompt"],
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        mode=gen_mode
    )
    if gen_mode == "reason":
        final_after, _ = extract_final(pred_after)
        a_for_score_after = final_after if final_after else pred_after
    else:
        a_for_score_after = pred_after

    rewrite_hit_after = judge_hit(judge_model, question=req["prompt"], final=a_for_score_after, golds=[req.get("target_new","")])

    # Locality（若提供）
    loc_rec: Dict[str, Any] = {}
    if isinstance(req.get("locality"), dict) and "nq" in req["locality"]:
        lp = req["locality"]["nq"].get("prompt", "")
        lg = req["locality"]["nq"].get("ground_truth", "")
        if lp and lg:
            loc_pred = generate_answer(
                edited_model, tok, lp,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                mode=gen_mode
            )
            loc_final, _ = extract_final(loc_pred) if gen_mode == "reason" else (loc_pred, "")
            loc_hit = judge_hit(judge_model, question=lp, final=(loc_final or loc_pred), golds=[lg])
            loc_rec = {
                "loc_prompt": lp,
                "loc_gold": lg,
                "pred_loc": loc_pred,
                "locality_hit": int(loc_hit),
            }

    # E) 组装记录（存在才写）
    rec: Dict[str, Any] = {
        "prompt": req["prompt"],
        "target_new": req["target_new"],
        "pred_after": pred_after,
        "rewrite_hit": int(rewrite_hit_after),
        "easyedit_metrics": metrics[0] if isinstance(metrics, list) and metrics else metrics,
    }
    if eval_before:
        rec["pred_before"] = pred_before
        rec["rewrite_hit_before"] = int(rewrite_hit_before)

    rec.update(loc_rec)
        # --- MINIMAL PATCH: always provide locality keys to avoid KeyError ---
    rec.setdefault("loc_prompt", "")
    rec.setdefault("loc_gold", "")
    rec.setdefault("pred_loc", "")
    rec.setdefault("locality_hit", None)
    # ---------------------------------------------------------------------

    return rec


# ===================== 主流程（支持 repeat） =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=SUPPORTED_ALG_NAMES, help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=True, help="ZSRE-like JSON path")
    ap.add_argument("--model", default="", help="HF model id/path to override YAML")
    ap.add_argument("--case_index", type=int, default=0, help="Start index in dataset")
    ap.add_argument("--repeat", type=int, default=1, help="How many consecutive cases to run")
    ap.add_argument("--wrap", action="store_true", help="Wrap around dataset if overflow")
    ap.add_argument("--no-reset_each", dest="reset_each", action="store_false",
                    help="Apply edits cumulatively on same process (default: reset each)")
    ap.add_argument("--judge_model", required=True, help="HF id/path for LLM Judge")

    # 生成控制
    ap.add_argument("--gen_mode", choices=["concise","reason"], default="concise")
    ap.add_argument("--max_new_tokens", type=int, default=9192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    # 评测控制
    ap.add_argument("--eval_before", action="store_true", help="Also judge before editing (to prove effect)")
    ap.add_argument("--show_eemetrics", action="store_true")

    # 输出
    ap.add_argument("--save_jsonl", default="", help="Where to append results (JSONL). If empty, just print.")
    ap.add_argument("--print_every", type=int, default=1)

    args = ap.parse_args()

    all_reqs = read_zsre_like(args.data_path)
    if not all_reqs:
        raise RuntimeError(f"No valid requests parsed from {args.data_path}")
    N = len(all_reqs)

    fw = None
    if args.save_jsonl:
        path = os.path.abspath(os.path.expanduser(args.save_jsonl))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fw = open(path, "a", encoding="utf-8")

    base_idx = args.case_index
    for t in range(args.repeat):
        idx = base_idx + t
        if idx >= N:
            if args.wrap:
                idx = idx % N
            else:
                break

        req = dict(all_reqs[idx])
        try:
            rec = run_one_case(
                req=req,
                alg=args.alg,
                hparams=args.hparams,
                model_override=args.model,
                judge_model=args.judge_model,
                gen_mode=args.gen_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eval_before=args.eval_before,
                show_eemetrics=args.show_eemetrics,
            )
            rec["case_index"] = idx
        except Exception as e:
            rec = {"case_index": idx, "error": str(e)}
            warnings.warn(f"[WARN] case {idx} failed: {e}")

        if fw:
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw.flush()
        if (t + 1) % max(1, args.print_every) == 0 or not fw:
            print(json.dumps(rec, ensure_ascii=False))

        # 显存清理（当前循环结束）
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # reset_each=True 时，每次 run_one_case 内都会重建 editor（等价于从干净基座开始）
        # 这里无需额外处理
        if args.reset_each:
            pass

    if fw:
        fw.close()

if __name__ == "__main__":
    main()
