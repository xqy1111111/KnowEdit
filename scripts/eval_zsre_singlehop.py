# scripts/eval_zsre_singlehop.py
# -*- coding: utf-8 -*-
import os, argparse, time, json, re, unicodedata
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from .utils_mquake import (  # 复用你的通用生成与工具
    load_mquake_json as load_json,
    generate_answer, split_reasoning_final,
    contains_any, sanitize_aliases, normalize_text
)

# -------------------- 小工具：规范化 & 模糊匹配 --------------------
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def norm_loose(s: str) -> str:
    s = s or ""
    s = strip_accents(s.lower())
    s = re.sub(r"[\W_]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

try:
    from rapidfuzz.fuzz import partial_ratio, WRatio
    HAVE_RAPID = True
except Exception:
    HAVE_RAPID = False

def fuzzy_hit(pred: str, golds: List[str], fuzzy_thr: int = 90) -> bool:
    p = norm_loose(pred)
    if not p:
        return False
    for g in golds:
        q = norm_loose(g)
        if not q:
            continue
        if q in p:
            return True
        if HAVE_RAPID:
            try:
                if partial_ratio(p, q) >= fuzzy_thr or WRatio(p, q) >= fuzzy_thr:
                    return True
            except Exception:
                pass
    return False

# -------------------- BERTScore（按需加载） --------------------
_BERT_SCORER = None
def get_bertscorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        try:
            from bert_score import BERTScorer
        except ImportError as e:
            raise RuntimeError("metric=bert 需安装：pip install bert-score") from e
        _BERT_SCORER = BERTScorer(
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _BERT_SCORER

def bert_hit_cached(pred: str, golds: List[str], thresh: float = 0.85) -> bool:
    scorer = get_bertscorer()
    P, R, F1 = scorer.score([pred] * len(golds), golds)
    best = float(F1.max()) if len(F1) else 0.0
    return best >= thresh

# -------------------- dtype / 加载模型（无 accelerate 依赖） --------------------
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # 新卡优先 bfloat16，不支持就 float16
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def load_tok_mdl(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # 有的 tokenizer 没有 pad，避免 generate 报 warning
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl

# -------------------- LLM-as-Judge（同样不依赖 accelerate） --------------------
_JUDGE_CACHE = {}
JUDGE_PROMPT = """You are a precise grader.
Only judge the FINAL_ANSWER string; IGNORE any hidden thoughts/reasoning if present.
Normalize: lowercase, remove punctuation/accents; minor typos are acceptable.
If FINAL_ANSWER semantically refers to ANY of the GOLD entities/aliases, answer YES; otherwise NO.

Return exactly one token: YES or NO.

QUESTION: {question}
GOLD: {golds}
FINAL_ANSWER: {final}

Decision (YES or NO):"""

def get_judge(model_id: str):
    if model_id in _JUDGE_CACHE:
        return _JUDGE_CACHE[model_id]
    tok, mdl = load_tok_mdl(model_id)
    _JUDGE_CACHE[model_id] = (tok, mdl)
    return tok, mdl

def judge_hit(judge_model_id: str, question: str, final: str, golds: List[str],
              max_new_tokens: int = 8, debug: bool = False) -> bool:
    tok, mdl = get_judge(judge_model_id)
    prompt = JUDGE_PROMPT.format(question=question, golds=", ".join(golds), final=final)
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    if debug:
        print(f"[LLM-JUDGE] {txt}")
    return "YES" in txt and "NO" not in txt

# -------------------- 读取 ZsRE（对齐你当前数据字段） --------------------
def load_zsre(path, n):
    raw = load_json(path)[:n]
    data = []
    for it in raw:
        q = it.get("rephrase") or it.get("rephrase_prompt") or it.get("src") or it.get("prompt") or ""
        data.append({
            "prompt": q,
            "answer": (it.get("ground_truth") or (it.get("answers", [None])[0])),
            "new_answer": (it.get("target_new") or it.get("alt") or ""),
        })
    return data

def get_golds(item, mode: str) -> List[str]:
    if mode == "new":
        golds = [item.get("new_answer","")]
    elif mode == "orig":
        golds = [item.get("answer","")]
    elif mode == "both":
        golds = [item.get("new_answer",""), item.get("answer","")]
    else:
        raise ValueError("gold_mode must be one of {new, orig, both}")
    return [g for g in golds if g]

# -------------------- 打印小工具 --------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _truncate(s: str, n: int = 160) -> str:
    s = _norm(s)
    return s if len(s) <= n else s[: n - 3] + "..."

# -------------------- 主评测流程 --------------------
def eval_model(
    model_path_or_id: str,
    data_path: str,
    n: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 64,
    print_every: int = 25,
    show_samples: int = 0,
    save_jsonl: str = "",
    gold_mode: str = "new",
    strict: bool = False,
    reveal_reasoning: bool = False,
    metric: str = "string",
    bert_thresh: float = 0.85,
    judge_model: str = "",
    reasoning_tokens: int = 512,
    two_pass_reasoning: bool = False,
    fuzzy: bool = True,
    fuzzy_thr: int = 90,
    debug_judge: bool = False,
):
    print(f"[INFO] Loading model: {model_path_or_id}")
    tok, mdl = load_tok_mdl(model_path_or_id)

    if torch.cuda.is_available():
        print(f"[INFO] CUDA devices: {torch.cuda.device_count()} | 0: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] dtype={mdl.dtype}  temp={temperature}  top_p={top_p}")
    print(f"[INFO] gold_mode={gold_mode}  strict={strict}")
    print(f"[INFO] metric={metric}  bert_thresh={bert_thresh}  judge_model={'(none)' if not judge_model else judge_model}")
    print(f"[INFO] fuzzy={fuzzy}  fuzzy_thr={fuzzy_thr}")
    if reveal_reasoning:
        print(f"[INFO] reveal_reasoning=True  reasoning_tokens={reasoning_tokens}  two_pass_reasoning={two_pass_reasoning}")

    if metric in ("judge", "hybrid") and not judge_model:
        raise RuntimeError("metric=judge/hybrid 需要 --judge_model")

    data = load_zsre(data_path, n)
    total = len(data)
    print(f"[INFO] Loaded {total} ZsRE cases from {data_path}")

    ok = 0
    start = time.time()

    fw = None
    if save_jsonl:
        path = os.path.abspath(os.path.expanduser(save_jsonl))
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 没有就新建
        fw = open(path, "w", encoding="utf-8")

    pbar = tqdm(range(total), ncols=100, desc="Evaluating ZsRE", unit="case")
    for i in pbar:
        it = data[i]
        q = it["prompt"]
        golds_raw = get_golds(it, gold_mode)
        golds = sanitize_aliases(golds_raw)

        # 1) 判分用短答
        short_ans = generate_answer(
            mdl, tok, f"Answer concisely.\nQ: {q}\nA:",
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            reveal_reasoning=False
        )
        a_for_score = short_ans

        # 2) 可选：再跑一遍长推理，仅记录
        final_for_record, reasoning_for_record, raw_long = "", "", ""
        if reveal_reasoning and two_pass_reasoning:
            raw_long = generate_answer(
                mdl, tok, f"Answer concisely.\nQ: {q}\nA:",
                max_new_tokens=reasoning_tokens, temperature=temperature, top_p=top_p,
                reveal_reasoning=True
            )
            try:
                ret_final, ret_reasoning = split_reasoning_final(raw_long)
            except Exception:
                ret_final, ret_reasoning = "", ""
            final_for_record, reasoning_for_record = ret_final, ret_reasoning

        # 3) 命中判断
        if strict:
            An = normalize_text(a_for_score)
            golds_text = [normalize_text(g) for g in golds if normalize_text(g)]
            hit = int(An in set(golds_text))
        else:
            if metric == "string":
                hit = int(contains_any(a_for_score, golds) or (fuzzy and fuzzy_hit(a_for_score, golds, fuzzy_thr)))
            elif metric == "bert":
                hit = int(bert_hit_cached(a_for_score, golds, thresh=bert_thresh))
            elif metric == "judge":
                a_for_judge = final_for_record if final_for_record else a_for_score
                hit = int(judge_hit(judge_model, q, a_for_judge, golds, debug=debug_judge))
            else:  # hybrid
                a_for_judge = final_for_record if final_for_record else a_for_score
                hit = int(
                    contains_any(a_for_score, golds) or
                    (fuzzy and fuzzy_hit(a_for_score, golds, fuzzy_thr)) or
                    bert_hit_cached(a_for_score, golds, thresh=bert_thresh) or
                    judge_hit(judge_model, q, a_for_judge, golds, debug=debug_judge)
                )

        ok += hit
        acc = ok / (i + 1)
        pbar.set_postfix(acc=f"{acc:.3f}")

        # 打印样例
        if (show_samples and i < show_samples) or ((i + 1) % max(1, print_every) == 0):
            print(f"\n[SAMPLE {i+1}] hit={hit}")
            print(f"  Q: {_truncate(q, 200)}")
            print(f"  Short: {_truncate(short_ans, 200)}")
            if reveal_reasoning:
                print(f"  Reasoning: {_truncate(reasoning_for_record, 200)}")
                print(f"  Final:     {_truncate(final_for_record, 200)}")
            print(f"  GOLD(s): {', '.join(golds)}")

        # 写结果
        if fw:
            qa_rec = {"q": q, "short": short_ans}
            if reveal_reasoning:
                qa_rec.update({"reasoning": reasoning_for_record, "final": final_for_record, "raw": raw_long})
            fw.write(json.dumps({
                "idx": i + 1,
                "hit": hit,
                "per_q_hits": hit,   # 单问，复用字段名便于通用聚合
                "golds": golds,
                "qas": [qa_rec],
            }, ensure_ascii=False) + "\n")
            fw.flush()

    if fw:
        fw.close()

    elapsed = time.time() - start
    print("\n====== SUMMARY (ZsRE) ======")
    print(f"Model: {model_path_or_id}")
    print(f"Cases: {total} | Acc: {ok/total:.3f} ({ok}/{total})")
    print(f"Time: {elapsed:.1f}s | {total/max(1,elapsed):.2f} cases/s")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--num_cases", type=int, default=200)
    ap.add_argument("--gold_mode", choices=["orig", "new", "both"], default="new")

    # 生成与显示
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--reasoning_tokens", type=int, default=512)
    ap.add_argument("--two_pass_reasoning", action="store_true")
    ap.add_argument("--show_samples", type=int, default=0)
    ap.add_argument("--save_jsonl", type=str, default="")
    ap.add_argument("--reveal_reasoning", action="store_true")

    # 匹配口径
    ap.add_argument("--strict", action="store_true")

    # 度量方式
    ap.add_argument("--metric", choices=["string", "bert", "judge", "hybrid"], default="string")
    ap.add_argument("--bert_thresh", type=float, default=0.85)
    ap.add_argument("--judge_model", type=str, default="")
    ap.add_argument("--fuzzy", action="store_true")
    ap.add_argument("--no-fuzzy", dest="fuzzy", action="store_false")
    ap.add_argument("--fuzzy_thr", type=int, default=90)
    ap.add_argument("--debug_judge", action="store_true")

    args = ap.parse_args()
    eval_model(
        model_path_or_id=args.model,
        data_path=args.data_path,
        n=args.num_cases,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        show_samples=args.show_samples,
        save_jsonl=args.save_jsonl,
        gold_mode=args.gold_mode,
        strict=args.strict,
        reveal_reasoning=args.reveal_reasoning,
        metric=args.metric,
        bert_thresh=args.bert_thresh,
        judge_model=args.judge_model,
        reasoning_tokens=args.reasoning_tokens,
        two_pass_reasoning=args.two_pass_reasoning,
        fuzzy=args.fuzzy,
        fuzzy_thr=args.fuzzy_thr,
        debug_judge=args.debug_judge,
    )

if __name__ == "__main__":
    main()
