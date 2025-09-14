# eval_mquake_multihop.py
import os, argparse, time, json, re, unicodedata, warnings
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from utils_mquake import (
    load_mquake_json, generate_answer, split_reasoning_final,
    contains_any, sanitize_aliases, normalize_text
)

# =============== Utils: 规范化 / 模糊匹配 =================

def strip_accents(s: str) -> str:
    # 去重音（é -> e；Í -> I）
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
    """模糊匹配：子串优先；否则 rapidfuzz 相似度阈值"""
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

# =============== BERTScore (单例缓存) =================

_BERT_SCORER = None
def get_bertscorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        try:
            from bert_score import BERTScorer
        except ImportError as e:
            raise RuntimeError("please pip install bert-score") from e
        _BERT_SCORER = BERTScorer(
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _BERT_SCORER

def bert_hit_cached(pred: str, golds: List[str], thresh: float = 0.85) -> bool:
    scorer = get_bertscorer()
    # 与 golds 分别算分，取最大
    P, R, F1 = scorer.score([pred] * len(golds), golds)
    best = float(F1.max()) if len(F1) else 0.0
    return best >= thresh

# =============== LLM-as-Judge (单例缓存 & 只看 FINAL) ==============

_JUDGE_CACHE = {}  # model_id -> (tok, mdl)

def get_judge(model_id: str):
    if model_id in _JUDGE_CACHE:
        return _JUDGE_CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    _JUDGE_CACHE[model_id] = (tok, mdl)
    return tok, mdl

JUDGE_PROMPT = """You are a precise grader.
Only judge the FINAL_ANSWER string; IGNORE any hidden thoughts/reasoning if present.
Normalize: lowercase, remove punctuation/accents; minor typos are acceptable.
If FINAL_ANSWER semantically refers to ANY of the GOLD entities/aliases, answer YES; otherwise NO.

Return exactly one token: YES or NO.

QUESTION: {question}
GOLD: {golds}
FINAL_ANSWER: {final}

Decision (YES or NO):"""

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

# =============== 其他小工具 =================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _truncate(s: str, n: int = 160) -> str:
    s = _norm(s)
    return s if len(s) <= n else s[: n - 3] + "..."

def get_golds(item: dict, mode: str) -> List[str]:
    if mode == "new":
        golds = [item.get("new_answer", "")] + item.get("new_answer_alias", [])
    elif mode == "orig":
        golds = [item.get("answer", "")] + item.get("answer_alias", [])
    elif mode == "both":
        golds = (
            [item.get("new_answer", "")] + item.get("new_answer_alias", []) +
            [item.get("answer", "")] + item.get("answer_alias", [])
        )
    else:
        raise ValueError("gold_mode must be one of {new, orig, both}")
    return [g for g in golds if g]

# =============== 主评测 =================

def eval_model(
    model_path_or_id: str,
    data_path: str,
    n: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 64,          # 判分用短答
    print_every: int = 25,
    show_samples: int = 0,
    save_jsonl: str = "",
    gold_mode: str = "new",
    strict: bool = False,
    require_k: int = 1,
    reveal_reasoning: bool = False,    # 仅记录推理
    metric: str = "string",            # string / bert / judge / hybrid
    bert_thresh: float = 0.85,
    judge_model: str = "",
    reasoning_tokens: int = 512,       # 推理答
    two_pass_reasoning: bool = True,   # 判分与推理分离
    fuzzy: bool = True,
    fuzzy_thr: int = 90,
    debug_judge: bool = False,
):
    print(f"[INFO] Loading model: {model_path_or_id}")
    tok = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path_or_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    if torch.cuda.is_available():
        print(f"[INFO] CUDA devices: {torch.cuda.device_count()} | 0: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] dtype={mdl.dtype}  temp={temperature}  top_p={top_p}")
    print(f"[INFO] gold_mode={gold_mode}  strict={strict}  require_k={require_k}")
    print(f"[INFO] metric={metric}  bert_thresh={bert_thresh}  judge_model={'(none)' if not judge_model else judge_model}")
    print(f"[INFO] fuzzy={fuzzy}  fuzzy_thr={fuzzy_thr}")
    if reveal_reasoning:
        print(f"[INFO] reveal_reasoning=True  reasoning_tokens={reasoning_tokens}  two_pass_reasoning={two_pass_reasoning}")

    if metric in ("judge", "hybrid") and not judge_model:
        raise RuntimeError("metric=judge/hybrid 需要 --judge_model")

    data = load_mquake_json(data_path)[:n]
    total = len(data)
    print(f"[INFO] Loaded {total} cases from {data_path}")

    ok = 0
    start = time.time()

    fw = None
    if save_jsonl:
        path = os.path.abspath(os.path.expanduser(save_jsonl))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fw = open(path, "w", encoding="utf-8")

    pbar = tqdm(range(total), ncols=100, desc="Evaluating", unit="case")
    for i in pbar:
        item = data[i]
        golds_raw = get_golds(item, gold_mode)
        golds = sanitize_aliases(golds_raw)  # 先按你现有的清洗规则处理一轮

        per_q_hits = 0
        qa_pairs = []

        for q in item["questions"]:
            # 1) 判分用“短答”（快且稳定）
            short_ans = generate_answer(
                mdl, tok, f"Answer concisely.\nQ: {q}\nA:",
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                reveal_reasoning=False
            )
            a_for_score = short_ans  # 默认用短答评分

            # 2) （可选）单独跑“长推理”用于记录，不影响判分
            final_for_record, reasoning_for_record, raw_long = "", "", ""
            if reveal_reasoning and two_pass_reasoning:
                raw_long = generate_answer(
                    mdl, tok, f"Answer concisely.\nQ: {q}\nA:",
                    max_new_tokens=reasoning_tokens, temperature=temperature, top_p=top_p,
                    reveal_reasoning=True
                )
                # 注意 split_reasoning_final 的返回顺序（你本地实现可能是 (final, reasoning)）
                try:
                    ret_final, ret_reasoning = split_reasoning_final(raw_long)
                except Exception:
                    # 容错：若解析失败，全部置空
                    ret_final, ret_reasoning = "", ""
                final_for_record, reasoning_for_record = ret_final, ret_reasoning

            # 3) 命中逻辑（严格等值 / 非严格四段式）
            if strict:
                An = normalize_text(a_for_score)
                golds_text = [normalize_text(g) for g in golds if normalize_text(g)]
                hit_this = An in set(golds_text)
            else:
                if metric == "string":
                    # 字符串包含 + （可选）模糊
                    hit_this = contains_any(a_for_score, golds) or (fuzzy and fuzzy_hit(a_for_score, golds, fuzzy_thr))
                elif metric == "bert":
                    hit_this = bert_hit_cached(a_for_score, golds, thresh=bert_thresh)
                elif metric == "judge":
                    # 判官只看 FINAL；若没有 FINAL，则退回短答
                    a_for_judge = final_for_record if final_for_record else a_for_score
                    hit_this = judge_hit(judge_model, q, a_for_judge, golds, debug=debug_judge)
                else:  # hybrid
                    a_for_judge = final_for_record if final_for_record else a_for_score
                    hit_this = (
                        contains_any(a_for_score, golds) or
                        (fuzzy and fuzzy_hit(a_for_score, golds, fuzzy_thr)) or
                        bert_hit_cached(a_for_score, golds, thresh=bert_thresh) or
                        judge_hit(judge_model, q, a_for_judge, golds, debug=debug_judge)
                    )

            if hit_this:
                per_q_hits += 1

            # 4) 记录
            rec = {"q": q, "short": short_ans}
            if reveal_reasoning:
                if two_pass_reasoning:
                    rec.update({"reasoning": reasoning_for_record, "final": final_for_record, "raw": raw_long})
                else:
                    rec.update({"reasoning": "", "final": "", "raw": ""})
            qa_pairs.append(rec)

        sample_hit = 1 if per_q_hits >= require_k else 0
        ok += sample_hit
        acc = ok / (i + 1)
        pbar.set_postfix(acc=f"{acc:.3f}")

        if print_every and ((i + 1) % print_every == 0):
            first = qa_pairs[0] if qa_pairs else {}
            ans_preview = first.get("short") or first.get("final") or ""
            print(f"\n[STEP {i+1}/{total}] hit={sample_hit} (per_q={per_q_hits})  running_acc={acc:.3f}")
            print(f"  Q: {_truncate(first.get('q',''))}")
            print(f"  A: {_truncate(ans_preview)}")
            if golds:
                print(f"  GOLD(s): {_truncate(', '.join(golds), 120)}")

        if show_samples and (i < show_samples):
            print(f"\n[SAMPLE {i+1}] hit={sample_hit}  per_q={per_q_hits}")
            for j, qa in enumerate(qa_pairs, 1):
                print(f"  Q{j}: {_truncate(qa.get('q',''), 200)}")
                print(f"  Short{j}: {_truncate(qa.get('short',''), 200)}")
                if reveal_reasoning:
                    print(f"  Reasoning{j}: {_truncate(qa.get('reasoning',''), 200)}")
                    print(f"  Final{j}:     {_truncate(qa.get('final',''), 200)}")
            if golds:
                print(f"  GOLD(s): {', '.join(golds)}")

        if fw:
            fw.write(json.dumps({
                "idx": i + 1,
                "hit": sample_hit,
                "per_q_hits": per_q_hits,
                "golds": golds,
                "qas": qa_pairs,
            }, ensure_ascii=False) + "\n")
            fw.flush()

    elapsed = time.time() - start
    if fw: fw.close()

    print("\n====== SUMMARY ======")
    print(f"Model: {model_path_or_id}")
    print(f"Cases: {total}")
    print(f"Multi-hop@{total}: {ok/total:.3f}  ({ok}/{total})")
    print(f"Time: {elapsed:.1f}s  |  {total/elapsed:.2f} cases/s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF 模型ID或本地编辑后目录")
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--num_cases", type=int, default=200)
    ap.add_argument("--gold_mode", type=str, default="new", choices=["new","orig","both"])

    # 生成与显示（判分短答与推理长答分离）
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=64, help="判分用短答的最大新标记数（建议 16~64）")
    ap.add_argument("--reasoning_tokens", type=int, default=9192, help="推理长答最大新标记数")
    ap.add_argument("--two_pass_reasoning", action="store_true", help="先短答判分，再长答仅记录推理")

    ap.add_argument("--print_every", type=int, default=25, help="每多少条样本打印一次简要进度日志（0=不打印）")
    ap.add_argument("--show_samples", type=int, default=0, help="打印前 N 条样本的完整 Q/A")
    ap.add_argument("--save_jsonl", type=str, default="", help="逐样本结果写入该 jsonl 文件路径")
    ap.add_argument("--reveal_reasoning", action="store_true", help="记录推理（建议配合 --two_pass_reasoning）")

    # 匹配口径
    ap.add_argument("--strict", action="store_true", help="严格等值匹配（默认关闭）")
    ap.add_argument("--require_k", type=int, default=1, help="至少命中多少问才算该样本 hit=1（默认 1）")

    # 度量方式
    ap.add_argument("--metric", type=str, default="string", choices=["string","bert","judge","hybrid"])
    ap.add_argument("--bert_thresh", type=float, default=0.85)
    ap.add_argument("--judge_model", type=str, default="", help="裁判模型（metric=judge/hybrid 必填）")

    # 新增：模糊匹配 & 调试
    ap.add_argument("--fuzzy", action="store_true", help="开启模糊匹配（默认开启）")
    ap.add_argument("--no-fuzzy", dest="fuzzy", action="store_false", help="关闭模糊匹配")
    ap.add_argument("--fuzzy_thr", type=int, default=90, help="rapidfuzz 阈值（0-100）")
    ap.add_argument("--debug_judge", action="store_true", help="打印判官原始输出")

    args = ap.parse_args()
    eval_model(
        model_path_or_id=args.model,
        data_path=args.data_path,
        n=args.num_cases,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        print_every=args.print_every,
        show_samples=args.show_samples,
        save_jsonl=args.save_jsonl,
        gold_mode=args.gold_mode,
        strict=args.strict,
        require_k=args.require_k,
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
