# -*- coding: utf-8 -*-
import json, os, re, unicodedata, torch
from typing import List, Dict, Any, Iterable, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------ 读取/统一格式 ------------------------
def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":   # JSON
            return json.load(f)
        else:             # JSONL
            return [json.loads(line) for line in f if line.strip()]

def _to_list(x):
    if not x: return []
    return x if isinstance(x, list) else [x]

def _cloze_to_question(prompt: str, subject: str) -> str:
    # 简单把完形模板转问句："{subj} was born in" → "Where was {subj} born?"
    s = (prompt or "").replace("{}", subject or "").strip()
    if not s.endswith("?"): s += "?"
    return s

def load_zsre_items(path: str) -> List[Dict[str, Any]]:
    """
    支持两种常见格式并统一为：
      { "questions": [q1, q2, ...], "golds_orig": [...], "golds_new": [...] }
    A) ROME/EasyEdit 的 ZSRE-MEND：含 requested_rewrite[0]，带 target_true/target_new/ question / paraphrase
    B) 纯 QA：{"question": "...", "answers": ["..."]} 或 {"question": "...", "answer": "..."}
    """
    raw = _read_json_or_jsonl(path)
    items = []
    for r in raw:
        # A) requested_rewrite
        if "requested_rewrite" in r and r["requested_rewrite"]:
            rr = r["requested_rewrite"][0]
            q0 = rr.get("question") or _cloze_to_question(rr.get("prompt",""), rr.get("subject",""))
            parap = r.get("paraphrase") or rr.get("paraphrase") or []
            questions = [q0] + [q for q in parap if q]
            golds_orig = _to_list(rr.get("target_true",{}).get("str"))
            golds_new  = _to_list(rr.get("target_new",{}).get("str"))
            items.append({
                "questions": questions,
                "golds_orig": golds_orig,
                "golds_new": golds_new
            })
            continue

        # B) 纯 QA
        if "question" in r:
            q0 = r["question"]
            parap = r.get("paraphrase") or []
            questions = [q0] + [q for q in parap if q]
            # 兼容 answers / answer / labels
            golds_orig = _to_list(r.get("answers") or r.get("answer") or r.get("labels"))
            items.append({
                "questions": questions,
                "golds_orig": golds_orig,
                "golds_new": []
            })
            continue

        # 其余结构忽略
    return items

# ------------------------ 文本/匹配（emoji 友好） ------------------------
def normalize_text(s: str) -> str:
    s = (s or "")
    # 先做一些常见符号的统一，兼容“D+C / D·C / D．C”等
    s = s.replace("＋", "+").replace("+", ".").replace("·", ".").replace("．", ".")
    s = unicodedata.normalize("NFKC", s)      # 全/半角统一
    s = re.sub(r"\W+", " ", s.strip().lower())
    return s

def sanitize_aliases(aliases: Iterable[str]) -> list:
    safe = []
    for n in aliases or []:
        if not n: continue
        g = normalize_text(n)
        # 丢掉易误判的 2 字母纯字母缩写（us/uk/it 等），emoji/符号不受影响
        if g and len(g) <= 2 and g.isalpha():
            continue
        safe.append(n)
    return safe

def contains_any(hay: str, needles: Iterable[str]) -> bool:
    raw = hay or ""
    H = normalize_text(raw)
    for n in needles or []:
        if not n: continue
        g = normalize_text(n)
        if g:  # 词边界匹配，避免 'sing' 命中 'singer'
            if re.search(rf"(?:^|\s){re.escape(g)}(?:\s|$)", H):
                return True
        else:  # 纯 emoji/符号：原样匹配
            if n in raw:
                return True
    return False

# ------------------------ 生成：与 MQuAKE 版一致 ------------------------
def _build_messages(prompt: str, reveal_reasoning: bool) -> list:
    if reveal_reasoning:
        sys_msg = (
            "You are a helpful assistant. "
            "Think step by step INSIDE <reasoning>...</reasoning>. "
            "Then output ONLY the final short answer INSIDE <final>...</final>. "
            "Do not include anything else outside these tags."
        )
    else:
        sys_msg = "Only output the final short answer. Do NOT explain."
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": prompt}
    ]

def generate_answer(model, tokenizer, prompt: str,
                    max_new_tokens=64, temperature=0.0, top_p=1.0,
                    reveal_reasoning: bool = False) -> str:
    messages = _build_messages(prompt, reveal_reasoning=reveal_reasoning)

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = (messages[0]["content"] + "\nQ: " + prompt + "\nA:")

    enc = tokenizer(text, return_tensors="pt").to(model.device)

    # 收集可能的 EOS
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
        temperature=temperature, top_p=top_p,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    )
    ans = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return ans.strip()

# ------------------------ 解析：分离推理与最终答案 ------------------------
_TAG_REASON = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
_TAG_FINAL   = re.compile(r"<final>\s*(.*?)\s*</final>",         re.IGNORECASE | re.DOTALL)

def _clean(s: str) -> str:
    return (s or "").strip().strip("：:").strip()

def split_reasoning_final(text: str) -> Tuple[str, str]:
    """
    返回 (final, reasoning)
    优先：
      1) <reasoning>...</reasoning> 与 <final>...</final>
      2) <think>...</think> + 'Final:' / '最终答案：'
      3) 只有 'Final:' / '最终答案：'
      4) 兜底：最后一行视作 final
    """
    raw = text or ""

    m_f = _TAG_FINAL.search(raw)
    m_r = _TAG_REASON.search(raw)
    if m_f:
        final = _clean(m_f.group(1))
        reasoning = _clean(m_r.group(1)) if m_r else _clean(_TAG_REASON.sub("", raw))
        return final, reasoning

    m_think = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.IGNORECASE | re.DOTALL)
    m_final_line = re.search(r"(?im)^\s*(final(?: answer)?|最终答案)\s*[:：]\s*(.+)\s*$", raw)
    if m_final_line:
        final = _clean(m_final_line.group(2))
        reasoning = _clean(m_think.group(1)) if m_think else _clean(raw[:m_final_line.start()])
        return final, reasoning

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return _clean(lines[-1]), _clean("\n".join(lines[:-1]))
    return "", ""
