import json, os, re, torch
from typing import List, Dict, Any, Iterable, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- IO ----------
def load_mquake_json(path: str) -> List[Dict[str, Any]]:
    """支持 .json / .jsonl"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":  # JSON
            return json.load(f)
        else:            # JSONL
            return [json.loads(line) for line in f if line.strip()]

# ---------- 文本与匹配（支持 emoji） ----------
def normalize_text(s: str) -> str:
    # 仅保留字母数字与空格，其他归一为空格，便于做词边界匹配
    return re.sub(r"\W+", " ", (s or "").strip().lower())

def sanitize_aliases(aliases: Iterable[str]) -> list:
    """
    过滤容易误判的别名：
    - 文本归一化后长度 <= 2 且全字母（如 it, us, uk）丢弃
    - emoji / 含非字母字符（比如 '🇮🇹', 'U.S.'）保留
    """
    safe = []
    for n in aliases:
        if not n:
            continue
        g = normalize_text(n)
        if g and len(g) <= 2 and g.isalpha():
            continue
        safe.append(n)
    return safe

def contains_any(hay: str, needles: Iterable[str]) -> bool:
    """
    - 字词类别名：在 normalize 后文本上做词边界匹配
    - 非字词（emoji/标点序列等，normalize 后为空）：在原文上做原样匹配
    """
    raw = hay or ""
    H = normalize_text(raw)
    for n in needles:
        if not n:
            continue
        g = normalize_text(n)
        if g:  # 词边界匹配
            if re.search(rf"(?:^|\s){re.escape(g)}(?:\s|$)", H):
                return True
        else:  # 纯 emoji / 标点：原样匹配
            if n in raw:
                return True
    return False

# ---------- HF 模型加载 ----------
def load_hf_model(model_name_or_path: str, dtype="bfloat16"):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return tok, mdl

def free_cuda(*objs):
    for o in objs:
        try: del o
        except: pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- 生成：支持“仅最终答案 / 显示推理+最终答案” ----------
def _build_messages(prompt: str, reveal_reasoning: bool) -> list:
    if reveal_reasoning:
        # 用标签辅助解析
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
                    max_new_tokens=128, temperature=0.0, top_p=1.0,
                    reveal_reasoning: bool = False) -> str:
    """
    对 Instruct/Chat 模型优先走 chat 模板；
    对 base 模型退化为普通续写。
    —— 不做“首行即停”等强截断；是否长输出交给调用处的 max_new_tokens 控制。
    """
    # 统一 pad_token，消除警告
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            if tokenizer.unk_token_id is None:
                tokenizer.add_special_tokens({"unk_token": "<unk>"})
            tokenizer.pad_token_id = tokenizer.unk_token_id
    try:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    messages = _build_messages(prompt, reveal_reasoning=reveal_reasoning)

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # 退化到纯文本
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
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        no_repeat_ngram_size=3  # 稍抑制自述重复
    )
    ans = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return ans.strip()

# ---------- 解析：把推理与最终答案分开 ----------
_TAG_REASON = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
_TAG_FINAL   = re.compile(r"<final>\s*(.*?)\s*</final>",         re.IGNORECASE | re.DOTALL)

def _clean(s: str) -> str:
    return (s or "").strip().strip("：:").strip()

def split_reasoning_final(text: str) -> Tuple[str, str]:
    """
    返回 (final, reasoning)
    解析优先级：
      1) <reasoning>...</reasoning> 与 <final>...</final>
      2) <think>...</think> + 'Final:' / '最终答案：'
      3) 只出现 'Final:' / '最终答案：'
      4) 兜底：把末行当 final，其余当 reasoning
    """
    raw = text or ""

    # 1) XML 标签
    m_f = _TAG_FINAL.search(raw)
    m_r = _TAG_REASON.search(raw)
    if m_f:
        final = _clean(m_f.group(1))
        reasoning = _clean(m_r.group(1)) if m_r else _clean(_TAG_REASON.sub("", raw))
        return final, reasoning

    # 2) DeepSeek 常见 <think>... </think> + Final:
    m_think = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.IGNORECASE | re.DOTALL)
    m_final_line = re.search(r"(?im)^\s*(final(?: answer)?|最终答案)\s*[:：]\s*(.+)\s*$", raw)
    if m_final_line:
        final = _clean(m_final_line.group(2))
        reasoning = _clean(m_think.group(1)) if m_think else _clean(raw[:m_final_line.start()])
        return final, reasoning

    # 3) 只有 Final: / 最终答案：
    m_final_alone = re.search(r"(?im)(final(?: answer)?|最终答案)\s*[:：]\s*(.+)$", raw)
    if m_final_alone:
        return _clean(m_final_alone.group(2)), _clean(raw[:m_final_alone.start()])

    # 4) 兜底：取最后一行或最后一句作为 final
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        final_guess = lines[-1]
        reasoning_guess = "\n".join(lines[:-1])
        return _clean(final_guess), _clean(reasoning_guess)

    return "", ""  # 空输出
