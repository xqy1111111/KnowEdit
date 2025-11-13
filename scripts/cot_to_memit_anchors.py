#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CoT-style ZSRE JSON/JSONL into MEMIT anchor requests.

Input records (JSON/JSONL) typically contain:
  - subject: string
  - src or prompt: the base question
  - alt or alt_answer: CoT text with an "Answer: ..." tail or the new target string
  - answers/ground_truth: original gold(s), optional

This script produces multiple anchor requests per case, suitable for MEMIT:
  {"prompt": "... {} ...", "subject": "...", "target_new": "...", "case_id": "..."}

By default, it uses an LLM (OpenAI) to generate diverse, CoT‑guided anchor
prompts that contain a single '{}' placeholder for the subject. Rephrases are
post‑processed with a token‑level Jaccard filter to reduce overlap. If
OPENAI_API_KEY is not set, it falls back to a deterministic set of safe
rephrases.

Environment (.env):
  OPENAI_API_KEY=sk-...
  OPENAI_BASE_URL=... (optional)
  OPENAI_MODEL=gpt-4.1 (default from .env.template)

Usage:
  python scripts/cot_to_memit_anchors.py --in data/cot.jsonl \
      --out_jsonl data/memit_anchors.jsonl --n_anchors 6

The output JSONL will contain one line per anchor. Anchors of the same case
share the same case_id so you can group them downstream.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _extract_answer_from_text(text: str) -> str:
    if not text:
        return ""
    # tags first
    m = re.search(r"<\s*(final|answer)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(2).strip().strip("：:")
    # Answer: ... / Final: ...
    last = None
    for mm in re.finditer(r"(?im)^(?:final(?:\s*answer)?|answer)\s*[:：]\s*(.+)$", text):
        last = mm
    if last:
        return last.group(1).strip()
    # fallback last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _split_sentences(text: str) -> List[str]:
    """Very light sentence splitter; avoids heavy deps.
    Splits on Chinese/English sentence punctuation and trims whitespace.
    """
    if not text:
        return []
    # Normalize line breaks
    s = re.sub(r"[\r\n]+", " ", text)
    # Split on ., ?, !, 。, ！, ？ and semicolons
    parts = re.split(r"(?<=[\.!?。！？；;])\s+", s)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 8:
            continue
        out.append(p)
    return out


def _ensure_placeholder(prompt: str, subject: str) -> str:
    """Ensure the prompt contains a single '{}' placeholder.
    If subject occurs verbatim, replace the first occurrence with '{}'.
    Otherwise, append ' (Subject: {})'.
    """
    if "{}" in prompt:
        return prompt
    if subject and subject in prompt:
        return prompt.replace(subject, "{}", 1)
    return f"{prompt} (Subject: {{}})"


def _token_set(s: str) -> set:
    toks = re.findall(r"[\w\u4e00-\u9fff]+", s.lower())
    return set(toks)


def _select_non_overlapping(sents: List[str], k: int, jaccard_thr: float = 0.6) -> List[str]:
    chosen: List[str] = []
    chosen_sets: List[set] = []
    for s in sents:
        ts = _token_set(s)
        if not ts:
            continue
        keep = True
        for cs in chosen_sets:
            inter = len(ts & cs)
            union = len(ts | cs) or 1
            if (inter / union) >= jaccard_thr:
                keep = False
                break
        if keep:
            chosen.append(s)
            chosen_sets.append(ts)
            if len(chosen) >= k:
                break
    return chosen


def _simple_rephrases(base: str) -> List[str]:
    # Very conservative fallback rephrases (no LLM). Keep '{}' out of rephrase; we add it later.
    templates = [
        "What is the mother tongue of {}?",
        "What is {}'s native language?",
        "Which language is {}'s first language?",
        "What language did {} learn first?",
        "Which language does {} speak natively?",
        "What is {}'s primary language?",
    ]
    return templates


def _build_llm_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    base = os.getenv("OPENAI_BASE_URL", "").strip() or None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base:
        kwargs["base_url"] = base
    try:
        client = OpenAI(**kwargs)
        return client
    except Exception:
        return None


def _llm_rephrases(client: Any, model: str, subject: str, src_prompt: str, cot_text: str, n_anchors: int, verbose: bool = False) -> List[str]:
    # CoT-guided, diversity-enforced rephrase request. We ask the LLM to
    # extract distinct hooks from the reasoning and craft non-overlapping
    # question templates that still ask the SAME thing as the base question.
    sys_msg = (
        "You rewrite questions. Return ONLY a JSON object with an array 'anchors' of "
        "distinct question rephrasings. Each anchor MUST contain exactly one '{}' placeholder "
        "for the subject and ask the same question as the base. Do not include any other keys."
    )
    user_msg = (
        "Subject: " + subject + "\n" +
        "Base question: " + src_prompt + "\n" +
        "Chain-of-thought (use it to derive distinct angles):\n" + cot_text + "\n\n" +
        f"Task: Propose {n_anchors} rephrasings that all ask the SAME question while being mutually non-overlapping. "
        "Guidelines: \n"
        "- Extract different angles/synonyms/constraints hinted by the reasoning (e.g., paraphrases, syntactic variants).\n"
        "- Avoid lexical overlap across anchors: do not reuse 3+ token sequences; minimize shared bigrams.\n"
        "- Keep each anchor concise (8–18 words) and natural.\n"
        "- Include exactly one '{}' placeholder for the subject; DO NOT insert the actual subject value.\n"
        "- Do NOT output numbering, explanations, or keys other than 'anchors'.\n\n"
        "Return JSON only: {\"anchors\": [\"...{}...\", ...]}"
    )
    try:
        resp = client.responses.create(
            model=model,
            temperature=0.3,
            max_output_tokens=512,
            input=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        )
        text = resp.output_text if hasattr(resp, "output_text") else None
        if verbose:
            print("[llm] got response text:\n" + (text or "<empty>"))
    except Exception as exc:
        sys.stderr.write(f"[LLM] request failed: {exc}\n")
        return []
    if not text:
        return []
    # Try to locate JSON (handle fenced code blocks and trailing backticks)
    # 1) Fenced code block ```json ... ```
    block = re.search(r"```\s*json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if not block:
        block = re.search(r"```\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if block:
        raw = block.group(1)
    else:
        # 2) Any JSON object substring
        m = re.search(r"\{[\s\S]*\}", text)
        raw = (m.group(0) if m else text.strip())
    if verbose:
        print("[llm] candidate JSON block:\n" + raw)
    try:
        obj = json.loads(raw)
        anchors = obj.get("anchors", [])
        if not isinstance(anchors, list):
            if verbose:
                print("[llm] JSON parsed but 'anchors' not a list; keys:", list(obj.keys()))
            return []
        # Ensure placeholder exists and deduplicate
        norm = []
        for a in anchors:
            if isinstance(a, str) and "{}" in a:
                s = a.strip()
                if s not in norm:
                    norm.append(s)
        if verbose:
            print("[llm] parsed anchors:", norm)
        return norm[:n_anchors]
    except Exception as e:
        if verbose:
            print("[llm] failed to parse JSON from response; error:", e)
        return []

def _pair_jaccard(a: str, b: str) -> float:
    sa = _token_set(a)
    sb = _token_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / (len(sa | sb) or 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CoT JSON/JSONL path")
    ap.add_argument("--out_jsonl", required=True, help="Output MEMIT anchors JSONL path")
    ap.add_argument("--n_anchors", type=int, default=6, help="Anchors per case")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit cases (0=all)")
    ap.add_argument(
        "--anchor_mode",
        choices=["rephrase", "cot_prefix", "cot_strict"],
        default="cot_prefix",
        help=(
            "How to form anchors: rephrase | cot_prefix | cot_strict. "
            "rephrase uses an LLM with CoT guidance + diversity filtering; "
            "cot_strict uses ordered, non-overlapping CoT sentences as raw prefixes (no connectors)."
        ),
    )
    ap.add_argument("--verbose", action="store_true", help="Print debug info about .env and fallback paths")
    args = ap.parse_args()

    if load_dotenv is not None:
        try:
            # Robustly locate .env from project root
            env_path = find_dotenv() if find_dotenv is not None else ""
            if env_path:
                load_dotenv(env_path, override=False)
            else:
                load_dotenv(override=False)
        except Exception:
            if args.verbose:
                print("[dotenv] Failed to load .env (continuing without).")

    client = _build_llm_client()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1").strip() or "gpt-4.1"
    if args.verbose:
        print(f"[env] OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
        if os.getenv('OPENAI_BASE_URL'):
            print(f"[env] OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")
        print(f"[env] OPENAI_MODEL: {model}")
        print(f"[mode] anchor_mode={args.anchor_mode}")

    rows = _read_json_or_jsonl(args.inp)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]

    out_path = os.path.abspath(os.path.expanduser(args.out_jsonl))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fw = open(out_path, "w", encoding="utf-8")
    n_written = 0

    for idx, r in enumerate(rows):
        subject = (r.get("subject") or r.get("sub_label") or r.get("subject_entity") or "").strip()
        src = (r.get("src") or r.get("prompt") or "").strip()
        if not subject or not src:
            continue

        # Extract target_new (alt_answer > target_new > parse alt)
        target_new = (r.get("alt_answer") or r.get("target_new") or "").strip()
        if not target_new:
            target_new = _extract_answer_from_text((r.get("alt") or "").strip())
        if not target_new:
            continue

        # Prepare anchors
        cot_text = (r.get("alt") or "").strip()
        base_templ = _ensure_placeholder(src, subject)

        anchors: List[str] = []
        if args.anchor_mode == "cot_prefix" and cot_text:
            # Derive anchors from CoT sentences as informative prefixes
            sents = _split_sentences(cot_text)
            # Cap sentence length to keep prompts concise
            sents = [re.sub(r"\s+", " ", s).strip()[:160] for s in sents]
            # Use a variety of connectives
            connectors = [
                "Given that {} , ",
                "Considering {} , ",
                "According to {} , ",
                "From the fact that {} , ",
                "Based on {} , ",
                "In view of {} , ",
            ]
            uniq: List[str] = []
            # Always include the base template as one anchor
            uniq.append(base_templ)
            for i, s in enumerate(sents):
                conn = connectors[i % len(connectors)]
                prefix = conn.format(s)
                templ = (prefix + base_templ).strip()
                if templ not in uniq:
                    uniq.append(templ)
                if len(uniq) >= args.n_anchors:
                    break
            anchors = uniq[: args.n_anchors]
        elif args.anchor_mode == "cot_strict" and cot_text:
            sents = _split_sentences(cot_text)
            sents = [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]
            # dedupe preserving order
            seen = set(); ordered = []
            for s in sents:
                if s not in seen:
                    seen.add(s); ordered.append(s)
            picked = _select_non_overlapping(ordered, args.n_anchors, jaccard_thr=0.6)
            anchors = [f"{s} {base_templ}".strip() for s in picked]
            if args.verbose:
                print("[cot_strict] picked sentences:", picked)
                print("[cot_strict] anchors count:", len(anchors))
        else:
            # Rephrase mode (LLM → fallback) with diversity filtering guided by CoT
            if client is not None:
                llm_out = _llm_rephrases(
                    client, model, subject, src, cot_text, args.n_anchors, verbose=args.verbose
                )
            else:
                llm_out = []

            # Start with base template
            anchors_build: List[str] = [base_templ]

            # Filter LLM outputs: ensure placeholder, filter too-similar to base, then enforce pairwise diversity
            candidates: List[str] = []
            if llm_out:
                for a in llm_out:
                    templ = _ensure_placeholder(a, subject)
                    # Discard if too similar to base
                    if _pair_jaccard(templ, base_templ) >= 0.55:
                        continue
                    if templ not in candidates:
                        candidates.append(templ)
                # Select diverse subset among candidates
                diverse = _select_non_overlapping(candidates, max(args.n_anchors - 1, 0), jaccard_thr=0.55)
                anchors_build.extend(diverse)

            # If still short, backfill with conservative templates filtered by diversity
            if len(anchors_build) < args.n_anchors:
                fallback = _simple_rephrases(src)
                for a in fallback:
                    templ = _ensure_placeholder(a, subject)
                    if any(_pair_jaccard(templ, exist) >= 0.55 for exist in anchors_build):
                        continue
                    anchors_build.append(templ)
                    if len(anchors_build) >= args.n_anchors:
                        break

            anchors = anchors_build[: args.n_anchors]

        case_id = str(r.get("case_id") or r.get("id") or (idx + 1))
        for templ in anchors[: args.n_anchors]:
            rec = {
                "case_id": case_id,
                "prompt": templ,
                "subject": subject,
                "target_new": target_new,
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    fw.close()
    print(f"[OK] Wrote {n_written} anchor lines to: {out_path}")


if __name__ == "__main__":
    main()
