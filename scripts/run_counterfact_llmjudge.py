import os
import json
import argparse
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from easyeditor import BaseEditor, ROMEHyperParams
from scripts.utils_mquake import generate_answer, free_cuda
from steer.evaluate.evaluate import Evaluator


def _pick_hparams(model_name: str, override: str = None) -> str:
    if override:
        return override
    m = model_name.lower()
    if "deepseek-ai/deepseek-r1-distill-qwen-7b" in m:
        return "hparams/ROME/deepseek-r1d-qwen-7b.yaml"
    if "qwen2.5-7b" in m or "qwen/qwen2.5-7b" in m:
        return "hparams/ROME/qwen2.5-7b.yaml"
    # 默认走 qwen2.5-7b 配置，避免硬中断
    return "hparams/ROME/qwen2.5-7b.yaml"


def _load_counterfact_first_case(data_path: str) -> Dict[str, Any]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = data[0]

    # 标准 CounterFact 结构
    rw = rec["requested_rewrite"]
    subject = rw.get("subject")
    prompt = rw.get("prompt", "")
    # 有的文件是带占位符的模板："Where was {} born" 需 format(subject) 再补 '?'
    if "{}" in prompt:
        prompt_fmt = prompt.format(subject)
    else:
        prompt_fmt = prompt
    if not prompt_fmt.strip().endswith("?"):
        prompt_fmt = prompt_fmt.strip() + "?"

    target_new = rw["target_new"]["str"]
    ground_truth = rw["target_true"]["str"]

    # locality：不同数据版本字段名可能不同，统一兜底
    locality_prompts = rec.get("neighborhood_prompts") or rec.get("locality_prompts") or rec.get("locality", [])
    if isinstance(locality_prompts, str):
        locality_prompts = [locality_prompts]
    locality_answers = [ground_truth] * len(locality_prompts)

    # rephrase/paraphrase
    rephrase_prompts = rec.get("paraphrase_prompts") or rec.get("rephrase_prompts") or []
    if isinstance(rephrase_prompts, str):
        rephrase_prompts = [rephrase_prompts]

    return {
        "subject": subject,
        "prompt": prompt_fmt,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "rephrase_prompts": rephrase_prompts,
        "locality_prompts": locality_prompts,
        "locality_answers": locality_answers,
    }


def _build_editor(model_name: str, hparams_path: str, fp16: bool = False):
    hps = ROMEHyperParams.from_hparams(hparams_path)
    hps.model_name = model_name
    hps.fp16 = fp16
    editor = BaseEditor.from_hparams(hps)
    return editor


def _generate_once(model, tokenizer, prompt: str) -> str:
    # 保守生成：低温采样，简短答案
    return generate_answer(
        model, tokenizer, prompt,
        max_new_tokens=128, temperature=0.1, top_p=0.9,
        reveal_reasoning=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="CounterFact JSON 路径")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="编辑用的 HF 模型名，如 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 或 Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--hparams", type=str, default=None, help="可选：覆盖默认的 ROME 超参路径")
    ap.add_argument("--judge_model", type=str, default="deepseek-v3-241226",
                    help="LLM Judge 使用的 API 模型名，如 deepseek-v3-241226 或 gpt-4o")
    ap.add_argument("--save_dir", type=str, default="results/counterfact_onecase",
                    help="保存评测结果的目录")
    args = ap.parse_args()

    case = _load_counterfact_first_case(args.data_path)
    print("[INFO] Loaded one CounterFact case:", {k: case[k] for k in ["subject", "prompt", "target_new", "ground_truth"]})

    hparams_path = _pick_hparams(args.model, args.hparams)
    print(f"[INFO] Using hparams: {hparams_path}")

    editor = _build_editor(args.model, hparams_path, fp16=False)

    locality_inputs = {
        "locality": {
            "prompt": case["locality_prompts"],
            "ground_truth": case["locality_answers"],
        }
    }

    metrics, edited_model, tok = editor.edit(
        prompts=[case["prompt"]],
        target_new=[case["target_new"]],
        ground_truth=[case["ground_truth"]],
        rephrase_prompts=[case["rephrase_prompts"]] if case["rephrase_prompts"] else None,
        locality_inputs=locality_inputs,
        subject=[case["subject"]],
        keep_original_weight=True,
        sequential_edit=False,
    )

    print("[INFO] Edit metrics:", json.dumps(metrics[:1], ensure_ascii=False, indent=2))

    # 生成一次，用于 LLM Judge
    gen_text = _generate_once(edited_model, tok, case["prompt"]).strip()
    print("[INFO] Generated response:", gen_text)

    # 组织 LLM Judge 输入
    results: List[Dict[str, Any]] = [{
        "input": case["prompt"],
        "pred": [gen_text],
    }]

    # 环境变量：API_KEY 与可选 BASE_URL 需要用户自行 export
    if os.getenv("API_KEY") is None:
        print("[WARN] 未检测到 API_KEY 环境变量，LLM Judge 将失败。请 export API_KEY=... 并视情况 export BASE_URL=...")

    evaluator = Evaluator(
        mode="direct",
        save_results=True,
        results_dir=args.save_dir,
        eval_methods=["llm"],
        llm_model=args.judge_model,
    )
    os.makedirs(args.save_dir, exist_ok=True)
    eval_out = evaluator.evaluate_from_direct(results, dataset_name="counterfact_onecase", concept=case["subject"])
    print("[INFO] LLM Judge:", json.dumps(eval_out, ensure_ascii=False, indent=2))

    free_cuda(editor, edited_model)


if __name__ == "__main__":
    main()

