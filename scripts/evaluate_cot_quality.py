#!/usr/bin/env python3
"""
Evaluate CoT quality and answer correctness using LLM.
CoT quality: 0.5 points
Answer correctness: 0.5 points
Total: 1.0 point
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client with custom endpoint
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


def evaluate_response(target_text: str, pred_text: str, case_idx: int) -> tuple[float, float, str, str]:
    """
    Evaluate both CoT quality and answer correctness in one call.
    Returns: (cot_score, answer_score, target_answer, pred_answer)
    """
    prompt = f"""You are evaluating a predicted response against a target response. Both contain reasoning (Chain-of-Thought) and a final answer.

TARGET RESPONSE (reference):
{target_text}

PREDICTED RESPONSE (to evaluate):
{pred_text}

Please evaluate the predicted response on two dimensions:

1. CoT Quality (0.0 - 0.5 points):
   - Logical coherence and reasoning quality
   - Relevance to the question
   - Completeness of reasoning
   - IMPORTANT: Tolerate minor typos and spelling errors - focus on reasoning quality

2. Answer Correctness (0.0 - 0.5 points):
   - Extract the final answer from both responses
   - Compare if they are semantically equivalent
   - IMPORTANT: Tolerate minor typos (e.g., "Tom andJerry" = "Tom and Jerry", missing spaces, small spelling mistakes)
   - Ignore capitalization and punctuation differences
   - Full match (including with minor typos) = 0.5, partial match = 0.1-0.4, completely different = 0.0

Respond in the following format ONLY:
CoT_Score: <number between 0.0 and 0.5>
Answer_Score: <number between 0.0 and 0.5>
Target_Answer: <extracted answer from target>
Pred_Answer: <extracted answer from prediction>
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Follow the format exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()
        print(f"[Case {case_idx}] LLM评估输出:\n{result_text}")

        # Parse the response
        cot_match = re.search(r'CoT_Score:\s*(0?\.\d+|0\.?5?)', result_text, re.IGNORECASE)
        answer_match = re.search(r'Answer_Score:\s*(0?\.\d+|0\.?5?)', result_text, re.IGNORECASE)
        target_ans_match = re.search(r'Target_Answer:\s*(.+?)(?:\n|$)', result_text, re.IGNORECASE | re.DOTALL)
        pred_ans_match = re.search(r'Pred_Answer:\s*(.+?)(?:\n|$)', result_text, re.IGNORECASE | re.DOTALL)

        cot_score = float(cot_match.group(1)) if cot_match else 0.0
        answer_score = float(answer_match.group(1)) if answer_match else 0.0
        target_answer = target_ans_match.group(1).strip() if target_ans_match else "N/A"
        pred_answer = pred_ans_match.group(1).strip() if pred_ans_match else "N/A"

        cot_score = max(0.0, min(0.5, cot_score))
        answer_score = max(0.0, min(0.5, answer_score))

        return cot_score, answer_score, target_answer, pred_answer

    except Exception as e:
        print(f"[Case {case_idx}] Error evaluating response: {e}")
        return 0.0, 0.0, "N/A", "N/A"


def evaluate_case(case: Dict, case_idx: int) -> Dict:
    """Evaluate a single case and return scores."""
    # Extract target and predicted data
    target_new = case.get("target_new", "")
    pred_after = case.get("pred_after", "")

    print(f"\n{'='*60}")
    print(f"[Case {case_idx}] 开始评估")
    print(f"{'='*60}")

    # Evaluate both CoT and answer in one call
    cot_score, answer_score, target_answer, pred_answer = evaluate_response(
        target_new, pred_after, case_idx
    )

    total_score = cot_score + answer_score

    print(f"[Case {case_idx}] CoT得分: {cot_score:.3f}, 答案得分: {answer_score:.3f}, 总分: {total_score:.3f}")
    print(f"[Case {case_idx}] 目标答案: {target_answer[:100]}...")
    print(f"[Case {case_idx}] 预测答案: {pred_answer[:100]}...")

    return {
        "cot_score": cot_score,
        "answer_score": answer_score,
        "total_score": total_score,
        "target_answer": target_answer,
        "pred_answer": pred_answer
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoT quality and answer correctness using LLM")
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", type=str, help="Path to output JSONL file (default: <input>_evaluated.jsonl)")

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)

    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_evaluated.jsonl"

    print(f"Loading cases from {input_file}")

    # Load all cases
    cases = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Loaded {len(cases)} cases")
    print(f"Using model: {MODEL}")
    print(f"Evaluating...")

    # Evaluate all cases
    results = []
    total_scores = []
    cot_scores = []
    answer_scores = []

    for idx, case in enumerate(cases):
        scores = evaluate_case(case, idx)
        case_result = {**case, **scores}
        results.append(case_result)
        total_scores.append(scores["total_score"])
        cot_scores.append(scores["cot_score"])
        answer_scores.append(scores["answer_score"])

        # Print running average every 10 cases
        if (idx + 1) % 10 == 0:
            running_avg = sum(total_scores) / len(total_scores)
            running_cot_avg = sum(cot_scores) / len(cot_scores)
            running_ans_avg = sum(answer_scores) / len(answer_scores)
            print(f"\n{'='*60}")
            print(f"[进度报告] 已完成 {idx + 1}/{len(cases)} cases")
            print(f"[当前平均] CoT: {running_cot_avg:.4f}/0.5, 答案: {running_ans_avg:.4f}/0.5, 总分: {running_avg:.4f}/1.0")
            print(f"{'='*60}\n")

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Calculate final statistics
    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
    avg_cot = sum(cot_scores) / len(cot_scores) if cot_scores else 0.0
    avg_answer = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0

    print("\n" + "="*60)
    print("最终评估结果 (FINAL EVALUATION RESULTS)")
    print("="*60)
    print(f"总评估案例数: {len(cases)}")
    print(f"平均CoT得分: {avg_cot:.4f} / 0.5 ({avg_cot/0.5*100:.1f}%)")
    print(f"平均答案得分: {avg_answer:.4f} / 0.5 ({avg_answer/0.5*100:.1f}%)")
    print(f"平均总分: {avg_score:.4f} / 1.0 ({avg_score*100:.1f}%)")
    print(f"\n结果已保存到: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
