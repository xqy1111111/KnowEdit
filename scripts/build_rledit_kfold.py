#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

from scripts.edit_once_with_judge import read_zsre_like


def _convert_to_rledit_sample(req: Dict[str, Any]) -> Dict[str, str]:
    prompt = req.get("prompt", "")
    rephrase_list = req.get("rephrase") or []
    if isinstance(rephrase_list, str):
        rephrase_list = [rephrase_list]
    rephrase = rephrase_list[0] if rephrase_list else prompt

    target_new = req.get("target_new", "")
    if not target_new:
        target_new = req.get("alt") or req.get("alt_answer") or ""

    locality_prompt = ""
    locality_answer = ""
    locality = req.get("locality") or {}
    if isinstance(locality, dict) and "nq" in locality:
        lp = locality["nq"].get("prompt", "")
        la = locality["nq"].get("ground_truth", "")
        locality_prompt = lp or locality_prompt
        locality_answer = la or locality_answer

    return {
        "src": prompt,
        "rephrase": rephrase or prompt,
        "ans": target_new,
        "loc": locality_prompt,
        "loc_ans": locality_answer,
    }


def _compute_fold_ranges(total: int, folds: int) -> List[range]:
    folds = max(1, folds)
    base = total // folds
    remainder = total % folds
    ranges: List[range] = []
    start = 0
    for i in range(folds):
        size = base + (1 if i < remainder else 0)
        end = start + size
        ranges.append(range(start, end))
        start = end
    return ranges


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare RLEdit training folds from dataset")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--start", type=int, default=0, help="Skip this many rows before taking data")
    ap.add_argument("--limit", type=int, default=0, help="Use at most this many rows after start (0 = all)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold_index", type=int, default=0, help="Which fold to hold out")
    ap.add_argument("--train_out", required=True, help="Where to write training JSON")
    ap.add_argument("--holdout_out", default="", help="Optional holdout JSON path")
    args = ap.parse_args()

    reqs = read_zsre_like(args.data_path)
    start = max(0, int(args.start))
    if start:
        reqs = reqs[start:]
    if args.limit > 0:
        reqs = reqs[: args.limit]
    if not reqs:
        raise SystemExit("No data available to build RLEdit folds")

    folds = max(1, int(args.folds))
    fold_ranges = _compute_fold_ranges(len(reqs), folds)
    fold_idx = max(0, min(int(args.fold_index), len(fold_ranges) - 1))
    holdout_range = fold_ranges[fold_idx]

    holdout_indices = set(holdout_range)
    train_data = [_convert_to_rledit_sample(req) for idx, req in enumerate(reqs) if idx not in holdout_indices]
    holdout_data = [_convert_to_rledit_sample(req) for idx, req in enumerate(reqs) if idx in holdout_indices]

    os.makedirs(os.path.dirname(os.path.abspath(args.train_out)), exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as fh:
        json.dump(train_data, fh, ensure_ascii=False, indent=2)

    if args.holdout_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.holdout_out)), exist_ok=True)
        with open(args.holdout_out, "w", encoding="utf-8") as fh:
            json.dump(holdout_data, fh, ensure_ascii=False, indent=2)

    total_after_start = len(train_data) + len(holdout_data)
    print(json.dumps({
        "start": start,
        "total_after_start": total_after_start,
        "train_size": len(train_data),
        "holdout_size": len(holdout_data),
        "fold_index": fold_idx,
        "folds": folds,
    }))


if __name__ == "__main__":
    main()
