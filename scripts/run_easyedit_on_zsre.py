import os
import json
import argparse
from typing import List, Tuple

from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams


def read_zsre(path: str, limit: int = None) -> Tuple[List[str], List[str], List[str]]:
    """Read ZsRE(MEND-style) JSON and return (prompts, ground_truths, target_news).
    Expected keys per item: prompt, ground_truth, target_new.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    prompts, gts, news = [], [], []
    for r in data:
        p = r.get("prompt") or r.get("src") or ""
        gt = r.get("ground_truth") or (r.get("answers") or [""])[0]
        nw = r.get("target_new") or r.get("alt") or ""
        if p and nw:
            prompts.append(p)
            gts.append(gt)
            news.append(nw)
    return prompts, gts, news


def save_edited_model(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to save edited model to {out_dir}: {e}")


def build_editor(alg: str, hparams_path: str, model_name: str):
    if alg.upper() == "ROME":
        hp = ROMEHyperParams.from_hparams(hparams_path)
        hp.model_name = model_name or hp.model_name
        editor = BaseEditor.from_hparams(hp)
        return editor
    elif alg.upper() == "WISE":
        hp = WISEHyperParams.from_hparams(hparams_path)
        hp.model_name = model_name or hp.model_name
        editor = BaseEditor.from_hparams(hp)
        return editor
    else:
        raise ValueError("alg must be one of {ROME, WISE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=["ROME", "WISE"], help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=True, help="Path to ZsRE MEND JSON file")
    ap.add_argument("--model", default="", help="HF model id or local path; overrides YAML")
    ap.add_argument("--edit_num", type=int, default=100, help="Number of edits to apply (sequential)")
    ap.add_argument("--out_dir", default="runs/edited_model", help="Directory to save the edited model")
    ap.add_argument("--sequential", action="store_true", help="Apply edits sequentially in one pass")
    args = ap.parse_args()

    prompts, gts, news = read_zsre(args.data_path, limit=args.edit_num)
    if not prompts:
        raise RuntimeError(f"No valid edits parsed from {args.data_path}")

    editor = build_editor(args.alg, args.hparams, args.model)

    # Apply edits. For ZsRE, a single pass sequential edit is typical.
    metrics, edited_model, tok = editor.edit(
        prompts=prompts,
        ground_truth=gts,
        target_new=news,
        sequential_edit=True,
    )

    print(f"[INFO] Editing finished. Saving to: {args.out_dir}/edited_model")
    save_edited_model(edited_model, tok, os.path.join(args.out_dir, "edited_model"))
    # Persist basic metrics
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[DONE] Edited model and metrics saved.")


if __name__ == "__main__":
    main()

