#!/usr/bin/env python
"""
Minimal RLEdit runner that loads an existing hypernetwork checkpoint, applies an
edit batch, and then calls model.generate() without relying on the higher-level
run_rledit_standalone.sh wrapper.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
RLEDIT_DIR = REPO_ROOT / "RLEdit"
if str(RLEDIT_DIR) not in sys.path:
    sys.path.insert(0, str(RLEDIT_DIR))

from data.base import make_loader  # noqa: E402
from data.zsre import ZSREDataset  # noqa: E402
from editor.rledit import RLEDIT  # noqa: E402
from model import make_model  # noqa: E402
from util import empty_cache  # noqa: E402


DATASET_REGISTRY = {
    "zsre": ZSREDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply an existing RLEdit hypernetwork and run generation.")
    parser.add_argument("--hparams", default="hparams/RLEdit/deepseek-r1d-llama-8b.yaml", help="Path to RLEdit hparams YAML.")
    parser.add_argument("--data-path", required=True, help="JSON file containing edit cases (zsre format).")
    parser.add_argument("--case-index", type=int, default=0, help="Start index inside data-path.")
    parser.add_argument("--count", type=int, default=1, help="Number of consecutive cases to edit.")
    parser.add_argument("--dataset-name", default="zsre", help="Dataset config key (defaults to zsre).")
    parser.add_argument("--ckpt-dir", required=True, help="Directory that stores *_net.pth and *_opt.pth.")
    parser.add_argument("--ckpt-tag", required=True, help="Checkpoint prefix, e.g. deepseek-r1d_rledit_20.")
    parser.add_argument("--prompt", default="", help="Optional manual prompt for generation; falls back to case prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="max_new_tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 == greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p for sampling.")
    parser.add_argument("--device", default=None, help="Torch device string, e.g. cuda:0 or cpu.")
    parser.add_argument("--output-jsonl", default="", help="Optional path to append generation results as JSONL.")
    parser.add_argument("--edit-batch-size", type=int, default=1, help="Batch size while applying edits.")
    return parser.parse_args()


def load_hparams(path: str) -> Dict:
    conf = OmegaConf.load(path)
    return OmegaConf.to_container(conf, resolve=True)  # type: ignore[return-value]


def build_cfg(hparams: Dict, dataset_name: str, device_override: str | None):
    config_dir = RLEDIT_DIR / "config"
    dataset_cfg = OmegaConf.load(config_dir / "dataset" / f"{dataset_name}.yaml")
    model_key = hparams.get("model_key") or "llama-3-instruct"
    model_cfg = OmegaConf.load(config_dir / "model" / f"{model_key}.yaml")
    editor_cfg = OmegaConf.load(config_dir / "editor" / "rledit.yaml")

    model_name = hparams.get("model_name") or model_cfg.get("name_or_path")
    if not model_name:
        raise ValueError("model_name must be provided in hparams (e.g. DeepSeek checkpoint id).")
    model_cfg.name_or_path = model_name
    model_cfg.name = Path(model_name.rstrip("/")).name or model_name
    if "inner_params" in hparams:
        model_cfg.edit_modules = hparams["inner_params"]
    if "half" in hparams:
        model_cfg.half = bool(hparams["half"])

    dataset_cfg.n_edits = max(1, int(hparams.get("dataset_n_edits", hparams.get("n_edits", dataset_cfg.n_edits))))
    dataset_cfg.batch_size = max(1, int(hparams.get("dataset_batch_size", hparams.get("batch_size", dataset_cfg.batch_size))))

    editor_mapping = {
        "rank": "rank",
        "n_blocks": "n_blocks",
        "lr": "lr",
        "meta_lr": "meta_lr",
        "token": "token",
        "loc_coef": "loc_coef",
        "time_decay": "time_decay",
        "back_depth": "back_depth",
        "full_curve": "full_curve",
        "save_checkpoint": "save_checkpoint",
        "load_checkpoint": "load_checkpoint",
    }
    for hp_key, cfg_key in editor_mapping.items():
        if hp_key in hparams:
            editor_cfg[cfg_key] = hparams[hp_key]
    if "editor_batch_size" in hparams:
        editor_cfg.batch_size = max(1, int(hparams["editor_batch_size"]))
    if "reg_lambda" in hparams:
        editor_cfg.reg_coef = hparams["reg_lambda"]
    if "cache_dir" in hparams:
        editor_cfg.cache_dir = str(Path(hparams["cache_dir"]).expanduser())

    device = device_override or hparams.get("device")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    cfg = OmegaConf.create(
        {
            "num_seq": int(hparams.get("num_seq", dataset_cfg.n_edits)),
            "glue_step": int(hparams.get("glue_step", 0)),
            "model_device": device,
            "editor_device": device,
        }
    )
    cfg.dataset = dataset_cfg
    cfg.model = model_cfg
    cfg.editor = editor_cfg
    return cfg


def slice_cases(data_path: str, start: int, count: int, out_dir: Path) -> Path:
    with open(data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if start < 0 or start >= len(data):
        raise ValueError(f"case_index {start} is outside the dataset with {len(data)} entries.")
    end = min(start + count, len(data))
    subset = data[start:end]
    if not subset:
        raise ValueError("No samples selected for editing.")
    out_path = out_dir / "selected_cases.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(subset, fh, ensure_ascii=False, indent=2)
    return out_path


def load_checkpoint(editor: RLEDIT, ckpt_dir: Path, ckpt_tag: str, device: str) -> None:
    net_path = ckpt_dir / f"{ckpt_tag}_net.pth"
    opt_path = ckpt_dir / f"{ckpt_tag}_opt.pth"
    if not net_path.exists():
        raise FileNotFoundError(f"Hypernetwork checkpoint not found: {net_path}")
    editor.net.load_state_dict(torch.load(net_path, map_location=device))
    if opt_path.exists():
        editor.opt.load_state_dict(torch.load(opt_path, map_location=device))


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, device: str) -> str:
    if not prompt:
        raise ValueError("Prompt for generation is empty.")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    hparams = load_hparams(args.hparams)
    cfg = build_cfg(hparams, args.dataset_name, args.device)

    dataset_cls = DATASET_REGISTRY.get(args.dataset_name)
    if dataset_cls is None:
        raise ValueError(f"Unsupported dataset '{args.dataset_name}'. Update DATASET_REGISTRY.")

    # Prepare temporary dataset slice for the requested cases.
    with tempfile.TemporaryDirectory(prefix="rledit_cases.") as tmpdir:
        subset_path = slice_cases(args.data_path, args.case_index, args.count, Path(tmpdir))
        cfg.dataset.n_edits = args.count
        cfg.dataset.batch_size = max(1, min(args.edit_batch_size, args.count))
        cfg.dataset.train_path = str(subset_path)
        cfg.dataset.valid_path = str(subset_path)

        train_loader, _ = make_loader(cfg, dataset_cls)

        model = make_model(cfg.model).to(cfg.model_device)
        editor = RLEDIT(cfg, model)
        load_checkpoint(editor, Path(args.ckpt_dir), args.ckpt_tag, cfg.editor_device)

        print(f"[RLEdit] Applying {args.count} case(s) starting from index {args.case_index}")
        editor.apply_loader_once(train_loader)
        print("[RLEdit] Edit applied. Running generation...")

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt_text = args.prompt
        if not prompt_text:
            with open(subset_path, "r", encoding="utf-8") as fh:
                sample = json.load(fh)[0]
            prompt_text = sample.get("prompt") or sample.get("src") or sample.get("question") or ""
        completion = run_generation(
            editor.model,
            tokenizer,
            prompt_text,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            cfg.model_device,
        )
        print("=" * 80)
        print(prompt_text.strip())
        print("-" * 80)
        print(completion.strip())
        print("=" * 80)

        if args.output_jsonl:
            record = {
                "case_index": args.case_index,
                "count": args.count,
                "prompt": prompt_text,
                "generation": completion,
                "ckpt_tag": args.ckpt_tag,
            }
            with open(args.output_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        empty_cache(cfg.editor.cache_dir, cfg)


if __name__ == "__main__":
    main()
