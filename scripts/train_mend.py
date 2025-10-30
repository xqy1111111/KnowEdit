#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple MEND training entry.

Usage:
  python -m scripts.train_mend \
    --hparams hparams/TRAINING/MEND/deepseek-r1d-llama-8b.yaml \
    --train_data data/zsre/zsre_mend_train.json \
    --eval_data data/zsre/zsre_mend_eval.json

Optional:
  --archive path/to/checkpoint.pt  # resume or evaluate-only
"""
import os
import argparse

from easyeditor import EditTrainer
from easyeditor import ZsreDataset
from easyeditor.trainer.training_hparams import MENDTrainingHparams


def ensure_hf_caches():
    base = os.path.join(os.getcwd(), "models", "hf_cache")
    os.environ.setdefault("HF_HOME", base)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(base, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(base, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(base, "datasets"))
    for p in [os.environ["HF_HOME"], os.environ["HUGGINGFACE_HUB_CACHE"], os.environ["TRANSFORMERS_CACHE"], os.environ["HF_DATASETS_CACHE"]]:
        os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", required=True, help="Path to training YAML (hparams/TRAINING/MEND/*.yaml)")
    ap.add_argument("--train_data", required=True, help="ZSRE train json path")
    ap.add_argument("--eval_data", required=True, help="ZSRE eval json path")
    ap.add_argument("--archive", default="", help="Optional checkpoint to resume/eval-only")
    args = ap.parse_args()

    ensure_hf_caches()

    cfg = MENDTrainingHparams.from_hparams(args.hparams)
    if args.archive:
        cfg.archive = args.archive

    train_ds = ZsreDataset(args.train_data, config=cfg)
    eval_ds = ZsreDataset(args.eval_data, config=cfg)
    trainer = EditTrainer(config=cfg, train_set=train_ds, val_set=eval_ds)
    trainer.run()


if __name__ == "__main__":
    main()

