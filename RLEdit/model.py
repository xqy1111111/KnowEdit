import os

# Respect existing CUDA_VISIBLE_DEVICES if user already set it; default to GPU 0 otherwise.
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from omegaconf import DictConfig

import torch
import torch.nn as nn

import transformers
from transformers import AutoModel

from util import get_module


def make_model(config: DictConfig):

    model_class = getattr(transformers, config.class_name)
    model = model_class.from_pretrained(config.name_or_path)

    if config.half:
        dtype_name = getattr(config, "dtype", None) or os.environ.get("RLEDIT_DTYPE") or "bfloat16"
        dtype_name = dtype_name.lower()
        if dtype_name in {"fp16", "float16", "half"}:
            model = model.half()
        elif dtype_name in {"bf16", "bfloat16"}:
            model = model.bfloat16()
        else:
            raise ValueError(f"Unsupported dtype '{dtype_name}' for RLEdit model")

    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model
