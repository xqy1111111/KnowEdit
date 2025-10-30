# scripts/edit_once_and_judge.py
# -*- coding: utf-8 -*-
import os
import re
import gc
import json
import copy
import argparse
import warnings
import unicodedata
import shutil
"""
Environment safety: ensure Hugging Face caches point to a writable location by default.
This avoids errors like:
  "There was a problem when trying to write in your cache folder ... You should set TRANSFORMERS_CACHE"

We set defaults inside the repository at models/hf_cache if the current env points to
an unwritable path or is unset. Users can still override via standard env vars.
"""

def _ensure_hf_caches():
    def _is_writable(p: str) -> bool:
        try:
            os.makedirs(p, exist_ok=True)
            test_path = os.path.join(p, ".write_test")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
            return True
        except Exception:
            return False

    # Base default under repo: <repo>/models/hf_cache
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    except Exception:
        repo_root = os.getcwd()
    base_default = os.path.join(repo_root, "models", "hf_cache")

    # Resolve desired paths, preferring existing env when writable
    desired_hf_home = os.environ.get("HF_HOME", os.path.join(base_default))
    desired_hub = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(desired_hf_home, "hub"))
    desired_tf_cache = os.environ.get("TRANSFORMERS_CACHE", os.path.join(desired_hf_home, "transformers"))
    desired_ds_cache = os.environ.get("HF_DATASETS_CACHE", os.path.join(desired_hf_home, "datasets"))

    # If any of these are not writable, fall back to repo-local defaults
    if not _is_writable(desired_hf_home):
        desired_hf_home = os.path.join(base_default)
    if not _is_writable(desired_hf_home):
        # Last resort: use CWD-based cache
        desired_hf_home = os.path.join(os.getcwd(), ".hf_home")
    os.makedirs(desired_hf_home, exist_ok=True)

    # Recompute subpaths now that HF_HOME is set
    desired_hub = os.path.join(desired_hf_home, "hub") if not _is_writable(desired_hub) else desired_hub
    desired_tf_cache = os.path.join(desired_hf_home, "transformers") if not _is_writable(desired_tf_cache) else desired_tf_cache
    desired_ds_cache = os.path.join(desired_hf_home, "datasets") if not _is_writable(desired_ds_cache) else desired_ds_cache

    for p in (desired_hub, desired_tf_cache, desired_ds_cache):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

    # Export effective values before importing HF libraries
    os.environ["HF_HOME"] = desired_hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = desired_hub
    os.environ["TRANSFORMERS_CACHE"] = desired_tf_cache
    os.environ["HF_DATASETS_CACHE"] = desired_ds_cache


# Ensure caches are set before any HF/torch imports that may consult them.
_ensure_hf_caches()
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import timedelta

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    StoppingCriteriaList,
    StoppingCriteria,
)
try:
    # Optional: if present, helps merge LoRA/QLoRA adapters before saving
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore
from easyeditor import (
    BaseEditor,
    ROMEHyperParams,
    WISEHyperParams,
    FTHyperParams,
    LoRAHyperParams,
    QLoRAHyperParams,
    MEMITHyperParams,
    MENDHyperParams,
    AlphaEditHyperParams,
    IKEHyperParams,
)
from easyeditor.models.memit import apply_memit_to_model  # type: ignore

# Improve CUDA memory fragmentation resilience unless user overrides.
# Effective as long as set before first CUDA allocation.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

@dataclass
class GenerationConfig:
    mode: str
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class DeepSpeedConfig:
    enabled: bool
    dtype: str
    max_out_tokens: int
    kernel_inject: bool
    mp_size: int


DEFAULT_STOP_SEQUENCES: Tuple[str, ...] = ("<<<END>>>",)


@dataclass
class JudgeConfig:
    model_id: str
    device_map: Optional[Union[str, Dict[str, Any]]]
    max_memory: Optional[Dict[str, Any]]
    keep_loaded: bool = False


def _clear_cuda_cache():
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _linear_pre_hook_align(module: nn.Linear, inputs):
    """Align device/dtype/layout before Linear forward to avoid cuBLASLt issues."""
    if not inputs:
        return inputs
    (x,) = inputs
    weight = module.weight
    if x.device != weight.device:
        x = x.to(weight.device, non_blocking=True)
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(weight.device)
    except Exception:
        pass
    return (x,)


def install_linear_safety_hooks(model: nn.Module, in_features: int = 3584, max_out_features: int = 64) -> int:
    """Install pre-hooks on narrow Linear-like layers to stabilize cuBLASLt matmul."""
    count = 0
    for name, mod in model.named_modules():
        weight = getattr(mod, "weight", None)
        if weight is None or not hasattr(weight, "shape"):
            continue
        if getattr(mod, "_safety_hook_installed", False):
            continue
        if weight.ndim != 2:
            continue
        out_features, in_feats = weight.shape
        if in_feats != in_features or out_features > max_out_features:
            continue
        # Install hook (works for nn.Linear and custom linear-likes taking single tensor input)
        try:
            mod.register_forward_pre_hook(_linear_pre_hook_align, with_kwargs=False)
            setattr(mod, "_safety_hook_installed", True)
            count += 1
            print(f"[hook] Installed safety pre-hook on {type(mod).__name__} {name} ({in_feats}->{out_features})")
        except Exception as hook_err:
            warnings.warn(f"[hook] failed to register on {name}: {hook_err}")
    return count


# ===================== 工具 =====================
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def _resolve_infer_dtype(ds_dtype: str) -> torch.dtype:
    """Choose a safe CUDA dtype for inference. If bf16 is requested/auto-picked but
    triangular ops are unsupported on this stack, fallback to fp16 to avoid
    "triu_tril_cuda_template not implemented for 'BFloat16'" errors.
    """
    # Explicit overrides
    if ds_dtype == "fp16":
        return torch.float16
    if ds_dtype == "bf16":
        candidate = torch.bfloat16
    else:  # auto
        candidate = torch.bfloat16 if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
    if candidate is torch.bfloat16:
        # Runtime op check: some stacks claim bf16 support but miss tri[u|l] kernels
        try:
            x = torch.zeros((2, 2), device=("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.bfloat16)
            _ = torch.triu(x)
            _ = torch.tril(x)
            return torch.bfloat16
        except Exception:
            return torch.float16
    return candidate

def ensure_pad_token(tokenizer: AutoTokenizer) -> bool:
    """Ensure tokenizer has a valid pad id; return True if new tokens were added."""
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return False
    before = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return len(tokenizer) != before

def _unwrap_tokenizer(tok_like: Any) -> Optional[PreTrainedTokenizerBase]:
    if tok_like is None:
        return None
    if isinstance(tok_like, PreTrainedTokenizerBase):
        return tok_like
    if hasattr(tok_like, "save_pretrained"):
        return tok_like
    if isinstance(tok_like, dict):
        for k in ("tokenizer", "tok", "fast_tokenizer", "hf_tokenizer"):
            v = tok_like.get(k)
            if isinstance(v, PreTrainedTokenizerBase) or hasattr(v, "save_pretrained"):
                return v
    return None


def _safe_load_tokenizer(
    preferred_paths: List[str],
    fallback_id: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """Prefer loading tokenizer from local directories; fallback to remote id.
    Tries fast tokenizer first; if it fails, retries with use_fast=False.
    """
    # Prefer local directories first
    for p in preferred_paths:
        if not p:
            continue
        try:
            if os.path.isdir(p):
                return AutoTokenizer.from_pretrained(p, trust_remote_code=trust_remote_code, use_fast=True)  # type: ignore
        except Exception:
            try:
                if os.path.isdir(p):
                    return AutoTokenizer.from_pretrained(p, trust_remote_code=trust_remote_code, use_fast=False)  # type: ignore
            except Exception:
                pass
    # Fallback to remote/base id
    if fallback_id:
        try:
            return AutoTokenizer.from_pretrained(fallback_id, trust_remote_code=trust_remote_code, use_fast=True)  # type: ignore
        except Exception:
            return AutoTokenizer.from_pretrained(fallback_id, trust_remote_code=trust_remote_code, use_fast=False)  # type: ignore
    raise RuntimeError("Unable to load tokenizer from local paths or fallback id.")


def ensure_tokenizer_model_vocab_alignment(tokenizer: Any, model: Any, context: str = "") -> bool:
    """Resize model embeddings if tokenizer length changed; return True if resized."""
    tok = _unwrap_tokenizer(tokenizer) if not isinstance(tokenizer, PreTrainedTokenizerBase) else tokenizer
    if tok is None or model is None:
        return False
    resize_fn = getattr(model, "resize_token_embeddings", None)
    get_inputs = getattr(model, "get_input_embeddings", None)
    if not callable(resize_fn) or not callable(get_inputs):
        return False
    try:
        tok_size = len(tok)
    except Exception:
        return False
    try:
        embeddings = get_inputs()
        num_embeddings = getattr(embeddings, "num_embeddings", None)
    except Exception:
        num_embeddings = None
    if not isinstance(num_embeddings, int):
        return False
    if tok_size == num_embeddings:
        return False
    prefix = f"{context} " if context else ""
    warnings.warn(f"{prefix}[vocab] tokenizer={tok_size}, embeddings={num_embeddings}; resizing to match.")
    resize_fn(tok_size)
    return True


class StopOnTokens(StoppingCriteria):
    """Stop generation when any of the stop token sequences appears at the tail."""

    def __init__(self, sequences: List[List[int]]):
        self.sequences = [seq for seq in sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
        if not self.sequences:
            return False
        if input_ids.numel() == 0:
            return False
        generated = input_ids[0].tolist()
        for seq in self.sequences:
            if len(generated) >= len(seq) and generated[-len(seq):] == seq:
                return True
        return False


_TEMPLATE_ANSWER_RE = re.compile(r"---ANSWER---\s*(.+?)(?:\r?\n|<<<END>>>|$)", re.IGNORECASE | re.DOTALL)


def _strip_stop_markers(text: str, markers: Tuple[str, ...]) -> str:
    if not text:
        return ""
    result = text
    for marker in markers:
        if not marker:
            continue
        idx = result.find(marker)
        if idx != -1:
            result = result[:idx]
            break
    return result.rstrip()


def _extract_template_answer(text: str) -> str:
    if not text:
        return ""
    m = _TEMPLATE_ANSWER_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_short_answer(text: str, mode: str) -> str:
    """Extract a concise final answer string for rule-based matching."""
    if not text:
        return ""
    final_txt, _ = extract_final(text)
    if final_txt:
        return final_txt.strip()
    templ = _extract_template_answer(text)
    if templ:
        return templ
    if mode == "noprompt":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    return ""


def _parse_device_map_arg(device_map: str) -> Optional[Union[str, Dict[str, Any]]]:
    if not device_map:
        return None
    device_map = device_map.strip()
    if not device_map:
        return None
    lowered = device_map.lower()
    if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return lowered
    try:
        parsed = json.loads(device_map)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析 gen_device_map: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("gen_device_map JSON 必须是对象")
    return parsed


def _parse_max_memory_arg(max_memory: str) -> Optional[Dict[str, Any]]:
    if not max_memory:
        return None
    max_memory = max_memory.strip()
    if not max_memory:
        return None
    try:
        parsed = json.loads(max_memory)
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析 gen_max_memory: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("gen_max_memory JSON 必须是对象，如 {'cuda:0': '20GiB'}")
    normalized: Dict[Any, Any] = {}
    for key, value in parsed.items():
        new_key = key
        if isinstance(key, str):
            stripped = key.strip()
            lower = stripped.lower()
            if lower.startswith("cuda:"):
                stripped = stripped.split(":", 1)[1]
                lower = stripped.lower()
            if stripped.isdigit():
                new_key = int(stripped)
            elif lower in {"cpu", "mps", "disk"}:
                new_key = lower
            else:
                new_key = stripped
        normalized[new_key] = value
    return normalized


def _maybe_dispatch_generation_model(model, device_map: Optional[Union[str, Dict[str, Any]]],
                                     max_memory: Optional[Dict[str, Any]] = None):
    if device_map is None:
        return model
    if not torch.cuda.is_available():
        warnings.warn("CUDA 不可用，无法执行多卡推理，退回单卡模式。")
        return model
    try:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory
    except ImportError:
        warnings.warn("未安装 accelerate，无法启用多卡推理，退回单卡模式。")
        return model

    current_map = getattr(model, "hf_device_map", None)
    if isinstance(current_map, dict):
        gpu_devices = set()
        for dev in current_map.values():
            if isinstance(dev, str):
                if dev not in {"cpu", "disk"}:
                    gpu_devices.add(dev)
            elif isinstance(dev, int):
                gpu_devices.add(f"cuda:{dev}")
        if len(gpu_devices) > 1:
            return model  # 已经是多卡

    resolved_map = device_map
    if isinstance(device_map, str):
        lowered = device_map.lower()
        no_split = getattr(model, "_no_split_modules", None)
        if lowered == "auto":
            resolved_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split,
            )
        elif lowered == "balanced":
            balanced_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split,
                low_zero=False,
            )
            resolved_map = infer_auto_device_map(
                model,
                max_memory=balanced_memory,
                no_split_module_classes=no_split,
            )
        elif lowered == "balanced_low_0":
            balanced_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split,
                low_zero=True,
            )
            resolved_map = infer_auto_device_map(
                model,
                max_memory=balanced_memory,
                no_split_module_classes=no_split,
            )
        elif lowered == "sequential":
            resolved_map = "sequential"
        else:
            warnings.warn(f"未知的 gen_device_map 选项: {device_map}，将忽略多卡配置。")
            return model

    try:
        dispatch_model(model, device_map=resolved_map, offload_dir=None)
    except Exception as exc:
        warnings.warn(f"多卡分发失败，仍使用单卡。原因: {exc}")
    return model


def _infer_model_device(model) -> torch.device:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, str):
        try:
            return torch.device(dev)
        except Exception:
            pass
    try:
        first_param = next(model.parameters())
        if isinstance(first_param, torch.nn.Parameter):
            return first_param.device
    except StopIteration:
        pass
    except Exception:
        pass
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, str) and value not in {"cpu", "disk"}:
                return torch.device(value)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== DeepSpeed Inference（可选） =====================
def _hf_device_map_multi_gpu(dm) -> bool:
    if not isinstance(dm, dict):
        return False
    gpu_set = set()
    for v in dm.values():
        if isinstance(v, str):
            if v not in {"cpu", "disk"}:
                gpu_set.add(v)
        elif isinstance(v, int):
            gpu_set.add(f"cuda:{v}")
    return len(gpu_set) > 1


def _maybe_apply_ds_inference(model, use_ds_infer: bool,
                              ds_dtype: str = "auto",
                              ds_max_out_tokens: int = 0,
                              ds_kernel_inject: bool = True):
    """在单进程/单卡场景下，对模型应用 DeepSpeed Inference kernel injection。
    - 当模型已通过 accelerate 分到多卡（hf_device_map 含多张 GPU）时跳过，避免冲突。
    - 当未安装 deepspeed 或 CPU 环境亦跳过。
    - 仅用于加速 generate，不改变权重数值。
    """
    if not use_ds_infer:
        return model
    # 避免重复注入导致显存翻倍
    try:
        if getattr(model, "_ds_infer_injected", False):
            return model
    except Exception:
        pass
    if not torch.cuda.is_available():
        warnings.warn("use_ds_infer=True 但 CUDA 不可用，跳过 DeepSpeed Inference。")
        return model

    # 若已经跨多卡分片，避免与 DS 注入冲突
    dm = getattr(model, "hf_device_map", None)
    if _hf_device_map_multi_gpu(dm):
        warnings.warn("检测到模型已跨多卡分片（accelerate device_map）。为避免冲突，将跳过 DeepSpeed kernel injection。")
        return model

    try:
        import deepspeed
    except Exception:
        warnings.warn("未安装 deepspeed 或导入失败，跳过 DeepSpeed Inference。")
        return model

    # 解析 dtype（带运行时降级，避免 bf16 缺三角核）
    target_dtype = _resolve_infer_dtype(ds_dtype)

    # ensure eval & cache for speed
    try:
        model.eval()
        if getattr(model, "config", None) is not None:
            setattr(model.config, "use_cache", True)
    except Exception:
        pass

    # DeepSpeed kernel injection（单进程 mp_size=1）
    kw = dict(
        mp_size=1,
        dtype=target_dtype,
        replace_method="auto",
        replace_with_kernel_inject=bool(ds_kernel_inject),
    )
    if isinstance(ds_max_out_tokens, int) and ds_max_out_tokens > 0:
        # 某些版本支持 max_out_tokens
        kw["max_out_tokens"] = ds_max_out_tokens

    try:
        engine = deepspeed.init_inference(model, **kw)
    except TypeError:
        # 兼容旧版本（不支持 max_out_tokens）
        kw.pop("max_out_tokens", None)
        engine = deepspeed.init_inference(model, **kw)
    except Exception as exc:
        warnings.warn(f"DeepSpeed Inference 初始化失败，跳过。原因: {exc}")
        return model

    wrapped = getattr(engine, "module", None)
    result = wrapped or model
    try:
        setattr(result, "_ds_infer_injected", True)
    except Exception:
        pass
    return result


def _init_distributed_if_needed(enable: bool) -> Tuple[bool, int, int, int]:
    """若需要，则初始化分布式环境，返回 (is_distributed, rank, world_size, local_rank)。
    要使多卡 + DeepSpeed 张量并行生效，请使用 torchrun 启动：
      torchrun --nproc_per_node=N python scripts/edit_once_with_judge.py ... --use_ds_infer --ds_mp_size N
    """
    if not enable:
        return False, 0, 1, 0
    try:
        import torch.distributed as dist
    except Exception:
        return False, 0, 1, 0

    if dist.is_initialized():
        try:
            rank = dist.get_rank()
            world = dist.get_world_size()
        except Exception:
            rank, world = 0, 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return world > 1, rank, world, local_rank

    # 尝试初始化
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        import torch.distributed as dist
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
    except Exception:
        try:
            import deepspeed
            deepspeed.init_distributed(dist_backend=backend)
        except Exception:
            return False, 0, 1, 0

    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
    except Exception:
        rank, world = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception:
        pass
    return world > 1, rank, world, local_rank

# ===================== 读取 ZsRE-like =====================
def _get_first(x, default=""):
    if x is None:
        return default
    if isinstance(x, list):
        return x[0] if x else default
    return x

def read_zsre_like(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    stripped = raw.lstrip()
    if not stripped:
        return []

    data: List[Any] = []
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, list):
            data = loaded
        elif isinstance(loaded, dict):
            data = [loaded]
        else:
            warnings.warn(f"[data] Unsupported JSON root type: {type(loaded).__name__}; expected list or dict.")
            return []
    except json.JSONDecodeError:
        # Treat as JSONL
        data = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"[data] 跳过第 {lineno} 行（JSON 解析失败）: {exc}")
                continue
            data.append(obj)

    reqs = []
    for r in data:
        prompt = r.get("prompt") or r.get("src") or ""
        if not prompt:
            continue
        target_new = r.get("target_new") or r.get("alt") or ""
        if not target_new:
            continue
        ground_truth = r.get("ground_truth")
        if ground_truth is None:
            ground_truth = _get_first(r.get("answers"), "")

        subject = r.get("subject") or r.get("sub_label") or r.get("subject_entity") or ""

        alt_answer = r.get("alt_answer") or ""

        rephrase = r.get("rephrase")
        if isinstance(rephrase, str):
            rephrase_list = [rephrase] if rephrase.strip() else []
        elif isinstance(rephrase, list):
            rephrase_list = [x for x in rephrase if isinstance(x, str) and x.strip()]
        else:
            rephrase_list = []

        lp = _get_first(r.get("loc"), "")
        la = _get_first(r.get("loc_ans"), "")

        req = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "target_new": target_new,
            "rephrase": rephrase_list,
            "locality": {"nq": {"prompt": lp, "ground_truth": la}} if (lp and la) else {},
            "portability": {},
        }
        if subject:
            req["subject"] = subject
        if alt_answer:
            req["alt_answer"] = alt_answer
        reqs.append(req)
    return reqs

# ===================== 主语 =====================
_SUBJ_PATTERNS = [
    re.compile(r'(?:did|does|was|is|were|has|have)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
    re.compile(r'^[Ww]ho (?:is|was)\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
    re.compile(r'\babout\s+([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})'),
]
def heuristic_extract_subject(prompt: str) -> str:
    for pat in _SUBJ_PATTERNS:
        m = pat.search(prompt)
        if m:
            return m.group(1).strip()
    caps = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+){0,6})', prompt)
    if caps:
        caps.sort(key=len, reverse=True)
        return caps[0].strip()
    print("[WARN] 无法从 prompt 中提取主语")
    return ""

# ===================== 读取 MEMIT Anchors =====================
def _read_memit_anchors_group(path: str, case_id: str, take: int = 6) -> Dict[str, Any]:
    """Read anchors JSONL produced by scripts/cot_to_memit_anchors.py and pack one case.

    Returns a ZSRE-like single request dict with extra key 'memit_bundle' listing
    multiple anchor requests for a single unified MEMIT update.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"anchors_jsonl not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    group = [r for r in rows if str(r.get("case_id", "")) == str(case_id)]
    if not group:
        raise RuntimeError(f"No anchors for case_id={case_id} in {path}")
    # Basic fields
    subject = (group[0].get("subject") or "").strip()
    target_new = (group[0].get("target_new") or "").strip()
    if not subject or not target_new:
        raise RuntimeError("anchors missing subject or target_new")

    # Render prompts with real subject for MEMIT (it will convert to '{}' internally)
    memit_reqs: List[Dict[str, Any]] = []
    eval_prompt = None
    for r in group[:max(1, int(take))]:
        p = (r.get("prompt") or "").strip()
        if not p:
            continue
        pr = p.format(subject) if "{}" in p else p
        if eval_prompt is None:
            eval_prompt = pr
        memit_reqs.append({
            "prompt": pr,
            "subject": subject,
            "target_new": target_new,
            "case_id": str(case_id),
        })
    if not memit_reqs:
        raise RuntimeError("No valid prompts in anchors group after formatting")
    if not eval_prompt:
        eval_prompt = memit_reqs[0]["prompt"]

    # Pack into single-case request for this runner
    req = {
        "prompt": eval_prompt,
        "target_new": target_new,
        "subject": subject,
        "locality": {},
        "portability": {},
        # extra payload consumed by run_one_case when alg==MEMIT
        "memit_bundle": memit_reqs,
    }
    return req

# ===================== 构建 Editor =====================
ALG_HP_MAP = {
    "ROME": (ROMEHyperParams, "ROME"),
    "WISE": (WISEHyperParams, "WISE"),
    "FT": (FTHyperParams, "FT"),
    "LORA": (LoRAHyperParams, "LoRA"),
    "QLORA": (QLoRAHyperParams, "QLoRA"),
    "MEMIT": (MEMITHyperParams, "MEMIT"),
    "MEND": (MENDHyperParams, "MEND"),
    "ALPHAEDIT": (AlphaEditHyperParams, "AlphaEdit"),
    "IKE": (IKEHyperParams, "IKE"),
}

SUPPORTED_ALG_NAMES = [display for _, display in ALG_HP_MAP.values()]


def build_editor(alg: str, hparams_path: str, model_name: str) -> BaseEditor:
    key = alg.upper()
    if key not in ALG_HP_MAP:
        supported = ", ".join(SUPPORTED_ALG_NAMES)
        raise ValueError(f"alg must be one of {{{supported}}}")

    hp_cls, _ = ALG_HP_MAP[key]
    hp = hp_cls.from_hparams(hparams_path)
    if model_name:
        hp.model_name = model_name
    # Under torchrun multi-process, avoid pinning all editors to one GPU.
    # If not using model_parallel editing, set device to local_rank.
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world > 1 and hasattr(hp, "model_parallel") and not getattr(hp, "model_parallel", False):
            if torch.cuda.is_available():
                hp.device = int(local_rank)
    except Exception:
        pass
    return BaseEditor.from_hparams(hp)

# ===================== 生成（两种模式） =====================
def _build_messages(prompt: str, mode: str) -> list:
    if mode == "reason":
        sys_msg = (
            "You are a helpful assistant. "
            "Think step by step INSIDE <think>...</think>. "
            "Then output ONLY the final short answer INSIDE <answer>...</answer>. "
            "Do not include anything else outside these tags."
        )
    elif mode == "concise":
        sys_msg = "Answer with a short noun phrase only. Do NOT explain."
    elif mode == "r1d":
        # DeepSeek-R1-Distill：只提供 user 消息，严格依赖 tokenizer.chat_template
        # 不额外注入任何 prompt/指令，由模板在 add_generation_prompt=True 时自动前置 <think>。
        return [
            {"role": "user", "content": prompt}
        ]
    else:  # "noprompt": 仅用户消息，不注入系统提示
        return [
            {"role": "user", "content": prompt}
        ]
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": prompt}
    ]

def generate_answer(model, tokenizer, prompt: str,
                    max_new_tokens=64, temperature=0.6, top_p=0.95,
                    mode: str = "concise") -> str:
    # "noprompt": 完全不注入系统提示，也不使用 chat template，直接把原始问题作为输入
    if mode == "noprompt":
        text = prompt
    else:
        messages = _build_messages(prompt, mode=mode)
        if hasattr(tokenizer, "apply_chat_template"):
            # 对 DeepSeek-R1-Distill，必须开启 add_generation_prompt 以触发模板自动插入 <think>
            add_gen = True if mode == "r1d" else False
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
        else:
            if mode == "reason":
                text = (
                    "You are a helpful assistant. Think step by step INSIDE <think>...</think>. less but useful steps are appriciated"
                    "Then output ONLY the final short answer INSIDE <answer>...</answer>.\n"
                    f"Q: {prompt}\nA:"
                )
            elif mode == "r1d":
                # 无 chat_template 的兜底：尽量模拟模板行为，以 <think> 起手
                text = f"{prompt}\n<think>"
            else:  # concise
                text = f"Answer with a short noun phrase only. Do NOT explain.\nQ: {prompt}\nA:"

    ensure_pad_token(tokenizer)
    ensure_tokenizer_model_vocab_alignment(tokenizer, model, context="[generate]")
    enc = tokenizer(text, return_tensors="pt")
    target_device = _infer_model_device(model)
    enc = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in enc.items()}

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

    stop_token_ids: List[List[int]] = []
    for seq in DEFAULT_STOP_SEQUENCES:
        if not seq:
            continue
        try:
            ids = tokenizer.encode(seq, add_special_tokens=False)
        except Exception:
            ids = []
        if ids:
            stop_token_ids.append(ids)
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)]) if stop_token_ids else None

    # 推理期禁用 autograd 以节省显存并提速
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=(1.07 if mode in {"reason", "r1d"} else 1.0),
            stopping_criteria=stopping_criteria,
        )
    gen_txt = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    gen_txt = _strip_stop_markers(gen_txt, DEFAULT_STOP_SEQUENCES)
    # 返回完整文本；分段由 extract_final 在上层完成
    return gen_txt.strip()

# ===================== 解析 <final> =====================
_FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
def extract_final(text: str) -> Tuple[str, str]:
    """通用解析：优先支持 <final>/<reasoning>；若无，则回退为 <think>…</think> + 其后的答案段。
    返回 (final_text, reasoning_or_think_text)。
    """
    if not text:
        return "", ""
    # 1) 优先匹配 <final>/<reasoning>
    m_f = _FINAL_RE.search(text)
    m_r = _REASON_RE.search(text)
    if m_f or m_r:
        final = (m_f.group(1).strip() if m_f else "").strip("：:").strip()
        reason = (m_r.group(1).strip() if m_r else "").strip()
        return final, reason
    # 2) R1D 新模板常把 <think> 放在提示词里，生成只包含 </think>
    #    若仅出现闭合标签，则把其前内容当作思维段，其后为最终答案
    m_close = _THINK_CLOSE_RE.search(text)
    if m_close and not _THINK_RE.search(text):
        reason_txt = text[: m_close.start()].strip()
        tail_txt = text[m_close.end():].strip()
        return tail_txt, reason_txt
    # 3) 标准 <think>…</think>：取最后一段作为思维，尾部为最终答案
    last = None
    for m in _THINK_RE.finditer(text):
        last = m
    if last is not None:
        think_txt = last.group(1).strip()
        tail = text[last.end():].strip()
        return tail, think_txt
    return "", ""

# ===================== LLM-Judge =====================
class JudgeManager:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

    def load(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._model is None:
            tok = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
            ensure_pad_token(tok)

            kwargs = dict(torch_dtype=pick_dtype(), trust_remote_code=True)
            if self.config.device_map is not None:
                kwargs["device_map"] = self.config.device_map
            if self.config.max_memory is not None:
                kwargs["max_memory"] = self.config.max_memory

            self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
            self._tokenizer = tok
            ensure_tokenizer_model_vocab_alignment(tok, self._model, context="[judge]")
        return self._tokenizer, self._model

    def release_if_needed(self, force: bool = False) -> None:
        if not force and self.config.keep_loaded:
            return
        if self._model is not None:
            try:
                del self._model
            finally:
                self._model = None
        if self._tokenizer is not None:
            try:
                del self._tokenizer
            finally:
                self._tokenizer = None
        _clear_cuda_cache()

    def shutdown(self) -> None:
        self.release_if_needed(force=True)

JUDGE_PROMPT = """You are a precise grader.
Only judge the FINAL_ANSWER string; ignore any hidden thoughts/reasoning if present.
Normalize: lowercase, strip accents/punctuation; minor typos are acceptable.
If FINAL_ANSWER semantically matches ANY of the GOLD items, answer YES; otherwise NO.

Return exactly one token: YES or NO.

GOLD: {golds}
FINAL_ANSWER: {final}

Decision (YES or NO):"""

def judge_hit(tok: AutoTokenizer, mdl: AutoModelForCausalLM,
              question: str, final: str, golds: List[str],
              max_new_tokens: int = 8, debug: bool = False) -> int:
    prompt = JUDGE_PROMPT.format(question=question, golds=", ".join(golds), final=final)
    enc = tok(prompt, return_tensors="pt")
    target_device = _infer_model_device(mdl)
    enc = {k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    if debug:
        print("[LLM-JUDGE RAW]:", txt)
    return int(("YES" in txt) and ("NO" not in txt))

# ===================== 单条：可选“编辑前” + 编辑 + 评测 =====================

def run_one_case(
    req: Dict[str, Any],
    alg: str,
    hparams: str,
    model_override: str,
    generation_cfg: GenerationConfig,
    judge_manager: Optional[JudgeManager],
    eval_before: bool = False,
    show_eemetrics: bool = False,
    gen_device_map: Optional[Union[str, Dict[str, Any]]] = None,
    gen_max_memory: Optional[Dict[str, Any]] = None,
    skip_locality: bool = False,
    ds_cfg: Optional[DeepSpeedConfig] = None,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    dist_local_rank: int = 0,
    no_judge: bool = False,
    save_edited_to: str = "",
    delete_saved_after: bool = False,
    save_tag: str = "",
    stage: str = "full",
    load_edited_from: str = "",
    train_ds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    gen_mode = generation_cfg.mode
    max_new_tokens = generation_cfg.max_new_tokens
    temperature = generation_cfg.temperature
    top_p = generation_cfg.top_p

    ds_enabled = bool(ds_cfg and ds_cfg.enabled)
    ds_dtype = ds_cfg.dtype if ds_cfg else "auto"
    ds_max_out_tokens = ds_cfg.max_out_tokens if ds_cfg else 0
    ds_kernel_inject = ds_cfg.kernel_inject if ds_cfg else True
    ds_mp_size = ds_cfg.mp_size if ds_cfg else 1

    stage = stage.lower()
    if stage not in {"full", "edit", "infer"}:
        raise ValueError("stage must be one of {full, edit, infer}")
    do_edit = stage in {"full", "edit"}
    do_infer = stage in {"full", "infer"}

    judge_tok: Optional[AutoTokenizer] = None
    judge_model: Optional[AutoModelForCausalLM] = None
    judge_loaded = False
    editor: Optional[BaseEditor] = None
    edited_model = None

    target_new_text: str = req.get("target_new", "") or ""
    alt_answer_text: str = req.get("alt_answer", "") or ""
    alt_answer_norm: str = _normalize_for_match(alt_answer_text) if alt_answer_text else ""

    try:
        # 延迟加载裁判模型：仅在真正需要判分时再加载，避免与编辑阶段抢显存
        def _ensure_judge_loaded():
            nonlocal judge_tok, judge_model, judge_loaded
            if no_judge:
                return
            if judge_loaded:
                return
            if dist_rank == 0 and judge_manager is not None:
                jt, jm = judge_manager.load()
                judge_tok, judge_model = jt, jm
                judge_loaded = True

        if alg.upper() in {"ROME", "MEMIT"}:
            s = req.get("subject", "")
            if not s:
                s = heuristic_extract_subject(req["prompt"])
                if s:
                    warnings.warn(f"[{alg.upper()}] subject missing; heuristically extracted '{s}'")
                else:
                    raise ValueError(f"[{alg.upper()}] cannot find subject for prompt: {req['prompt']}")
            if s not in req["prompt"]:
                req = dict(req)
                req["prompt"] = f"{req['prompt']} (Subject: {s})"
                req["subject"] = s

        # For WISE, ensure 'loc_prompt' exists and matches a substring in prompt.
        # This string is used to locate the subject span for activation masks.
        if alg.upper() == "WISE":
            if not isinstance(req, dict):
                req = dict(req)
            if not req.get("loc_prompt"):
                prompt_text = req.get("prompt", "")
                subj = req.get("subject", "") or heuristic_extract_subject(prompt_text)
                chosen = subj or prompt_text
                # Try to pick the exact substring as it appears in prompt (case-insensitive match)
                if prompt_text and subj:
                    try:
                        import re as _re
                        m = _re.search(_re.escape(subj), prompt_text, flags=_re.IGNORECASE)
                        if m:
                            chosen = prompt_text[m.start():m.end()]
                    except Exception:
                        pass
                req["loc_prompt"] = chosen

        is_multi_rank = dist_world_size > 1
        main_rank = (not is_multi_rank) or dist_rank == 0

        hp_cls, _ = ALG_HP_MAP[alg.upper()]
        base_hp = hp_cls.from_hparams(hparams)
        if model_override:
            base_hp.model_name = model_override
        base_model_id = base_hp.model_name
        try:
            world_env = int(os.environ.get("WORLD_SIZE", "1"))
            local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))
            if world_env > 1 and hasattr(base_hp, "model_parallel") and not getattr(base_hp, "model_parallel", False):
                if torch.cuda.is_available():
                    base_hp.device = int(local_rank_env)
        except Exception:
            pass

        if do_edit and (is_multi_rank or bool(ds_enabled and (ds_mp_size or 0) > 1)) and not save_edited_to:
            raise RuntimeError("--save_edited_to is required when using distributed or tensor-parallel inference.")

        editor = None
        if do_edit and main_rank:
            hp_for_editor = copy.deepcopy(base_hp)
            editor = BaseEditor.from_hparams(hp_for_editor)
        model_for_edit = getattr(editor, "model", None) if editor is not None else None
        tok_for_edit = _unwrap_tokenizer(getattr(editor, "tok", None)) if editor is not None else None
        if tok_for_edit is not None:
            ensure_pad_token(tok_for_edit)
        if isinstance(model_for_edit, nn.Module) and tok_for_edit is not None:
            ensure_tokenizer_model_vocab_alignment(tok_for_edit, model_for_edit, context="[editor]")

        # 安装 Linear 安全前置钩子，避免 cublasLt Matmul 在窄层上崩溃
        try:
            if isinstance(model_for_edit, nn.Module):
                installed = install_linear_safety_hooks(model_for_edit, in_features=3584, max_out_features=64)
                if installed == 0:
                    print("[hook] No Linear layers (3584-><=64) found; safety hook skipped.")
        except Exception as hook_exc:
            warnings.warn(f"[hook] failed to install safety hooks: {hook_exc}")

        # 在编辑阶段降低显存占用：关闭 use_cache、开启梯度检查点；生成阶段再恢复
        use_cache_prev = None
        gckpt_enabled = False

        pred_before, a_for_score_before, rewrite_hit_before = "", "", None
        llm_judge_before: Optional[int] = None
        answer_match_before: Optional[int] = None
        final_short_before = ""
        if eval_before and main_rank and editor is not None:
            if not (ds_enabled and (ds_mp_size or 0) > 1 and dist_world_size > 1 and dist_rank != 0):
                mdl0 = getattr(editor, "model", None)
                tok0 = _unwrap_tokenizer(getattr(editor, "tok", None))
                if mdl0 is None or tok0 is None:
                    bm_id = base_model_id
                    tok0 = AutoTokenizer.from_pretrained(bm_id, trust_remote_code=True)
                    ensure_pad_token(tok0)
                    mdl0 = AutoModelForCausalLM.from_pretrained(
                        bm_id,
                        torch_dtype=pick_dtype(),
                        trust_remote_code=True,
                        device_map=(gen_device_map if gen_device_map else "auto"),
                        max_memory=gen_max_memory,
                    )
                    need_release = True
                else:
                    ensure_pad_token(tok0)
                    need_release = False

                ensure_tokenizer_model_vocab_alignment(tok0, mdl0, context="[eval_before]")

                pred_before = generate_answer(
                    mdl0,
                    tok0,
                    req["prompt"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    mode=gen_mode,
                )
                if gen_mode in {"reason", "r1d"}:
                    final_b, _ = extract_final(pred_before)
                    a_for_score_before = final_b if final_b else pred_before
                else:
                    a_for_score_before = pred_before
                final_short_before = _extract_short_answer(pred_before, gen_mode)
                if dist_rank == 0 and not no_judge:
                    _ensure_judge_loaded()
                if (not no_judge) and dist_rank == 0 and judge_model is not None and judge_tok is not None:
                    judge_golds_before: List[str] = []
                    if target_new_text:
                        judge_golds_before.append(target_new_text)
                    if alt_answer_text:
                        judge_golds_before.append(alt_answer_text)
                    if judge_golds_before:
                        llm_judge_before = judge_hit(
                            judge_tok,
                            judge_model,
                            question=req["prompt"],
                            final=a_for_score_before,
                            golds=judge_golds_before,
                        )
                if alt_answer_norm and final_short_before:
                    answer_match_before = int(_normalize_for_match(final_short_before) == alt_answer_norm)
                if llm_judge_before is not None or answer_match_before is not None:
                    rewrite_hit_before = int(
                        int(llm_judge_before or 0) > 0 or int(answer_match_before or 0) > 0
                    )

                if need_release:
                    try:
                        del mdl0
                        del tok0
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        # 编辑前：对 editor 挂载的模型进行显存友好设置
        try:
            if isinstance(model_for_edit, nn.Module) and getattr(model_for_edit, "config", None) is not None:
                use_cache_prev = getattr(model_for_edit.config, "use_cache", None)
                try:
                    model_for_edit.config.use_cache = False
                except Exception:
                    pass
                try:
                    if hasattr(model_for_edit, "gradient_checkpointing_enable"):
                        model_for_edit.gradient_checkpointing_enable()
                        gckpt_enabled = True
                    if hasattr(model_for_edit, "enable_input_require_grads"):
                        model_for_edit.enable_input_require_grads()
                except Exception:
                    pass
        except Exception:
            pass
        metrics: Any = []
        edited_model = None
        if do_edit and main_rank and editor is not None:
            if alg.upper() == "MEMIT" and isinstance(req, dict) and req.get("memit_bundle"):
                # Use the packed multi-anchor requests for a single unified MEMIT update
                try:
                    edited_model, _ = apply_memit_to_model(
                        editor.model,
                        tok_for_edit if tok_for_edit is not None else _unwrap_tokenizer(editor.tok),
                        req["memit_bundle"],
                            base_hp,  # MEMITHyperParams
                        return_orig_weights=True,
                    )
                except Exception as memit_exc:
                    raise RuntimeError(f"MEMIT apply failed: {memit_exc}")
            else:
                # For IKE, pass train_ds via kwargs
                kwargs = {}
                if alg.upper() == "IKE" and train_ds is not None:
                    kwargs['train_ds'] = train_ds
                metrics, edited_model, _ = editor.edit_requests(requests=[req], sequential_edit=True, **kwargs)

        # 编辑后：恢复以便生成阶段加速
        try:
            if isinstance(edited_model, nn.Module):
                if gckpt_enabled and hasattr(edited_model, "gradient_checkpointing_disable"):
                    try:
                        edited_model.gradient_checkpointing_disable()
                    except Exception:
                        pass
                if getattr(edited_model, "config", None) is not None:
                    try:
                        edited_model.config.use_cache = True if use_cache_prev is None else use_cache_prev
                    except Exception:
                        pass
        except Exception:
            pass

        # 释放编辑期对象，避免占用生成阶段显存：删除 editor/原始编辑模型，清空缓存
        try:
            if 'model_for_edit' in locals():
                del model_for_edit
        except Exception:
            pass
        try:
            if editor is not None:
                del editor
        except Exception:
            pass
        _clear_cuda_cache()
        if tok_for_edit is not None:
            ensure_tokenizer_model_vocab_alignment(tok_for_edit, edited_model, context="[editor.after_edit]")
        if show_eemetrics and dist_rank == 0:
            print("[EASYEDIT METRICS]", json.dumps(metrics, ensure_ascii=False))

        # ===== 保存编辑后模型，并重新加载用于推理（可选） =====
        case_dir = ""
        if do_edit and save_edited_to:
            dist_inited = False
            try:
                import torch.distributed as dist  # type: ignore
                dist_inited = dist.is_initialized()
            except Exception:
                pass

            base_dir = os.path.abspath(os.path.expanduser(save_edited_to))
            if main_rank:
                os.makedirs(base_dir, exist_ok=True)
            tag = save_tag or str(req.get("case_id", ""))
            subdir_name = f"{alg.lower()}-{tag}" if tag else f"{alg.lower()}"
            case_dir = os.path.join(base_dir, subdir_name)

            def _maybe_merge_lora(m):
                try:
                    if PeftModel is not None and isinstance(m, PeftModel):
                        try:
                            return m.merge_and_unload()
                        except Exception:
                            return m
                except Exception:
                    pass
                return m

            # 仅 rank0 执行保存
            if main_rank:
                try:
                    edited_to_save = _maybe_merge_lora(edited_model)
                    if hasattr(edited_to_save, "save_pretrained"):
                        edited_to_save.save_pretrained(case_dir)
                    else:
                        torch.save(getattr(edited_to_save, "state_dict", lambda: {})(), os.path.join(case_dir, "pytorch_model.bin"))
                    # 同步保存 tokenizer（优先编辑阶段的 tokenizer）
                    tok_for_save = tok_for_edit
                    if tok_for_save is not None and hasattr(tok_for_save, "save_pretrained"):
                        try:
                            tok_for_save.save_pretrained(case_dir)
                        except Exception:
                            pass
                    else:
                        # 若编辑阶段未提供 tokenizer，则从 base_model_id 加载一个并保存，确保本地可用
                        try:
                            _tok_tmp = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
                        except Exception:
                            try:
                                _tok_tmp = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True, use_fast=False)
                            except Exception as _tok_load_err:
                                _tok_tmp = None
                                warnings.warn(f"[SAVE] 无法为 {case_dir} 加载 tokenizer 进行保存：{_tok_load_err}")
                        if _tok_tmp is not None:
                            try:
                                _tok_tmp.save_pretrained(case_dir)
                            except Exception as _tok_save_err:
                                warnings.warn(f"[SAVE] 分词器保存失败：{_tok_save_err}")
                    print(f"[SAVE] edited model saved to: {case_dir}")
                except Exception as exc:
                    warnings.warn(f"[SAVE] failed to save edited model: {exc}")

            # 同步各进程
            try:
                if dist_inited:
                    dist.barrier()  # type: ignore[name-defined]
            except Exception:
                pass

            # 释放编辑期对象，减小显存压力
            try:
                if edited_model is not None:
                    del edited_model
            except Exception:
                pass
            _clear_cuda_cache()

            # 重新加载干净的推理模型与分词器
            try:
                tok_for_reload = AutoTokenizer.from_pretrained(case_dir, trust_remote_code=True)
                ensure_pad_token(tok_for_reload)
                tok_for_edit = tok_for_reload
            except Exception:
                pass  # 失败则后续按 base_model_id 重新取
            try:
                # Use safe dtype chosen for DS/accelerate inference
                infer_dtype = _resolve_infer_dtype(ds_dtype)
                edited_model = AutoModelForCausalLM.from_pretrained(
                    case_dir,
                    torch_dtype=infer_dtype,
                    trust_remote_code=True,
                )
            except Exception as exc:
                raise RuntimeError(f"[LOAD] failed to reload edited model from {case_dir}: {exc}")

            # 让后续 tokenizer 加载从保存目录取
            base_model_id = case_dir

        if not do_edit and do_infer:
            if not load_edited_from:
                raise RuntimeError("stage=infer requires --load_edited_from")
            case_dir = os.path.abspath(os.path.expanduser(load_edited_from))
            base_model_id = case_dir
            try:
                tok_for_edit = AutoTokenizer.from_pretrained(case_dir, trust_remote_code=True)
                ensure_pad_token(tok_for_edit)
            except Exception:
                tok_for_edit = None
            infer_dtype = _resolve_infer_dtype(ds_dtype)
            edited_model = AutoModelForCausalLM.from_pretrained(
                case_dir,
                torch_dtype=infer_dtype,
                trust_remote_code=True,
            )

        if do_edit and not do_infer:
            rec = {
                "prompt": req["prompt"],
                "target_new": req["target_new"],
                "saved_model_dir": case_dir,
                "easyedit_metrics": metrics[0] if isinstance(metrics, list) and metrics else metrics,
            }
            return rec

        use_ds_tensor_parallel = bool(ds_enabled and (ds_mp_size or 0) > 1 and dist_world_size > 1)
        if use_ds_tensor_parallel:
            if gen_device_map is not None:
                warnings.warn("已启用 DeepSpeed 张量并行，将忽略 gen_device_map/accelerate 分片。")
            try:
                import deepspeed  # type: ignore
            except Exception:
                warnings.warn("未安装 deepspeed，无法进行多卡 DS 推理，退回单卡/accelerate 模式。")
                use_ds_tensor_parallel = False

        if use_ds_tensor_parallel:
            target_dtype = _resolve_infer_dtype(ds_dtype)
            try:
                import deepspeed  # type: ignore

                # Prefer new API: tensor_parallel over deprecated mp_size/replace_method
                # Avoid double tensor-parallel: disable Transformers AutoTP when using DS TP
                os.environ.setdefault("TRANSFORMERS_AUTO_TP_DISABLE", "1")

                # Provide an injection policy for Llama to improve DS kernel injection stability
                inj_policy = None
                if bool(ds_kernel_inject):
                    try:
                        from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # type: ignore
                        inj_policy = {LlamaDecoderLayer: ["self_attn.o_proj", "mlp.down_proj"]}
                    except Exception:
                        inj_policy = None
                try:
                    engine = deepspeed.init_inference(
                        edited_model,
                        dtype=target_dtype,
                        tensor_parallel={"tp_size": dist_world_size},
                        replace_with_kernel_inject=bool(ds_kernel_inject),
                        injection_policy=inj_policy,
                    )
                except TypeError:
                    # Fallback for older DS: use mp_size without kernel inject deprecations
                    engine = deepspeed.init_inference(
                        edited_model,
                        mp_size=dist_world_size,
                        dtype=target_dtype,
                        replace_with_kernel_inject=bool(ds_kernel_inject),
                    )
                edited_model = getattr(engine, "module", engine)
            except Exception as exc:
                # Second-chance: retry without kernel injection
                try:
                    engine = deepspeed.init_inference(
                        edited_model,
                        dtype=target_dtype,
                        tensor_parallel={"tp_size": dist_world_size},
                        replace_with_kernel_inject=False,
                    )
                    edited_model = getattr(engine, "module", engine)
                except Exception:
                    warnings.warn(f"DeepSpeed 张量并行初始化失败，退回 accelerate/单卡：{exc}")
                    use_ds_tensor_parallel = False
                    if is_multi_rank:
                        raise RuntimeError(f"DeepSpeed inference failed under distributed setup: {exc}") from exc

        if not use_ds_tensor_parallel:
            edited_model = _maybe_dispatch_generation_model(edited_model, gen_device_map, gen_max_memory)
            edited_model = _maybe_apply_ds_inference(
                edited_model,
                use_ds_infer=ds_enabled,
                ds_dtype=("fp16" if _resolve_infer_dtype(ds_dtype) is torch.float16 else "bf16"),
                ds_max_out_tokens=ds_max_out_tokens,
                ds_kernel_inject=ds_kernel_inject,
            )

        # 生成阶段获取 tokenizer：优先本地（case_dir/save_edited_to），再退 base id；fast->slow
        try:
            tok = _safe_load_tokenizer([case_dir, save_edited_to], base_model_id or "")
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        ensure_pad_token(tok)
        ensure_tokenizer_model_vocab_alignment(tok, edited_model, context="[after_edit]")
        pred_after = generate_answer(
            edited_model,
            tok,
            req["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            mode=gen_mode,
        )
        if gen_mode in {"reason", "r1d"}:
            final_after, _ = extract_final(pred_after)
            a_for_score_after = final_after if final_after else pred_after
        else:
            a_for_score_after = pred_after
        final_short_after = _extract_short_answer(pred_after, gen_mode)

        llm_judge_after: Optional[int] = None
        if dist_rank == 0 and not no_judge:
            _ensure_judge_loaded()
        if (not no_judge) and dist_rank == 0 and judge_model is not None and judge_tok is not None:
            judge_golds_after: List[str] = []
            if target_new_text:
                judge_golds_after.append(target_new_text)
            if alt_answer_text:
                judge_golds_after.append(alt_answer_text)
            if judge_golds_after:
                llm_judge_after = judge_hit(
                    judge_tok,
                    judge_model,
                    question=req["prompt"],
                    final=a_for_score_after,
                    golds=judge_golds_after,
                )

        answer_match_after: Optional[int] = None
        if alt_answer_norm and final_short_after:
            answer_match_after = int(_normalize_for_match(final_short_after) == alt_answer_norm)

        rewrite_hit_after: Optional[int] = None
        if llm_judge_after is not None or answer_match_after is not None:
            rewrite_hit_after = int(
                int(llm_judge_after or 0) > 0 or int(answer_match_after or 0) > 0
            )

        loc_rec: Dict[str, Any] = {}
        if (not skip_locality) and isinstance(req.get("locality"), dict) and "nq" in req["locality"]:
            lp = req["locality"]["nq"].get("prompt", "")
            lg = req["locality"]["nq"].get("ground_truth", "")
            if lp and lg:
                loc_pred = generate_answer(
                    edited_model,
                    tok,
                    lp,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    mode=gen_mode,
                )
                loc_final, _ = extract_final(loc_pred) if gen_mode in {"reason", "r1d"} else (loc_pred, "")
                if dist_rank == 0 and not no_judge:
                    _ensure_judge_loaded()
                loc_hit = (
                    judge_hit(judge_tok, judge_model, question=lp, final=(loc_final or loc_pred), golds=[lg])
                    if (not no_judge) and dist_rank == 0 and judge_model is not None and judge_tok is not None
                    else None
                )
                loc_rec = {
                    "loc_prompt": lp,
                    "loc_gold": lg,
                    "pred_loc": loc_pred,
                    "final_loc": loc_final or "",
                    "locality_hit": (int(loc_hit) if loc_hit is not None else None),
                }

        rec_rewrite_hit = int(rewrite_hit_after) if rewrite_hit_after is not None else None
        # 记录完整输出，同时附加分段字段（final/think or reasoning）
        rec: Dict[str, Any] = {
            "prompt": req["prompt"],
            "target_new": req["target_new"],
            "pred_after": pred_after,
            "rewrite_hit": rec_rewrite_hit,
            "easyedit_metrics": metrics[0] if isinstance(metrics, list) and metrics else metrics,
        }
        if isinstance(req, dict) and req.get("memit_bundle"):
            try:
                rec["memit_anchor_count"] = len(req["memit_bundle"])  # type: ignore[index]
                rec["case_id"] = req["memit_bundle"][0].get("case_id", "")  # type: ignore[index]
            except Exception:
                pass
        if gen_mode in {"reason", "r1d"}:
            fa, rs = extract_final(pred_after)
            if fa:
                rec["final_after"] = fa
            if rs:
                # 对 reason 模式是 <reasoning>，对 r1d 模式是 <think>
                rec["reasoning_or_think_after"] = rs
        elif final_short_after:
            rec["final_after"] = final_short_after
        if llm_judge_after is not None:
            rec["llm_judge_after"] = int(llm_judge_after)
        if answer_match_after is not None:
            rec["answer_match_after"] = int(answer_match_after)
        if eval_before:
            rec["pred_before"] = pred_before
            rec["rewrite_hit_before"] = int(rewrite_hit_before) if rewrite_hit_before is not None else None
            if llm_judge_before is not None:
                rec["llm_judge_before"] = int(llm_judge_before)
            if answer_match_before is not None:
                rec["answer_match_before"] = int(answer_match_before)
            if gen_mode in {"reason", "r1d"}:
                fb, rb = extract_final(pred_before)
                if fb:
                    rec["final_before"] = fb
                if rb:
                    rec["reasoning_or_think_before"] = rb
            elif final_short_before:
                rec["final_before"] = final_short_before

        rec.update(loc_rec)
        rec.setdefault("loc_prompt", "")
        rec.setdefault("loc_gold", "")
        rec.setdefault("pred_loc", "")
        rec.setdefault("locality_hit", None)

        # ===== 生成结束后按需删除保存目录 =====
        if delete_saved_after and case_dir:
            dist_inited = False
            try:
                import torch.distributed as dist  # type: ignore
                dist_inited = dist.is_initialized()
            except Exception:
                pass
            # 仅 rank0 删除
            if (not dist_inited) or dist_rank == 0:
                try:
                    if os.path.isdir(case_dir):
                        shutil.rmtree(case_dir)
                        print(f"[CLEANUP] removed saved dir: {case_dir}")
                except Exception as exc:
                    warnings.warn(f"[CLEANUP] failed to remove {case_dir}: {exc}")
            try:
                if dist_inited:
                    dist.barrier()  # type: ignore[name-defined]
            except Exception:
                pass

        return rec
    finally:
        # 防止变量已被提前 del，使用 locals() 安全释放
        try:
            if 'edited_model' in locals() and locals().get('edited_model') is not None:
                del edited_model  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            if 'editor' in locals() and locals().get('editor') is not None:
                del editor  # type: ignore[name-defined]
        except Exception:
            pass
        if judge_loaded and judge_manager is not None:
            judge_manager.release_if_needed()
        _clear_cuda_cache()


# ===================== 主流程（支持 repeat） =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=SUPPORTED_ALG_NAMES, help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=False, default="", help="ZSRE-like JSON path (ignored if --anchors_jsonl is set)")
    ap.add_argument("--train_data_path", required=False, default="", help="Path to training examples for IKE (example pool for in-context retrieval)")
    ap.add_argument("--model", default="", help="HF model id/path to override YAML")
    ap.add_argument("--case_index", type=int, default=0, help="Start index in dataset")
    ap.add_argument("--repeat", type=int, default=1, help="How many consecutive cases to run")
    ap.add_argument("--wrap", action="store_true", help="Wrap around dataset if overflow")
    ap.add_argument("--no-reset_each", dest="reset_each", action="store_false",
                    help="Apply edits cumulatively on same process (default: reset each)")
    ap.add_argument("--judge_model", required=False, help="HF id/path for LLM Judge")
    ap.add_argument("--no_judge", action="store_true", help="跳过全部裁判打分与加载，仅进行编辑与生成")

    # 生成控制
    ap.add_argument("--gen_mode", choices=["concise","reason","noprompt","r1d"], default="concise")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gen_device_map", default="auto", help="多卡推理 device_map 配置，支持 auto/balanced/balanced_low_0/sequential 或 JSON 映射")
    ap.add_argument("--gen_max_memory", default="", help="多卡推理时传给 accelerate 的 max_memory JSON，例如 '{\"cuda:0\": \"20GiB\", \"cuda:1\": \"20GiB\"}'")
    # Judge placement
    ap.add_argument("--judge_device_map", default="auto", help="裁判模型多卡 device_map（auto/… 或 JSON），默认 auto")
    ap.add_argument("--judge_max_memory", default="", help="裁判模型 max_memory JSON，如 '{\"cuda:2\": \"20GiB\", \"cuda:3\": \"20GiB\"}'")
    ap.add_argument("--keep_judge_loaded", action="store_true", help="多条 case 共用同一个裁判模型实例，不在每次后释放显存")
    # DeepSpeed Inference（单卡 kernel injection）
    ap.add_argument("--use_ds_infer", action="store_true", help="编辑后生成阶段启用 DeepSpeed Inference kernel injection（单进程/单卡优先）")
    ap.add_argument("--ds_dtype", choices=["auto","bf16","fp16"], default="auto", help="DeepSpeed Inference 计算精度")
    ap.add_argument("--ds_max_out_tokens", type=int, default=0, help="DeepSpeed 预分配最大输出 token 数（>0 启用，可减少动态再分配开销）")
    ap.add_argument("--no_ds_kernel_inject", dest="ds_kernel_inject", action="store_false", help="关闭 DeepSpeed kernel injection，仅做空包装")
    ap.set_defaults(ds_kernel_inject=True)
    ap.add_argument("--ds_mp_size", type=int, default=1, help="DeepSpeed 张量并行大小（>1 需配合 torchrun 多进程启动）")

    # 评测控制
    ap.add_argument("--eval_before", action="store_true", help="Also judge before editing (to prove effect)")
    ap.add_argument("--show_eemetrics", action="store_true")
    ap.add_argument("--skip_locality", action="store_true", help="Skip locality eval to save time")

    # 输出
    ap.add_argument("--save_jsonl", default="", help="Where to append results (JSONL). If empty, just print.")
    ap.add_argument("--print_every", type=int, default=1)
    # 保存/清理
    ap.add_argument("--save_edited_to", default="", help="保存编辑后模型到该目录（每个case建子目录）")
    ap.add_argument("--delete_saved_after", action="store_true", help="生成后删除该case保存目录")
    ap.add_argument("--stage", choices=["full", "edit", "infer"], default="full", help="Run full pipeline, edit-only, or infer-only stage.")
    ap.add_argument("--load_edited_from", default="", help="When stage=infer, load edited model from this directory")
    # Connectivity
    ap.add_argument("--offline", action="store_true", help="Offline mode: do not contact Hugging Face Hub; require local files")

    # MEMIT anchors (optional): if provided, override data_path and build a single case
    ap.add_argument("--anchors_jsonl", default="", help="Path to MEMIT anchors JSONL (from scripts/cot_to_memit_anchors.py)")
    ap.add_argument("--anchors_case_id", default="", help="Case id to pick from anchors JSONL")
    ap.add_argument("--anchors_take", type=int, default=6, help="Max anchors to take for one MEMIT update")

    args = ap.parse_args()

    # Honor offline mode early to avoid network retries/timeouts
    if getattr(args, "offline", False):
        os.environ["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE", "1")
        os.environ["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE", "1")

    if not args.no_judge and not args.judge_model:
        raise SystemExit("需要提供 --judge_model，或使用 --no_judge 跳过裁判。")

    if args.stage == "infer" and not args.load_edited_from:
        raise SystemExit("stage=infer 需要提供 --load_edited_from。")

    # 速度小优化：如支持则启用 TF32（对大多数 A100/RTX40 有收益）
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    try:
        gen_device_map = _parse_device_map_arg(args.gen_device_map)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        gen_max_memory = _parse_max_memory_arg(args.gen_max_memory)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        judge_device_map = _parse_device_map_arg(args.judge_device_map)
    except ValueError as exc:
        raise SystemExit(str(exc))
    try:
        judge_max_memory = _parse_max_memory_arg(args.judge_max_memory)
    except ValueError as exc:
        raise SystemExit(str(exc))

    generation_cfg = GenerationConfig(
        mode=args.gen_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    ds_cfg = DeepSpeedConfig(
        enabled=args.use_ds_infer,
        dtype=args.ds_dtype,
        max_out_tokens=args.ds_max_out_tokens,
        kernel_inject=args.ds_kernel_inject,
        mp_size=args.ds_mp_size,
    )
    judge_manager = None
    if not args.no_judge:
        judge_cfg = JudgeConfig(
            model_id=args.judge_model,
            device_map=judge_device_map,
            max_memory=judge_max_memory,
            keep_loaded=args.keep_judge_loaded,
        )
        judge_manager = JudgeManager(judge_cfg)

    # If user didn't provide max_memory and has multi-GPU, derive a safe default for gen.
    if gen_device_map and isinstance(gen_device_map, str) and gen_device_map in {"auto","balanced","balanced_low_0"}:
        if not gen_max_memory and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                mm = {}
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    # leave ~10% headroom; use total for stable cap
                    cap_gib = max(1, int((total / (1024**3)) * 0.9))
                    mm[i] = f"{cap_gib}GiB"
                gen_max_memory = mm
            except Exception:
                pass

    # 若用户要求 DeepSpeed 多卡推理，则尝试初始化分布式
    is_dist, dist_rank, dist_world, dist_local = _init_distributed_if_needed(
        enable=bool(args.use_ds_infer and (args.ds_mp_size or 0) > 1)
    )
    if args.use_ds_infer and (args.ds_mp_size or 0) > 1 and not is_dist:
        warnings.warn("要求 ds_mp_size>1 但未检测到分布式环境（需 torchrun 启动），将退回单进程模式。")

    # Build requests source
    if args.anchors_jsonl:
        if not args.anchors_case_id:
            raise SystemExit("--anchors_jsonl 需要配合 --anchors_case_id")
        one = _read_memit_anchors_group(args.anchors_jsonl, args.anchors_case_id, take=args.anchors_take)
        all_reqs = [one]
    else:
        if not args.data_path:
            raise SystemExit("需要提供 --data_path 或 (--anchors_jsonl + --anchors_case_id)")
        all_reqs = read_zsre_like(args.data_path)
        if not all_reqs:
            raise RuntimeError(f"No valid requests parsed from {args.data_path}")
    N = len(all_reqs)

    # Load training examples for IKE (in-context retrieval pool)
    train_ds = None
    if args.train_data_path and args.alg.upper() == "IKE":
        try:
            with open(args.train_data_path, "r", encoding="utf-8") as f:
                train_ds_raw = json.load(f)

            # Convert field names to match IKE expectations: src/alt -> prompt/target_new
            train_ds = []
            for item in train_ds_raw:
                converted = {
                    "prompt": item.get("prompt") or item.get("src", ""),
                    "target_new": item.get("target_new") or item.get("alt", ""),
                }
                # Optionally preserve other fields that IKE might use
                if "rephrase" in item:
                    converted["rephrase_prompt"] = item["rephrase"]
                if "loc" in item:
                    converted["locality_prompt"] = item.get("loc", "")
                    converted["locality_ground_truth"] = item.get("loc_ans", "")
                train_ds.append(converted)

            print(f"[IKE] Loaded {len(train_ds)} training examples from {args.train_data_path}")

            # Generate embedding cache for IKE
            from easyeditor.models.ike import encode_ike_facts
            from sentence_transformers import SentenceTransformer

            # Load hyperparameters to get sentence_model_name and results_dir
            hp_cls, _ = ALG_HP_MAP["IKE"]
            ike_hp = hp_cls.from_hparams(args.hparams)
            if args.model:
                ike_hp.model_name = args.model

            # Check if embedding cache already exists
            safe_model_name = ike_hp.sentence_model_name.rsplit('/', 1)[-1]
            cache_path = f'{ike_hp.results_dir}/{ike_hp.alg_name}/embedding/{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl'

            if not os.path.exists(cache_path):
                print(f"[IKE] Generating embedding cache at {cache_path}...")
                sentence_model = SentenceTransformer(ike_hp.sentence_model_name).to(f'cuda:{ike_hp.device}')
                encode_ike_facts(sentence_model, train_ds, ike_hp)
                print(f"[IKE] Embedding cache generated successfully")
            else:
                print(f"[IKE] Using existing embedding cache at {cache_path}")
        except Exception as e:
            warnings.warn(f"Failed to load train_data from {args.train_data_path}: {e}")
            train_ds = None

    fw = None
    if args.save_jsonl:
        path = os.path.abspath(os.path.expanduser(args.save_jsonl))
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fw = open(path, "a", encoding="utf-8")

    base_idx = args.case_index
    for t in range(args.repeat):
        idx = base_idx + t
        if idx >= N:
            if args.wrap:
                idx = idx % N
            else:
                break

        req = copy.deepcopy(all_reqs[idx])
        try:
            rec = run_one_case(
                req=req,
                alg=args.alg,
                hparams=args.hparams,
                model_override=args.model,
                generation_cfg=generation_cfg,
                judge_manager=judge_manager,
                eval_before=args.eval_before,
                show_eemetrics=args.show_eemetrics,
                gen_device_map=gen_device_map,
                gen_max_memory=gen_max_memory,
                skip_locality=args.skip_locality,
                ds_cfg=ds_cfg,
                dist_rank=dist_rank,
                dist_world_size=dist_world,
                dist_local_rank=dist_local,
                no_judge=bool(args.no_judge),
                save_edited_to=args.save_edited_to,
                delete_saved_after=args.delete_saved_after,
                save_tag=str(idx),
                stage=args.stage,
                load_edited_from=args.load_edited_from,
                train_ds=train_ds,
            )
            rec["case_index"] = idx
        except Exception as e:
            rec = {"case_index": idx, "error": str(e)}
            warnings.warn(f"[WARN] case {idx} failed: {e}")

        # 仅在 rank0 打印/写文件，避免多进程重复输出
        if (not is_dist) or dist_rank == 0:
            if fw:
                fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fw.flush()
            if (t + 1) % max(1, args.print_every) == 0 or not fw:
                print(json.dumps(rec, ensure_ascii=False))

        # 显存清理（当前循环结束）
        _clear_cuda_cache()

        # reset_each=True 时，每次 run_one_case 内都会重建 editor（等价于从干净基座开始）
        # 这里无需额外处理
        if args.reset_each:
            pass

    if fw:
        fw.close()

    if judge_manager is not None:
        judge_manager.shutdown()

if __name__ == "__main__":
    main()
