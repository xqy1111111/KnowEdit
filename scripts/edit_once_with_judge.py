# scripts/edit_once_and_judge.py
# -*- coding: utf-8 -*-
import os
import re
import gc
import json
import copy
import argparse
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import timedelta

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from easyeditor import (
    BaseEditor,
    ROMEHyperParams,
    WISEHyperParams,
    FTHyperParams,
    LoRAHyperParams,
    QLoRAHyperParams,
)

# Improve CUDA memory fragmentation resilience unless user overrides.
# Effective as long as set before first CUDA allocation.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Optional judge placement controls (set in main)
JUDGE_DEVICE_MAP: Optional[Union[str, Dict[str, Any]]] = None
JUDGE_MAX_MEMORY: Optional[Dict[str, Any]] = None

# Judge/device management toggles (tweaked via CLI in main)
DEFAULT_MIN_FREE_GPU_GIB = 3.0
FORCE_CPU_JUDGE = False
JUDGE_CACHE_ENABLED = True
MIN_FREE_GPU_GIB = DEFAULT_MIN_FREE_GPU_GIB


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


def _is_cpu_device_map(device_map: Optional[Union[str, Dict[str, Any]]]) -> bool:
    if device_map is None:
        return False
    if isinstance(device_map, str):
        return device_map.lower() == "cpu"
    if isinstance(device_map, dict):
        return all(
            (isinstance(v, str) and v.lower() in {"cpu", "disk"})
            or (isinstance(v, int) and v < 0)
            for v in device_map.values()
        )
    return False


def _should_force_cpu_for_judge(user_device_map: Optional[Union[str, Dict[str, Any]]]) -> bool:
    if FORCE_CPU_JUDGE:
        return True
    if user_device_map is not None and not (
        isinstance(user_device_map, str) and user_device_map.lower() in {"", "auto", "balanced", "balanced_low_0", "sequential"}
    ):
        # User provided explicit mapping; respect it.
        return False
    if not torch.cuda.is_available():
        return True
    if MIN_FREE_GPU_GIB <= 0:
        return False
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / (1024 ** 3)
    except Exception:
        return False
    return free_gib < MIN_FREE_GPU_GIB

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

def ensure_pad_token(tokenizer: AutoTokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

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

    # 解析 dtype
    if ds_dtype == "bf16":
        target_dtype = torch.bfloat16
    elif ds_dtype == "fp16":
        target_dtype = torch.float16
    else:
        # auto：尽量用 bf16，其次 fp16
        target_dtype = pick_dtype()

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
        data = json.load(f)
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
    return ""

# ===================== 构建 Editor =====================
ALG_HP_MAP = {
    "ROME": (ROMEHyperParams, "ROME"),
    "WISE": (WISEHyperParams, "WISE"),
    "FT": (FTHyperParams, "FT"),
    "LORA": (LoRAHyperParams, "LoRA"),
    "QLORA": (QLoRAHyperParams, "QLoRA"),
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
            "Think step by step INSIDE <reasoning>...</reasoning>. "
            "Then output ONLY the final short answer INSIDE <final>...</final>. "
            "Do not include anything else outside these tags."
        )
    else:
        sys_msg = "Answer with a short noun phrase only. Do NOT explain."
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": prompt}
    ]

def generate_answer(model, tokenizer, prompt: str,
                    max_new_tokens=64, temperature=0.6, top_p=0.95,
                    mode: str = "concise") -> str:
    messages = _build_messages(prompt, mode=mode)
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if mode == "reason":
            text = (
                "You are a helpful assistant. Think step by step INSIDE <reasoning>...</reasoning>. less but useful steps are appriciated"
                "Then output ONLY the final short answer INSIDE <final>...</final>.\n"
                f"Q: {prompt}\nA:"
            )
        else:
            text = f"Answer with a short noun phrase only. Do NOT explain.\nQ: {prompt}\nA:"

    ensure_pad_token(tokenizer)
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
        )
    ans = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return ans.strip()

# ===================== 解析 <final> =====================
_FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)
def extract_final(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    m_f = _FINAL_RE.search(text)
    m_r = _REASON_RE.search(text)
    final = (m_f.group(1).strip() if m_f else "").strip("：:").strip()
    reason = (m_r.group(1).strip() if m_r else "").strip()
    return final, reason

# ===================== LLM-Judge =====================
_JUDGE_CACHE = {}
def load_judge(model_id: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    if JUDGE_CACHE_ENABLED and model_id in _JUDGE_CACHE:
        return _JUDGE_CACHE[model_id]

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    ensure_pad_token(tok)

    # Respect optional judge placement controls; default to 'auto' if unset.
    user_dm = JUDGE_DEVICE_MAP if JUDGE_DEVICE_MAP is not None else "auto"
    force_cpu = _should_force_cpu_for_judge(user_dm)
    resolved_dm: Optional[Union[str, Dict[str, Any]]] = "cpu" if force_cpu else user_dm
    cpu_target = _is_cpu_device_map(resolved_dm)

    def _load(resolved_map, dtype, max_mem=None, low_cpu=False):
        kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
        if resolved_map is not None:
            kwargs["device_map"] = resolved_map
        if max_mem is not None:
            kwargs["max_memory"] = max_mem
        if low_cpu:
            kwargs["low_cpu_mem_usage"] = True
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    dtype = torch.float32 if cpu_target else pick_dtype()
    max_mem = None if cpu_target else JUDGE_MAX_MEMORY

    try:
        mdl = _load(resolved_dm, dtype, max_mem=max_mem, low_cpu=cpu_target)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if not cpu_target and ("cuda out of memory" in msg or "cublas" in msg or "mmalloc" in msg):
            warnings.warn("裁判模型加载触发显存不足，将退回 CPU 运行。")
            dtype = torch.float32
            mdl = _load("cpu", dtype, max_mem=None, low_cpu=True)
            cpu_target = True
            resolved_dm = "cpu"
        else:
            raise
    except Exception as exc:
        warnings.warn(f"加载裁判模型时多卡/内存配置失败，将退回单设备：{exc}")
        dtype = torch.float32 if force_cpu else pick_dtype()
        fallback_dm = "cpu" if force_cpu else None
        mdl = _load(fallback_dm, dtype, max_mem=None if fallback_dm == "cpu" else JUDGE_MAX_MEMORY, low_cpu=(fallback_dm == "cpu"))

    if JUDGE_CACHE_ENABLED:
        _JUDGE_CACHE[model_id] = (tok, mdl)
    return tok, mdl


def release_judge(model_id: Optional[str] = None) -> None:
    if model_id is not None:
        entries = [model_id]
    else:
        entries = list(_JUDGE_CACHE.keys())
    for mid in entries:
        pair = _JUDGE_CACHE.pop(mid, None)
        if pair is None:
            continue
        tok, mdl = pair
        try:
            del tok
        except Exception:
            pass
        try:
            del mdl
        except Exception:
            pass
    _clear_cuda_cache()

JUDGE_PROMPT = """You are a precise grader.
Only judge the FINAL_ANSWER string; ignore any hidden thoughts/reasoning if present.
Normalize: lowercase, strip accents/punctuation; minor typos are acceptable.
If FINAL_ANSWER semantically matches ANY of the GOLD items, answer YES; otherwise NO.

Return exactly one token: YES or NO.

GOLD: {golds}
FINAL_ANSWER: {final}

Decision (YES or NO):"""

def judge_hit(judge_model_id: str, question: str, final: str, golds: List[str],
              max_new_tokens: int = 8, debug: bool = False) -> int:
    tok, mdl = load_judge(judge_model_id)
    prompt = JUDGE_PROMPT.format(question=question, golds=", ".join(golds), final=final)
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    if debug:
        print("[LLM-JUDGE RAW]:", txt)
    return int(("YES" in txt) and ("NO" not in txt))

# ===================== 单条：可选“编辑前” + 编辑 + 评测 =====================
def run_one_case(
    req: Dict[str, Any],
    alg: str,
    hparams: str,
    model_override: str,
    judge_model: str,
    gen_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eval_before: bool = False,
    show_eemetrics: bool = False,
    gen_device_map: Optional[Union[str, Dict[str, Any]]] = None,
    gen_max_memory: Optional[Dict[str, Any]] = None,
    skip_locality: bool = False,
    use_ds_infer: bool = False,
    ds_dtype: str = "auto",
    ds_max_out_tokens: int = 0,
    ds_kernel_inject: bool = True,
    ds_mp_size: int = 1,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    dist_local_rank: int = 0,
) -> Dict[str, Any]:

    # 0) ROME subject 准备
    if alg.upper() == "ROME":
        s = req.get("subject", "")
        if not s:
            s = heuristic_extract_subject(req["prompt"])
            if s:
                warnings.warn(f"[ROME] subject missing; heuristically extracted '{s}'")
            else:
                raise ValueError(f"[ROME] cannot find subject for prompt: {req['prompt']}")
        if s not in req["prompt"]:
            req = dict(req)
            req["prompt"] = f"{req['prompt']} (Subject: {s})"
            req["subject"] = s

    # 构建编辑器（优先一次性加载，避免重复占显存）
    base_model_id = model_override  # 若为空，后面会从 editor.hparams 取
    editor = build_editor(alg, hparams, model_override)
    if not base_model_id:
        base_model_id = getattr(editor.hparams, "model_name", "")

    #（可选）编辑前评测（尽量直接用已加载的 editor.model 与 editor.tok）
    pred_before, a_for_score_before, rewrite_hit_before = "", "", None
    if eval_before:
        if not (use_ds_infer and (ds_mp_size or 0) > 1 and dist_world_size > 1 and dist_rank != 0):
            mdl0 = getattr(editor, "model", None)
            tok0 = _unwrap_tokenizer(getattr(editor, "tok", None))
            if mdl0 is None or tok0 is None:
                # 回退到单独加载一次基座模型
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

            pred_before = generate_answer(
                mdl0, tok0, req["prompt"],
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                mode=gen_mode
            )
            if gen_mode == "reason":
                final_b, _ = extract_final(pred_before)
                a_for_score_before = final_b if final_b else pred_before
            else:
                a_for_score_before = pred_before
            # judge 仅在 rank0 评测输出
            if dist_rank == 0:
                rewrite_hit_before = judge_hit(judge_model, question=req["prompt"], final=a_for_score_before, golds=[req.get("target_new","")])

            # 如有必要释放回退加载的模型
            if need_release:
                try:
                    del mdl0; del tok0
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception:
                    pass

    # 执行编辑（单条）。关键：sequential_edit=True 以保留编辑后的权重供外部生成。
    metrics, edited_model, _ = editor.edit_requests(requests=[req], sequential_edit=True)
    if show_eemetrics and dist_rank == 0:
        print("[EASYEDIT METRICS]", json.dumps(metrics, ensure_ascii=False))
    
    # 选择推理并行方案
    use_ds_tensor_parallel = bool(use_ds_infer and (ds_mp_size or 0) > 1 and dist_world_size > 1)
    if use_ds_tensor_parallel:
        # 若选择 DS 张量并行，多数情况下不要再用 accelerate 分片
        if gen_device_map is not None:
            warnings.warn("已启用 DeepSpeed 张量并行，将忽略 gen_device_map/accelerate 分片。")
        try:
            import deepspeed
        except Exception:
            warnings.warn("未安装 deepspeed，无法进行多卡 DS 推理，退回单卡/accelerate 模式。")
            use_ds_tensor_parallel = False

    if use_ds_tensor_parallel:
        # 在多进程环境下，每个 rank 都有一份相同的 edited_model，然后做 DS TP 注入。
        target_dtype = torch.bfloat16 if ds_dtype == "bf16" else (torch.float16 if ds_dtype == "fp16" else pick_dtype())
        try:
            import deepspeed
            engine = deepspeed.init_inference(
                edited_model,
                mp_size=dist_world_size,
                dtype=target_dtype,
                replace_method="auto",
                replace_with_kernel_inject=bool(ds_kernel_inject),
            )
            edited_model = engine.module
        except Exception as exc:
            warnings.warn(f"DeepSpeed 张量并行初始化失败，退回 accelerate/单卡：{exc}")
            use_ds_tensor_parallel = False

    if not use_ds_tensor_parallel:
        # 仍然可用 accelerate 多卡分片，或单卡
        edited_model = _maybe_dispatch_generation_model(edited_model, gen_device_map, gen_max_memory)
        # 单卡 kernel 注入（若开启）
        edited_model = _maybe_apply_ds_inference(
            edited_model,
            use_ds_infer=use_ds_infer,
            ds_dtype=ds_dtype,
            ds_max_out_tokens=ds_max_out_tokens,
            ds_kernel_inject=ds_kernel_inject,
        )
    # 这里不再重复注入 DeepSpeed，避免显存重复占用

    # 用编辑后模型生成（直接复用 editor.tok，保持与训练一致）
    tok = _unwrap_tokenizer(getattr(editor, "tok", None)) or AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    ensure_pad_token(tok)
    pred_after = generate_answer(
        edited_model, tok, req["prompt"],
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
        mode=gen_mode
    )
    if gen_mode == "reason":
        final_after, _ = extract_final(pred_after)
        a_for_score_after = final_after if final_after else pred_after
    else:
        a_for_score_after = pred_after

    rewrite_hit_after = judge_hit(judge_model, question=req["prompt"], final=a_for_score_after, golds=[req.get("target_new","")]) if dist_rank == 0 else None

    # Locality（若提供）
    loc_rec: Dict[str, Any] = {}
    if (not skip_locality) and isinstance(req.get("locality"), dict) and "nq" in req["locality"]:
        lp = req["locality"]["nq"].get("prompt", "")
        lg = req["locality"]["nq"].get("ground_truth", "")
        if lp and lg:
            loc_pred = generate_answer(
                edited_model, tok, lp,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                mode=gen_mode
            )
            loc_final, _ = extract_final(loc_pred) if gen_mode == "reason" else (loc_pred, "")
            loc_hit = judge_hit(judge_model, question=lp, final=(loc_final or loc_pred), golds=[lg]) if dist_rank == 0 else None
            loc_rec = {
                "loc_prompt": lp,
                "loc_gold": lg,
                "pred_loc": loc_pred,
                "locality_hit": (int(loc_hit) if loc_hit is not None else None),
            }

    # E) 组装记录（存在才写）
    rec_rewrite_hit = int(rewrite_hit_after) if rewrite_hit_after is not None else None
    rec: Dict[str, Any] = {
        "prompt": req["prompt"],
        "target_new": req["target_new"],
        "pred_after": pred_after,
        "rewrite_hit": rec_rewrite_hit,
        "easyedit_metrics": metrics[0] if isinstance(metrics, list) and metrics else metrics,
    }
    if eval_before:
        rec["pred_before"] = pred_before
        rec["rewrite_hit_before"] = (int(rewrite_hit_before) if rewrite_hit_before is not None else None)

    rec.update(loc_rec)
        # --- MINIMAL PATCH: always provide locality keys to avoid KeyError ---
    rec.setdefault("loc_prompt", "")
    rec.setdefault("loc_gold", "")
    rec.setdefault("pred_loc", "")
    rec.setdefault("locality_hit", None)
    # ---------------------------------------------------------------------

    # 资源清理，避免多次 case 叠加显存/内存
    try:
        del edited_model
    except Exception:
        pass
    try:
        del editor
    except Exception:
        pass
    if not JUDGE_CACHE_ENABLED:
        release_judge(judge_model)
    _clear_cuda_cache()

    return rec


# ===================== 主流程（支持 repeat） =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=SUPPORTED_ALG_NAMES, help="Editing algorithm")
    ap.add_argument("--hparams", required=True, help="Path to YAML under hparams/")
    ap.add_argument("--data_path", required=True, help="ZSRE-like JSON path")
    ap.add_argument("--model", default="", help="HF model id/path to override YAML")
    ap.add_argument("--case_index", type=int, default=0, help="Start index in dataset")
    ap.add_argument("--repeat", type=int, default=1, help="How many consecutive cases to run")
    ap.add_argument("--wrap", action="store_true", help="Wrap around dataset if overflow")
    ap.add_argument("--no-reset_each", dest="reset_each", action="store_false",
                    help="Apply edits cumulatively on same process (default: reset each)")
    ap.add_argument("--judge_model", required=True, help="HF id/path for LLM Judge")

    # 生成控制
    ap.add_argument("--gen_mode", choices=["concise","reason"], default="concise")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gen_device_map", default="auto", help="多卡推理 device_map 配置，支持 auto/balanced/balanced_low_0/sequential 或 JSON 映射")
    ap.add_argument("--gen_max_memory", default="", help="多卡推理时传给 accelerate 的 max_memory JSON，例如 '{\"cuda:0\": \"20GiB\", \"cuda:1\": \"20GiB\"}'")
    # Judge placement
    ap.add_argument("--judge_device_map", default="auto", help="裁判模型多卡 device_map（auto/… 或 JSON），默认 auto")
    ap.add_argument("--judge_max_memory", default="", help="裁判模型 max_memory JSON，如 '{\"cuda:2\": \"20GiB\", \"cuda:3\": \"20GiB\"}'")
    ap.add_argument("--force_cpu_judge", action="store_true", help="强制裁判模型在 CPU 上运行，避免与编辑模型争抢显存")
    ap.add_argument("--min_free_gpu_gib", type=float, default=DEFAULT_MIN_FREE_GPU_GIB,
                    help="当检测到可用显存低于该阈值 (GiB) 时，自动将裁判模型回退到 CPU；设为 <=0 可关闭自动回退")
    ap.add_argument("--no_judge_cache", dest="judge_cache", action="store_false",
                    help="每条 case 结束后释放裁判模型，减少持续显存占用（会重复加载，略慢）")
    ap.set_defaults(judge_cache=True)
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

    args = ap.parse_args()

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

    # Stash judge placement knobs for load_judge()
    global JUDGE_DEVICE_MAP, JUDGE_MAX_MEMORY, FORCE_CPU_JUDGE, JUDGE_CACHE_ENABLED, MIN_FREE_GPU_GIB
    JUDGE_DEVICE_MAP = judge_device_map
    JUDGE_MAX_MEMORY = judge_max_memory
    FORCE_CPU_JUDGE = bool(args.force_cpu_judge)
    JUDGE_CACHE_ENABLED = bool(args.judge_cache)
    MIN_FREE_GPU_GIB = float(args.min_free_gpu_gib)

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

    all_reqs = read_zsre_like(args.data_path)
    if not all_reqs:
        raise RuntimeError(f"No valid requests parsed from {args.data_path}")
    N = len(all_reqs)

    fw = None
    if args.save_jsonl:
        path = os.path.abspath(os.path.expanduser(args.save_jsonl))
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
                judge_model=args.judge_model,
                gen_mode=args.gen_mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eval_before=args.eval_before,
                show_eemetrics=args.show_eemetrics,
                gen_device_map=gen_device_map,
                gen_max_memory=gen_max_memory,
                skip_locality=args.skip_locality,
                use_ds_infer=args.use_ds_infer,
                ds_dtype=args.ds_dtype,
                ds_max_out_tokens=args.ds_max_out_tokens,
                ds_kernel_inject=args.ds_kernel_inject,
                ds_mp_size=args.ds_mp_size,
                dist_rank=dist_rank,
                dist_world_size=dist_world,
                dist_local_rank=dist_local,
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

    # 保守起见：主流程结束后尝试释放裁判模型缓存
    release_judge()

if __name__ == "__main__":
    main()
