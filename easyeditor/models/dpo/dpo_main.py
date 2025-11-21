from copy import deepcopy
from typing import Any, Dict, List, Tuple
from peft import get_peft_model, AdaLoraConfig, TaskType, get_peft_model_state_dict, set_peft_model_state_dict, LoraConfig
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dpo_hparams import DPOHyperParams

def apply_dpo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    """
    weights_copy = {}
    if copy:
        # If you need to copy the model, handle it here
        pass  # Avoid deep copying to save memory

    device = torch.device(f'cuda:{hparams.device}')
    print(f"Using device: {device}")

    # Configure LoRA
    Config = LoraConfig

    peft_config = Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules
    )
    # Add LoRA modules to the model
    peft_model = get_peft_model(model, peft_config)

    # Manually set only LoRA parameters to be trainable
    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    peft_model.to(device)

    # Execute the DPO algorithm
    edited_model = execute_dpo(peft_model, tok, requests, hparams)

    return edited_model, weights_copy


def execute_dpo(
        peft_model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DPOHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    """
    Executes the DPO algorithm for the specified updates.
    """
    peft_model.train()
    device = next(peft_model.parameters()).device

    # Define the optimizer
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )

    loss_meter = AverageMeter()

    # Prepare data
    texts = [r["prompt"] for r in requests]
    targets_pos = [r["target_new"] for r in requests]  # Positive samples
    targets_neg = [r["target_neg"] for r in requests]  # Negative samples

    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt_batch, tgt_pos_batch, tgt_neg_batch in zip(
                chunks(texts, hparams.batch_size),
                chunks(targets_pos, hparams.batch_size),
                chunks(targets_neg, hparams.batch_size),
        ):
            mask_token = -100
            opt.zero_grad()
            # Build inputs for positive samples (mask prompt tokens; only predict target part)
            full_pos = [f"{p} {t}" for p, t in zip(txt_batch, tgt_pos_batch)]
            sent_pos = tok(full_pos, return_tensors="pt", padding=True, truncation=True)
            targ_pos = tok(tgt_pos_batch, return_tensors="pt", padding=True, truncation=True)
            labels_pos = sent_pos["input_ids"].clone()
            for i in range(labels_pos.size(0)):
                tgt_len = int(targ_pos["attention_mask"][i].sum().item())
                pad_len = int(sent_pos["input_ids"].size(1) - sent_pos["attention_mask"][i].sum().item())
                if tgt_len + pad_len < labels_pos.size(1):
                    labels_pos[i, : labels_pos.size(1) - tgt_len - pad_len] = mask_token
                labels_pos[i, labels_pos[i] == tok.pad_token_id] = mask_token
            sent_pos = {**sent_pos, "labels": labels_pos}
            sent_pos = {k: v.to(device) for k, v in sent_pos.items()}

            # Build inputs for negative samples
            full_neg = [f"{p} {t}" for p, t in zip(txt_batch, tgt_neg_batch)]
            sent_neg = tok(full_neg, return_tensors="pt", padding=True, truncation=True)
            targ_neg = tok(tgt_neg_batch, return_tensors="pt", padding=True, truncation=True)
            labels_neg = sent_neg["input_ids"].clone()
            for i in range(labels_neg.size(0)):
                tgt_len = int(targ_neg["attention_mask"][i].sum().item())
                pad_len = int(sent_neg["input_ids"].size(1) - sent_neg["attention_mask"][i].sum().item())
                if tgt_len + pad_len < labels_neg.size(1):
                    labels_neg[i, : labels_neg.size(1) - tgt_len - pad_len] = mask_token
                labels_neg[i, labels_neg[i] == tok.pad_token_id] = mask_token
            sent_neg = {**sent_neg, "labels": labels_neg}
            sent_neg = {k: v.to(device) for k, v in sent_neg.items()}

            # Forward policy (LoRA-enabled)
            outputs_pos = peft_model(**sent_pos)
            outputs_neg = peft_model(**sent_neg)

            # Reference model: disable LoRA adapters
            peft_model.eval()
            peft_model.disable_adapter_layers()
            with torch.no_grad():
                ref_outputs_pos = peft_model(**sent_pos)
                ref_outputs_neg = peft_model(**sent_neg)
            peft_model.train()
            peft_model.enable_adapter_layers()

            # Helper to compute sequence log-prob sum over non-masked labels (teacher forcing)
            def seq_logp_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                # mask positions where labels are valid
                mask = (shift_labels != mask_token).to(shift_logits.dtype)
                logp = F.log_softmax(shift_logits, dim=-1)
                # replace masked labels to zero index to avoid gather errors
                safe_labels = torch.where(shift_labels >= 0, shift_labels, torch.zeros_like(shift_labels))
                picked = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                return (picked * mask).sum(dim=1)

            policy_pos = seq_logp_sum(outputs_pos.logits, labels_pos)
            policy_neg = seq_logp_sum(outputs_neg.logits, labels_neg)
            ref_pos = seq_logp_sum(ref_outputs_pos.logits, labels_pos)
            ref_neg = seq_logp_sum(ref_outputs_neg.logits, labels_neg)

            beta = hparams.beta
            dpo_advantage = beta * ((policy_pos - policy_neg) - (ref_pos - ref_neg))
            dpo_loss = (-F.logsigmoid(dpo_advantage)).mean()

            # Optional CE on positive to stabilize
            lora_loss = outputs_pos.loss
            loss = hparams.alpha * lora_loss + (1.0 - hparams.alpha) * dpo_loss

            loss.backward()
            # Stabilize training for single-sample updates
            try:
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
            except Exception:
                pass
            opt.step()

            bs = len(txt_batch)
            loss_meter.update(loss.item(), n=bs)

        print(f"Total loss {loss_meter.avg}")

    return peft_model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n]
