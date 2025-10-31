import os, argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from easyeditor import BaseEditor, ROMEHyperParams
from utils_mquake import load_mquake_json, generate_answer, contains_any, free_cuda

def build_editor(model_name: str, use_fp16: bool = False, layers=None):
    if layers is None:
        layers = [20]
    hps = ROMEHyperParams.from_hparams("hparams/ROME/qwen2.5-7b")
    hps.model_name = model_name
    hps.fp16 = use_fp16           # ROME 编辑阶段建议 False
    hps.layers = layers
    editor = BaseEditor.from_hparams(hps)
    return editor

def normalize(s):
    import re
    return re.sub(r"\W+", " ", (s or "").strip().lower())

def baseline_multihop(model, tok, item, max_new_tokens=128, temperature=0.2, top_p=0.9):
    golds = [item.get("new_answer", "")]
    golds += item.get("new_answer_alias", [])
    for q in item["questions"]:
        prompt = f"Answer concisely.\nQ: {q}\nA:"
        a = generate_answer(model, tok, prompt, max_new_tokens, temperature, top_p)
        if contains_any(a, golds):
            return 1
    return 0

def eval_single_hop_cloze(edited_model, tok, item, max_new_tokens=64):
    ok, tot = 0, 0
    for rw in item["requested_rewrite"]:
        q = rw["prompt"]
        tgt_new = rw["target_new"]["str"]
        a = generate_answer(edited_model, tok, q, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        ok += 1 if contains_any(a, [tgt_new]) else 0
        tot += 1
    return ok, tot

def eval_single_hop_locality(pre_model, pre_tok, post_model, post_tok, item, max_new_tokens=64):
    if "single_hops" not in item or not item["single_hops"]:
        return 0, 0
    ok, tot = 0, 0
    for sh in item["single_hops"]:
        q = sh["question"]
        prompt = f"Answer concisely.\nQ: {q}\nA:"
        a_pre = generate_answer(pre_model, pre_tok, prompt, max_new_tokens=max_new_tokens)
        a_post = generate_answer(post_model, post_tok, prompt, max_new_tokens=max_new_tokens)
        ok += 1 if normalize(a_pre)==normalize(a_post) else 0
        tot += 1
    return ok, tot

def post_multihop(edited_model, tok, item, max_new_tokens=128, temperature=0.2, top_p=0.9):
    return baseline_multihop(edited_model, tok, item, max_new_tokens, temperature, top_p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="MQuAKE/datasets/MQuAKE-CF-3k-v2.json")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--save_dir", type=str, default="runs/qwen7b_rome")
    ap.add_argument("--num_cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data = load_mquake_json(args.data_path)[:args.num_cases]  # 不shuffle，保证与基线同批

    # 未编辑模型（用于Baseline与Locality前向）
    tok_pre = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    mdl_pre = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    base_mh_ok = 0
    for it in data:
        base_mh_ok += baseline_multihop(mdl_pre, tok_pre, it)

    sum_edit_ok = sum_edit_tot = 0
    sum_loc_ok  = sum_loc_tot  = 0
    post_mh_ok  = 0

    for idx, item in enumerate(data, 1):
        editor = build_editor(args.model_name, use_fp16=False, layers=[20])

        prompts, gt, new = [], [], []
        for rw in item["requested_rewrite"]:
            prompts.append(rw["prompt"])
            gt.append(rw["target_true"]["str"])
            new.append(rw["target_new"]["str"])

        metrics, edited_model, tok_post = editor.edit(
            prompts=prompts,
            ground_truth=gt,
            target_new=new,
            sequential_edit=True
        )

        e_ok, e_tot = eval_single_hop_cloze(edited_model, tok_post, item)
        sum_edit_ok += e_ok; sum_edit_tot += e_tot

        l_ok, l_tot = eval_single_hop_locality(mdl_pre, tok_pre, edited_model, tok_post, item)
        sum_loc_ok += l_ok; sum_loc_tot += l_tot

        post_mh_ok += post_multihop(edited_model, tok_post, item)

        print(f"[ROME-QWEN7B][{idx}/{len(data)}] rewrite={e_ok}/{e_tot} locality={l_ok}/{l_tot} | easyedit={metrics}")
        free_cuda(editor, edited_model)

    def ratio(a,b): return 0.0 if b==0 else a/b
    with open(os.path.join(args.save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "cases": len(data),
            "baseline_multihop": [base_mh_ok, len(data), ratio(base_mh_ok, len(data))],
            "post_multihop":     [post_mh_ok, len(data), ratio(post_mh_ok, len(data))],
            "delta_multihop":    ratio(post_mh_ok, len(data)) - ratio(base_mh_ok, len(data)),
            "edit_success":      [sum_edit_ok, sum_edit_tot, ratio(sum_edit_ok, sum_edit_tot)],
            "locality":          [sum_loc_ok,  sum_loc_tot,  ratio(sum_loc_ok,  sum_loc_tot)]
        }, f, indent=2, ensure_ascii=False)

    print("=== ROME@Qwen2.5-7B Done ===")
    print(f"Baseline Multi-hop: {base_mh_ok/len(data):.3f} | Post Multi-hop: {post_mh_ok/len(data):.3f} | Δ={post_mh_ok/len(data)-base_mh_ok/len(data):.3f}")
    print(f"Edit Success: {sum_edit_ok}/{sum_edit_tot} = {ratio(sum_edit_ok,sum_edit_tot):.3f}")
    print(f"Locality:     {sum_loc_ok}/{sum_loc_tot} = {ratio(sum_loc_ok,sum_loc_tot):.3f}")

if __name__ == "__main__":
    main()
