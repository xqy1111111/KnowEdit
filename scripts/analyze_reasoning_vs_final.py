import json, argparse, re
from utils_mquake import contains_any, sanitize_aliases

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_jsonl", required=True, help="eval_* 输出（含 reasoning/final/raw）")
    ap.add_argument("--golds_from_jsonl", action="store_true", default=True)
    args = ap.parse_args()

    n=0; both=0; think_only=0; final_only=0; none=0
    with open(args.pred_jsonl,"r",encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            row = json.loads(ln)
            golds = sanitize_aliases(row.get("golds", []))
            ok_reason=False; ok_final=False
            for qa in row.get("qas", []):
                reason = (qa.get("reasoning") or "")
                final  = (qa.get("final") or qa.get("a") or "")
                if contains_any(reason, golds): ok_reason=True
                if contains_any(final,  golds): ok_final=True
            if ok_reason and ok_final: both+=1
            elif ok_reason and not ok_final: think_only+=1
            elif not ok_reason and ok_final: final_only+=1
            else: none+=1
            n+=1
    print(f"Total={n}")
    print(f"reasoning+final 都含金答案: {both}")
    print(f"只在 reasoning 出现金答案(但最终结论错): {think_only}")
    print(f"只在 final 出现金答案: {final_only}")
    print(f"均未出现: {none}")

if __name__ == "__main__":
    main()
