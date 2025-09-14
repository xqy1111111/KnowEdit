import json
import argparse


def load_hits(path: str, key_name: str = "hit"):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                xs.append(int(json.loads(ln).get(key_name, 0)))
            except Exception:
                xs.append(0)
    return xs


def acc(x):
    return sum(x) / max(1, len(x))


def pair(a, b):
    n01 = n10 = n11 = n00 = 0
    for x, y in zip(a, b):
        if x == 0 and y == 1:
            n01 += 1
        elif x == 1 and y == 0:
            n10 += 1
        elif x == 1 and y == 1:
            n11 += 1
        else:
            n00 += 1
    return n01, n10, n11, n00


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_vs_orig", required=True)
    ap.add_argument("--base_vs_new", required=True)
    ap.add_argument("--edit_vs_new", required=True)
    ap.add_argument("--edit_vs_orig", required=True)
    ap.add_argument("--judge_key", default="hit", choices=["hit", "judge_hit"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    avo = load_hits(args.base_vs_orig, args.judge_key)
    avn = load_hits(args.base_vs_new, args.judge_key)
    evn = load_hits(args.edit_vs_new, args.judge_key)
    evo = load_hits(args.edit_vs_orig, args.judge_key)

    print(f"\n=== {args.tag} ===")
    print(f"Base vs ORIG: {acc(avo):.3f} | {args.base_vs_orig}")
    print(f"Base vs NEW : {acc(avn):.3f} | {args.base_vs_new}")
    print(f"Edit vs NEW : {acc(evn):.3f} | {args.edit_vs_new}")
    print(f"Edit vs ORIG: {acc(evo):.3f} | {args.edit_vs_orig}")

    n01, n10, _, _ = pair(avn, evn)
    print(f"\n[NEW] 0→1={n01} 1→0={n10}  Δ={n01 - n10}")

    n01o, n10o, _, _ = pair(avo, evo)
    print(f"[ORIG] 0→1={n01o} 1→0={n10o} Δ={n01o - n10o}")

    print("\nES(编辑成功)=Edit vs NEW  高越好")
    print("AR(对旧事实的依赖)=Edit vs ORIG  低越好")


if __name__ == "__main__":
    main()

