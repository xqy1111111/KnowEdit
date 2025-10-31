# scripts/summarize_jsonl.py
import argparse, json, os, glob

def read_acc(path: str):
    ok = 0; total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            total += 1
            ok += int(obj.get("hit", 0))
    return ok, total, (ok/total if total else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="jsonl file(s) or glob(s)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    files = sorted(set(files))

    print(f"{'file':70} | {'ok':>5} / {'n':<5} | acc")
    print("-"*92)
    for fp in files:
        ok, n, acc = read_acc(fp)
        print(f"{os.path.basename(fp):70} | {ok:>5} / {n:<5} | {acc:.3f}")

if __name__ == "__main__":
    main()
