"""Summarise the end-to-end private tuning benchmark against the paper's training-run-only numbers.
usage: python3 analyze_ptune.py <dataset>     -> prints markdown tables, writes results/ptune_summary_<dataset>.json
Mean AUC with Nadeau-Bengio 95% intervals (variance inflated by 1/J + n_test/n_train, t distribution with J-1 df), as in the paper."""
import sys, os, json, numpy as np
from scipy.stats import t as tdist
ROOT = os.path.dirname(os.path.abspath(__file__)); EPS = [0.5, 1, 2, 4, 8]; MS = [1, 2, 6]

def nb(vals, ratio):
    v = np.asarray(vals, float); J = len(v)
    if J < 2: return float(v.mean()), float("nan")
    se = np.sqrt((1.0/J + ratio)*v.var(ddof=1)); return float(v.mean()), float(tdist.ppf(0.975, J-1)*se)

def main(name):
    pt = json.load(open(os.path.join(ROOT, "results", f"ptune_{name}.json"))); cv = json.load(open(os.path.join(ROOT, "results", f"cv_{name}.json")))
    J = len(pt); ratio = np.mean([r["n_test"]/r["n_train"] for r in pt]); out = {}
    tags = sorted({k.split("_")[0] for k in pt[0]["models"]}, key=lambda s: s != "PO")
    for tag in tags:
        print(f"\n### {name}, {tag}: test AUC, mean ± 95% NB interval over {J} splits (eps = TOTAL, tuning included)\n")
        print("| eps | paper (tuning free, training-only eps) | m=1 no tuning | m=2 lr only | m=6 full grid | most common choice at m=6 (share of splits) |"); print("|---|---|---|---|---|---|")
        for e in EPS:
            free = [cv[i]["models"][f"dp_mlp_{tag}_eps{e}"]["auc"] for i in range(J)]
            row = {"free": nb(free, ratio)}
            cells = [f"{row['free'][0]:.3f} ± {row['free'][1]:.3f}"]
            for m in MS:
                a = [pt[i]["models"][f"{tag}_eps{e}_m{m}"]["auc"] for i in range(J)]; row[f"m{m}"] = nb(a, ratio)
                d = [x-y for x, y in zip(a, free)]; row[f"m{m}_minus_free"] = nb(d, ratio)
                cells.append(f"{row[f'm{m}'][0]:.3f} ± {row[f'm{m}'][1]:.3f}")
            # how often does the noisy selection pick the candidate with the best test AUC among the m=6 grid? (not available -> report index entropy instead)
            ch = [tuple(pt[i]["info"][f"{tag}_eps{e}_m6"]["chosen"]) for i in range(J)]; top = max(set(ch), key=ch.count); share = 100*ch.count(top)/J
            row["m6_most_common_choice"] = [list(top), share]; cells.append(f"{top} in {share:.0f}%")
            out[f"{tag}_eps{e}"] = row; print(f"| {e} | " + " | ".join(cells) + " |")
        print(f"\nPaired difference to the paper's tuning-free number (negative = price of private tuning):\n")
        print("| eps | m=1 | m=2 | m=6 |"); print("|---|---|---|---|")
        for e in EPS: r = out[f"{tag}_eps{e}"]; print(f"| {e} | " + " | ".join(f"{r[f'm{m}_minus_free'][0]:+.3f} ± {r[f'm{m}_minus_free'][1]:.3f}" for m in MS) + " |")
    json.dump(out, open(os.path.join(ROOT, "results", f"ptune_summary_{name}.json"), "w"), indent=1)
if __name__ == "__main__": main(sys.argv[1])
