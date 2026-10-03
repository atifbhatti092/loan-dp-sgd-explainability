import json, numpy as np
from collections import Counter
from scipy import stats
R1 = json.load(open("results/exp1.json")); R2 = json.load(open("results/exp2.json")); J = 25; RHO = 123/491; tc = stats.t.ppf(.975, J-1)
def v(R, m, k): return np.array([r["models"][m][k] for r in R])
def ci(R, m, k): x = v(R, m, k); return x.mean(), tc*np.sqrt((1/J+RHO)*x.var(ddof=1))
def line(R, m):
    a, ca = ci(R, m, "acc"); u, cu = ci(R, m, "auc"); b, _ = ci(R, m, "bacc"); s, _ = ci(R, m, "spec")
    return f"{m:22s} acc {a:.3f}±{ca:.3f} | AUC {u:.3f}±{cu:.3f} | bal.acc {b:.3f} | spec {s:.3f}"
print("Non-private reference:"); [print(line(R1, m)) for m in ("credit_history_rule", "logreg_L2", "mlp_tuned")]
for n in ("FB100", "PO300"):
    print(f"\nTuned DP {n}:"); [print(line(R2, f"dpT_{n}_eps{e}")) for e in (0.5, 1, 2, 4, 8)]
print("\nChosen (C,lr) counts @eps=1 PO300:", Counter(map(tuple, (r['chosen']['PO300_1'] for r in R2))).most_common(3))
print("Chosen (C,lr) counts @eps=1 FB100:", Counter(map(tuple, (r['chosen']['FB100_1'] for r in R2))).most_common(3))
def nb(a, b, k):
    d = a - b; se = np.sqrt((1/J+RHO)*d.var(ddof=1)); t = d.mean()/se; return d.mean(), 2*stats.t.sf(abs(t), J-1)
print("\nNB tests vs non-private tuned MLP (AUC / acc):")
for n in ("FB100", "PO300"):
    for e in (0.5, 1, 2, 4, 8):
        m = f"dpT_{n}_eps{e}"; da, pa = nb(v(R2, m, "auc"), v(R1, "mlp_tuned", "auc"), "auc"); dc, pc = nb(v(R2, m, "acc"), v(R1, "mlp_tuned", "acc"), "acc")
        print(f"{m:18s} dAUC={da:+.3f} (p={pa:.2f}) dAcc={dc:+.3f} (p={pc:.2f})")
d, p = nb(v(R2, "dpT_FB100_eps1", "auc"), v(R2, "dpT_PO300_eps1", "auc"), "auc"); print(f"\nFB100 vs PO300 @eps1 dAUC={d:+.3f} p={p:.2f}")
d, p = nb(v(R2, "dpT_FB100_eps4", "auc"), v(R2, "dpT_PO300_eps4", "auc"), "auc"); print(f"FB100 vs PO300 @eps4 dAUC={d:+.3f} p={p:.2f}")
