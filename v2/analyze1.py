import json, numpy as np
from scipy import stats
R = json.load(open("results/exp1.json")); J = len(R); RHO = 123/491
tc = stats.t.ppf(0.975, J-1)
def vec(m, k): return np.array([r["models"][m][k] for r in R])
def ci(m, k):
    v = vec(m, k); se = np.sqrt((1/J + RHO)*v.var(ddof=1)); return v.mean(), tc*se
def nb_test(a, b, k="auc"):
    d = vec(a, k) - vec(b, k); se = np.sqrt((1/J + RHO)*d.var(ddof=1))
    t = d.mean()/se; return d.mean(), 2*stats.t.sf(abs(t), J-1)
def row(m):
    a, ca = ci(m, "acc"); u, cu = ci(m, "auc"); b, _ = ci(m, "bacc"); s, _ = ci(m, "spec"); c, _ = ci(m, "mcc")
    return f"{m:26s} acc {a:.3f}±{ca:.3f} | AUC {u:.3f}±{cu:.3f} | bal.acc {b:.3f} | MCC {c:.3f} | spec(reject) {s:.3f}"
print(f"J={J} splits; ± = 95% Nadeau-Bengio CI\n")
print("== Non-DP models =="); [print(row(m)) for m in ["majority","credit_history_rule","logreg_L2","random_forest","mlp_paper_cfg","mlp_tuned"]]
sel = [tuple(r["mlp_selected_cfg"]) for r in R]; from collections import Counter; print("MLP configs picked by inner CV:", Counter(sel).most_common(4))
for name in ["FB30","FB100","PO300"]:
    print(f"\n== DP {name} =="); [print(row(f"dp_{name}_eps{e}")) for e in [0.25,0.5,1,2,4,8]]
print("\n== Clip sensitivity (PO300) =="); 
for e in (1,4):
    for C in ("0.25","0.5",None,"3.0"):
        m = f"dpC{C}_PO300_eps{e}" if C else f"dp_PO300_eps{e}"; print(("C=%s"%(C or "1.0")).ljust(8), row(m))
print("\n== Paired NB tests on AUC (diff, p) ==")
for a,b in [("mlp_tuned","logreg_L2"),("mlp_tuned","credit_history_rule"),("random_forest","logreg_L2"),("dp_FB30_eps2","mlp_tuned"),("dp_FB30_eps1","mlp_tuned"),("dp_FB30_eps8","mlp_tuned"),("dp_FB30_eps4","dp_FB30_eps1"),("dp_FB30_eps1","dp_PO300_eps1"),("dp_FB30_eps4","dp_PO300_eps4")]:
    d,p = nb_test(a,b,"auc"); print(f"{a:16s} vs {b:20s} dAUC={d:+.3f} p={p:.3f}")
print("\n== Paired NB tests on ACCURACY ==")
for a,b in [("mlp_tuned","credit_history_rule"),("mlp_tuned","logreg_L2"),("dp_FB30_eps2","credit_history_rule"),("dp_FB30_eps1","mlp_tuned"),("dp_FB30_eps8","mlp_tuned"),("mlp_tuned","majority")]:
    d,p = nb_test(a,b,"acc"); print(f"{a:16s} vs {b:20s} dAcc={d:+.3f} p={p:.3f}")
