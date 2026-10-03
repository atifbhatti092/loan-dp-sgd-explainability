"""Explanation stability under DP-SGD: IG and grouped permutation importance of DP models vs a non-private reference.
usage: python3 run_expl.py <dataset> <n_splits> <budget_s>   (needs results/cv_<dataset>.json for the tuned hyper-parameters)"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, sigma_for, feature_partition, unit_partition, ROOT
from dpnet import Net
from explain import *
from sklearn.metrics import roc_auc_score
from run_cv import CFG, EPS
M_STEPS, N_ROWS, DP_SEEDS = 128, 300, 3

def train_np(Xtr, ytr, cfg, chosen, seed):
    lr, st = chosen["mlp_np"]; q = None if cfg["np_q"] is None else cfg["np_q"]/len(ytr)
    return Net(Xtr.shape[1], seed=seed).fit(Xtr, ytr, steps=st, lr=lr, q=q, seed=seed)

def train_dp(Xtr, ytr, cfg, chosen, eps, seed):
    n = len(ytr); q = cfg["B"]/n; C, lr = chosen[f"mlp_PO_{eps}"]
    return Net(Xtr.shape[1], seed=seed).fit(Xtr, ytr, steps=cfg["T"], lr=lr, clip=C, sigma=sigma_for(cfg["T"], q, eps), q=q, seed=seed)

def explain_model(net, Xtr, Xe, ye, fparts, uparts, seed):
    out = {"auc": float(roc_auc_score(ye, net.predict_proba(Xe))), "ig": {}, "vec": {}}
    for bname, base in baselines(Xtr).items():
        A, resid = integrated_gradients(net, Xe, base, m=M_STEPS)
        af, au = group_sum(A, fparts), group_sum(A, uparts)
        out["ig"][bname] = dict(A=A, af=af, au=au, resid=resid)
    out["pi_f"] = perm_importance(net, Xe, ye, fparts, seed=seed); out["pi_u"] = perm_importance(net, Xe, ye, uparts, seed=seed)
    return out

def summarise(ref, cur, base_zero, net, Xe, fparts, rng):
    r = {"auc": cur["auc"]}
    for b in ("zero", "median", "modal"):
        R, Cc = ref["ig"][b], cur["ig"][b]
        d = {}
        for lvl in ("af", "au"):
            d[lvl] = compare(np.abs(R[lvl]).mean(0), np.abs(Cc[lvl]).mean(0))
        d["local_cos"], d["sign_agree"] = local_similarity(R["A"], Cc["A"])
        d["resid_max"] = float(np.abs(Cc["resid"]).max()); d["resid_med"] = float(np.median(np.abs(Cc["resid"])))
        d["faith"] = comprehensiveness(net, Xe, base_zero if b == "zero" else baselines_cache[b], Cc["af"], fparts)
        rand = np.stack([rng.permutation(Cc["af"].shape[1]) for _ in range(len(Xe))])          # random feature order as control
        d["faith_random"] = comprehensiveness(net, Xe, base_zero if b == "zero" else baselines_cache[b], rand.astype(float), fparts)
        r[f"ig_{b}"] = d
    r["pi_f"] = compare(ref["pi_f"], cur["pi_f"]); r["pi_u"] = compare(ref["pi_u"], cur["pi_u"])
    r["ig_vs_pi_f"] = compare(np.abs(cur["ig"]["zero"]["af"]).mean(0), cur["pi_f"]); r["ig_vs_pi_u"] = compare(np.abs(cur["ig"]["zero"]["au"]).mean(0), cur["pi_u"])
    r["vec_ig_u"] = np.abs(cur["ig"]["zero"]["au"]).mean(0).tolist(); r["vec_pi_u"] = cur["pi_u"].tolist()
    return r

baselines_cache = {}
def main(name, nsplits, budget):
    global baselines_cache
    t0 = time.time(); ds = load(name); cfg = CFG[name]; cv = json.load(open(os.path.join(ROOT, "results", f"cv_{name}.json")))
    path = os.path.join(ROOT, "results", f"expl_{name}.json"); out = json.load(open(path)) if os.path.exists(path) else []
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg["repeats"], random_state=0).split(ds.df, ds.split_y))[:nsplits]
    for si, (tr, te) in enumerate(splits):
        if si < len(out): continue
        if time.time()-t0 > budget: print(f"PAUSED at {si}/{len(splits)}"); return
        chosen = cv[si]["chosen"]; dtr, dte = ds.df.iloc[tr], ds.df.iloc[te]
        prep = ds.make_prep().fit(dtr); Xtr, Xte = prep.transform(dtr), prep.transform(dte); ytr, yte = ds.y[tr], ds.y[te]
        sel = np.random.default_rng(si).choice(len(te), size=min(N_ROWS, len(te)), replace=False); Xe, ye = Xte[sel], yte[sel]
        fparts, uparts = feature_partition(name, prep.cols), unit_partition(name, prep.cols); baselines_cache = baselines(Xtr)
        rng = np.random.default_rng(si); rows = []
        ref_net = train_np(Xtr, ytr, cfg, chosen, 0); ref = explain_model(ref_net, Xtr, Xe, ye, fparts, uparts, 0)
        rows.append({"kind": "reference", "eps": None, "seed": 0, **summarise(ref, ref, np.zeros(Xtr.shape[1]), ref_net, Xe, fparts, rng)})
        for s in (1, 2):
            net = train_np(Xtr, ytr, cfg, chosen, s); ex = explain_model(net, Xtr, Xe, ye, fparts, uparts, s)
            rows.append({"kind": "nonprivate_reseed", "eps": None, "seed": s, **summarise(ref, ex, np.zeros(Xtr.shape[1]), net, Xe, fparts, rng)})
        for eps in EPS:
            for s in range(DP_SEEDS):
                net = train_dp(Xtr, ytr, cfg, chosen, eps, 100+s); ex = explain_model(net, Xtr, Xe, ye, fparts, uparts, s)
                rows.append({"kind": "dp", "eps": eps, "seed": s, **summarise(ref, ex, np.zeros(Xtr.shape[1]), net, Xe, fparts, rng)})
        out.append({"split": si, "feature_names": list(fparts.keys()), "unit_names": list(uparts.keys()), "rows": rows})
        json.dump(out, open(path, "w")); print(f"{name} expl split {si+1}/{len(splits)} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", name)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
