"""Empirical privacy audit: loss-threshold and LiRA (fixed-variance, leave-one-model-out) membership inference.
Shadow models are trained on random halves of the data; each model is attacked in turn using the statistics of the others.
usage: python3 run_mia.py <dataset> <budget_s>"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from collections import Counter
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, sigma_for, DELTA, ROOT
from dpnet import Net
from run_cv import CFG
NMOD, MIA_EPS, B = 96, [1, 4, 8], 64

def phi(net, X, y): z = net.logit(X); return np.where(y == 1, z, -z)     # logit of the confidence in the true label

def tpr_at(fpr, tpr, a): return float(np.interp(a, fpr, tpr))

def eps_lower_bound(scores, member, delta=DELTA, conf=0.05):
    """Largest ln((TPR_lo - delta)/FPR_hi) over thresholds (Clopper-Pearson one-sided 95% bounds).
    Heuristic: pairs from the same model/record are not independent, so the interval is approximate."""
    order = np.argsort(-scores); m = member[order]; P, Nn = m.sum(), (~m).sum()
    tp, fp = np.cumsum(m), np.cumsum(~m); best = 0.0
    idx = np.unique(np.linspace(20, len(m)-1, 400).astype(int))
    for k in idx:
        if fp[k] < 5: continue
        tpr_lo = beta.ppf(conf, tp[k], P-tp[k]+1) if tp[k] > 0 else 0.0
        fpr_hi = beta.ppf(1-conf, fp[k]+1, Nn-fp[k]) if fp[k] < Nn else 1.0
        if tpr_lo > delta: best = max(best, np.log((tpr_lo-delta)/fpr_hi))
    return float(best)

def audit(Phi, Mk, boot=200, seed=0):
    N, n = Phi.shape; res = {}
    def stats(score, member):
        fpr, tpr, _ = roc_curve(member.ravel(), score.ravel()); return roc_auc_score(member.ravel(), score.ravel()), tpr_at(fpr, tpr, 0.01), tpr_at(fpr, tpr, 0.1)
    # loss-threshold attack
    a, t1, t10 = stats(Phi, Mk); res["loss"] = dict(auc=a, tpr_fpr1=t1, tpr_fpr10=t10, eps_lb=eps_lower_bound(Phi.ravel(), Mk.ravel()))
    # LiRA, fixed global variance, leave-one-model-out
    Sin = (Mk*Phi).sum(0); Nin = Mk.sum(0); Sout = ((~Mk)*Phi).sum(0); Nout = (~Mk).sum(0)
    mu_in = (Sin[None]-Mk*Phi)/np.maximum(Nin[None]-Mk, 1); mu_out = (Sout[None]-(~Mk)*Phi)/np.maximum(Nout[None]-(~Mk), 1)
    var = np.mean([Phi[Mk[:, i], i].var() if Mk[:, i].sum() > 1 else 0 for i in range(n)] + [Phi[~Mk[:, i], i].var() if (~Mk[:, i]).sum() > 1 else 0 for i in range(n)]) + 1e-6
    lira = ((Phi-mu_out)**2 - (Phi-mu_in)**2)/(2*var)
    a, t1, t10 = stats(lira, Mk); res["lira"] = dict(auc=a, tpr_fpr1=t1, tpr_fpr10=t10, eps_lb=eps_lower_bound(lira.ravel(), Mk.ravel()))
    # bootstrap over records for CI
    rng = np.random.default_rng(seed); ba, bt = [], []
    for _ in range(boot):
        cols = rng.integers(0, n, n); s, m = lira[:, cols], Mk[:, cols]
        fpr, tpr, _ = roc_curve(m.ravel(), s.ravel()); ba.append(roc_auc_score(m.ravel(), s.ravel())); bt.append(tpr_at(fpr, tpr, 0.01))
    res["lira"]["auc_ci"] = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]; res["lira"]["tpr1_ci"] = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    return res

def main(name, budget):
    t0 = time.time(); ds = load(name)
    if name == "taiwan":
        idx = np.random.default_rng(0).choice(len(ds.df), 1000, replace=False); df, y = ds.df.iloc[idx].reset_index(drop=True), ds.y[idx]
    else: df, y = ds.df, ds.y
    prep = ds.make_prep().fit(df); X = prep.transform(df); n = len(y); half = n//2
    cv = json.load(open(os.path.join(ROOT, "results", f"cv_{name}.json"))); cfg = CFG[name]
    mode = lambda key: Counter(tuple(r["chosen"][key]) for r in cv).most_common(1)[0][0]
    path = os.path.join(ROOT, "results", f"mia_{name}.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    rng = np.random.default_rng(1); masks = np.zeros((NMOD, n), bool)
    for j in range(NMOD): masks[j, rng.permutation(n)[:half]] = True
    lr_np, st_np = Counter(tuple(r["chosen"]["mlp_np"]) for r in cv).most_common(1)[0][0]
    mechs = {"nonprivate": lambda Xt, yt, s: Net(X.shape[1], seed=s).fit(Xt, yt, steps=st_np, lr=lr_np, seed=s, q=None if cfg["np_q"] is None else cfg["np_q"]/half),
             "overfit_control": lambda Xt, yt, s: Net(X.shape[1], hidden=(64,), seed=s).fit(Xt, yt, steps=600, lr=1.0, seed=s)}
    for eps in MIA_EPS:
        C, lr = mode(f"mlp_PO_{eps}"); q = cfg["B"]/half; sg = sigma_for(cfg["T"], q, eps)
        mechs[f"dp_eps{eps}"] = (lambda C, lr, q, sg: (lambda Xt, yt, s: Net(X.shape[1], seed=s).fit(Xt, yt, steps=cfg["T"], lr=lr, clip=C, sigma=sg, q=q, seed=s)))(C, lr, q, sg)
    for mname, trainer in mechs.items():
        if mname in out: continue
        if time.time()-t0 > budget: print("PAUSED before", mname); return
        Phi = np.zeros((NMOD, n))
        for j in range(NMOD):
            tr = masks[j]; net = trainer(X[tr], y[tr], j); Phi[j] = phi(net, X, y)
        res = audit(Phi, masks); res["train_acc_gap"] = None
        res["n_models"], res["n_records"] = NMOD, n
        out[mname] = res; json.dump(out, open(path, "w")); print(name, mname, {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("auc", "tpr_fpr1", "eps_lb")} for k, v in res.items() if k in ("loss", "lira")}, f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", name)
if __name__ == "__main__": main(sys.argv[1], float(sys.argv[2]))
