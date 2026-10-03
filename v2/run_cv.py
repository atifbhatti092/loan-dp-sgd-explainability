"""Nested, leak-free repeated-CV benchmark for one dataset (loan | german | taiwan).
usage: python3 run_cv.py <dataset> <time_budget_seconds>   (resumable; rerun until it prints DONE)"""
import sys, os, json, time, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("absl").setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from interpret.privacy import DPExplainableBoostingClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, sigma_for, rule_pred, DELTA, ROOT
from dpnet import Net
from accounting import eps_fullbatch_exact, solve_sigma

EPS = [0.5, 1, 2, 4, 8]
DPGRID = [(C, lr) for C in (0.1, 0.25, 0.5) for lr in (0.5, 2.0)]
CFG = {"loan":   dict(repeats=5, B=64,  T=300, fb=True,  np_grid=[(lr, st) for lr in (.1, .5, 1.) for st in (30, 100, 300)], np_q=None),
       "german": dict(repeats=3, B=64,  T=300, fb=False, np_grid=[(lr, st) for lr in (.1, .5, 1.) for st in (30, 100, 300)], np_q=None),
       "taiwan": dict(repeats=2, B=256, T=800, fb=False, np_grid=[(lr, st) for lr in (.1, .5) for st in (500, 1500)],          np_q=256)}

def fit_net(X, y, steps, lr, seed, hidden=(16, 8), **kw): return Net(X.shape[1], hidden=hidden, seed=seed).fit(X, y, steps=steps, lr=lr, seed=seed, **kw)

def tune_np(X, y, seed, grid, npq):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed); best, bl = None, 9e9
    for lr, st in grid:
        ls = []
        for a, b in skf.split(X, y):
            q = None if npq is None else npq/len(a)
            p = fit_net(X[a], y[a], st, lr, seed, q=q).predict_proba(X[b]); ls.append(log_loss(y[b], np.clip(p, 1e-6, 1-1e-6)))
        if np.mean(ls) < bl: bl, best = np.mean(ls), (lr, st)
    return best

def tune_dp(X, y, seed, steps, q, sg, hidden):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed); sc = []
    for C, lr in DPGRID:
        a_ = []
        for a, b in skf.split(X, y):
            qq = q if q is None else min(1.0, q*len(X)/len(a))   # keep expected batch size fixed inside inner folds
            p = fit_net(X[a], y[a], steps, lr, seed, hidden=hidden, clip=C, sigma=sg, q=qq).predict_proba(X[b]); a_.append(roc_auc_score(y[b], p))
        sc.append(np.mean(a_))
    return DPGRID[int(np.argmax(sc))]

def input_perturb_lr(Xtr, ytr, Xte, eps, seed):
    """Local (per-record) Gaussian perturbation of features, as in feature-level DP baselines. Record clipped to L2 <= R,
    replace-one sensitivity 2R, analytic Gaussian calibration for one release; LR is trained on the noisy records (post-processing)."""
    d = Xtr.shape[1]; R = 2*np.sqrt(d); nrm = np.linalg.norm(Xtr, axis=1, keepdims=True)
    Xc = Xtr*np.minimum(1, R/np.maximum(nrm, 1e-12)); sg = solve_sigma(eps, 1, DELTA, None)*2*R
    Xn = Xc + np.random.default_rng(seed).normal(0, sg, Xc.shape)
    return LogisticRegression(C=1.0, max_iter=500).fit(Xn, ytr).predict_proba(Xte)[:, 1]

def one_split(ds, cfg, si, tr, te):
    dtr, dte = ds.df.iloc[tr], ds.df.iloc[te]; ytr, yte = ds.y[tr], ds.y[te]
    prep = ds.make_prep().fit(dtr); Xtr = prep.transform(dtr); Xte = prep.transform(dte); n = len(ytr)
    rec = {"split": si, "n_train": n, "n_test": len(yte), "models": {}, "chosen": {}}; M = rec["models"]; tm = {}
    def T(name, t0): tm[name] = round(time.time()-t0, 2)
    t0 = time.time(); M["majority"] = metrics(yte, np.full(len(yte), float(ytr.mean() >= .5)))
    r = rule_pred(ds, dte, prep)
    if r is not None: M["single_feature_rule"] = metrics(yte, r)
    M["logreg_L2"] = metrics(yte, LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]); T("lr", t0)
    t0 = time.time(); M["random_forest"] = metrics(yte, RandomForestClassifier(300, max_depth=6, random_state=si, n_jobs=1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]); T("rf", t0)
    t0 = time.time(); M["catboost"] = metrics(yte, CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, verbose=0, random_seed=si, thread_count=1).fit(Xtr, ytr).predict_proba(Xte)[:, 1]); T("cat", t0)
    t0 = time.time(); M["xgboost"] = metrics(yte, XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, n_jobs=1, random_state=si, verbosity=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]); T("xgb", t0)
    t0 = time.time(); lr_, st_ = tune_np(Xtr, ytr, si, cfg["np_grid"], cfg["np_q"]); rec["chosen"]["mlp_np"] = [lr_, st_]
    M["mlp_nonprivate"] = metrics(yte, fit_net(Xtr, ytr, st_, lr_, si, q=None if cfg["np_q"] is None else cfg["np_q"]/n).predict_proba(Xte)); T("mlp_np", t0)
    B, Tn = cfg["B"], cfg["T"]; q = B/n
    variants = [("PO", Tn, q)] + ([("FB100", 100, None)] if cfg["fb"] else [])
    for eps in EPS:
        for tag, steps, qq in variants:
            t0 = time.time(); sg = sigma_for(steps, qq, eps); C, lr = tune_dp(Xtr, ytr, si, steps, qq, sg, (16, 8)); rec["chosen"][f"mlp_{tag}_{eps}"] = [C, lr]
            M[f"dp_mlp_{tag}_eps{eps}"] = metrics(yte, fit_net(Xtr, ytr, steps, lr, si*100, hidden=(16, 8), clip=C, sigma=sg, q=qq).predict_proba(Xte)); M[f"dp_mlp_{tag}_eps{eps}"]["sigma"] = sg; T(f"dpmlp{tag}", t0)
        t0 = time.time(); sg = sigma_for(Tn, q, eps); C, lr = tune_dp(Xtr, ytr, si, Tn, q, sg, ()); rec["chosen"][f"lr_PO_{eps}"] = [C, lr]
        M[f"dp_lr_PO_eps{eps}"] = metrics(yte, fit_net(Xtr, ytr, Tn, lr, si*100, hidden=(), clip=C, sigma=sg, q=q).predict_proba(Xte)); T("dplr", t0)
        t0 = time.time(); Xa, Xb = np.clip(Xtr, -4, 4), np.clip(Xte, -4, 4); bounds = {i: (-4.0, 4.0) for i in range(Xa.shape[1])}
        try:
            m = DPExplainableBoostingClassifier(epsilon=eps, delta=DELTA, privacy_bounds=bounds, random_state=si).fit(Xa, ytr.astype(int))
            M[f"dp_ebm_eps{eps}"] = metrics(yte, m.predict_proba(Xb)[:, 1])
        except Exception as e: M[f"dp_ebm_eps{eps}"] = {"error": str(e)[:80]}
        T("ebm", t0)
        t0 = time.time(); M[f"local_perturb_lr_eps{eps}"] = metrics(yte, input_perturb_lr(Xtr, ytr, Xte, eps, si)); T("pert", t0)
    rec["timing"] = tm
    return rec

def main(name, budget):
    t0 = time.time(); ds = load(name); cfg = CFG[name]; out_path = os.path.join(ROOT, "results", f"cv_{name}.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else []
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg["repeats"], random_state=0).split(ds.df, ds.split_y))
    for si, (tr, te) in enumerate(splits):
        if si < len(out): continue
        if time.time()-t0 > budget: print(f"PAUSED at {si}/{len(splits)}"); return
        out.append(one_split(ds, cfg, si, tr, te)); json.dump(out, open(out_path, "w"))
        print(f"{name} split {si+1}/{len(splits)} ({time.time()-t0:.0f}s) timing={out[-1]['timing']}", flush=True)
    print("DONE", name, len(out))
if __name__ == "__main__": main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 240)
