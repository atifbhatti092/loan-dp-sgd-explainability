"""Small experiments: Credit_History treatments (comment 6), accountant comparison + Monte-Carlo check of the composition
theorem (comments 13-14, option C), training-loss curves (comment 8c)."""
import sys, os, json, warnings, logging
warnings.filterwarnings("ignore"); logging.getLogger("absl").setLevel(logging.ERROR)
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, sigma_for, ROOT
from dpnet import Net
from accounting import *
res = {}
# ---- (a) Credit_History treatments, non-private, 5x5 CV
ds = load("loan"); rows = {m: {"logreg": [], "mlp": []} for m in ("mode", "indicator", "category")}
for si, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(ds.df, ds.split_y)):
    for mode in rows:
        prep = ds.make_prep(credit_mode=mode).fit(ds.df.iloc[tr]); Xtr, Xte = prep.transform(ds.df.iloc[tr]), prep.transform(ds.df.iloc[te])
        rows[mode]["logreg"].append(metrics(ds.y[te], LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ds.y[tr]).predict_proba(Xte)[:, 1]))
        rows[mode]["mlp"].append(metrics(ds.y[te], Net(Xtr.shape[1], seed=si).fit(Xtr, ds.y[tr], steps=100, lr=0.5, seed=si).predict_proba(Xte)))
res["credit_history"] = rows
# ---- (b) accountant comparison, full batch k=30
k, tab = 30, []
for s in (200, 100, 60, 40, 25, 15):
    e_ms, d_ms = eps_advanced_paper(s, k, 1/491)
    d = 1e-5; dp = d/2; d0 = (d-dp)/k; e0 = np.sqrt(2*np.log(1.25/d0))/s          # advanced composition with a correct delta split (classical bound, valid only for e0<1)
    tab.append(dict(sigma=s, manuscript_adv=e_ms, manuscript_delta_actual=d_ms, adv_corrected_delta1e5=(np.sqrt(2*k*np.log(1/dp))*e0 + k*e0*(np.exp(e0)-1)), adv_corrected_valid=bool(e0 < 1),
                    rdp_1e5=eps_rdp(s, k, 1e-5), exact_1e5=eps_fullbatch_exact(s, k, 1e-5), pld_1e5=eps_pld(s, k, 1e-5),
                    rdp_1_491=eps_rdp(s, k, 1/491), exact_1_491=eps_fullbatch_exact(s, k, 1/491)))
res["accountants_fullbatch"] = tab
res["accountants_poisson"] = [dict(sigma=s, q=64/491, steps=300, pld=eps_pld(s, 300, 1e-5, 64/491), rdp=eps_rdp(s, 300, 1e-5, 64/491)) for s in (1.5, 2.0, 3.0, 5.0, 8.5)]
# ---- (c) Monte-Carlo check: privacy-loss of k full-batch releases vs closed form  (worst case: every clipped gradient moves by C in the same direction)
rng = np.random.default_rng(0); mc = []
for s, kk in ((15, 30), (8, 30), (3, 30), (2, 100)):
    x = rng.normal(0, s, size=(300000, kk)); L = (kk - 2*x.sum(1))/(2*s*s)            # log p_P(x)/p_Q(x),  P = N(0, s^2 I), Q = N(C*1, s^2 I), C = 1
    for eps in (0.25, 0.5, 1.0):
        mc.append(dict(sigma=s, k=kk, eps=eps, delta_monte_carlo=float(np.mean(np.maximum(0.0, 1-np.exp(eps-L)))), delta_closed_form=float(norm.cdf(-eps/(np.sqrt(kk)/s) + np.sqrt(kk)/s/2) - np.exp(eps)*norm.cdf(-eps/(np.sqrt(kk)/s) - np.sqrt(kk)/s/2))))
res["composition_monte_carlo"] = mc
# ---- (d) training-loss curves on one loan training fold
tr, te = next(iter(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0).split(ds.df, ds.split_y)))
prep = ds.make_prep().fit(ds.df.iloc[tr]); Xtr = prep.transform(ds.df.iloc[tr]); ytr = ds.y[tr]; n = len(ytr); q = 64/n; curves = {}
def rec(store): return lambda t, net: store.append(float(-np.mean(ytr*np.log(np.clip(net.predict_proba(Xtr), 1e-7, 1)) + (1-ytr)*np.log(np.clip(1-net.predict_proba(Xtr), 1e-7, 1))))) if t % 5 == 0 else None
c = []; Net(Xtr.shape[1], seed=0).fit(Xtr, ytr, steps=300, lr=0.5, seed=0, callback=rec(c)); curves["non-private (full batch, lr 0.5)"] = c
for eps in (1, 4, 8):
    c = []; Net(Xtr.shape[1], seed=0).fit(Xtr, ytr, steps=300, lr=0.5, clip=0.25, sigma=sigma_for(300, q, eps), q=q, seed=0, callback=rec(c)); curves[f"DP-SGD eps={eps} (Poisson, C=0.25, lr 0.5)"] = c
res["loss_curves"] = curves
json.dump(res, open(os.path.join(ROOT, "results", "misc.json"), "w"), default=float); print("misc saved")
for r in tab: print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()})
for r in mc[:6]: print({k_: (round(v, 6) if isinstance(v, float) else v) for k_, v in r.items()})
