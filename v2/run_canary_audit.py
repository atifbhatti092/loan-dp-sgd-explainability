"""White-box canary audit of the DP-SGD implementation + accountant (full batch, exact Gaussian-DP accounting).
For each target epsilon: R runs with the canary and R without; the statistic Z is described in dpnet_audit.py.
Reported: mu_hat (shift estimate) vs the theoretical mu = sqrt(T)/sigma, the epsilon implied by mu_hat (and its 95% CI),
and the one-sided Clopper-Pearson empirical lower bound on epsilon (threshold test, no distributional assumption).
usage: python3 run_canary_audit.py <dataset> <R> <budget_seconds>      (resumable)"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import beta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, DELTA, ROOT
from dpnet_audit import AuditNet
from accounting import solve_sigma
from private_tuning import eps_from_mu
STEPS, CLIP, EPS = 100, 0.25, [1, 2, 4, 8]

def eps_lower_bound(z_in, z_out, delta=DELTA, conf=0.05):
    """largest ln((TPR_lo - delta)/FPR_hi) over thresholds; one-sided Clopper-Pearson bounds, TPR/FPR of 'Z > thr' as 'member'."""
    thr = np.unique(np.concatenate([z_in, z_out])); best = 0.0
    P, N = len(z_in), len(z_out)
    for t in thr[np.linspace(0, len(thr)-1, 300).astype(int)]:
        tp, fp = int((z_in > t).sum()), int((z_out > t).sum())
        if fp < 3: continue
        lo = beta.ppf(conf, tp, P-tp+1) if tp > 0 else 0.0
        hi = beta.ppf(1-conf, fp+1, N-fp) if fp < N else 1.0
        if lo > delta: best = max(best, float(np.log((lo-delta)/hi)))
    return best

def main(name, R, budget):
    t0 = time.time(); ds = load(name); prep = ds.make_prep().fit(ds.df); X = prep.transform(ds.df); y = ds.y; d = X.shape[1]
    rng = np.random.default_rng(7); u = rng.normal(size=d); xc = 5.0*u/np.linalg.norm(u)*np.sqrt(d); yc = 1.0   # far-out canary -> gradient norm >> C
    path = os.path.join(ROOT, "results", f"canary_{name}.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    for eps in EPS:
        key = str(eps)
        if key in out: continue
        if time.time()-t0 > budget: print("PAUSED before eps", eps); return
        sg = solve_sigma(eps, STEPS, DELTA, None); zin, zout, gin, gout, mx = [], [], [], [], 0.0
        for r in range(R):
            for member, zl, gl in ((True, zin, gin), (False, zout, gout)):
                net = AuditNet(d, seed=r); Z, gs, m = net.fit_audit(X, y, STEPS, 0.5, CLIP, sg, xc, yc, member, seed=1000+2*r+int(member))
                zl.append(Z); gl.append(gs); mx = max(mx, m)
        zin, zout = np.array(zin), np.array(zout)
        mu_theory = np.sqrt(STEPS)/sg
        mu_hat = (zin.mean()-zout.mean())/np.sqrt(STEPS); se = np.sqrt(2.0/R)        # Z has unit variance per step -> std sqrt(T)
        lo, hi = mu_hat-1.96*se, mu_hat+1.96*se
        out[key] = dict(sigma=sg, R=R, mu_theory=mu_theory, mu_hat=float(mu_hat), mu_ci=[float(lo), float(hi)],
                        mean_canary_norm_over_C=float(np.mean(gin)/STEPS), max_canary_norm_over_C=float(mx),
                        eps_theory=float(eps), eps_from_mu_hat=float(eps_from_mu(max(mu_hat, 1e-6), DELTA)),
                        eps_from_mu_ci=[float(eps_from_mu(max(lo, 1e-6), DELTA)), float(eps_from_mu(max(hi, 1e-6), DELTA))],
                        eps_lower_bound_threshold=eps_lower_bound(zin, zout))
        json.dump(out, open(path, "w")); print(name, eps, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[key].items() if k in ("mu_theory", "mu_hat", "eps_from_mu_hat", "eps_lower_bound_threshold", "max_canary_norm_over_C")}, f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", name)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
