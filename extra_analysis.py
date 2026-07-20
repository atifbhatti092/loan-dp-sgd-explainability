import numpy as np, json
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss)
from scipy import stats
import sys
sys.path.insert(0, ".")
from dpmlp import DPNet, privacy_epsilon

data = np.load("prep_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
n_in = X_train.shape[1]
DELTA = 1.0 / len(y_train)
SEEDS = [1, 2, 3, 4, 5]

def eval_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred), precision=precision_score(y_true, pred),
                recall=recall_score(y_true, pred), f1=f1_score(y_true, pred),
                roc_auc=roc_auc_score(y_true, proba), brier=brier_score_loss(y_true, proba))

# =========================================================
# A. Classical ML baselines (Random Forest, HistGradientBoosting)
#    trained on the IDENTICAL 80/20 split, 5 seeds each
# =========================================================
def run_classical(model_fn, name):
    per_seed = {k: [] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]}
    for s in SEEDS:
        m = model_fn(s)
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_test)[:, 1]
        met = eval_metrics(y_test, proba)
        for k, v in met.items():
            per_seed[k].append(v)
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "raw": v} for k, v in per_seed.items()}
    print(name, "acc:", [round(x,4) for x in per_seed["accuracy"]], "mean", round(summary["accuracy"]["mean"],4))
    return summary

rf_results = run_classical(lambda s: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=s), "RandomForest")
hgb_results = run_classical(lambda s: HistGradientBoostingClassifier(max_depth=4, random_state=s), "HistGradientBoosting")

# =========================================================
# B. Re-run non-private MLP baseline and DP-SGD sweep, this time
#    saving RAW per-seed accuracy (needed for paired t-tests)
# =========================================================
Xb, yb = X_train, y_train  # "no balancing" was the best strategy (Table 2); reproduce baseline exactly

mlp_baseline_raw = {k: [] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]}
for s in SEEDS:
    net = DPNet(n_in, seed=s)
    net.fit(Xb, yb, epochs=30, batch_size=len(Xb), lr=0.5, seed=s)
    proba = net.predict_proba(X_test)
    met = eval_metrics(y_test, proba)
    for k, v in met.items():
        mlp_baseline_raw[k].append(v)
print("MLP baseline acc:", [round(x,4) for x in mlp_baseline_raw["accuracy"]])

CLIP_NORM = 3.0
sweep_raw = {}
for sigma in [200.0, 100.0, 60.0, 40.0, 25.0, 15.0]:
    per_seed = {k: [] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]}
    n_steps = None
    for s in SEEDS:
        net = DPNet(n_in, seed=s)
        net.fit(Xb, yb, epochs=30, batch_size=len(Xb), lr=0.5,
                clip_norm=CLIP_NORM, noise_multiplier=sigma, seed=s)
        n_steps = net.n_steps
        proba = net.predict_proba(X_test)
        met = eval_metrics(y_test, proba)
        for k, v in met.items():
            per_seed[k].append(v)
    sweep_raw[sigma] = {"metrics": per_seed, "n_steps": n_steps}
    print(f"sigma={sigma} acc:", [round(x,4) for x in per_seed["accuracy"]])

# =========================================================
# C. Renyi-DP accounting, compared against advanced composition
# =========================================================
def rdp_epsilon(sigma, k_steps, delta, alphas=None):
    """
    RDP of the (non-subsampled) Gaussian mechanism at order alpha is
        RDP(alpha) = alpha / (2*sigma^2)
    (Mironov, 2017, Prop. 7), composed additively over k_steps compositions:
        RDP_total(alpha) = k_steps * alpha / (2*sigma^2)
    converted to (epsilon,delta)-DP via
        epsilon = RDP_total(alpha) + ln(1/delta) / (alpha - 1)
    minimised over alpha > 1.
    """
    if alphas is None:
        alphas = np.concatenate([np.linspace(1.01, 10, 200), np.linspace(10, 512, 200)])
    best_eps = np.inf
    best_alpha = None
    for a in alphas:
        rdp = k_steps * a / (2 * sigma ** 2)
        eps = rdp + np.log(1 / delta) / (a - 1)
        if eps < best_eps:
            best_eps = eps
            best_alpha = a
    return best_eps, best_alpha

privacy_table = []
for sigma, d in sweep_raw.items():
    k = d["n_steps"]
    eps_adv = privacy_epsilon(sigma, k, DELTA)
    eps_rdp, alpha_star = rdp_epsilon(sigma, k, DELTA)
    privacy_table.append({"sigma": sigma, "n_steps": k, "delta": DELTA,
                           "epsilon_advanced_composition": eps_adv,
                           "epsilon_rdp": eps_rdp, "best_alpha": alpha_star})
    print(f"sigma={sigma:6.1f}  eps_advComp={eps_adv:9.3f}  eps_RDP={eps_rdp:7.3f}  (alpha*={alpha_star:.2f})")

# =========================================================
# D. Paired t-tests (5 seeds = 5 paired observations)
# =========================================================
def paired_ttest(a, b):
    t, p = stats.ttest_rel(a, b)
    return {"t_stat": float(t), "p_value": float(p), "mean_diff": float(np.mean(a) - np.mean(b))}

sig_tests = {}
sig_tests["MLP_vs_RandomForest"] = paired_ttest(mlp_baseline_raw["accuracy"], rf_results["accuracy"]["raw"])
sig_tests["MLP_vs_HistGB"] = paired_ttest(mlp_baseline_raw["accuracy"], hgb_results["accuracy"]["raw"])
sig_tests["MLP_nonprivate_vs_DPSGD_eps7.27"] = paired_ttest(mlp_baseline_raw["accuracy"], sweep_raw[15.0]["metrics"]["accuracy"])
sig_tests["MLP_nonprivate_vs_DPSGD_eps2.20"] = paired_ttest(mlp_baseline_raw["accuracy"], sweep_raw[40.0]["metrics"]["accuracy"])
sig_tests["MLP_nonprivate_vs_DPSGD_eps0.39"] = paired_ttest(mlp_baseline_raw["accuracy"], sweep_raw[200.0]["metrics"]["accuracy"])
sig_tests["DPSGD_eps7.27_vs_eps0.39"] = paired_ttest(sweep_raw[15.0]["metrics"]["accuracy"], sweep_raw[200.0]["metrics"]["accuracy"])
for k, v in sig_tests.items():
    print(k, {kk: round(vv,4) if isinstance(vv,float) else vv for kk,vv in v.items()})

# =========================================================
# Save everything
# =========================================================
out = {
    "rf": {k: {"mean": v["mean"], "std": v["std"], "raw": v["raw"]} for k, v in rf_results.items()},
    "hgb": {k: {"mean": v["mean"], "std": v["std"], "raw": v["raw"]} for k, v in hgb_results.items()},
    "mlp_baseline_raw": mlp_baseline_raw,
    "dp_sweep_raw": {str(k): v["metrics"] for k, v in sweep_raw.items()},
    "privacy_accounting_comparison": privacy_table,
    "significance_tests": sig_tests,
}
with open("extra_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved extra_results.json")
