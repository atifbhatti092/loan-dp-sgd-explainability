import numpy as np, json, sys
sys.path.insert(0, ".")
from dpmlp import DPNet, smote, privacy_epsilon, evaluate

data = np.load("prep_data.npz", allow_pickle=True)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_cols = list(data["feature_cols"])
n_in = X_train.shape[1]
DELTA = 1.0 / len(y_train)   # standard convention: delta < 1/n

results = {}

# =========================================================
# 1. Class-imbalance handling comparison (SMOTE vs class-weight)
#    -- trained on the SAME non-private MLP architecture --
# =========================================================
FULL_BATCH_KW = dict(epochs=30, lr=0.5)

def fit_eval(Xtr, ytr, sample_weight=None, seed=42, **kw):
    net = DPNet(n_in, seed=seed)
    if sample_weight is None:
        kw.setdefault("batch_size", len(Xtr))
        net.fit(Xtr, ytr, **kw)
    else:
        # emulate class weighting by oversampling gradients proportionally
        # (equivalent in expectation to a weighted BCE loss for this batch size)
        w = sample_weight
        reps = np.round(w / w.min()).astype(int)
        Xtr_w = np.repeat(Xtr, reps, axis=0)
        ytr_w = np.repeat(ytr, reps, axis=0)
        idx = np.random.default_rng(seed).permutation(len(ytr_w))
        kw.setdefault("batch_size", len(Xtr_w))
        net.fit(Xtr_w[idx], ytr_w[idx], **kw)
    return net, evaluate(net, X_test, y_test)

X_sm, y_sm = smote(X_train, y_train, k=5, seed=42)
net_smote, m_smote = fit_eval(X_sm, y_sm, seed=42, **FULL_BATCH_KW)

classes, counts = np.unique(y_train, return_counts=True)
class_w = {c: len(y_train) / (2 * cnt) for c, cnt in zip(classes, counts)}
sw = np.array([class_w[v] for v in y_train])
net_cw, m_cw = fit_eval(X_train, y_train, sample_weight=sw, seed=42, **FULL_BATCH_KW)

net_plain, m_plain = fit_eval(X_train, y_train, seed=42, **FULL_BATCH_KW)

results["class_balance_comparison"] = {
    "no_balancing": {k: v for k, v in m_plain.items() if k != "proba"},
    "smote": {k: v for k, v in m_smote.items() if k != "proba"},
    "class_weighting": {k: v for k, v in m_cw.items() if k != "proba"},
}
print("class balance comparison done")
for k, v in results["class_balance_comparison"].items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items() if kk not in ("confusion_matrix",)})

# best balancing strategy carried forward
best_key = max(results["class_balance_comparison"],
                key=lambda k: results["class_balance_comparison"][k]["f1"])
print("Best balancing strategy:", best_key)
if best_key == "smote":
    X_bal, y_bal = X_sm, y_sm
    balance_kwargs = {}
elif best_key == "class_weighting":
    X_bal, y_bal = X_train, y_train
    balance_kwargs = {"sample_weight": sw}
else:
    X_bal, y_bal = X_train, y_train
    balance_kwargs = {}

# =========================================================
# 2. Non-private baseline (epsilon = infinity) -- full training
#    averaged over 5 seeds for a stable estimate
# =========================================================
if best_key == "class_weighting":
    reps = np.round(sw / sw.min()).astype(int)
    Xb = np.repeat(X_train, reps, axis=0); yb = np.repeat(y_train, reps, axis=0)
    idx = np.random.default_rng(42).permutation(len(yb))
    Xb, yb = Xb[idx], yb[idx]
else:
    Xb, yb = X_bal, y_bal

BASE_SEEDS = [1, 2, 3, 4, 5]
base_runs = []
net_base = None
loss_hist_base = None
for s in BASE_SEEDS:
    net = DPNet(n_in, seed=s)
    lh = net.fit(Xb, yb, epochs=30, batch_size=len(Xb), lr=0.5, seed=s)
    m = evaluate(net, X_test, y_test)
    base_runs.append({k: v for k, v in m.items() if k not in ("proba", "confusion_matrix")})
    if s == BASE_SEEDS[0]:
        net_base, loss_hist_base, m_base = net, lh, m   # keep one representative model for explainability plots

base_metric_names = list(base_runs[0].keys())
results["baseline_nonprivate"] = {
    "mean": {k: float(np.mean([r[k] for r in base_runs])) for k in base_metric_names},
    "std": {k: float(np.std([r[k] for r in base_runs])) for k in base_metric_names},
    "runs": base_runs,
}
print("Baseline (non-private, mean of 5 seeds):",
      {k: round(v, 4) for k, v in results["baseline_nonprivate"]["mean"].items()})

# =========================================================
# 3. DP-SGD sweep across noise multipliers -> epsilon values
#    each point averaged over 5 seeds
# =========================================================
CLIP_NORM = 3.0
DP_SEEDS = [1, 2, 3, 4, 5]
sweep = []
for sigma in [200.0, 100.0, 60.0, 40.0, 25.0, 15.0]:
    runs = []
    eps = None
    for s in DP_SEEDS:
        net_dp = DPNet(n_in, seed=s)
        net_dp.fit(Xb, yb, epochs=30, batch_size=len(Xb), lr=0.5,
                   clip_norm=CLIP_NORM, noise_multiplier=sigma, seed=s)
        eps = privacy_epsilon(sigma, net_dp.n_steps, DELTA)
        m_dp = evaluate(net_dp, X_test, y_test)
        runs.append({k: v for k, v in m_dp.items() if k not in ("proba", "confusion_matrix")})
    mean_row = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    std_row = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    row = {"sigma": sigma, "epsilon": eps, "delta": DELTA, "n_steps": 30,
           "mean": mean_row, "std": std_row}
    sweep.append(row)
    print(f"sigma={sigma:6.2f}  eps={eps:8.3f}  "
          f"acc={mean_row['accuracy']:.4f}+-{std_row['accuracy']:.4f}  "
          f"f1={mean_row['f1']:.4f}  auc={mean_row['roc_auc']:.4f}+-{std_row['roc_auc']:.4f}")

results["dp_sgd_sweep"] = sweep

# =========================================================
# 4. 5-fold stratified CV for the non-private model (stability check)
# =========================================================
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc, cv_f1, cv_auc = [], [], []
Xall = np.vstack([X_train, X_test]); yall = np.concatenate([y_train, y_test])
for tr_idx, te_idx in skf.split(Xall, yall):
    net_cv = DPNet(n_in, seed=42)
    net_cv.fit(Xall[tr_idx], yall[tr_idx], epochs=30, batch_size=len(tr_idx), lr=0.5)
    m_cv = evaluate(net_cv, Xall[te_idx], yall[te_idx])
    cv_acc.append(m_cv["accuracy"]); cv_f1.append(m_cv["f1"]); cv_auc.append(m_cv["roc_auc"])
results["cv_5fold"] = {
    "accuracy_mean": float(np.mean(cv_acc)), "accuracy_std": float(np.std(cv_acc)),
    "f1_mean": float(np.mean(cv_f1)), "f1_std": float(np.std(cv_f1)),
    "auc_mean": float(np.mean(cv_auc)), "auc_std": float(np.std(cv_auc)),
    "folds_acc": cv_acc,
}
print("5-fold CV:", {k: round(v, 4) if isinstance(v, float) else v for k, v in results["cv_5fold"].items() if k != "folds_acc"})

# =========================================================
# 5. Explainability: permutation importance + integrated gradients
#    on the non-private baseline model
# =========================================================
def permutation_importance(model, X, y, n_repeats=20, seed=42):
    rng = np.random.default_rng(seed)
    base_auc = evaluate(model, X, y)["roc_auc"]
    importances = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            auc = evaluate(model, Xp, y)["roc_auc"]
            drops.append(base_auc - auc)
        importances[j] = np.mean(drops)
    return importances, base_auc

perm_imp, base_auc_for_perm = permutation_importance(net_base, X_test, y_test)
results["permutation_importance"] = {
    "base_auc": base_auc_for_perm,
    "importances": dict(zip(feature_cols, perm_imp.tolist())),
}

def integrated_gradients(model, x, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(x)
    alphas = np.linspace(0, 1, steps)
    grads = []
    for a in alphas:
        xi = baseline + a * (x - baseline)
        eps = 1e-4
        g = np.zeros_like(x)
        f0 = model.predict_proba(xi.reshape(1, -1))[0]
        for k in range(len(x)):
            xi2 = xi.copy(); xi2[k] += eps
            f1 = model.predict_proba(xi2.reshape(1, -1))[0]
            g[k] = (f1 - f0) / eps
        grads.append(g)
    avg_grad = np.mean(grads, axis=0)
    return avg_grad * (x - baseline)

baseline_x = X_train.mean(axis=0)
sample_idx = np.random.default_rng(0).choice(len(X_test), size=40, replace=False)
ig_attrs = np.array([integrated_gradients(net_base, X_test[i], baseline_x, steps=20) for i in sample_idx])
ig_mean_abs = np.abs(ig_attrs).mean(axis=0)
results["integrated_gradients"] = dict(zip(feature_cols, ig_mean_abs.tolist()))

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
np.savez("artifacts.npz", loss_hist_base=loss_hist_base,
         proba_base=m_base["proba"], y_test=y_test,
         perm_imp=perm_imp, ig_mean_abs=ig_mean_abs,
         feature_cols=np.array(feature_cols))
print("\nSaved results.json and artifacts.npz")
