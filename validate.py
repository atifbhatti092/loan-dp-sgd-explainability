import numpy as np, pandas as pd, json, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss)
sys.path.insert(0, "..")
from dpmlp import DPNet, privacy_epsilon

cols = ['checking_status','duration','credit_history','purpose','credit_amount','savings',
        'employment','installment_rate','personal_status','other_debtors','residence_since',
        'property','age','other_installment_plans','housing','existing_credits','job',
        'num_liable','telephone','foreign_worker','target']
df = pd.read_csv('german_500.csv', header=None, names=cols)
y = (df['target'] == 1).astype(int).values  # 1 = good credit risk, 0 = bad

cat_cols = ['checking_status','credit_history','purpose','savings','employment',
            'personal_status','other_debtors','property','other_installment_plans',
            'housing','job','telephone','foreign_worker']
num_cols = ['duration','credit_amount','installment_rate','residence_since','age',
            'existing_credits','num_liable']

Xdf = df[cat_cols + num_cols].copy()
for c in cat_cols:
    Xdf[c] = LabelEncoder().fit_transform(Xdf[c])
# log-transform the skewed monetary field, mirroring the primary pipeline's treatment
Xdf['credit_amount'] = np.log1p(Xdf['credit_amount'])

X = Xdf.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_in = X_train.shape[1]
print("Train:", X_train.shape, "Test:", X_test.shape, "train class balance:", np.bincount(y_train))

def eval_metrics(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y_true, pred), precision=precision_score(y_true, pred),
                recall=recall_score(y_true, pred), f1=f1_score(y_true, pred),
                roc_auc=roc_auc_score(y_true, proba), brier=brier_score_loss(y_true, proba))

SEEDS = [1, 2, 3, 4, 5]
DELTA = 1.0 / len(y_train)

# Non-private baseline
base_raw = {k: [] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]}
for s in SEEDS:
    net = DPNet(n_in, seed=s)
    net.fit(X_train, y_train, epochs=30, batch_size=len(X_train), lr=0.5, seed=s)
    met = eval_metrics(y_test, net.predict_proba(X_test))
    for k, v in met.items():
        base_raw[k].append(v)
base_summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in base_raw.items()}
print("Non-private baseline:", {k: round(v["mean"],4) for k,v in base_summary.items()})

# Two DP-SGD points: strongest privacy (sigma=200) and weakest (sigma=15), same as primary sweep
CLIP_NORM = 3.0
dp_results = {}
for sigma in [200.0, 15.0]:
    raw = {k: [] for k in ["accuracy","precision","recall","f1","roc_auc","brier"]}
    n_steps = None
    for s in SEEDS:
        net = DPNet(n_in, seed=s)
        net.fit(X_train, y_train, epochs=30, batch_size=len(X_train), lr=0.5,
                clip_norm=CLIP_NORM, noise_multiplier=sigma, seed=s)
        n_steps = net.n_steps
        met = eval_metrics(y_test, net.predict_proba(X_test))
        for k, v in met.items():
            raw[k].append(v)
    eps = privacy_epsilon(sigma, n_steps, DELTA)
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in raw.items()}
    dp_results[sigma] = {"epsilon": eps, "n_steps": n_steps, "metrics": summary}
    print(f"sigma={sigma} eps={eps:.3f}:", {k: round(v["mean"],4) for k,v in summary.items()})

out = {
    "dataset": {"name": "German Credit (Statlog), 572-record subsample", "n_train": len(y_train),
                "n_test": len(y_test), "n_features": n_in,
                "train_good": int(np.bincount(y_train)[1]), "train_bad": int(np.bincount(y_train)[0])},
    "baseline_nonprivate": base_summary,
    "dp_sgd": {str(k): v for k, v in dp_results.items()},
}
with open("german_validation_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved german_validation_results.json")
