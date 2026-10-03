"""Unified data loading, leak-free preprocessing, metrics and sigma cache for all three datasets.
Label convention in this module: y = 1 means the ADVERSE outcome (loan rejected / borrower defaults / bad credit risk)."""
import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(ROOT, "..")
from data import Prep as LoanPrep, load_raw as _load_loan
from accounting import solve_sigma

# ---------------------------------------------------------------- generic tabular preprocessing
class TabPrep:
    def __init__(self, num, cat, log=(), slog=()): self.num, self.cat, self.log, self.slog = list(num), list(cat), list(log), list(slog)
    def _mat(self, d):
        M = d[self.num].astype(float).copy()
        for c in self.log:  M[c] = np.log1p(np.clip(M[c], 0, None))
        for c in self.slog: M[c] = np.sign(M[c]) * np.log1p(np.abs(M[c]))
        oh = pd.DataFrame({f"{c}={l}": (d[c].astype(str) == l).astype(float) for c in self.cat for l in self.levels[c]}, index=d.index)
        return pd.concat([M, oh], axis=1)
    def fit(self, d):
        self.levels = {c: sorted(d[c].astype(str).unique()) for c in self.cat}
        Z = self._mat(d); self.cols = list(Z.columns); self.scaler = StandardScaler().fit(Z.values); return self
    def transform(self, d): return self.scaler.transform(self._mat(d)[self.cols].values)

GERMAN_COLS = ["status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", "installment_rate",
               "personal_status_sex", "other_debtors", "present_residence", "property", "age", "other_installment_plans", "housing",
               "number_credits", "job", "people_liable", "telephone", "foreign_worker"]
G_NUM = ["duration", "amount", "installment_rate", "present_residence", "age", "number_credits", "people_liable"]
G_CAT = [c for c in GERMAN_COLS if c not in G_NUM]

class Dataset:
    def __init__(self, name, df, y_adv, split_labels, make_prep, groups, rule=None, note=""):
        self.name, self.df, self.y, self.split_y, self.make_prep, self.groups, self.rule, self.note = name, df, y_adv, split_labels, make_prep, groups, rule, note

def load(name):
    if name == "loan":
        df, y_appr = _load_loan(os.path.join(DATA, "Train.csv")); y_adv = 1.0 - y_appr
        g = lambda d: {"Gender": d["Gender"].fillna("Unknown").values, "Married": d["Married"].fillna("Unknown").values, "Property_Area": d["Property_Area"].values}
        return Dataset("loan", df, y_adv, y_appr, lambda **k: LoanPrep(**k), g, rule="Credit_History", note="Loan_Status=N (rejected) is the positive/adverse class")
    if name == "german":
        d = pd.read_csv(os.path.join(DATA, "data_extra", "german_full_1000.csv"), header=None); d.columns = GERMAN_COLS + ["risk"]
        y = (d.pop("risk").values == 2).astype(float)
        g = lambda x: {"Sex/marital": np.where(x["personal_status_sex"].isin(["A92", "A95"]), "female-coded", "male-coded"),
                       "Age": np.where(x["age"] < 25, "<25", ">=25"), "Foreign": x["foreign_worker"].values}
        return Dataset("german", d, y, y, lambda **k: TabPrep([c for c in G_NUM if not (k.get("drop_protected") and c == "age")], [c for c in G_CAT if not (k.get("drop_protected") and c in ("personal_status_sex", "foreign_worker"))], log=["amount", "duration"]), g, note="risk=bad is the positive class")
    if name == "taiwan":
        d = pd.read_csv(os.path.join(DATA, "data_extra", "taiwan_credit_default_30000.csv")).drop(columns=["ID"])
        y = d.pop(d.columns[-1]).values.astype(float)
        d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})
        pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]; bill = [f"BILL_AMT{i}" for i in range(1, 7)]; pa = [f"PAY_AMT{i}" for i in range(1, 7)]
        g = lambda x: {"Sex": np.where(x["SEX"] == 1, "male", "female"), "Marriage": x["MARRIAGE"].astype(str).values,
                       "Age": np.where(x["AGE"] < 30, "<30", ">=30")}
        return Dataset("taiwan", d, y, y, lambda **k: TabPrep(["LIMIT_BAL"] + ([] if k.get("drop_protected") else ["AGE"]) + pay + bill + pa, ["EDUCATION"] + ([] if k.get("drop_protected") else ["SEX", "MARRIAGE"]), log=["LIMIT_BAL"] + pa, slog=bill),
                       g, rule="PAY_0", note="default next month is the positive class")
    raise ValueError(name)

def rule_pred(ds, dtest, prep):
    if ds.name == "loan": return (prep.raw_credit == 0).astype(float)      # Credit_History==0 -> predict reject
    if ds.name == "taiwan": return (dtest["PAY_0"].values >= 1).astype(float)  # delayed last month -> predict default
    return None

# ---------------------------------------------------------------- metrics
def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int); y = y.astype(int)
    try: auc = roc_auc_score(y, p)
    except ValueError: auc = float("nan")
    tpr = float(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)); tnr = float(((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return dict(acc=accuracy_score(y, pred), auc=auc, bacc=balanced_accuracy_score(y, pred),
                mcc=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0, rec_adv=tpr, tnr=tnr)

# ---------------------------------------------------------------- sigma cache (persisted)
DELTA = 1e-5
_SC = os.path.join(ROOT, "results", "sigma_cache.json")
_sig = json.load(open(_SC)) if os.path.exists(_SC) else {}
def sigma_for(steps, q, eps, delta=DELTA):
    k = f"{steps}|{None if q is None else round(q, 6)}|{eps}|{delta}"
    if k not in _sig:
        _sig[k] = solve_sigma(eps, steps, delta, q); json.dump(_sig, open(_SC, "w"))
    return _sig[k]

# ---------------------------------------------------------------- explanation units
def feature_partition(name, cols):
    """one-hot columns of the same original field are merged -> 'feature' level"""
    parts = {}
    for i, c in enumerate(cols):
        key = c.split("=")[0] if "=" in c else c
        if name == "loan" and c.startswith("PA_"): key = "Property_Area"
        parts.setdefault(key, []).append(i)
    return parts

def unit_partition(name, cols):
    """correlated fields are merged into one explanation unit -> 'unit' level"""
    units = {}
    for k, idx in feature_partition(name, cols).items():
        if name == "loan" and (k == "Income_to_Loan_Ratio" or k.endswith("_log")): u = "Income_and_amounts"
        elif name == "german" and k in ("duration", "amount"): u = "Loan_size"
        elif name == "taiwan" and k.startswith("PAY_AMT"): u = "PAY_AMT_series"
        elif name == "taiwan" and k.startswith("PAY_"): u = "PAY_status_series"
        elif name == "taiwan" and k.startswith("BILL_AMT"): u = "BILL_series"
        else: u = k
        units.setdefault(u, []).extend(idx)
    return units
