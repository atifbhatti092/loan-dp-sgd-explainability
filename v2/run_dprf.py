"""Differentially private random forest baseline (IBM diffprivlib) on the same outer splits as run_cv.py.
diffprivlib's RandomForestClassifier builds random trees whose splits do not depend on the data and chooses each leaf label with
the permute-and-flip mechanism (pure epsilon-DP, delta = 0). Number of trees (10) and depth (5) are fixed a priori, so epsilon is end-to-end:
no hyper-parameter search touches the private data. Features are clipped to [-4, 4] after standardisation (as for DP-EBM).
usage: python3 run_dprf.py <dataset> <budget_seconds>      (pip install diffprivlib; resumable)"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from diffprivlib.models import RandomForestClassifier as DPRF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import load, metrics, ROOT
EPS = [0.5, 1, 2, 4, 8]; REPEATS = {"loan": 5, "german": 3, "taiwan": 2}

def one_split(ds, si, tr, te):
    dtr, dte = ds.df.iloc[tr], ds.df.iloc[te]; ytr, yte = ds.y[tr], ds.y[te]
    prep = ds.make_prep().fit(dtr); Xa, Xb = np.clip(prep.transform(dtr), -4, 4), np.clip(prep.transform(dte), -4, 4)
    rec = {"split": si, "models": {}}
    for eps in EPS:
        try:
            m = DPRF(n_estimators=10, max_depth=5, epsilon=eps, bounds=(-4.0, 4.0), classes=[0, 1], random_state=si).fit(Xa, ytr.astype(int))
            rec["models"][f"dp_rf_eps{eps}"] = metrics(yte, m.predict_proba(Xb)[:, 1])
        except Exception as e: rec["models"][f"dp_rf_eps{eps}"] = {"error": str(e)[:120]}
    return rec

def main(name, budget):
    t0 = time.time(); ds = load(name); path = os.path.join(ROOT, "results", f"dprf_{name}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    for si, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS[name], random_state=0).split(ds.df, ds.split_y)):
        if si < len(out): continue
        if time.time()-t0 > budget: print(f"PAUSED at {si}"); return
        out.append(one_split(ds, si, tr, te)); json.dump(out, open(path, "w")); print(f"{name} split {si+1} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", name)
if __name__ == "__main__": main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 240)
