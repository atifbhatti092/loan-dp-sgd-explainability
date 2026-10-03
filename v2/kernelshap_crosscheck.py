"""MANUAL STEP (optional, comment 25): KernelSHAP as a second explanation reference for the loan data.
Install:  pip install shap     Run:  python3 kernelshap_crosscheck.py
Compares the unit-level ranking of KernelSHAP with Integrated Gradients for the non-private and DP (eps = 1, 4) networks on split 0."""
import json, numpy as np, shap
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold
from pipeline import load, sigma_for, unit_partition
from dpnet import Net
from explain import integrated_gradients, group_sum
ds = load("loan"); cv = json.load(open("results/cv_loan.json")); tr, te = next(iter(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(ds.df, ds.split_y)))
prep = ds.make_prep().fit(ds.df.iloc[tr]); Xtr, Xte = prep.transform(ds.df.iloc[tr]), prep.transform(ds.df.iloc[te]); ytr = ds.y[tr]; parts = unit_partition("loan", prep.cols); n = len(ytr); q = 64/n
ch = cv[0]["chosen"]; lr, st = ch["mlp_np"]
nets = {"non-private": Net(Xtr.shape[1], seed=0).fit(Xtr, ytr, steps=st, lr=lr, seed=0)}
for eps in (1, 4):
    C, lr2 = ch[f"mlp_PO_{eps}"]; nets[f"DP eps={eps}"] = Net(Xtr.shape[1], seed=100).fit(Xtr, ytr, steps=300, lr=lr2, clip=C, sigma=sigma_for(300, q, eps), q=q, seed=100)
bg = shap.kmeans(Xtr, 25)
for name, net in nets.items():
    sv = shap.KernelExplainer(net.predict_proba, bg).shap_values(Xte, nsamples=300)
    ks = np.abs(group_sum(np.asarray(sv), parts)).mean(0); ig = np.abs(group_sum(integrated_gradients(net, Xte, np.zeros(Xtr.shape[1]), 128)[0], parts)).mean(0)
    print(f"{name:12s} Spearman(KernelSHAP units, IG units) = {spearmanr(ks, ig)[0]:.2f}")
