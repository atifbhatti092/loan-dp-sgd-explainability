"""MANUAL STEP (not run by the author of this file: needs torch + opacus, which could not be installed in the analysis sandbox).
Cross-checks (a) the privacy accountants of this repository against Opacus and (b) trains the same 17-16-8-1 MLP with Opacus.
Install:  pip install torch opacus dp-accounting     Run:  python3 opacus_crosscheck.py
Expected: the PLD/PRV epsilons agree to about 2 decimals; the RDP epsilons agree closely. Report any larger gap."""
import numpy as np
from opacus.accountants import RDPAccountant, PRVAccountant
from accounting import eps_pld, eps_rdp
q, T, delta = 64/491, 300, 1e-5
print("sigma | ours PLD | ours RDP | Opacus PRV | Opacus RDP")
for sigma in (1.5, 2.0, 3.0, 5.0, 8.5):
    rdp, prv = RDPAccountant(), PRVAccountant()
    for acc in (rdp, prv):
        for _ in range(T): acc.step(noise_multiplier=sigma, sample_rate=q)
    print(f"{sigma:5} | {eps_pld(sigma, T, delta, q):9.3f} | {eps_rdp(sigma, T, delta, q):8.3f} | {prv.get_epsilon(delta):10.3f} | {rdp.get_epsilon(delta):10.3f}")
# (b) training comparison on one loan split
import torch, torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pipeline import load
ds = load("loan"); tr, te = next(iter(StratifiedKFold(5, shuffle=True, random_state=0).split(ds.df, ds.split_y)))
prep = ds.make_prep().fit(ds.df.iloc[tr]); Xtr = torch.tensor(prep.transform(ds.df.iloc[tr]), dtype=torch.float32); ytr = torch.tensor(ds.y[tr], dtype=torch.float32)
Xte = prep.transform(ds.df.iloc[te]); yte = ds.y[te]
model = nn.Sequential(nn.Linear(Xtr.shape[1], 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
opt = torch.optim.SGD(model.parameters(), lr=0.5); sigma, C = 2.0, 0.25
loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
pe = PrivacyEngine(accountant="prv"); model, opt, loader = pe.make_private(module=model, optimizer=opt, data_loader=loader, noise_multiplier=sigma, max_grad_norm=C, poisson_sampling=True)
steps = 0; lossf = nn.BCEWithLogitsLoss()
while steps < T:
    for xb, yb in loader:
        opt.zero_grad(); lossf(model(xb).squeeze(1), yb).backward(); opt.step(); steps += 1
        if steps >= T: break
with torch.no_grad(): p = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).squeeze(1).numpy()
print("Opacus-trained DP MLP: test AUC", round(roc_auc_score(yte, p), 3), "| epsilon (PRV, delta=1e-5):", round(pe.get_epsilon(1e-5), 3))
print("Compare with this repository: v2/results/cv_loan.json (dp_mlp_PO_eps*) and eps_pld(sigma=2.0, T=300, q=64/491).")
