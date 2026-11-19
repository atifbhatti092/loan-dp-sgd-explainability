"""Tests for the additions (private tuning, canary audit). Run: python3 tests_ext.py"""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from dpnet import Net
from dpnet_audit import AuditNet
from private_tuning import calibrate, eps_from_mu, mu_from_eps, eps_total_rdp, noisy_select, clipped_loss_sum
from accounting import eps_fullbatch_exact

# 1. mu <-> epsilon, agrees with accounting.py
for e in (0.5, 1, 2, 4, 8):
    assert abs(eps_from_mu(mu_from_eps(e, 1e-5), 1e-5) - e) < 1e-6
assert abs(eps_from_mu(np.sqrt(30)/15, 1e-5) - eps_fullbatch_exact(15, 30, 1e-5)) < 1e-8
print("1. Gaussian-DP conversions agree with accounting.py")
# 2. total epsilon of the tuned pipeline equals the target (full batch, exact), and tuning costs noise
for m in (1, 2, 6):
    for e in (0.5, 1, 4, 8):
        c = calibrate(e, 100, None, m); assert abs(c["eps_check"] - e) < 1e-6
s = [calibrate(1, 100, None, m)["sigma_run"] for m in (1, 2, 6)]; assert s[0] < s[1] < s[2]
print("2. calibration hits the target epsilon; sigma_run grows with the number of candidates:", [round(x, 1) for x in s])
# 3. RDP fallback reproduces the RDP column of Table B2 of the paper within 2% and upper-bounds its PLD column
for sg, pld, rdp in ((8.5, 1.005, 1.099), (5, 1.832, 1.998), (3, 3.359, 3.657)):
    e = eps_total_rdp(sg, 300, 64/491, 1, None, 1e-5); assert e >= pld and abs(e - rdp)/rdp < 0.02
print("3. RDP fallback matches Table B2 (RDP column) and is above the PLD column")
# 4. selection mechanism: noise-free limit picks the argmin; sensitivity of clipped loss sum is <= L per record
rng = np.random.default_rng(0)
assert noisy_select([5, 3, 9], 3, 1e-9, 2.0, rng) == 1
p = rng.random(50); y = (rng.random(50) < .5).astype(float); L = 2.0
base = clipped_loss_sum(p, y, L)
for i in range(50):                                         # remove one record -> change <= L
    assert abs(base - clipped_loss_sum(np.delete(p, i), np.delete(y, i), L)) <= L + 1e-12
print("4. selection: argmin in the noise-free limit; clipped-loss sum has sensitivity <= L")
# 5. AuditNet with member=False reproduces Net.fit exactly (same seed)
X = rng.normal(size=(60, 12)); yy = (rng.random(60) < .4).astype(float); xc = rng.normal(size=12)*20
a = Net(12, seed=2).fit(X, yy, steps=7, lr=0.5, clip=0.3, sigma=4.0, seed=11)
b = AuditNet(12, seed=2); b.fit_audit(X, yy, 7, 0.5, 0.3, 4.0, xc, 1.0, False, seed=11)
assert all(np.allclose(u, v) for u, v in zip(a.W + a.b, b.W + b.b)); print("5. AuditNet(member=False) == Net.fit")
# 6. canary gradient is clipped to C and the statistic has the theoretical shift
net = AuditNet(12, seed=2); Zs = []
sigma, C, T = 3.0, 0.3, 5
zi, zo, mx = [], [], 0
R = 1500
for r in range(R):
    for mem, l in ((True, zi), (False, zo)):
        Z, gs, m = AuditNet(12, seed=2).fit_audit(X, yy, T, 0.5, C, sigma, xc, 1.0, mem, seed=r*2+int(mem)); l.append(Z); mx = max(mx, m)
assert mx <= 1 + 1e-9; mu_hat = (np.mean(zi) - np.mean(zo))/np.sqrt(T); mu = np.sqrt(T)/sigma
print(f"6. max canary norm/C = {mx:.6f}; mu_hat = {mu_hat:.3f} vs theory {mu:.3f}"); se = np.sqrt(2.0/R); assert abs(mu_hat - mu) < 4*se, (mu_hat, mu, se)
print("ALL EXTENSION TESTS PASSED")
