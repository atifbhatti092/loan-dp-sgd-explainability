"""White-box canary audit of full-batch DP-SGD (Nasr et al. 2023 style; adversary sees every noisy update).

Neighbouring datasets: D (no canary) and D' = D + one canary record. In every step the adversary knows D, the current
weights theta_t and the canary, so it can compute the clipped gradient sum of D and the clipped canary gradient g_t.
It observes the released noisy sum S_t and forms
        z_t = < S_t - sum_{i in D} clip(g_i), g_t > / (sigma * C * ||g_t||)
which is N(0,1) without the canary and N(||g_t||/(sigma*C), 1) with it (||g_t|| <= C after clipping). Summing over T steps,
Z = sum_t z_t ~ N(0, T) vs N(sum_t ||g_t|| / (sigma*C), T): a Gaussian shift mu_hat = sum_t ||g_t|| / (sigma*C*sqrt(T)).
Theory (worst case, every canary gradient has norm exactly C): mu = sqrt(T)/sigma. The audit therefore checks, end to end,
(i) that the clipping bound holds, (ii) that the injected noise has std sigma*C, and (iii) how close real data gets to the
worst case that the accountant assumes.
AuditNet.fit_audit with member=False and no statistic reproduces Net.fit exactly (see tests_ext.py)."""
import numpy as np
from dpnet import Net

class AuditNet(Net):
    def fit_audit(self, X, y, steps, lr, clip, sigma, xc, yc, member, seed=0):
        """Full-batch DP-SGD with B = len(X) (same convention as Net.fit). Returns (Z, sum_t ||g_t||/C, max clipped canary norm / C)."""
        rng = np.random.default_rng(seed); n = len(X); P = self.n_params()
        xc = np.asarray(xc, float).reshape(1, -1); yc = np.asarray([yc], float)
        Z = 0.0; gsum = 0.0; maxn = 0.0
        for t in range(steps):
            G = self.per_example_grads(X, y)
            G = G * np.minimum(1.0, clip/np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12))
            cs = G.sum(0)
            gc = self.per_example_grads(xc, yc)[0]
            nrm = np.linalg.norm(gc); gc = gc * min(1.0, clip/max(nrm, 1e-12)); gn = np.linalg.norm(gc)
            S = cs + (gc if member else 0.0) + rng.normal(0, sigma*clip, P)
            if gn > 0: Z += float((S - cs) @ gc) / (sigma*clip*gn)
            gsum += gn/clip; maxn = max(maxn, gn/clip)
            self._step(S/n, lr)
        return Z, gsum, maxn
