import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              brier_score_loss)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------
# Manual SMOTE (k-NN interpolation on the minority class only)
# ---------------------------------------------------------------
def smote(X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_count = counts.max()
    minority_count = counts.min()
    n_needed = majority_count - minority_count

    X_min = X[y == minority_class]
    n_min = X_min.shape[0]

    # pairwise distances within minority class
    dists = np.linalg.norm(X_min[:, None, :] - X_min[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    knn_idx = np.argsort(dists, axis=1)[:, :k]

    synthetic = np.zeros((n_needed, X.shape[1]))
    for i in range(n_needed):
        base_i = rng.integers(0, n_min)
        neighbor_i = knn_idx[base_i, rng.integers(0, k)]
        gap = rng.random()
        synthetic[i] = X_min[base_i] + gap * (X_min[neighbor_i] - X_min[base_i])

    X_res = np.vstack([X, synthetic])
    y_res = np.concatenate([y, np.full(n_needed, minority_class)])
    perm = rng.permutation(len(y_res))
    return X_res[perm], y_res[perm]


# ---------------------------------------------------------------
# Small feed-forward network trained by hand (numpy), supporting
# plain mini-batch SGD or per-example-clipped, noise-added DP-SGD
# (Abadi et al., 2016 style).
# ---------------------------------------------------------------
class DPNet:
    def __init__(self, n_in, h1=16, h2=8, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, h1))
        self.b1 = np.zeros(h1)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2))
        self.b2 = np.zeros(h2)
        self.W3 = rng.normal(0, np.sqrt(2 / h2), (h2, 1))
        self.b3 = np.zeros(1)

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1 / (1 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1 + ez)
        return out

    def _forward_single(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        a3 = self._sigmoid(z3)
        return z1, a1, z2, a2, z3, a3

    def forward_batch(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        a3 = self._sigmoid(z3)
        return a3.ravel()

    def predict_proba(self, X):
        return self.forward_batch(X)

    def _grad_single(self, x, y):
        x = x.reshape(1, -1)
        z1, a1, z2, a2, z3, a3 = self._forward_single(x)
        dz3 = a3 - y                      # (1,1)
        dW3 = a2.T @ dz3
        db3 = dz3.ravel()
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.ravel()
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.ravel()
        return [dW1, db1, dW2, db2, dW3, db3]

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _apply_update(self, grads, lr):
        params = self._params()
        for p, g in zip(params, grads):
            p -= lr * g

    def fit(self, X, y, epochs=60, batch_size=32, lr=0.08,
            clip_norm=None, noise_multiplier=0.0, seed=42, verbose=False):
        """
        clip_norm is None            -> ordinary mini-batch SGD (no DP)
        clip_norm is a float value C -> per-example gradient clipping to
                                         L2 norm C; if noise_multiplier>0,
                                         Gaussian noise N(0,(sigma*C)^2) is
                                         added to the summed gradient before
                                         averaging (DP-SGD, Abadi et al. 2016)
        """
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        loss_history = []
        n_steps = 0
        for epoch in range(epochs):
            idx = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                bx, by = X[batch_idx], y[batch_idx]
                bsz = len(batch_idx)

                if clip_norm is None:
                    # plain batched backprop (fast path, no privacy)
                    z1, a1, z2, a2, z3, a3 = self._forward_single(bx)
                    a3c = np.clip(a3, 1e-7, 1 - 1e-7)
                    epoch_loss += -np.mean(by.reshape(-1, 1) * np.log(a3c) +
                                            (1 - by.reshape(-1, 1)) * np.log(1 - a3c))
                    dz3 = (a3 - by.reshape(-1, 1)) / bsz
                    dW3 = a2.T @ dz3; db3 = dz3.sum(0)
                    da2 = dz3 @ self.W3.T
                    dz2 = da2 * (z2 > 0)
                    dW2 = a1.T @ dz2; db2 = dz2.sum(0)
                    da1 = dz2 @ self.W2.T
                    dz1 = da1 * (z1 > 0)
                    dW1 = bx.T @ dz1; db1 = dz1.sum(0)
                    grads = [dW1, db1, dW2, db2, dW3, db3]
                else:
                    # per-example clipped gradients -> DP-SGD
                    summed = None
                    for j in range(bsz):
                        g = self._grad_single(bx[j], by[j])
                        flat = np.concatenate([gg.ravel() for gg in g])
                        norm = np.linalg.norm(flat) + 1e-12
                        scale = min(1.0, clip_norm / norm)
                        g = [gg * scale for gg in g]
                        if summed is None:
                            summed = g
                        else:
                            summed = [s + gg for s, gg in zip(summed, g)]
                    if noise_multiplier > 0:
                        noisy = []
                        for s in summed:
                            noise = rng.normal(0, noise_multiplier * clip_norm, s.shape)
                            noisy.append((s + noise) / bsz)
                        grads = noisy
                    else:
                        grads = [s / bsz for s in summed]
                    n_steps += 1

                self._apply_update(grads, lr)
            loss_history.append(epoch_loss)
        self.n_steps = n_steps
        return loss_history


def privacy_epsilon(noise_multiplier, n_steps, delta):
    """
    Single-step (eps0, delta0)-DP of the Gaussian mechanism (analytic form,
    Dwork & Roth 2014, Thm A.1):  eps0 = sqrt(2 ln(1.25/delta0)) / sigma
    composed over n_steps releases using the advanced composition theorem
    (Dwork, Rothblum & Vadhan, 2010): for any delta' > 0, k adaptive
    (eps0,delta0)-DP mechanisms compose into a (eps_total, k*delta0+delta')-DP
    mechanism with
        eps_total = sqrt(2k ln(1/delta')) * eps0 + k * eps0 * (e^eps0 - 1)
    Advanced composition scales roughly as sqrt(k) rather than the k of
    naive (linear) composition. It is still a looser bound than the
    moments-accountant / RDP accounting used by tools such as Opacus, which
    is noted as a limitation of this implementation.
    """
    if noise_multiplier <= 0:
        return np.inf
    delta0 = delta / 2.0
    delta_prime = delta / 2.0
    eps0 = np.sqrt(2 * np.log(1.25 / delta0)) / noise_multiplier
    k = n_steps
    eps_total = np.sqrt(2 * k * np.log(1 / delta_prime)) * eps0 + k * eps0 * (np.exp(eps0) - 1)
    return eps_total


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "proba": proba,
    }
