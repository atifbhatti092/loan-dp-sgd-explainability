"""Vectorised NumPy MLP with per-example gradients and DP-SGD (clip + Gaussian noise).
Same architecture as the paper: in -> 16 ReLU -> 8 ReLU -> 1 sigmoid, He init, BCE loss.
Supports full-batch or Poisson-subsampled DP-SGD. Add/remove adjacency: sensitivity = C.
NOTE: numpy's default_rng is NOT a cryptographically secure noise source."""
import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-np.abs(z))), np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))

class Net:
    def __init__(self, n_in, hidden=(16, 8), seed=0):
        rng = np.random.default_rng(seed)
        dims = [n_in, *hidden, 1]
        self.W = [rng.normal(0, np.sqrt(2/dims[i]), (dims[i], dims[i+1])) for i in range(len(dims)-1)]
        self.b = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
        self.shapes = [w.shape for w in self.W]

    # ---- forward ----
    def _forward(self, X):
        Zs, As = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            Zs.append(z)
            a = np.maximum(z, 0) if i < len(self.W)-1 else _sigmoid(z)
            As.append(a)
        return Zs, As
    def predict_proba(self, X):
        return self._forward(X)[1][-1].ravel()
    def logit(self, X):
        return self._forward(X)[0][-1].ravel()

    # ---- per-example gradients, flattened to (n, P) ----
    def per_example_grads(self, X, y):
        Zs, As = self._forward(X)
        n, L = len(X), len(self.W)
        dz = As[-1] - y.reshape(-1, 1)                    # (n,1) BCE + sigmoid
        parts = [None]*(2*L)
        for i in range(L-1, -1, -1):
            parts[2*i]   = (As[i][:, :, None] * dz[:, None, :]).reshape(n, -1)   # dW_i
            parts[2*i+1] = dz                                                     # db_i
            if i > 0:
                dz = (dz @ self.W[i].T) * (Zs[i-1] > 0)
        return np.concatenate(parts, axis=1)

    def mean_grad(self, X, y):
        """Average gradient without materialising per-example gradients (non-private fast path)."""
        Zs, As = self._forward(X); n, L = len(X), len(self.W)
        dz = (As[-1] - y.reshape(-1, 1)) / max(n, 1); parts = [None]*(2*L)
        for i in range(L-1, -1, -1):
            parts[2*i] = As[i].T @ dz; parts[2*i+1] = dz.sum(0)
            if i > 0: dz = (dz @ self.W[i].T) * (Zs[i-1] > 0)
        return np.concatenate([p.ravel() for p in parts])

    def _unflatten(self, g):
        out, k = [], 0
        for W, b in zip(self.W, self.b):
            out.append(g[k:k+W.size].reshape(W.shape)); k += W.size
            out.append(g[k:k+b.size]);                  k += b.size
        return out
    def _step(self, g, lr):
        parts = self._unflatten(g)
        for i in range(len(self.W)):
            self.W[i] -= lr*parts[2*i]; self.b[i] -= lr*parts[2*i+1]
    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # ---- training ----
    def fit(self, X, y, steps, lr, clip=None, sigma=0.0, q=None, seed=0, wd=0.0, callback=None):
        """clip=None -> non-private. q=None -> full batch. q in (0,1) -> Poisson sampling.
        DP update:  g = ( sum_i clip(g_i, C) + N(0, sigma^2 C^2 I) ) / B,  B = n (full) or q*n (Poisson)."""
        rng = np.random.default_rng(seed)
        n = len(X)
        for t in range(steps):
            if q is None:
                Xb, yb, B = X, y, float(n)
            else:
                m = rng.random(n) < q
                Xb, yb, B = X[m], y[m], q*n
            if len(yb) == 0 and clip is not None:
                Xb, yb = X[:0], y[:0]
            if clip is None:
                g = self.mean_grad(Xb, yb) if len(yb) else np.zeros(self.n_params())
            else:
                G = self.per_example_grads(Xb, yb) if len(yb) else np.zeros((0, self.n_params()))
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G * np.minimum(1.0, clip/np.maximum(norms, 1e-12))
                noise = rng.normal(0, sigma*clip, self.n_params()) if sigma > 0 else 0.0
                g = (G.sum(0) + noise) / B
            if wd: g = g + wd*np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in zip(self.W, self.b)])
            self._step(g, lr)
            if callback is not None: callback(t, self)
        return self
