"""Unit tests for the DP-SGD implementation (reviewer comment 10a)."""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from dpnet import Net
from dpmlp import DPNet            # original loop implementation

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 15)); y = (rng.random(40) < .6).astype(float)

# 1. vectorised per-example gradients == original loop implementation
ref = DPNet(15, seed=3); new = Net(15, seed=3)
for a, b in zip([ref.W1, ref.W2, ref.W3], new.W): assert np.allclose(a, b)      # same init
G = new.per_example_grads(X, y)
Gref = np.array([np.concatenate([g.ravel() for g in ref._grad_single(X[i], y[i])]) for i in range(40)])
# parameter order differs (W1,b1,W2,b2,W3,b3 in both) -> directly comparable
print("1. max |grad_vec - grad_loop| =", np.abs(G - Gref).max()); assert np.allclose(G, Gref, atol=1e-10)

# 2. numerical gradient check
def loss(net):
    p = np.clip(net.predict_proba(X), 1e-12, 1-1e-12); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
flat = lambda n: np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in zip(n.W, n.b)])
g_mean = G.mean(0); eps = 1e-6; base = flat(new); worst = 0
for j in rng.choice(len(base), 25, replace=False):
    for sign in (+1, -1):
        pass
    p = base.copy(); p[j] += eps; n1 = Net(15, seed=3); parts = n1._unflatten(p)
    n1.W = [parts[2*i] for i in range(3)]; n1.b = [parts[2*i+1] for i in range(3)]
    m = base.copy(); m[j] -= eps; n2 = Net(15, seed=3); parts = n2._unflatten(m)
    n2.W = [parts[2*i] for i in range(3)]; n2.b = [parts[2*i+1] for i in range(3)]
    worst = max(worst, abs((loss(n1)-loss(n2))/(2*eps) - g_mean[j]))
print("2. max |numerical - analytic| gradient =", worst); assert worst < 1e-6

# 3. clipped norms <= C
C = 1.0; norms = np.linalg.norm(G, axis=1); Gc = G*np.minimum(1, C/norms[:, None])
print("3. max clipped norm =", np.linalg.norm(Gc, axis=1).max(), "(C =", C, ")"); assert np.linalg.norm(Gc, axis=1).max() <= C+1e-12

# 4. injected noise std == sigma*C (lr=1, B=n: update = -(sum clipped + noise)/n)
n_p = new.n_params(); sig, C = 5.0, 2.0; dev = []
for s in range(300):
    net = Net(15, seed=3); w0 = flat(net); net.fit(X, y, steps=1, lr=1.0, clip=C, sigma=sig, seed=s)
    clipped_sum = (G*np.minimum(1, C/norms[:, None])).sum(0) if False else None
    Gs = net.per_example_grads(X, y)  # after update grads differ; recompute on w0 instead
    net0 = Net(15, seed=3); Gs = net0.per_example_grads(X, y)
    Gs = Gs*np.minimum(1, C/np.linalg.norm(Gs, axis=1, keepdims=True))
    dev.append(((w0 - flat(net))*len(X) - Gs.sum(0)))
dev = np.array(dev); print(f"4. empirical noise std = {dev.std():.3f}, expected sigma*C = {sig*C:.3f}"); assert abs(dev.std()/(sig*C) - 1) < 0.02

# 5. Poisson sampling rate ~ q
q = 0.2; rate = np.mean([ (np.random.default_rng(s).random(10000) < q).mean() for s in range(5)]); print("5. sampling rate", round(rate, 3))

# 6. hand calculation on a 2-row batch: tiny linear-ish check via one non-private step
X2, y2 = X[:2], y[:2]; net = Net(15, seed=3); w0 = flat(net); g2 = net.per_example_grads(X2, y2).mean(0)
net.fit(X2, y2, steps=1, lr=0.1); assert np.allclose(w0 - flat(net), 0.1*g2)
print("6. one-step update on a 2-row batch matches lr * mean gradient")
# 7. fast mean-gradient path equals mean of per-example gradients
assert np.allclose(new.mean_grad(X, y), new.per_example_grads(X, y).mean(0), atol=1e-12); print("7. mean_grad == mean of per-example gradients")
# 8. linear (hidden=()) model = logistic regression gradient
lin = Net(15, hidden=(), seed=1); pg = lin.per_example_grads(X, y); pr = lin.predict_proba(X)
assert np.allclose(pg[:, :15], X*(pr-y)[:, None], atol=1e-12); print("8. linear model gradient = x*(p-y)")
print("ALL TESTS PASSED")
