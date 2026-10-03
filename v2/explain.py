"""Integrated Gradients (analytic input gradients, completeness check), grouped permutation importance and stability metrics."""
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score

def input_grad(net, X):
    """d p / d x for every row (p = sigmoid output)."""
    Zs, As = net._forward(X); p = As[-1]; dz = p*(1-p)
    for i in range(len(net.W)-1, -1, -1):
        da = dz @ net.W[i].T
        if i == 0: return da
        dz = da*(Zs[i-1] > 0)

def integrated_gradients(net, X, base, m=64):
    """midpoint Riemann sum on the straight path base -> x; returns attributions (n,d) and the completeness residual."""
    diff = X - base; G = np.zeros_like(X)
    for a in (np.arange(m)+0.5)/m: G += input_grad(net, base + a*diff)
    attr = diff*G/m
    resid = attr.sum(1) - (net.predict_proba(X) - net.predict_proba(np.broadcast_to(base, X.shape)))
    return attr, resid

def baselines(Xtr):
    modal = np.array([np.unique(Xtr[:, j], return_counts=True)[0][np.argmax(np.unique(Xtr[:, j], return_counts=True)[1])] if len(np.unique(Xtr[:, j])) <= 3
                      else np.median(Xtr[:, j]) for j in range(Xtr.shape[1])])
    return {"zero": np.zeros(Xtr.shape[1]), "median": np.median(Xtr, 0), "modal": modal}

def group_sum(attr, parts):           # additivity of IG: a group's attribution is the sum of its members
    return np.stack([attr[:, idx].sum(1) for idx in parts.values()], 1)

def perm_importance(net, X, y, parts, repeats=5, seed=0):
    rng = np.random.default_rng(seed); base = roc_auc_score(y, net.predict_proba(X)); out = []
    for idx in parts.values():
        d = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, idx] = X[rng.permutation(len(X))][:, idx]; d.append(base - roc_auc_score(y, net.predict_proba(Xp)))
        out.append(np.mean(d))
    return np.array(out)

def compare(a, b, k=3):
    """agreement between two importance vectors"""
    top = lambda v: set(np.argsort(-v)[:k])
    return dict(spearman=float(spearmanr(a, b)[0]), kendall=float(kendalltau(a, b)[0]), top3=len(top(a) & top(b))/k,
                cosine=float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)))

def local_similarity(A, B):
    cos = (A*B).sum(1)/(np.linalg.norm(A, axis=1)*np.linalg.norm(B, axis=1) + 1e-12)
    return float(cos.mean()), float((np.sign(A) == np.sign(B)).mean())

def comprehensiveness(net, X, base, attr_feat, parts, K=3, rng=None):
    """mean |F(x) - F(x with the K most important features reset to baseline)|, averaged over k=1..K (higher = explanation points at what the model uses)"""
    ranks = np.argsort(np.argsort(-np.abs(attr_feat), axis=1), axis=1)      # 0 = most important
    p0 = net.predict_proba(X); tot = 0.0
    for k in range(1, K+1):
        Xk = X.copy()
        for j, idx in enumerate(parts.values()):
            m = ranks[:, j] < k; Xk[np.ix_(m, idx)] = base[idx]
        tot += np.abs(p0 - net.predict_proba(Xk)).mean()
    return tot/K
