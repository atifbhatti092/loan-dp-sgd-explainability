"""End-to-end differentially private hyper-parameter selection for DP-SGD.

The study in the paper reports epsilon for the final training run only; the (clip norm C, learning rate) search
reads the private data and is NOT covered. This module removes that gap with the simplest construction whose
accounting is exact and easy to audit:

  1. Train every candidate (C_j, lr_j), j = 1..m, with its own DP-SGD run on the whole training set
     (m independent Gaussian DP-SGD runs -> composition).
  2. Score each trained candidate on the training data with the SUM of per-record losses, each loss clipped to [0, L].
     Adding or removing one record changes every score by at most L, so the m-vector of scores has L2 sensitivity L*sqrt(m).
  3. Release scores + N(0, (sigma_sel * L * sqrt(m))^2) per coordinate (Gaussian mechanism, mu_sel = 1/sigma_sel)
     and keep the candidate with the smallest noisy score (post-processing).

Total privacy = composition of m*T subsampled-Gaussian steps and one Gaussian mechanism.
  * full batch: everything is Gaussian, so  mu_total^2 = m*T/sigma_run^2 + 1/sigma_sel^2  (exact, Dong-Roth-Su 2022)
  * Poisson sampling: PLD accountant (dp_accounting) if installed, otherwise an RDP bound implemented below
    (always an upper bound on the PLD epsilon, so results obtained with the fallback are conservative).
sigma_sel is tied to the target through  sigma_sel = 1/(rho * mu_target)  so the selection step uses a fraction rho^2
of the Gaussian-DP budget; sigma_run is then solved so that the TOTAL epsilon equals the target.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import logsumexp, gammaln

try:
    import dp_accounting as dpa
    from dp_accounting import pld as _pld
    HAVE_DPA = hasattr(_pld, "PLDAccountant") and hasattr(dpa, "PoissonSampledDpEvent")
except Exception:                      # pragma: no cover
    HAVE_DPA = False

GRID6 = [(C, lr) for C in (0.1, 0.25, 0.5) for lr in (0.5, 2.0)]      # same grid as run_cv.py
GRID2 = [(0.25, 0.5), (0.25, 2.0)]                                      # learning rate only, C fixed a priori
GRID1 = [(0.25, 0.5)]                                                   # no tuning: one a-priori default
GRIDS = {1: GRID1, 2: GRID2, 6: GRID6}

# ------------------------------------------------------------------ Gaussian-DP helpers
def eps_from_mu(mu, delta):
    f = lambda e: norm.cdf(-e/mu + mu/2) - np.exp(e + norm.logcdf(-e/mu - mu/2)) - delta
    if f(0.0) <= 0: return 0.0
    hi = 1.0
    while f(hi) > 0: hi *= 2
    return brentq(f, 1e-12, hi)

def mu_from_eps(eps, delta):
    """the mu for which a mu-Gaussian mechanism is exactly (eps, delta)-DP"""
    return brentq(lambda mu: eps_from_mu(mu, delta) - eps, 1e-4, 60.0, xtol=1e-10)

# ------------------------------------------------------------------ RDP fallback for the sampled Gaussian (integer orders)
_ORDERS = np.arange(2, 257)
def rdp_sampled_gaussian(q, sigma, alphas=_ORDERS):
    """Mironov-Talwar-Zhang (2019) closed form for integer alpha (equation A4 of the paper)."""
    out = []
    for a in alphas:
        if q >= 1.0: out.append(a/(2*sigma**2)); continue
        j = np.arange(0, a+1)
        logc = gammaln(a+1) - gammaln(j+1) - gammaln(a-j+1)
        terms = logc + (a-j)*np.log1p(-q) + j*np.log(q) + (j*j-j)/(2*sigma**2)
        out.append(logsumexp(terms)/(a-1))
    return np.array(out)

def eps_total_rdp(sigma_run, steps, q, m, sigma_sel, delta):
    r = m*steps*rdp_sampled_gaussian(q if q is not None else 1.0, sigma_run)
    if sigma_sel is not None: r = r + _ORDERS/(2*sigma_sel**2)
    a = _ORDERS.astype(float)       # tighter RDP -> (eps, delta) conversion (Balle et al. 2020), as used by dp_accounting
    return float(max(0.0, np.min(r + np.log1p(-1.0/a) - (np.log(delta) + np.log(a))/(a-1))))

def eps_total_pld(sigma_run, steps, q, m, sigma_sel, delta):
    g = dpa.GaussianDpEvent(sigma_run)
    if q is not None and q < 1.0: g = dpa.PoissonSampledDpEvent(q, g)
    ev = [dpa.SelfComposedDpEvent(g, m*steps)]
    if sigma_sel is not None: ev.append(dpa.GaussianDpEvent(sigma_sel))
    acc = _pld.PLDAccountant(); acc.compose(dpa.ComposedDpEvent(ev))
    return acc.get_epsilon(delta)

# ------------------------------------------------------------------ calibration
def calibrate(target_eps, steps, q, m, delta=1e-5, rho=0.5, accountant="auto"):
    """returns dict(sigma_run, sigma_sel, eps_check, accountant). m = 1 means 'no selection' (single run, no tuning)."""
    mu_t = mu_from_eps(target_eps, delta)
    if m == 1:
        sigma_sel, rho_eff = None, 0.0
    else:
        sigma_sel, rho_eff = 1.0/(rho*mu_t), rho
    if q is None or q >= 1.0:                                   # exact, scipy only
        mu_run2 = mu_t**2*(1 - rho_eff**2)
        sigma_run = float(np.sqrt(m*steps/mu_run2))
        tot = np.sqrt(m*steps/sigma_run**2 + (0 if sigma_sel is None else 1/sigma_sel**2))
        return dict(sigma_run=sigma_run, sigma_sel=sigma_sel, eps_check=eps_from_mu(tot, delta), accountant="gaussian-exact")
    use = (accountant == "pld") or (accountant == "auto" and HAVE_DPA)
    fn = eps_total_pld if use else eps_total_rdp
    f = lambda s: fn(s, steps, q, m, sigma_sel, delta) - target_eps
    lo = 0.5
    while f(lo) < 0 and lo > 0.15: lo *= 0.8
    hi = 500.0
    if f(hi) > 0: raise ValueError("target epsilon too small for this configuration")
    sigma_run = brentq(f, lo, hi, xtol=1e-3)
    return dict(sigma_run=float(sigma_run), sigma_sel=sigma_sel, eps_check=fn(sigma_run, steps, q, m, sigma_sel, delta), accountant="pld" if use else "rdp-upper-bound")

# ------------------------------------------------------------------ selection
def clipped_loss_sum(p, y, L=2.0):
    p = np.clip(p, 1e-6, 1-1e-6)
    return float(np.clip(-(y*np.log(p) + (1-y)*np.log(1-p)), 0.0, L).sum())

def noisy_select(scores, m, sigma_sel, L, rng):
    """scores: array of m clipped-loss sums. Gaussian mechanism with L2 sensitivity L*sqrt(m); returns argmin of noisy scores."""
    s = np.asarray(scores, float)
    if sigma_sel is None: return 0
    return int(np.argmin(s + rng.normal(0.0, sigma_sel*L*np.sqrt(m), size=s.shape)))

def train_and_select(Net, X, y, m, target_eps, steps, q, seed, delta=1e-5, rho=0.5, L=2.0, hidden=(16, 8), accountant="auto"):
    """Runs the whole private pipeline on one training set. Returns (selected_net, info)."""
    cal = calibrate(target_eps, steps, q, m, delta, rho, accountant)
    grid = GRIDS[m]; nets, scores = [], []
    for j, (C, lr) in enumerate(grid):
        net = Net(X.shape[1], hidden=hidden, seed=seed).fit(X, y, steps=steps, lr=lr, clip=C, sigma=cal["sigma_run"], q=q, seed=seed*100 + j)
        nets.append(net); scores.append(clipped_loss_sum(net.predict_proba(X), y, L))
    rng = np.random.default_rng(10_000 + seed)
    k = noisy_select(scores, m, cal["sigma_sel"], L, rng)
    return nets[k], dict(chosen=list(grid[k]), index=k, scores=scores, **cal)
