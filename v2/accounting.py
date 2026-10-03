"""Privacy accounting for Gaussian DP-SGD (add/remove adjacency, sensitivity C).

Full-batch, k steps  -> exact: k Gaussian releases = one Gaussian mechanism with
                        mu = sqrt(k)/sigma (Gaussian DP, Dong-Roth-Su 2022);
                        delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2).
Poisson-subsampled   -> PLD accountant (Google dp_accounting), cross-checked with RDP.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import dp_accounting as dpa
from dp_accounting import pld, rdp

def eps_fullbatch_exact(sigma, steps, delta):
    mu = np.sqrt(steps) / sigma
    # stable form: e^eps * Phi(-eps/mu - mu/2) = exp(eps + logcdf(...))
    f = lambda e: norm.cdf(-e/mu + mu/2) - np.exp(e + norm.logcdf(-e/mu - mu/2)) - delta
    if f(0.0) <= 0: return 0.0
    hi = 1.0
    while f(hi) > 0: hi *= 2
    return brentq(f, 1e-12, hi)

def _event(sigma, steps, q=None):
    g = dpa.GaussianDpEvent(sigma)
    if q is not None and q < 1.0:
        g = dpa.PoissonSampledDpEvent(q, g)
    return dpa.SelfComposedDpEvent(g, steps)

def eps_pld(sigma, steps, delta, q=None):
    a = pld.PLDAccountant(); a.compose(_event(sigma, steps, q))
    return a.get_epsilon(delta)

def eps_rdp(sigma, steps, delta, q=None):
    a = rdp.RdpAccountant(); a.compose(_event(sigma, steps, q))
    return a.get_epsilon(delta)

def eps_advanced_paper(sigma, steps, delta):
    """The accountant used in the original manuscript (kept only for comparison).
    Returns (epsilon, delta_actually_achieved = k*delta0 + delta')."""
    d0 = dp = delta / 2.0
    e0 = np.sqrt(2*np.log(1.25/d0)) / sigma
    eps = np.sqrt(2*steps*np.log(1/dp))*e0 + steps*e0*(np.exp(e0)-1)
    return eps, steps*d0 + dp

def solve_sigma(target_eps, steps, delta, q=None, hi=500.0):
    """Smallest sigma whose PLD epsilon is <= target_eps (bisection)."""
    if q is None or q >= 1.0:      # exact closed form (verified equal to PLD); no discretisation, no memory issue
        return brentq(lambda s: eps_fullbatch_exact(s, steps, delta) - target_eps, 0.02, 1e5, xtol=1e-4)
    fn = (lambda s: eps_pld(s, steps, delta, q))
    lo = 0.7
    while fn(lo) < target_eps and lo > 0.2: lo *= 0.8        # small sampling rates reach large eps only at small sigma
    if fn(hi) > target_eps: raise ValueError("target too small")
    return brentq(lambda s: fn(s) - target_eps, lo, hi, xtol=1e-3)
