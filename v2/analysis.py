"""Tables, statistics and figures built ONLY from the saved result files in results/. Used by the notebooks and the paper builder."""
import os, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
DS = ["loan", "german", "taiwan"]; NAMES = {"loan": "Loan approval (n=614)", "german": "German Credit (n=1000)", "taiwan": "Taiwan card default (n=30000)"}
EPS = [0.5, 1, 2, 4, 8]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
load = lambda f: json.load(open(os.path.join(RES, f)))
def cv(ds): return load(f"cv_{ds}.json")
def vec(R, m, k): return np.array([r["models"][m][k] for r in R if m in r["models"] and k in r["models"][m]], float)

# ---- Nadeau-Bengio corrected resampled statistics (Nadeau & Bengio 2003): variance inflated by (1/J + n_test/n_train)
def rho_of(R): return float(np.mean([r["n_test"]/r["n_train"] for r in R]))
def nb_ci(x, rho):
    J = len(x); se = np.sqrt((1/J + rho)*np.var(x, ddof=1)); return float(np.mean(x)), float(stats.t.ppf(0.975, J-1)*se)
def nb_test(a, b, rho):
    d = np.asarray(a) - np.asarray(b); J = len(d); se = np.sqrt((1/J + rho)*np.var(d, ddof=1))
    if se == 0: return float(d.mean()), 1.0, (0.0, 0.0)
    t = d.mean()/se; p = 2*stats.t.sf(abs(t), J-1); h = stats.t.ppf(0.975, J-1)*se; return float(d.mean()), float(p), (float(d.mean()-h), float(d.mean()+h))
def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m-i)*ps[j]); adj[j] = min(1.0, run)
    return adj
def fmt(m, c, d=3): return f"{m:.{d}f} ± {c:.{d}f}"

def model_row(R, m, keys=("acc", "auc", "bacc", "mcc", "rec_adv")):
    rho = rho_of(R); out = {}
    for k in keys:
        v = vec(R, m, k)
        if len(v) == 0 or np.isnan(v).all(): out[k] = (np.nan, np.nan); continue
        out[k] = nb_ci(v, rho)
    return out

def nonprivate_table(ds):
    R = cv(ds); rows = []
    order = [("majority", "Majority class"), ("single_feature_rule", "Single-feature rule"), ("logreg_L2", "Logistic regression (L2)"), ("random_forest", "Random forest"),
             ("catboost", "CatBoost"), ("xgboost", "XGBoost"), ("mlp_nonprivate", "MLP (16-8), tuned")]
    for m, lab in order:
        if m not in R[0]["models"]: continue
        r = model_row(R, m); rows.append({"Model": lab, "Accuracy": fmt(*r["acc"]), "AUC": fmt(*r["auc"]), "Bal. acc.": fmt(*r["bacc"], d=3), "MCC": fmt(*r["mcc"], d=3), "Recall (adverse)": fmt(*r["rec_adv"], d=3)})
    return pd.DataFrame(rows)

FAMILIES = [("dp_mlp_PO", "DP-SGD MLP (Poisson)"), ("dp_mlp_FB100", "DP-SGD MLP (full batch, 100 steps)"), ("dp_lr_PO", "DP-SGD logistic regression"), ("dp_ebm", "DP-EBM"), ("local_perturb_lr", "Local feature perturbation + LR")]
def dp_table(ds, key="auc"):
    R = cv(ds); rows = []
    for fam, lab in FAMILIES:
        if f"{fam}_eps1" not in R[0]["models"] and f"{fam}_eps0.5" not in R[0]["models"]: continue
        row = {"Method": lab}
        for e in EPS:
            m = f"{fam}_eps{e}"; v = vec(R, m, key)
            row[f"ε={e}"] = fmt(*nb_ci(v, rho_of(R))) if len(v) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)

def dp_means(ds, fam, key="auc"):
    R = cv(ds); rho = rho_of(R); mu, ci = [], []
    for e in EPS:
        v = vec(R, f"{fam}_eps{e}", key); a, c = nb_ci(v, rho) if len(v) > 1 else (np.nan, np.nan); mu.append(a); ci.append(c)
    return np.array(mu), np.array(ci)

def fig_privacy_utility(fname="fig_privacy_utility.png", key="auc"):
    fig, ax = plt.subplots(1, len(DS), figsize=(10.5, 3.2)); sty = {"dp_mlp_PO": ("o-", "#1f4e79"), "dp_mlp_FB100": ("s--", "#7f9fbf"), "dp_lr_PO": ("^-", "#c55a11"), "dp_ebm": ("d-", "#548235"), "local_perturb_lr": ("x:", "#7f7f7f")}
    for a, ds in zip(ax, DS):
        R = cv(ds); rho = rho_of(R)
        for fam, lab in FAMILIES:
            if f"{fam}_eps1" not in R[0]["models"]: continue
            mu, ci = dp_means(ds, fam, key)
            if fam == "local_perturb_lr": a.plot(EPS, mu, sty[fam][0], color=sty[fam][1], ms=4, lw=1.2, label=lab)
            else: a.errorbar(EPS, mu, yerr=ci, fmt=sty[fam][0], color=sty[fam][1], ms=4, lw=1.2, capsize=2, label=lab)
        for m, lab, ls in (("mlp_nonprivate", "non-private MLP", "-"), ("logreg_L2", "non-private LR", "--")):
            mu = nb_ci(vec(R, m, key), rho)[0]; a.axhline(mu, color="k", lw=0.8, ls=ls, alpha=0.7, label=lab)
        a.set_xscale("log", base=2); a.set_xticks(EPS); a.set_xticklabels(EPS); a.set_xlabel("ε (δ = 1e-5)"); a.set_title(NAMES[ds], fontsize=9)
    ax[0].set_ylabel("test AUC" if key == "auc" else "test accuracy"); h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.17)); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

def fig_accountants(fname="fig_accountants.png"):
    M = load("misc.json")["accountants_fullbatch"]; s = [r["sigma"] for r in M]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(s, [r["manuscript_adv"] for r in M], "o-", c="#c00000", label="advanced composition as in the first draft (true δ = 0.032)")
    ax.plot(s, [r["adv_corrected_delta1e5"] for r in M], "s--", c="#e08080", label="advanced composition, δ split corrected (δ = 1e-5)")
    ax.plot(s, [r["rdp_1e5"] for r in M], "^-", c="#7f9fbf", label="Rényi DP (δ = 1e-5)"); ax.plot(s, [r["exact_1e5"] for r in M], "d-", c="#1f4e79", label="exact Gaussian composition (δ = 1e-5)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("noise multiplier σ (30 full-batch steps)"); ax.set_ylabel("ε"); ax.legend(fontsize=6.3, frameon=False); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

def fig_loss(fname="fig_loss.png"):
    C = load("misc.json")["loss_curves"]; fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for (k, v), c in zip(C.items(), ["k", "#1f4e79", "#548235", "#c55a11"]): ax.plot(np.arange(len(v))*5, v, label=k, c=c, lw=1.2)
    ax.set_xlabel("training step"); ax.set_ylabel("training loss (BCE)"); ax.legend(fontsize=6.3, frameon=False); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

# =============================================================== explanation stability
def expl(ds): return load(f"expl_{ds}.json")
def _per_split(E, kind, eps, getter):
    out = []
    for sp in E:
        vals = [getter(r) for r in sp["rows"] if r["kind"] == kind and (eps is None or r["eps"] == eps)]
        if vals: out.append(np.mean(vals))
    return np.array(out)
EXPL_METRICS = {"IG rank corr. (features)": lambda r: r["ig_zero"]["af"]["spearman"], "IG rank corr. (units)": lambda r: r["ig_zero"]["au"]["spearman"],
                "IG top-3 overlap (units)": lambda r: r["ig_zero"]["au"]["top3"], "IG local cosine": lambda r: r["ig_zero"]["local_cos"],
                "PI rank corr. (features)": lambda r: r["pi_f"]["spearman"], "PI rank corr. (units)": lambda r: r["pi_u"]["spearman"], "PI top-3 overlap (units)": lambda r: r["pi_u"]["top3"]}
def expl_table(ds, metric):
    E = expl(ds); g = EXPL_METRICS[metric]; rho = 0.25; rows = []
    fl = _per_split(E, "nonprivate_reseed", None, g); rows.append({"Model": "Non-private, re-seeded (noise floor)", "mean": fl.mean(), "ci": nb_ci(fl, rho)[1]})
    for e in EPS:
        v = _per_split(E, "dp", e, g); rows.append({"Model": f"DP-SGD ε={e}", "mean": v.mean(), "ci": nb_ci(v, rho)[1]})
    return pd.DataFrame(rows)
def expl_summary(ds):
    E = expl(ds); out = {}
    for m, g in EXPL_METRICS.items():
        t = expl_table(ds, m); out[m] = {r["Model"]: fmt(r["mean"], r["ci"], 2) for _, r in t.iterrows()}
    return pd.DataFrame(out).T
def expl_extra(ds):
    E = expl(ds); r = {}
    resid = [x["ig_zero"]["resid_med"] for sp in E for x in sp["rows"]]; rmax = [x["ig_zero"]["resid_max"] for sp in E for x in sp["rows"]]
    r["IG completeness residual (median, max over rows) "] = (float(np.median(resid)), float(np.max(rmax)))
    for b in ("zero", "median", "modal"):
        r[f"IG baseline {b}: top-3 overlap with reference, DP eps=1"] = float(_per_split(E, "dp", 1, lambda x, b=b: x[f"ig_{b}"]["au"]["top3"]).mean())
    r["faithfulness ratio (explanation order / random order), non-private"] = float(np.mean([x["ig_zero"]["faith"]/max(x["ig_zero"]["faith_random"], 1e-9) for sp in E for x in sp["rows"] if x["kind"] == "reference"]))
    r["faithfulness ratio, DP eps=1"] = float(np.mean([x["ig_zero"]["faith"]/max(x["ig_zero"]["faith_random"], 1e-9) for sp in E for x in sp["rows"] if x["kind"] == "dp" and x["eps"] == 1]))
    r["IG vs PI agreement (units, Spearman), non-private"] = float(np.mean([x["ig_vs_pi_u"]["spearman"] for sp in E for x in sp["rows"] if x["kind"] == "reference"]))
    return r
def fig_expl(fname="fig_expl_stability.png"):
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for a, (metric, ttl) in zip(ax, (("IG rank corr. (units)", "Integrated Gradients"), ("PI rank corr. (units)", "Permutation importance"))):
        for ds, c in zip(DS, ["#1f4e79", "#c55a11", "#548235"]):
            if not os.path.exists(os.path.join(RES, f"expl_{ds}.json")): continue
            t = expl_table(ds, metric); mu = t["mean"].values[1:]; ci = t["ci"].values[1:]
            a.errorbar(EPS, mu, yerr=ci, fmt="o-", color=c, ms=3.5, lw=1.2, capsize=2, label=ds); a.axhline(t["mean"].values[0], color=c, ls=":", lw=1)
        a.set_xscale("log", base=2); a.set_xticks(EPS); a.set_xticklabels(EPS); a.set_xlabel("ε"); a.set_title(ttl, fontsize=9)
    ax[0].set_ylabel("Spearman ρ with non-private ranking"); ax[0].legend(fontsize=7, frameon=False, title="dotted: noise floor", title_fontsize=7); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)
def fig_expl_heat(ds, fname=None, eps_list=(1, 4)):
    E = expl(ds); un = E[0]["unit_names"]; fname = fname or f"fig_expl_heat_{ds}.png"
    def avg(kind, eps, key):
        V = [np.mean([r[key] for r in sp["rows"] if r["kind"] == kind and (eps is None or r["eps"] == eps)], 0) for sp in E]; V = np.array(V); V = V/np.maximum(V.sum(1, keepdims=True), 1e-12); return V.mean(0)
    cols = [("non-private", avg("reference", None, "vec_ig_u"))] + [(f"DP ε={e}", avg("dp", e, "vec_ig_u")) for e in eps_list]
    M = np.stack([c[1] for c in cols], 1); order = np.argsort(-M[:, 0]); fig, ax = plt.subplots(figsize=(3.6, 0.32*len(un)+0.9)); im = ax.imshow(M[order], cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=7); ax.set_yticks(range(len(un))); ax.set_yticklabels([un[i] for i in order], fontsize=7)
    for i in range(len(un)):
        for j in range(len(cols)): ax.text(j, i, f"{M[order][i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"IG share per unit - {ds}", fontsize=8); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

# =============================================================== membership inference
def mia_table(ds):
    M = load(f"mia_{ds}.json"); rows = []
    claimed = {"nonprivate": "-", "overfit_control": "-", "dp_eps1": 1, "dp_eps4": 4, "dp_eps8": 8}
    for m, r in M.items():
        rows.append({"Model": m, "ε claimed": claimed.get(m, "-"), "Loss-threshold AUC": f"{r['loss']['auc']:.3f}", "LiRA AUC (95% CI)": f"{r['lira']['auc']:.3f} [{r['lira']['auc_ci'][0]:.3f}, {r['lira']['auc_ci'][1]:.3f}]",
                     "LiRA TPR@1%FPR": f"{r['lira']['tpr_fpr1']:.3f}", "audit ε lower bound (LiRA)": f"{r['lira']['eps_lb']:.2f}"})
    return pd.DataFrame(rows)
def fig_mia(fname="fig_mia.png"):
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0)); order = ["nonprivate", "overfit_control", "dp_eps8", "dp_eps4", "dp_eps1"]; lab = ["non-private", "overfit\ncontrol", "DP ε=8", "DP ε=4", "DP ε=1"]; w = 0.26
    for k, (ds, c) in enumerate(zip(DS, ["#1f4e79", "#c55a11", "#548235"])):
        p = os.path.join(RES, f"mia_{ds}.json")
        if not os.path.exists(p): continue
        M = json.load(open(p)); a = [M[o]["lira"]["auc"] if o in M else np.nan for o in order]; t = [M[o]["lira"]["tpr_fpr1"] if o in M else np.nan for o in order]
        ax[0].bar(np.arange(5)+(k-1)*w, a, w, color=c, label=ds); ax[1].bar(np.arange(5)+(k-1)*w, t, w, color=c)
    ax[0].axhline(0.5, color="k", lw=0.7, ls="--"); ax[0].set_ylabel("LiRA attack AUC"); ax[0].set_ylim(0.45, None); ax[1].axhline(0.01, color="k", lw=0.7, ls="--"); ax[1].set_ylabel("LiRA TPR at 1% FPR")
    for a_ in ax: a_.set_xticks(range(5)); a_.set_xticklabels(lab, fontsize=7)
    ax[0].legend(fontsize=7, frameon=False); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

# =============================================================== fairness
def fair_gaps(ds, nboot=300, seed=0):
    F = load(f"fair_{ds}.json"); rng = np.random.default_rng(seed); models = list(F[0]["models"].keys()); rows = []
    attrs = sorted({k.split("=")[0] for k in F[0]["models"][models[0]]})
    def gaps(idx, mname, attr):
        agg = {}
        for i in idx:
            for k, c in F[i]["models"][mname].items():
                if k.split("=")[0] == attr: agg[k] = agg.get(k, np.zeros(4)) + np.array(c)
        st = {k: v for k, v in agg.items() if v.sum() >= 30}
        if len(st) < 2: return np.nan, np.nan, np.nan
        sel = [(v[0]+v[1])/v.sum() for v in st.values()]; tpr = [v[0]/max(v[0]+v[2], 1) for v in st.values()]; fpr = [v[1]/max(v[1]+v[3], 1) for v in st.values()]
        return max(sel)-min(sel), max(tpr)-min(tpr), max(fpr)-min(fpr)
    for mname in models:
        for attr in attrs:
            full = gaps(range(len(F)), mname, attr); bs = np.array([gaps(rng.integers(0, len(F), len(F)), mname, attr) for _ in range(nboot)])
            ci = np.nanpercentile(bs, [2.5, 97.5], axis=0)
            tot = np.zeros(4)
            for i in range(len(F)):
                for k_, c_ in F[i]["models"][mname].items():
                    if k_.split("=")[0] == attr: tot += np.array(c_)
            rows.append({"Model": mname, "Attribute": attr, "Flagged overall": f"{(tot[0]+tot[1])/tot.sum():.3f}", "Selection-rate gap": f"{full[0]:.3f} [{ci[0][0]:.3f}, {ci[1][0]:.3f}]", "TPR gap": f"{full[1]:.3f} [{ci[0][1]:.3f}, {ci[1][1]:.3f}]", "FPR gap": f"{full[2]:.3f} [{ci[0][2]:.3f}, {ci[1][2]:.3f}]"})
    return pd.DataFrame(rows)

# =============================================================== learning curve
def lc_table():
    L = load("lc_taiwan.json"); rows = []
    for n in sorted({r["n"] for r in L}):
        rr = [r for r in L if r["n"] == n]; row = {"n": n, "reps": len(rr)}
        for m in rr[0]["models"]:
            v = np.array([r["models"][m]["auc"] for r in rr]); row[m] = f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else "")
        rows.append(row)
    return pd.DataFrame(rows)
def fig_lc(fname="fig_learning_curve.png"):
    L = load("lc_taiwan.json"); ns = sorted({r["n"] for r in L}); fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for m, lab, st, c in (("logreg_nonprivate", "non-private LR", "k--", None), ("mlp_nonprivate", "non-private MLP", "k-", None), ("dp_mlp_eps1", "DP-MLP ε=1", "o-", "#1f4e79"), ("dp_mlp_eps4", "DP-MLP ε=4", "s-", "#7f9fbf"),
                          ("dp_lr_eps1", "DP-LR ε=1", "^-", "#c55a11"), ("dp_ebm_eps1", "DP-EBM ε=1", "d-", "#548235")):
        mu = [np.mean([r["models"][m]["auc"] for r in L if r["n"] == n]) for n in ns]; ax.plot(ns, mu, st, color=c, ms=4, lw=1.2, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("training-set size"); ax.set_ylabel("test AUC (Taiwan, 6,000 held-out)"); ax.legend(fontsize=6.5, frameon=False); fig.savefig(os.path.join(FIG, fname)); plt.close(fig)

# =============================================================== misc
def credit_history_table():
    M = load("misc.json")["credit_history"]; rows = []
    for mode, lab in (("mode", "Mode imputation"), ("indicator", "Mode + missing flag"), ("category", "Missing as its own state")):
        row = {"Credit_History handling": lab}
        for mdl in ("logreg", "mlp"):
            for k in ("acc", "auc"): row[f"{mdl} {k}"] = fmt(*nb_ci(np.array([r[k] for r in M[mode][mdl]]), 0.25))
        rows.append(row)
    return pd.DataFrame(rows)
def mlptune_table(ds):
    T = load(f"mlptune_{ds}.json"); R = cv(ds)[:len(T)]; rho = rho_of(R); rows = []
    for lab, v in (("MLP, small grid (lr, steps)", vec(R, "mlp_nonprivate", "auc")), ("MLP, wide grid (width, depth, lr, steps, weight decay)", np.array([t["metrics"]["auc"] for t in T])), ("Logistic regression", vec(R, "logreg_L2", "auc")), ("CatBoost", vec(R, "catboost", "auc"))):
        rows.append({"Model": lab, "AUC": fmt(*nb_ci(v, rho))})
    return pd.DataFrame(rows)


# =============================================================== paired comparisons with Holm correction
def paired_table(ds):
    R = cv(ds); rho = rho_of(R); has = lambda m: m in R[0]["models"]
    comps = []
    if has("logreg_L2"): comps.append(("Non-private MLP vs logistic regression", "mlp_nonprivate", "logreg_L2"))
    if has("single_feature_rule"): comps.append(("Non-private MLP vs single-feature rule", "mlp_nonprivate", "single_feature_rule"))
    for e in (1, 4, 8): comps.append((f"DP-SGD MLP (ε={e}) vs non-private MLP", f"dp_mlp_PO_eps{e}", "mlp_nonprivate"))
    for e in (1, 4): comps += [(f"DP-SGD LR (ε={e}) vs DP-SGD MLP (ε={e})", f"dp_lr_PO_eps{e}", f"dp_mlp_PO_eps{e}"), (f"DP-EBM (ε={e}) vs DP-SGD LR (ε={e})", f"dp_ebm_eps{e}", f"dp_lr_PO_eps{e}")]
    if has("dp_mlp_FB100_eps1"): comps.append(("Full-batch vs Poisson DP-SGD MLP (ε=1)", "dp_mlp_FB100_eps1", "dp_mlp_PO_eps1"))
    rows = []; pa, pc = [], []
    for lab, a, b in comps:
        da, p1, ci1 = nb_test(vec(R, a, "auc"), vec(R, b, "auc"), rho); dc, p2, ci2 = nb_test(vec(R, a, "acc"), vec(R, b, "acc"), rho)
        rows.append([lab, da, ci1, p1, dc, ci2, p2]); pa.append(p1); pc.append(p2)
    ha, hc = holm(pa), holm(pc); out = []
    for r, x, y in zip(rows, ha, hc): out.append({"Comparison": r[0], "ΔAUC [95% CI]": f"{r[1]:+.3f} [{r[2][0]:+.3f}, {r[2][1]:+.3f}]", "p (Holm)": ("<0.001" if x < 0.0005 else f"{x:.3f}"), "ΔAccuracy [95% CI]": f"{r[4]:+.3f} [{r[5][0]:+.3f}, {r[5][1]:+.3f}]", "p (Holm) ": ("<0.001" if y < 0.0005 else f"{y:.3f}")})
    return pd.DataFrame(out)

def chosen_hparams(ds):
    from collections import Counter
    R = cv(ds); rows = []
    for key in sorted({k for r in R for k in r["chosen"] if k.startswith("mlp_PO") or k.startswith("lr_PO")}):
        c = Counter(tuple(r["chosen"][key]) for r in R if key in r["chosen"]).most_common(2); rows.append({"setting": key, "most frequent (C, lr)": f"{c[0][0]} in {c[0][1]}/{len(R)}"})
    return pd.DataFrame(rows)
