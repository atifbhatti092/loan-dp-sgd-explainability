"""Exports every table and number used in the paper from the saved result files to paper_data.json."""
import os, sys, json, platform, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
from pipeline import load as load_ds
import sklearn, scipy, catboost, xgboost, interpret, dp_accounting, importlib.metadata as im
have = lambda ds, f: os.path.exists(os.path.join(A.RES, f.format(ds=ds)))
rows = lambda df: [list(df.columns)] + df.astype(str).values.tolist()
out = {}
out["versions"] = f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, scikit-learn {sklearn.__version__}, CatBoost {catboost.__version__}, XGBoost {xgboost.__version__}, interpret {im.version('interpret')}, dp_accounting {im.version('dp-accounting')}"
# ---------------- datasets table
loan = load_ds("loan"); miss = loan.df.isna().mean()
out["ch_missing_pct"] = f"{100*miss['Credit_History']:.1f}%"
dsr = [["Dataset", "Rows", "Raw fields", "Model inputs", "Adverse class", "Source and licence"]]
info = {"loan": ("Loan approval", "11", "Loan rejected: 31.3%", "Analytics Vidhya loan-prediction file (Kaggle, GitHub mirrors); licence not stated"),
        "german": ("German Credit", "20", "Bad risk: 30.0%", "Statlog German Credit, UCI [10]; CC BY 4.0"),
        "taiwan": ("Taiwan card default", "23", "Default: 22.1%", "Yeh and Lien [9]; UCI; CC BY 4.0")}
ninp = {}
for ds in A.DS:
    d = load_ds(ds); p = d.make_prep().fit(d.df); ninp[ds] = len(p.cols); n = len(d.df)
    dsr.append([info[ds][0], f"{n:,}", info[ds][1], str(ninp[ds]), info[ds][2], info[ds][3]])
out["t_datasets"] = dsr
def npar(n_in): return n_in*16+16+16*8+8+8+1
out["nparams"] = ", ".join(f"{npar(ninp[d]):,}" for d in A.DS[:2]) + f" and {npar(ninp['taiwan']):,}"
out["t_hyper"] = [["Setting", "Loan approval", "German Credit", "Taiwan default"],
    ["Architecture", "17-16-8-1 ReLU", "61-16-8-1 ReLU", "29-16-8-1 ReLU"],
    ["Outer CV", "5 × 5 stratified", "5 × 3 stratified", "5 × 2 stratified"],
    ["Poisson batch B (rate q = B/n)", "64", "64", "256"],
    ["DP steps T", "300 (and 100 full batch)", "300", "800"],
    ["DP grid: clip C", "0.1, 0.25, 0.5", "0.1, 0.25, 0.5", "0.1, 0.25, 0.5"],
    ["DP grid: learning rate", "0.5, 2.0", "0.5, 2.0", "0.5, 2.0"],
    ["Non-private grid", "lr {0.1, 0.5, 1}, steps {30, 100, 300}", "lr {0.1, 0.5, 1}, steps {30, 100, 300}", "lr {0.1, 0.5}, steps {500, 1500}, B = 256"],
    ["Inner selection", "3-fold (log-loss; AUC for DP)", "3-fold (log-loss; AUC for DP)", "3-fold (log-loss; AUC for DP)"],
    ["Target ε; δ", "0.5, 1, 2, 4, 8; 10^{-5}", "0.5, 1, 2, 4, 8; 10^{-5}", "0.5, 1, 2, 4, 8; 10^{-5}"]]
# ---------------- accountants
M = A.load("misc.json"); acc = M["accountants_fullbatch"]
out["t_acc"] = [["σ", "Draft formula ε (true δ)", "Advanced comp., δ split fixed (δ=1e-5)", "RDP (δ=1e-5)", "Exact (δ=1e-5)", "Exact (δ=1/491)"]] + [
    [f"{r['sigma']}", f"{r['manuscript_adv']:.3f} (δ={r['manuscript_delta_actual']:.3f})", f"{r['adv_corrected_delta1e5']:.2f}", f"{r['rdp_1e5']:.3f}", f"{r['exact_1e5']:.3f}", f"{r['exact_1_491']:.3f}"] for r in acc]
out["acc_raw"] = acc; out["acc_pois"] = M["accountants_poisson"]
ratios = [r["rdp_1e5"]/r["exact_1e5"]-1 for r in acc] + [r["rdp"]/r["pld"]-1 for r in M["accountants_poisson"]]
out["rdp_note"] = f"between {100*min(ratios):.0f}% and {100*max(ratios):.0f}%"
out["delta_tot"] = f"{acc[-1]['manuscript_delta_actual']:.4f}"; out["delta_ratio"] = f"{acc[-1]['manuscript_delta_actual']*491:.1f}"
out["overstate_range"] = [min(r["manuscript_adv"]/r["exact_1_491"] for r in acc), max(r["manuscript_adv"]/r["exact_1_491"] for r in acc)]
out["t_mc"] = [["σ", "k", "ε", "δ, Monte Carlo", "δ, closed form"]] + [[str(r["sigma"]), str(r["k"]), str(r["eps"]), f"{r['delta_monte_carlo']:.5f}", f"{r['delta_closed_form']:.5f}"] for r in M["composition_monte_carlo"]]
out["mc_maxrel"] = max(abs(r["delta_monte_carlo"]/r["delta_closed_form"]-1) for r in M["composition_monte_carlo"] if r["delta_closed_form"] > 1e-3)
# ---------------- main tables per dataset
out["np"], out["dp_auc"], out["dp_acc"], out["paired"], out["raw"], out["hp"] = {}, {}, {}, {}, {}, {}
for ds in A.DS:
    if not have(ds, "cv_{ds}.json"): continue
    R = A.cv(ds); rho = A.rho_of(R); out["np"][ds] = rows(A.nonprivate_table(ds)); out["dp_auc"][ds] = rows(A.dp_table(ds, "auc")); out["dp_acc"][ds] = rows(A.dp_table(ds, "acc"))
    out["paired"][ds] = rows(A.paired_table(ds)); out["hp"][ds] = rows(A.chosen_hparams(ds)); out["raw"][ds] = {"n_splits": len(R), "rho": rho}
    for m in R[0]["models"]:
        out["raw"][ds][m] = {k: list(A.nb_ci(A.vec(R, m, k), rho)) for k in ("acc", "auc", "bacc", "mcc", "rec_adv") if len(A.vec(R, m, k)) > 1}
out["credit_history"] = rows(A.credit_history_table())
out["mlptune"] = {ds: rows(A.mlptune_table(ds)) for ds in ("loan", "german") if have(ds, "mlptune_{ds}.json")}
out["lc"] = rows(A.lc_table()) if have("taiwan", "lc_taiwan.json") else None
# ---------------- explanations
out["expl"], out["expl_extra"], out["expl_raw"] = {}, {}, {}
for ds in A.DS:
    if not have(ds, "expl_{ds}.json"): continue
    out["expl"][ds] = rows(A.expl_summary(ds).reset_index().rename(columns={"index": "Metric"})); out["expl_extra"][ds] = {k: v for k, v in A.expl_extra(ds).items()}
    out["expl_raw"][ds] = {m: {r["Model"]: [r["mean"], r["ci"]] for _, r in A.expl_table(ds, m).iterrows()} for m in A.EXPL_METRICS}
# ---------------- MIA, fairness
out["mia"], out["mia_raw"], out["fair"] = {}, {}, {}
for ds in A.DS:
    if have(ds, "mia_{ds}.json"): out["mia"][ds] = rows(A.mia_table(ds)); out["mia_raw"][ds] = A.load(f"mia_{ds}.json")
    if have(ds, "fair_{ds}.json"): out["fair"][ds] = rows(A.fair_gaps(ds))
json.dump(out, open(os.path.join(A.ROOT, "paper_data.json"), "w"), indent=1, default=float); print("paper_data.json written; datasets:", list(out["np"].keys()))

# ---------------- extra appendix tables: loan fields (comment 4) and seeds/versions (comment 38)
miss = load_ds("loan").df.isna().mean()
fields = [("Gender", "nominal (2)", "binary, male = 1", "Gender"), ("Married", "nominal (2)", "binary", "Married"), ("Dependents", "ordinal (0, 1, 2, 3+)", "integer 0 to 3", "Dependents"),
          ("Education", "nominal (2)", "binary, graduate = 1", "Education"), ("Self_Employed", "nominal (2)", "binary", "Self_Employed"),
          ("ApplicantIncome", "numeric", "signed log, also summed into TotalIncome", "ApplicantIncome"), ("CoapplicantIncome", "numeric", "signed log, also summed into TotalIncome", "CoapplicantIncome"),
          ("LoanAmount", "numeric (thousands)", "signed log; basis of EMI and ratio inputs", "LoanAmount"), ("Loan_Amount_Term", "numeric (months)", "standardised", "Loan_Amount_Term"),
          ("Credit_History", "binary", "0/1 (three treatments for missing values compared)", "Credit_History"), ("Property_Area", "nominal (3)", "one-hot, three columns", "Property_Area")]
rows_ = [["Field", "Type", "Encoding", "Missing", "Status"]] + [[a, b, c, f"{100*miss[k]:.1f}%", "kept"] for a, b, c, k in fields] + [["Loan_ID", "identifier", "none", "0.0%", "dropped"],
         ["Derived inputs", "numeric", "TotalIncome, EMI, income-to-loan ratio, balance after EMI, all in signed log form", "n/a", "added"]]
out["t_loan_cols"] = rows_
out["t_repro"] = [["Item", "Value"], ["Outer CV split seed", "RepeatedStratifiedKFold, random_state = 0 (same folds for every model of a dataset)"], ["Network initialisation and Poisson sampling", "seed = split index (DP training: split index × 100 + run index)"],
    ["Inner 3-fold selection", "StratifiedKFold with random_state = split index"], ["Membership audit", "96 shadow models; subset masks from default_rng(1); model seeds 0 to 95"],
    ["Explanations", "128 IG steps; 3 DP seeds per ε and split; permutation importance with 5 shuffles"], ["Software", out["versions"]], ["Hardware", "one CPU core, no GPU"]]
json.dump(out, open(os.path.join(A.ROOT, "paper_data.json"), "w"), indent=1, default=float); print("extra tables added")
