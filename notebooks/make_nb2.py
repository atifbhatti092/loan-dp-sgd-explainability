import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# Notebook 2 - The corrected study: every table and figure of the paper

All numbers shown here are read from the JSON files in `v2/results/` that the experiment scripts wrote, or recomputed live where that is cheap (privacy accounting, unit tests, a full re-run of one cross-validation split). Nothing is typed in by hand.

**Contents**

1. Provenance and file integrity
2. Unit tests of the DP-SGD code
3. Privacy accounting, with a Monte-Carlo check of the theorem
4. Non-private baselines on three datasets
5. Utility under DP-SGD, private baselines, paired tests
6. Sensitivity: Credit_History handling, MLP tuning, training loss
7. Learning curve on the 30,000-row data
8. Explanation stability
9. Membership-inference audit
10. Group-level diagnostics
11. One split re-run from scratch and compared with the saved result""")
code("""import os, sys, json, subprocess, hashlib, platform, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 70)
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
V2 = os.path.join(ROOT, 'v2'); sys.path.insert(0, V2); os.chdir(V2)
import analysis as A
from IPython.display import Image, display, Markdown
fig = lambda name, w=620: display(Image(os.path.join(V2, 'figures', name), width=w))
print('python', platform.python_version())""")
md("## 1. Provenance and file integrity\n\nThe manifest was written by `make_manifest.py` after the last experiment finished. Here every file is hashed again and compared with it.")
code("""M = json.load(open(os.path.join(V2, 'MANIFEST.json')))
print('manifest created (UTC):', M['created_utc']); print('git commit            :', M['git_commit'][:12]); print('library versions      :', M['versions'])
bad = []
for rel, h in M['sha256'].items():
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p) or hashlib.sha256(open(p, 'rb').read()).hexdigest() != h: bad.append(rel)
print(f'{len(M["sha256"])} files checked, {len(bad)} differ from the manifest', bad[:5])
assert not bad, 'files changed after the manifest was written'""")
code("""from pipeline import load
rows = []
for d in A.DS:
    ds = load(d); p = ds.make_prep().fit(ds.df); rows.append({'dataset': d, 'rows': len(ds.df), 'raw fields': ds.df.shape[1], 'model inputs': len(p.cols), 'adverse-class rate': round(float(ds.y.mean()), 4), 'missing cells': int(ds.df.isna().sum().sum())})
pd.DataFrame(rows)""")
md("## 2. Unit tests of the DP-SGD implementation\n\nPer-example gradients against the original loop code, a numerical gradient check, the clipping bound, the injected-noise standard deviation, the Poisson sampling rate and a hand-computed one-step update.")
code("""r = subprocess.run([sys.executable, 'tests.py'], capture_output=True, text=True, cwd=V2); print(r.stdout[-1400:]); assert r.returncode == 0 and 'ALL TESTS PASSED' in r.stdout""")
md("""## 3. Privacy accounting

For 30 full-batch steps the privacy loss is exactly Gaussian (Theorem 1 of the paper, proof in Appendix A). The table compares the advanced-composition formula with the exact accountant, our closed form against Google's PLD accountant, and the Rényi accountant. The live computation below does not read any saved file.""")
code("""from accounting import eps_fullbatch_exact, eps_pld, eps_rdp, eps_advanced_paper
rows = []
for s in (200, 100, 60, 40, 25, 15):
    e_adv, d_act = eps_advanced_paper(s, 30, 1/491)
    rows.append({'sigma': s, 'advanced comp. eps (nominal delta 1/491)': round(e_adv, 3), 'delta actually achieved': round(d_act, 4), 'exact eps, delta=1/491': round(eps_fullbatch_exact(s, 30, 1/491), 3),
                 'exact eps, delta=1e-5': round(eps_fullbatch_exact(s, 30, 1e-5), 3), 'PLD eps, delta=1e-5': round(eps_pld(s, 30, 1e-5), 3), 'RDP eps, delta=1e-5': round(eps_rdp(s, 30, 1e-5), 3)})
T = pd.DataFrame(rows); display(T)
assert np.allclose(T['exact eps, delta=1e-5'], T['PLD eps, delta=1e-5'], atol=2e-3); print('closed form == PLD to 3 decimals')
print('RDP exceeds the exact value by %.0f%% to %.0f%%' % (100*(T['RDP eps, delta=1e-5']/T['exact eps, delta=1e-5']-1).min(), 100*(T['RDP eps, delta=1e-5']/T['exact eps, delta=1e-5']-1).max()))""")
code("""# Monte-Carlo check of equation (1), done from scratch here (150,000 samples per row)
from scipy.stats import norm
rng = np.random.default_rng(1); rows = []
for s, k in ((15, 30), (8, 30), (3, 30)):
    x = rng.normal(0, s, size=(150000, k)); L = (k - 2*x.sum(1))/(2*s*s)
    for eps in (0.25, 0.5, 1.0):
        mu = np.sqrt(k)/s; cf = norm.cdf(-eps/mu + mu/2) - np.exp(eps)*norm.cdf(-eps/mu - mu/2)
        rows.append({'sigma': s, 'k': k, 'eps': eps, 'delta Monte Carlo': round(float(np.mean(np.maximum(0, 1-np.exp(eps-L)))), 5), 'delta closed form': round(float(cf), 5)})
mc = pd.DataFrame(rows); display(mc); print('largest relative deviation (rows with delta > 1e-3): %.3f' % (abs(mc['delta Monte Carlo']/mc['delta closed form']-1)[mc['delta closed form'] > 1e-3]).max())
fig('fig_accountants.png', 420)""")
md("## 4. Non-private baselines\n\nMean ± 95% Nadeau-Bengio interval over the outer splits. Positive class = adverse outcome.")
code("""for d in A.DS:
    R = A.cv(d); print(f'--- {A.NAMES[d]}: {len(R)} outer splits'); display(A.nonprivate_table(d))""")
md("## 5. Utility under DP-SGD\n\nTest AUC by method and ε (δ = 1e-5). The noise multiplier is solved per fold from the target ε.")
code("""fig('fig_privacy_utility.png', 760)
for d in A.DS: print('---', A.NAMES[d], '(AUC)'); display(A.dp_table(d, 'auc'))""")
code("""for d in A.DS: print('---', A.NAMES[d], '(accuracy)'); display(A.dp_table(d, 'acc'))""")
md("### Paired comparisons (corrected resampled t-test, Holm-adjusted within each dataset)")
code("""for d in A.DS: print('---', A.NAMES[d]); display(A.paired_table(d))""")
code("""for d in A.DS: print('---', d, '- most frequent (C, lr) chosen by the inner search'); display(A.chosen_hparams(d))""")
md("## 6. Sensitivity analyses")
code("""print('Credit_History handling (loan data, non-private, 5x5 CV)'); display(A.credit_history_table())
for d in ('loan', 'german'): print('--- wider MLP search,', d); display(A.mlptune_table(d))
fig('fig_loss.png', 420)""")
md("## 7. Learning curve (Taiwan, fixed 6,000-row test set)")
code("""display(A.lc_table()); fig('fig_learning_curve.png', 460)""")
md("## 8. Explanation stability\n\nSpearman rank correlation between a DP model's global importance ranking and a non-private reference. The *noise floor* column compares the reference with two non-private networks that differ only in the random seed.")
code("""for d in A.DS: print('---', A.NAMES[d]); display(A.expl_summary(d))
fig('fig_expl_stability.png', 700)""")
code("""for d in A.DS:
    print('---', d)
    for k, v in A.expl_extra(d).items(): print(f'  {k}: {v}')
fig('fig_expl_heat_loan.png', 360)""")
md("## 9. Membership-inference audit\n\nLiRA (fixed variance) over 96 shadow models per setting. `overfit_control` is a deliberately over-trained network and shows that the attack works when there is something to find.")
code("""for d in A.DS: print('---', A.NAMES[d]); display(A.mia_table(d))
fig('fig_mia.png', 700)""")
md("## 10. Group-level diagnostics\n\nPooled counts over splits; intervals are bootstrap percentiles over splits. Groups with fewer than 30 pooled rows are dropped.")
code("""for d in A.DS:
    t = A.fair_gaps(d); print('---', A.NAMES[d]); display(t[t.Model.str.contains('nonprivate|dp_eps1$|dp_eps4$')])""")
md("## 11. Re-running one split from scratch\n\nThe first loan split is re-run from the raw CSV with the code in `run_cv.py` and compared with the saved result. Differences below 1e-6 count as identical.")
code("""import time, run_cv
from sklearn.model_selection import RepeatedStratifiedKFold
ds = load('loan'); tr, te = next(iter(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0).split(ds.df, ds.split_y)))
t0 = time.time(); fresh = run_cv.one_split(ds, run_cv.CFG['loan'], 0, tr, te); saved = json.load(open(os.path.join(V2, 'results', 'cv_loan.json')))[0]
worst, nm = 0.0, 0
for m, d in saved['models'].items():
    for k, v in d.items():
        if isinstance(v, float) and m in fresh['models'] and k in fresh['models'][m]: worst = max(worst, abs(v - fresh['models'][m][k])); nm += 1
print(f'{nm} metric values compared in {time.time()-t0:.0f}s; largest absolute difference = {worst:.2e}')
assert worst < 1e-6""")
md("""## What this notebook shows

Each statement in the paper's Results section corresponds to a table or figure produced above from files whose hashes are checked in section 1, and the code that produced those files reproduces a split exactly (section 11). To recreate everything from the raw data run `v2/run_all.sh`.""")
nb.cells = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "02_corrected_study.ipynb"); print("written")
