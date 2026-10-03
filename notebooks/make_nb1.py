import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Notebook 1 - Reproducing and auditing the original (first-draft) pipeline

This notebook does three things, in this order.

1. Re-runs the **unchanged** first-draft code (`prep.py`, `run_experiments.py`) in a scratch folder and checks that it reproduces the numbers printed in the first draft. That shows the audit below is about the real pipeline and not about a re-implementation.
2. Re-computes the privacy parameters of the first draft with an exact accountant.
3. Quantifies how much of the headline accuracy was a property of one lucky 123-row test split.

Everything here is computed when the notebook runs. Nothing is typed in by hand except the first-draft values we compare against, which are read from the shipped `results.json`.""")

code("""import os, sys, json, shutil, subprocess, hashlib, platform, tempfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'v2'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
print('python', platform.python_version(), '| numpy', np.__version__, '| pandas', pd.__version__)
print('git HEAD :', subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
for f in ['Train.csv', 'prep.py', 'dpmlp.py', 'run_experiments.py', 'results.json']: print(f'{f:22s} sha256[:16] = {sha(os.path.join(ROOT, f))}')""")

md("## 1. Re-run the first-draft code, unchanged\n\nThe scripts are copied to a scratch directory and executed there. The shipped `results.json` is not touched.")
code("""scratch = tempfile.mkdtemp()
for f in ['Train.csv', 'prep.py', 'dpmlp.py', 'run_experiments.py']: shutil.copy(os.path.join(ROOT, f), scratch)
for script in ['prep.py', 'run_experiments.py']:
    r = subprocess.run([sys.executable, script], cwd=scratch, capture_output=True, text=True); print(script, 'exit code', r.returncode)
new = json.load(open(os.path.join(scratch, 'results.json'))); old = json.load(open(os.path.join(ROOT, 'results.json')))
def flat(d, p=''):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, p + k + '.')
        elif isinstance(v, (int, float)): yield p + k, v
fo, fn = dict(flat(old)), dict(flat(new)); common = [k for k in fo if k in fn]
diff = max(abs(fo[k] - fn[k]) for k in common)
print(f'{len(common)} numeric entries compared, largest absolute difference = {diff:.2e}')
assert diff < 1e-9, 'first-draft pipeline did not reproduce'""")

md("### What the first draft reported")
code("""b = old['baseline_nonprivate']; cvd = old['cv_5fold']
print('Single split (5 seeds)  accuracy %.4f  AUC %.4f' % (b['mean']['accuracy'], b['mean']['roc_auc']))
print('5-fold CV on all data   accuracy %.4f  AUC %.4f' % (cvd['accuracy_mean'], cvd['auc_mean']))
sw = pd.DataFrame([{'sigma': s['sigma'], 'epsilon_reported': s['epsilon'], 'delta_nominal': s['delta'], 'accuracy': s['mean']['accuracy'], 'AUC': s['mean']['roc_auc']} for s in old['dp_sgd_sweep']])
sw""")

md("""## 2. Audit A - privacy accounting

The first draft computes epsilon with the advanced-composition theorem and splits delta as `delta0 = delta' = delta/2`. Two things follow from that arithmetic, both checked below:

* the delta that is actually achieved is `k*delta0 + delta'`, which is far larger than the nominal 1/491;
* for 30 full-batch Gaussian steps there is an **exact** accountant: the 30 releases form one Gaussian mechanism with `mu = sqrt(30)/sigma`, so epsilon can be read off the closed form (derivation and Monte-Carlo check are in Notebook 2 and in the paper's appendix).

We use two independent exact accountants: our own closed form and Google's `dp_accounting` PLD accountant. If they agree, the number is not an artefact of one implementation.""")
code("""from accounting import eps_fullbatch_exact, eps_pld, eps_rdp, eps_advanced_paper
rows = []
for s in old['dp_sgd_sweep']:
    sg = s['sigma']; e_adv, d_act = eps_advanced_paper(sg, 30, 1/491)
    rows.append({'sigma': sg, 'eps first draft': round(s['epsilon'], 3), 'recomputed first-draft formula': round(e_adv, 3), 'delta actually achieved': round(d_act, 4),
                 'exact eps @ delta=1/491': round(eps_fullbatch_exact(sg, 30, 1/491), 3), 'exact eps @ delta=1e-5': round(eps_fullbatch_exact(sg, 30, 1e-5), 3),
                 'PLD eps @ delta=1e-5': round(eps_pld(sg, 30, 1e-5), 3), 'RDP eps @ delta=1e-5': round(eps_rdp(sg, 30, 1e-5), 3)})
acc = pd.DataFrame(rows); acc""")
code("""assert np.allclose(acc['eps first draft'], acc['recomputed first-draft formula'], atol=2e-3), 'we do not reproduce the first-draft epsilon'
assert np.allclose(acc['exact eps @ delta=1e-5'], acc['PLD eps @ delta=1e-5'], atol=2e-3), 'closed form and PLD disagree'
print('first-draft epsilon reproduced; closed form and PLD agree to 3 decimals')
r1 = acc['eps first draft'] / acc['exact eps @ delta=1/491']; r2 = acc['eps first draft'] / acc['exact eps @ delta=1e-5']
print('Overstatement at the SAME nominal delta (1/491): %.1fx to %.1fx   <- the figure used in the paper' % (r1.min(), r1.max()))
print('Overstatement against the exact eps at the stricter delta=1e-5 used in the paper: %.1fx to %.1fx' % (r2.min(), r2.max()))""")

md("""## 3. Audit B - where the leakage is in the code

These are the exact lines of the first-draft scripts (printed from the files, not retyped).""")
code("""def show(path, needles, width=1):
    L = open(os.path.join(ROOT, path)).read().splitlines()
    for i, line in enumerate(L):
        if any(n in line for n in needles): print(f'{path}:{i+1:>3}: {line.strip()[:150]}')
print('-- imputation / encoding is done on the whole file before any split:')
show('prep.py', ['fillna', 'train_test_split', 'StandardScaler', 'fit_transform'])
print('\\n-- the balancing strategy is picked by looking at test-set F1:')
show('run_experiments.py', ['best_key', 'max('])
print('\\n-- clipping norm used for DP-SGD:')
show('run_experiments.py', ['clip'])""")

md("""## 4. Audit C - how much does one 123-row split tell us?

The first draft's single split has 123 test applicants. With an accuracy near 0.83 the binomial standard error alone is about 3.4 percentage points. Below we run the same MLP configuration of the first draft (lr 0.5, 30 epochs) on 25 leak-free train/test splits and see where 0.8325 falls.""")
code("""n_test, p = 123, old['baseline_nonprivate']['mean']['accuracy']
se = np.sqrt(p * (1 - p) / n_test); print(f'accuracy {p:.4f} on n={n_test}: standard error {100*se:.1f} points, 95% interval roughly [{p-1.96*se:.3f}, {p+1.96*se:.3f}]')
exp1 = json.load(open(os.path.join(ROOT, 'v2', 'results', 'exp1.json')))
accs = np.array([r['models']['mlp_paper_cfg']['acc'] for r in exp1]); aucs = np.array([r['models']['mlp_paper_cfg']['auc'] for r in exp1])
print(f'first-draft MLP configuration, leak-free 5x5 CV: accuracy {accs.mean():.3f} (sd {accs.std(ddof=1):.3f}), AUC {aucs.mean():.3f}')
print(f'share of the 25 splits whose accuracy is >= 0.8325: {np.mean(accs >= 0.8325):.0%}')
print(f'majority-class accuracy {np.mean([r["models"]["majority"]["acc"] for r in exp1]):.3f};  rule "Credit_History == 1 -> approve": {np.mean([r["models"]["credit_history_rule"]["acc"] for r in exp1]):.3f}')""")
code("""import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 2.8)); ax.hist(accs, bins=10, color='#7f9fbf', edgecolor='white'); ax.axvline(p, color='#c00000', lw=2, label='first-draft single split (0.8325)')
ax.axvline(accs.mean(), color='k', ls='--', label='mean over 25 leak-free splits'); ax.set_xlabel('test accuracy of the first-draft MLP configuration'); ax.set_ylabel('splits'); ax.legend(fontsize=7, frameon=False); plt.show()""")

md("""## Summary of what this notebook establishes

* The first-draft pipeline reproduces bit-for-bit from the shipped code and data.
* Its epsilon values are overstated by a large factor relative to two independent exact accountants, and the delta it actually achieves is not the nominal one.
* Imputation, scaling statistics and the balancing strategy were fitted on data that included the test rows.
* A single 123-row split cannot support a claim at the two-decimal level; the leak-free 25-split mean is lower and the single-feature rule is competitive.

The corrected study is in `notebooks/02_corrected_study.ipynb`.""")
nb.cells = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "01_original_pipeline_audit.ipynb"); print("written")
