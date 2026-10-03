# v2 findings (Phase 1): leak-free evaluation + exact privacy accounting

Protocol: 5x5 repeated stratified CV (25 splits). All preprocessing fitted on the training fold only.
delta = 1e-5. Intervals are 95% Nadeau-Bengio corrected. Unit tests: `python3 tests.py`.

## Accounting (full-batch, 30 steps; original manuscript vs exact)
| sigma | manuscript eps (delta actually 0.0316) | exact eps, delta=1e-5 (GDP = PLD) | RDP |
|---|---|---|---|
| 200 | 0.394 | 0.083 | 0.093 |
| 40 | 2.197 | 0.480 | 0.527 |
| 15 | 7.269 | 1.406 | 1.532 |

## Non-private (leak-free)
| model | accuracy | AUC |
|---|---|---|
| majority | 0.687 | 0.500 |
| Credit_History rule | 0.809 ± 0.021 | 0.705 ± 0.034 |
| logistic regression | 0.802 ± 0.026 | 0.752 ± 0.043 |
| random forest | 0.808 ± 0.037 | 0.779 ± 0.061 |
| MLP (nested-tuned) | 0.798 ± 0.024 | 0.744 ± 0.043 |

## DP-MLP, (C, lr) tuned on inner CV (tuning privacy cost NOT included in eps)
| eps | FB100 acc / AUC | PO300 acc / AUC |
|---|---|---|
| 0.5 | 0.692 / 0.631 | 0.706 / 0.661 |
| 1 | 0.758 / 0.702 | 0.760 / 0.693 |
| 2 | 0.783 / 0.721 | 0.778 / 0.709 |
| 4 | 0.793 / 0.726 | 0.792 / 0.711 |
| 8 | 0.803 / 0.729 | 0.801 / 0.723 |

Files: accounting.py, dpnet.py, data.py, tests.py, exp1.py, exp2.py, analyze1.py, analyze2.py, results/*.json
