# Additions to the corrected study (v2): end-to-end privacy, canary audit, DP random forest

All new files are additive; no existing file was changed. Run from the `v2/` folder.

| File | What it does |
|---|---|
| `private_tuning.py` | End-to-end private hyper-parameter selection: m candidate (C, lr) settings, each trained with its own DP-SGD run, scored by a noisy clipped-loss sum (Gaussian mechanism); the total epsilon covers all runs and the selection. Exact for full batch; PLD (dp_accounting) for Poisson sampling, with an RDP upper bound as fallback. |
| `run_private_tuning.py` | Benchmark on the paper's outer splits for m = 1 (no tuning), 2 (learning rate only), 6 (full grid) at total epsilon 0.5, 1, 2, 4, 8. Output `results/ptune_<dataset>.json`. |
| `analyze_ptune.py` | Tables: AUC with Nadeau-Bengio intervals and the paired "price of private tuning" against the paper's numbers. |
| `dpnet_audit.py`, `run_canary_audit.py` | White-box canary audit of the full-batch DP-SGD implementation and the exact accountant (Nasr et al. style): estimates the Gaussian shift mu_hat and compares it with the theoretical sqrt(T)/sigma. Output `results/canary_<dataset>.json`. |
| `run_dprf.py` | DP random forest baseline (diffprivlib, pure epsilon-DP, no tuning). Output `results/dprf_<dataset>.json`. |
| `tests_ext.py` | Tests for all of the above. |

## Run on Colab (about 1 to 2 hours in total on a CPU runtime)

```bash
pip install -r requirements_v2.txt diffprivlib
cd v2
python3 tests.py && python3 tests_ext.py
for d in loan german taiwan; do python3 run_private_tuning.py $d 1000000 pld; python3 analyze_ptune.py $d; done
python3 run_canary_audit.py loan 2000 1000000       # R = 2000 runs per class; add german if desired
for d in loan german taiwan; do python3 run_dprf.py $d 1000000; done
```
`pld` makes the Poisson-sampling runs use the PLD accountant (tighter than the RDP bound used in the preliminary runs that are included here).
Every script is resumable: if Colab stops, run the same command again.

## Preliminary results included (single CPU core, RDP upper bound for Poisson runs)
`results/ptune_loan.json`, `ptune_summary_loan.json`, `canary_loan.json` (R = 300, epsilon 1 and 2 only). They are smoke-test quality: re-run with the commands above for the numbers that go into the paper.
