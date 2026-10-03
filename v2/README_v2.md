# v2: corrected study (exact accounting, leak-free evaluation, three datasets)

Everything in the paper is produced by the code in this folder from the raw data.

| Step | Command | Time (1 CPU core) |
|---|---|---|
| Everything, from raw data | `bash run_all.sh` | about 2 to 3 hours |
| Unit tests only | `python3 tests.py` | seconds |
| Re-create tables/figures from saved results | `python3 export_paper_data.py` (needs `results/*.json`) | seconds |
| Verify nothing changed | open `notebooks/02_corrected_study.ipynb` (section 1 re-hashes every file) | seconds |

Layout: `accounting.py` (exact, PLD and RDP accountants), `dpnet.py` (NumPy DP-SGD), `pipeline.py` and `data.py` (datasets, leak-free preprocessing), `run_cv.py` (nested CV benchmark), `run_expl.py` (explanation stability), `run_mia.py` (membership audit), `run_fair.py`, `run_lc.py`, `run_mlp_tune.py`, `run_misc.py`, `analysis.py` (all tables and figures), `results/` (every raw result as JSON), `figures/`, `MANIFEST.json` (SHA-256 of every file, library versions, git commit).
Data: `../Train.csv` (loan), `../data_extra/german_full_1000.csv`, `../data_extra/taiwan_credit_default_30000.csv`.
Manual steps that could not be run in the analysis environment: `opacus_crosscheck.py`, `kernelshap_crosscheck.py`.
Label convention: y = 1 is the adverse outcome. Results are deterministic given the library versions in `MANIFEST.json`; re-running split 0 of the loan benchmark reproduces the saved metrics exactly (notebook 2, section 11).
