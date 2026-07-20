# Differentially Private Deep Learning for Loan Default Risk Prediction

Code and data accompanying the paper *"Differentially Private Deep Learning for Loan
Default Risk Prediction: A DP-SGD and Integrated-Gradient Explainability Framework"*
(Muhammad Atif Sultan, Shaheer Arif Nazeer; Department of Computer Sciences, University
of Lahore, Sargodha Campus).

Everything in this repository is implemented from first principles in NumPy / scikit-learn
(no PyTorch, TensorFlow, or Opacus dependency) so that every gradient-clipping and
noise-injection step in DP-SGD, and every privacy-accounting formula, is fully visible
and auditable in plain Python.

## Repository structure

```
.
├── Train.csv                  # Primary dataset (Loan Prediction benchmark, 614 records)
├── german_500.csv             # External validation dataset (German Credit subsample, 572 records)
├── prep.py                    # Step 1: preprocessing, feature engineering, train/test split
├── dpmlp.py                   # Core library: DPNet (hand-rolled MLP + DP-SGD), SMOTE,
│                               #   privacy_epsilon (advanced composition)
├── run_experiments.py         # Step 2: main experiment pipeline (baseline, DP-SGD sweep,
│                               #   class-balance comparison, 5-fold CV, explainability)
├── make_figures.py            # Step 3: generates all figures used in the paper
├── extra_analysis.py          # Step 4: Random Forest / HistGradientBoosting baselines,
│                               #   Rényi-DP accounting, paired t-tests
├── german/
│   ├── german_500.csv
│   └── validate.py            # Step 5: external validation on German Credit
├── results/                   # Saved JSON outputs and PNG figures from the runs above
└── requirements.txt
```

## Reproducing the results

```bash
pip install -r requirements.txt

python prep.py                 # -> prep_data.npz
python run_experiments.py      # -> results.json, artifacts.npz
python make_figures.py         # -> fig_*.png
python extra_analysis.py       # -> extra_results.json
cd german && python validate.py  # -> german_validation_results.json
```

All randomness is seeded (seeds 1–5 for the multi-seed sweeps, seed 42 elsewhere), so
re-running the pipeline should reproduce the numbers reported in the paper's tables
exactly, modulo platform-level floating-point differences.

## Data sources

- `Train.csv`: the public Loan Prediction benchmark dataset, originally released as a
  practice problem by Analytics Vidhya and widely mirrored (e.g. on Kaggle).
- `german_500.csv`: a 572-record subsample of the Statlog (German Credit Data) dataset,
  Hofmann, H., UCI Machine Learning Repository, 1994, https://doi.org/10.24432/C5NC77
  (CC BY 4.0).

## Note on classical ML baselines

`extra_analysis.py` uses scikit-learn's `RandomForestClassifier` and
`HistGradientBoostingClassifier` as the classical-ML comparison point. CatBoost/XGBoost
were not available in the environment this code was developed in; if you want a direct
CatBoost comparison, swap in `catboost.CatBoostClassifier` trained on the same
`prep_data.npz` split and seeds — the rest of the pipeline does not need to change.

## License

Code: MIT License (add a `LICENSE` file with your names before publishing).
Data: redistributed under the original datasets' terms (Analytics Vidhya loan-prediction
practice dataset; German Credit Data, CC BY 4.0).
