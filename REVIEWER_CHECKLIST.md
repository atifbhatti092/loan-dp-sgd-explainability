# Reviewer 2 report: status of all 38 comments (honest version)

D = done and evidenced, P = partly done, N = not done, M = needs a manual step by the authors.
"Where" points to the paper section or file that shows the evidence.

| # | Status | What was done / what is missing | Where |
|---|---|---|---|
| 1 | D | Retitled as credit decisions; loan target called approval, not default | Title, Sec 3.2 |
| 2 | D | One claim: exact accounting + private baselines + explanation stability | Sec 1 |
| 3 | D | Threat model, privacy scope figure, post-processing proposition | Sec 3.1, Fig 1, App A.6 |
| 4 | P | Full field table for the loan data; German/Taiwan described in text and Table 1 | Table B5, Table 1 |
| 5 | D | All statistics fitted on the training fold | Sec 3.3 |
| 6 | D | Three Credit_History treatments | Table 7 |
| 7 | P | Group diagnostics with/without protected fields at eps = 1 and 4; no mitigation, small groups | Sec 4.8, Table 12, B7 |
| 8 | P | Wide non-private MLP search and per-eps DP tuning of C and lr; DP width/depth not searched | Table 8, Sec 5.3 |
| 9 | D | Majority, single-feature rule, LR, RF, CatBoost, XGBoost | Table 4 |
| 10 | M | Opacus / JAX-Privacy cross-check could not be run here (no torch). Script prepared | `v2/opacus_crosscheck.py` |
| 11 | P | C and lr chosen by inner CV, stated as outside eps; no private tuning method | Sec 3.1, 5.3 |
| 12 | D | No resampling, class weights or threshold tuning; reason given (threshold tuning reads private data) | Sec 3.4 |
| 13 | D | Draft delta arithmetic shown wrong (15.5x), corrected analysis | Sec 3.6, 4.1, App A.4 |
| 14 | D | Exact (Gaussian DP), PLD and RDP accountants, cross-checked | Table 3, Fig 2 |
| 15 | D | Target eps first, sigma solved per fold | Sec 3.6 |
| 16 | D | Poisson sampling, 300 to 800 steps, PLD accounting | Sec 3.5, 3.6 |
| 17 | P | Repeated nested CV with inner selection; no separate untouched final test set | Sec 3.7, 5.3 |
| 18 | P | DP-LR, DP-EBM, local perturbation done; no DP random forest | Table 5 |
| 19 | P | Corrected resampled t-test with Holm; McNemar and DeLong not done | Table 6, Sec 5.3 |
| 20 | P | Recall, balanced accuracy, MCC reported; PR-AUC and threshold moving not done | Table 4, Sec 5.3 |
| 21 | D | Single-split vs CV discrepancy of the first draft is shown in notebook 1 | notebook 1 |
| 22 | D | Loss-threshold and LiRA, positive control, audit eps lower bound | Sec 4.7, Table 11 |
| 23 | D | 30,000-row Taiwan data and learning curve (Give Me Some Credit and Home Credit were not reachable) | Sec 4.5 |
| 24 | D | German Credit, all 1,000 rows, same nested CV and eps sweep | Tables 4, 5 |
| 25 | P | IG and grouped PI at every eps and seed; KernelSHAP not run here (script prepared) | Sec 4.6, `v2/kernelshap_crosscheck.py` |
| 26 | D | Three baselines, 128 steps, completeness error; up to 300 test rows per split | Sec 3.8, 4.6 |
| 27 | D | Three cases separated | Sec 3.1, App A.6 |
| 28 | D | Cross-references checked by script; re-read in Word after any edit | whole paper |
| 29 | P | No comparison with literature numbers at unmatched settings; none claimed | Sec 2.2 |
| 30 | D | Limits written with consequences | Sec 5.3 |
| 31 | D | Section 2 rewritten with corrected and recent sources | Sec 2 |
| 32 | P | Contributions stated as findings in Sec 1; not every one is tied to a numbered table in Sec 1 | Sec 1 |
| 33 | M | EU rules cited. If the target context is Pakistan, add State Bank of Pakistan guidance | Sec 1 |
| 34 | D | Abstract 225 words, numbers with eps, delta | Abstract |
| 35 | D + M | Mechanical checks pass (no em dashes, no stock phrases, long sentences split). Authors must read and own the text | whole paper |
| 36 | D | Findings, limits, two open questions | Sec 6 |
| 37 | P + M | Web-verified: refs 5, 6, 17, 18, 19. All other references are from memory: verify DOI, volume, pages | References |
| 38 | D | `v2/run_all.sh`, seeds and versions table | Table B6 |

## Where I disagree with the reviewer
* Comment 12: the report says threshold tuning on validation data costs no privacy. It does, because the validation labels are private data. The paper explains this.
* Comment 15/16 implied that Poisson subsampling is what makes DP-SGD usable on this data. On the loan data, full-batch (100 steps) and Poisson (300 steps) were not different (Sec 4.3).

## Manual tasks for the authors (cannot be done by the assistant)
1. DONE: re-run on Google Colab with `run_all.py`; all results matched to within 2e-7 except the Taiwan membership audit (up to 1.9e-2, floating-point sensitivity). Update Table B6 (hardware) with the Colab machine if you want it to describe your run.
2. Verify every reference against the publisher record and add DOIs (see comment 37).
3. Run `opacus_crosscheck.py` and, optionally, `kernelshap_crosscheck.py`; add one sentence with the outcome.
4. Disclose the use of generative AI tools exactly as the journal policy requires. This manuscript text was drafted with an AI assistant.
5. Add the regional regulator reference if needed (comment 33).
6. Push the new folders to GitHub (`git add v2 notebooks data_extra REVIEWER_CHECKLIST.md && git commit -m "Corrected study v2" && git push`) so that the Data Availability link is true.
