# improve/ — methodology fixes for Traditional_ML (M2.1 / M2.2 / M2.3 / M2.4)

Standalone, same pattern as `../experiment_vocab/`: this folder **imports pure
helpers** from the main pipeline (TF-IDF fit/transform, the leak-free feature
recipe, the metric functions, the four `build_estimator` factories) and does
its own CV wiring. It does **not** edit or monkeypatch `model/*.py`,
`TF_IDF.py` or `model/common.py`, and writes only into this folder.

## What each piece does

| ID | Where | Question it answers |
|---|---|---|
| **M2.1** | `run_improve.py` — `complement_nb` variant | Does `ComplementNB` beat `MultinomialNB` on this imbalanced 3-class set? (Rennie et al. 2003 — Complement NB is built for class-imbalanced text; here `positive` is 23% of rows.) |
| **M2.2** | `repeated_cv.py` | Replace the single 5-fold number with **mean ± std over 10 repeated splits**, so a ~0.02 macro-F1 gap is visibly inside the noise. |
| **M2.3** | `mcnemar.py` | Is model A really better than model B, or is it the fold split? McNemar's test (Dietterich 1998), hand-rolled on `scipy` — `statsmodels` is not a project dependency. Tests the **accuracy** gap (per-row correct/wrong). |
| **M2.4** | `bootstrap.py` | McNemar tests accuracy; the models are ranked by **macro-F1**, which on this set hinges on the 35-row `positive` class. Resample the 152 eval rows with replacement → 95% CI on each model's macro-F1 and on every pairwise `Δ(macro-F1)`. One shared set of resamples is reused for every model so `Δ` is paired. A CI straddling 0 ⇒ the ranking is split noise. |
| **M2.5** | `nadeau_bengio.py` + `run_nadeau_bengio.py` | Does the naive `std/sqrt(n_repeats)` in M2.2 understate the true uncertainty? Nadeau & Bengio (2003): repeated-CV folds share overlapping training rows, so they are not independent draws — the naive SE is a lower bound. Reconstructs per-fold (not per-repeat) macro-F1 from the existing OOF arrays (no retraining) and applies the corrected-variance estimator, both standalone (CI on one model's mean) and paired (corrected resampled t-test on `Δ` between two models). |
| **M2.6** | `calibration_check.py` | Is RF's raw vote-fraction `predict_proba` / SVM's Platt-scaled `predict_proba` actually calibrated? Reliability diagram (binned confidence vs empirical accuracy) + Brier score + ECE, compared before/after a calibration change on the SAME leak-free CV. Promoted into production off this: `model/random_forest.py` now uses isotonic calibration (ECE 0.061→0.027), `model/svm.py` dropped Platt (macro-F1 up but ECE worse — kept anyway since models are ranked by macro-F1, not calibration; see `IMPROVEMENTS.md` section D). |

## Run

```bash
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py --repeats 3 --n-boot 400   # quick
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py --models multinomial_nb complement_nb
```

Outputs (all in `improve/`):

- `RESULTS.txt` — the report
- `data/repeated_cv_per_repeat.csv` — every (model, repeat) row
- `data/repeated_cv_summary.csv` — mean/std per model per metric
- `data/mcnemar_pairwise.csv` — pairwise McNemar
- `data/bootstrap_macro_f1_ci.csv` — per-model macro-F1 point + 95% CI
- `data/bootstrap_delta.csv` — pairwise `Δ(macro-F1)` point + 95% CI + bootstrap p

## How the CV harness stays faithful

`repeated_cv.stratified_folds(y, n_splits, seed)` is the exact round-robin
stratification of `model/common.build_stratified_folds`, with the RNG seed
exposed as an argument. Checked while building this folder:

- `stratified_folds(y, 5, RANDOM_SEED)` is byte-identical to
  `build_stratified_folds(y)`.
- `run_single_cv(MultinomialNB, fold_seed=RANDOM_SEED)` reproduces the main
  pipeline's out-of-fold predictions exactly (152/152 rows).
- Per-fold train vocabulary excludes terms that occur only in the held-out
  fold (no leakage).
- `run_single_cv` carries the same `model.classes_ == [0,1,2]` guard as
  `model/common.run_cross_validation`, so a fold that ever lost a class fails
  loudly instead of misaligning `predict_proba` columns.
- `bootstrap.macro_f1_per_repeat` matches `model/common.compute_metrics`'
  macro-F1 to machine precision (checked on the real out-of-fold vectors).

In repeat `r` every model is scored on the **same** split, so both the McNemar
comparison and the paired bootstrap `Δ(macro-F1)` are properly paired.

## M2.5 result (419-row tune set, 10 repeats)

```bash
python News/Build_sentiment_label/Traditional_ML/improve/run_nadeau_bengio.py
```

The correction widens every CI by the same factor (~3.65× here — it depends
only on `n_repeats`, `n_splits` and row count, not on the model or its
variance) and **never changes a point estimate**. On the tune set it does
change which pairwise gaps hold up:

- `logistic_regression`/`multinomial_nb` vs `svm` — corrected CI still
  excludes 0 (corrected `p` ≈ 0.005–0.010). Survives.
- `random_forest` vs `svm` (naive `p<0.05`) and `multinomial_nb` vs
  `random_forest` (naive `p<0.05`) — corrected CI **includes 0**. Do not cite
  these as reliable; the naive repeated-CV variance made them look more
  solid than they are.

Practical read: "every model beats SVM" only holds cleanly for
`logistic_regression`/`multinomial_nb`; `random_forest`'s edge over `svm` is
not distinguishable from split noise once the fold-overlap is accounted for.
Full table: `nadeau_bengio_out/RESULTS.txt` (not committed by default — same
`--out-dir` pattern as `run_improve.py`).

## Not done here

- Nested-CV hyperparameter tuning — see `../IMPROVEMENTS.md` section D. Worth
  revisiting as ground truth keeps growing (599 rows now; the tuning-variance
  concern that blocked it at 152 rows is smaller but not gone).
- Isotonic-calibrated RF, the SVM Platt→margin-softmax switch, and the
  averaged-probability ensemble (LR+NB+RF) are **promoted already** (see
  M2.6 above and `../IMPROVEMENTS.md` section D) — `model/random_forest.py`,
  `model/svm.py`, `model/ensemble.py`. Not left here as an experiment.

## Promotion path

The pipeline's default ground truth is now `data_news/ground_truth_combined.csv`
(599 rows, via `prepare_ground_truth.py`) - the old 152-row
`ground_truth_labeled.csv` is a strict subset, still on disk, no longer the
default input.

`RESULTS_SUMMARY.txt` (generated by `run_pipeline.py`) still reports **one
single-seed macro-F1 per model** on this 599-row set. It should not be cited
as the final word — e.g. the ensemble's single-seed macro-F1 (0.593) sits
*below* `logistic_regression` alone (0.596) in this one split; M2.2/M2.4-style
repeated CV + bootstrap on the ensemble is needed before concluding whether it
actually helps. To make `RESULTS_SUMMARY.txt` honest, `run_pipeline.py` has to
move to repeated CV and report mean ± std plus the bootstrap CI - deferred
until a model choice is actually being locked in.

When a change here is confirmed (e.g. `ComplementNB` reliably ≥ `MultinomialNB`
across repeats, McNemar **and** the bootstrap `Δ` CI), fold it into
`model/naive_bayes.py`, switch `compare_models.py` / `RESULTS_SUMMARY.txt` to
report mean ± std + CI, and add the McNemar / bootstrap tables there.
