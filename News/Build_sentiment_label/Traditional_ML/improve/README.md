# improve/ — methodology fixes for Traditional_ML (M2.1 / M2.2 / M2.3)

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
| **M2.3** | `mcnemar.py` | Is model A really better than model B, or is it the fold split? McNemar's test (Dietterich 1998), hand-rolled on `scipy` — `statsmodels` is not a project dependency. |

## Run

```bash
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py --repeats 3          # quick
python News/Build_sentiment_label/Traditional_ML/improve/run_improve.py --models multinomial_nb complement_nb
```

Outputs (all in `improve/`):

- `RESULTS.txt` — the report
- `data/repeated_cv_per_repeat.csv` — every (model, repeat) row
- `data/repeated_cv_summary.csv` — mean/std per model per metric
- `data/mcnemar_pairwise.csv` — pairwise McNemar

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

In repeat `r` every model is scored on the **same** split, so the McNemar
comparison is properly paired.

## Not done here (needs more ground truth first)

Nested-CV hyperparameter tuning, calibrated RF, an averaged-probability
ensemble — see `../IMPROVEMENTS.md` section D. With 152 rows the tuning
variance swamps the gains.

## Promotion path

When a change here is confirmed (e.g. `ComplementNB` reliably ≥ `MultinomialNB`
across repeats and McNemar), fold it into `model/naive_bayes.py`, switch
`compare_models.py` / `RESULTS_SUMMARY.txt` to report mean ± std, and add the
McNemar table there.
