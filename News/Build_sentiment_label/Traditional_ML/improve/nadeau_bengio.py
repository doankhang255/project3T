"""Nadeau & Bengio (2003) corrected variance for repeated k-fold CV.

``improve/repeated_cv.py`` reports ``mean +/- std`` of the macro-F1 across
``n_repeats`` *repeat-level* scores (each already an aggregate over 5 folds).
That naive ``std/sqrt(n_repeats)`` treats the repeats as independent draws.
They are not: every repeat's 5 training folds overlap with every other
repeat's training folds by construction, so the naive standard error
under-estimates the true uncertainty (Nadeau & Bengio 2003; the "corrected
resampled t-test" popularized by Bouckaert & Frank 2004 is the standard fix).

This computes the correction at the *fold* level (``n_repeats * n_splits``
individual scores, not the repeat-level aggregates):

    corrected_var(mean) = (1/n + n_test/n_train) * sample_var(fold_scores)

``n_test`` / ``n_train`` are the held-out / training set sizes of a single
fold. ``n_test/n_train > 0`` always, so the corrected variance is always
>= the naive one - this can only widen a CI or shrink a t-statistic, never
the reverse. It does not change the point estimate (the mean) at all.

Also provides the paired version (corrected resampled paired t-test) for
comparing two models on the same fold structure - the parametric counterpart
to ``bootstrap.py``'s resampling-based delta CI.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from News.Build_sentiment_label.Traditional_ML.improve.repeated_cv import stratified_folds


def _fold_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int = 3) -> float:
    f1s = []
    for label_id in range(n_labels):
        true_pos = int(np.sum((y_true == label_id) & (y_pred == label_id)))
        false_pos = int(np.sum((y_true != label_id) & (y_pred == label_id)))
        false_neg = int(np.sum((y_true == label_id) & (y_pred != label_id)))
        denom = 2 * true_pos + false_pos + false_neg
        f1s.append(0.0 if denom == 0 else 2 * true_pos / denom)
    return float(np.mean(f1s))


def per_fold_scores(y: np.ndarray, oof: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """One macro-F1 per (repeat, fold) - length ``oof.shape[0] * n_splits`` -
    reconstructed from the already-computed out-of-fold predictions, using
    the same deterministic ``stratified_folds(y, n_splits, repeat_id)`` the
    real run used. No retraining."""
    y = np.asarray(y, dtype=int)
    scores = []
    for repeat_id in range(oof.shape[0]):
        for fold in stratified_folds(y, n_splits, repeat_id):
            scores.append(_fold_macro_f1(y[fold], oof[repeat_id][fold]))
    return np.asarray(scores, dtype=float)


def nadeau_bengio_ci(
    y: np.ndarray, oof: np.ndarray, n_splits: int = 5, alpha: float = 0.05
) -> dict:
    """Naive vs Nadeau-Bengio-corrected CI for one model's mean macro-F1."""
    y = np.asarray(y, dtype=int)
    n_rows = len(y)
    n_test = n_rows // n_splits
    n_train = n_rows - n_test

    fold_scores = per_fold_scores(y, oof, n_splits)
    n = len(fold_scores)
    mean = float(fold_scores.mean())
    sample_var = float(fold_scores.var(ddof=1))

    naive_se = float(np.sqrt(sample_var / n))
    corrected_se = float(np.sqrt(sample_var * (1.0 / n + n_test / n_train)))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))

    return {
        "mean": mean,
        "n_folds": n,
        "n_test": n_test,
        "n_train": n_train,
        "naive_se": naive_se,
        "corrected_se": corrected_se,
        "naive_ci": (mean - t_crit * naive_se, mean + t_crit * naive_se),
        "corrected_ci": (mean - t_crit * corrected_se, mean + t_crit * corrected_se),
        "widen_factor": corrected_se / naive_se if naive_se > 0 else float("inf"),
    }


def nadeau_bengio_paired_test(
    y: np.ndarray, oof_a: np.ndarray, oof_b: np.ndarray, n_splits: int = 5, alpha: float = 0.05
) -> dict:
    """Corrected resampled paired t-test (Nadeau & Bengio 2003 / Bouckaert &
    Frank 2004) for delta = model_a - model_b, fold by fold. Parametric
    counterpart to ``bootstrap.bootstrap_delta`` - same question ("is the
    macro-F1 gap real"), different machinery (t-distribution instead of
    resampling)."""
    y = np.asarray(y, dtype=int)
    n_rows = len(y)
    n_test = n_rows // n_splits
    n_train = n_rows - n_test

    scores_a = per_fold_scores(y, oof_a, n_splits)
    scores_b = per_fold_scores(y, oof_b, n_splits)
    deltas = scores_a - scores_b
    n = len(deltas)
    mean_delta = float(deltas.mean())
    sample_var = float(deltas.var(ddof=1))

    naive_se = float(np.sqrt(sample_var / n))
    corrected_se = float(np.sqrt(sample_var * (1.0 / n + n_test / n_train)))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))

    corrected_t = mean_delta / corrected_se if corrected_se > 0 else float("inf")
    corrected_p = float(2 * stats.t.sf(abs(corrected_t), df=n - 1))

    return {
        "mean_delta": mean_delta,
        "n_folds": n,
        "naive_ci": (mean_delta - t_crit * naive_se, mean_delta + t_crit * naive_se),
        "corrected_ci": (mean_delta - t_crit * corrected_se, mean_delta + t_crit * corrected_se),
        "corrected_p": corrected_p,
        "corrected_crosses_zero": bool(
            mean_delta - t_crit * corrected_se <= 0.0 <= mean_delta + t_crit * corrected_se
        ),
    }
