"""Nested-CV hyperparameter tuning (ML_SUMMARY.qmd section 4).

    python News/Build_sentiment_label/Traditional_ML/Common/tune_hyperparameters.py

Cawley & Talbot (2010): picking a hyperparameter using the same fold you then
report performance on is optimistic - the outer validation fold leaks into
model selection. Nested CV fixes this: for each of the 5 OUTER folds, the
hyperparameter is chosen by an INNER 3-fold CV restricted to that fold's
training rows only (TF-IDF vocabulary refit per inner-train split too - same
leak-free discipline as ``model/common.run_cross_validation``); the outer
validation fold is touched exactly once, after the winning hyperparameter for
that outer fold is already fixed. Fitting the TF-IDF features once per inner
fold and reusing them across every grid candidate (vocabulary fit does not
depend on the model's hyperparameter) keeps this from being ``outer x inner x
grid`` TF-IDF fits - it is ``outer x inner``.

This alone only answers "does tuning help, honestly measured" via a paired
bootstrap of nested-CV macro-F1 vs. the current hardcoded default (same
out-of-fold arrays both sides, same pattern as ``experiment_Lexicon_features/``).
It does NOT hand you a single hyperparameter to hardcode - nested CV can (and
often does) pick a different winner per outer fold, by design. If a model's
delta CI excludes 0, the second step (``--select-final``) does one plain
(non-nested) grid search over the WHOLE tune set to pick that single value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import (
    build_document_term_counts,
    fit_tfidf_vocabulary,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.Common.bootstrap import (
    bootstrap_samples,
    ci,
    two_sided_p,
)
from News.Build_sentiment_label.Traditional_ML.Common.repeated_cv import (
    load_stopword_set,
    run_repeated_cv,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    MAX_FEATURES,
    RANDOM_SEED,
    VALID_LABELS,
    build_stratified_folds,
    compute_metrics,
    encode_labels,
    load_ground_truth_frame,
    resolve_n_splits,
    run_cross_validation,
    select_top_features,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    build_lexicon_feature_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.logistic_regression import (
    build_estimator as build_logistic_regression_default,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.naive_bayes import (
    build_estimator as build_multinomial_nb_default,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.random_forest import (
    N_ESTIMATORS,
    build_estimator as build_random_forest_default,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.svm import (
    MarginSoftmaxSVC,
    build_estimator as build_svm_default,
)

SCRIPT_DIR = Path(__file__).resolve().parent
N_SPLITS = 5
INNER_SPLITS = 3


def build_logistic_regression(random_state: int, C: float) -> LogisticRegression:
    return LogisticRegression(
        C=C, max_iter=5000, class_weight="balanced", random_state=random_state
    )


def build_svm(random_state: int, C: float) -> MarginSoftmaxSVC:
    return MarginSoftmaxSVC(random_state=random_state, C=C)


def build_multinomial_nb(random_state: int, alpha: float) -> MultinomialNB:
    del random_state
    return MultinomialNB(alpha=alpha)


def build_random_forest(
    random_state: int, min_samples_leaf: int, max_features
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )


# grid ranges per ML_SUMMARY.qmd section 4; kept modest (5-6 candidates) so
# a nested search (outer x inner x grid model fits) finishes in minutes, not
# hours - this is a first pass, not an exhaustive search.
MODELS = {
    "logistic_regression": {
        "build_fn": build_logistic_regression,
        "grid": [{"C": c} for c in (0.1, 0.3, 1.0, 3.0, 10.0)],
        "default_params": {"C": 1.0},
        "build_default": build_logistic_regression_default,
        "uses_lexicon": False,
    },
    "svm": {
        "build_fn": build_svm,
        "grid": [{"C": c} for c in (0.1, 0.3, 1.0, 3.0, 10.0)],
        "default_params": {"C": 1.0},
        "build_default": build_svm_default,
        "uses_lexicon": False,
    },
    "multinomial_nb": {
        "build_fn": build_multinomial_nb,
        "grid": [{"alpha": a} for a in (0.1, 0.3, 0.5, 1.0, 2.0)],
        "default_params": {"alpha": 1.0},
        "build_default": build_multinomial_nb_default,
        "uses_lexicon": False,
    },
    "random_forest": {
        "build_fn": build_random_forest,
        "grid": [
            {"min_samples_leaf": msl, "max_features": mf}
            for msl in (1, 2, 5)
            for mf in ("sqrt", 0.4)
        ],
        "default_params": {"min_samples_leaf": 1, "max_features": "sqrt"},
        "build_default": build_random_forest_default,
        "uses_lexicon": True,
    },
}


def _fit_fold_features(
    train_counts,
    val_counts,
    stopwords: set[str],
    max_features: int = MAX_FEATURES,
):
    """Leak-free TF-IDF fit (train rows only) + top-feature cut, transformed
    for both train and val - the same 3 calls ``run_cross_validation`` makes,
    factored out so both the inner and outer loop below share it verbatim."""
    vocabulary_df = fit_tfidf_vocabulary(
        train_counts, total_documents=len(train_counts), stopwords=stopwords
    )
    x_train, _ = transform_tfidf(train_counts, vocabulary_df)
    x_val, _ = transform_tfidf(val_counts, vocabulary_df)
    x_train_selected, _, selected_indices = select_top_features(
        x_train, vocabulary_df, max_features=max_features
    )
    x_val_selected = x_val[:, selected_indices]
    return x_train_selected, x_val_selected


def _with_extra_features(
    x_train_selected, x_val_selected, extra_features, train_indices, val_indices
):
    if extra_features is None:
        return x_train_selected, x_val_selected
    train_extra = extra_features[train_indices]
    val_extra = extra_features[val_indices]
    extra_mean = train_extra.mean(axis=0)
    extra_std = train_extra.std(axis=0)
    extra_std = np.where(extra_std > 1e-8, extra_std, 1.0)
    return (
        np.hstack([x_train_selected, (train_extra - extra_mean) / extra_std]),
        np.hstack([x_val_selected, (val_extra - extra_mean) / extra_std]),
    )


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics = compute_metrics(y_true, y_pred)
    return float(metrics.loc[metrics["metric_scope"].eq("overall"), "f1"].iloc[0])


def nested_cv_tune(
    build_fn,
    grid: list[dict],
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    extra_features: np.ndarray | None = None,
    n_splits: int = N_SPLITS,
    inner_splits: int = INNER_SPLITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    y = np.asarray(y, dtype=int)
    n_rows = len(y)
    resolved_splits = resolve_n_splits(y, n_splits)
    all_indices = np.arange(n_rows)
    probabilities = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    fold_of_row = np.full(n_rows, -1, dtype=int)
    chosen_params_per_fold: list[dict] = []

    outer_folds = build_stratified_folds(y, n_splits=resolved_splits)
    for fold_id, val_indices in enumerate(outer_folds, start=1):
        fold_of_row[val_indices] = fold_id
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[val_indices] = False
        train_indices = all_indices[train_mask]

        # --- inner CV: pick the grid candidate using train_indices only ---
        inner_resolved = resolve_n_splits(y[train_indices], inner_splits)
        inner_folds_local = build_stratified_folds(y[train_indices], n_splits=inner_resolved)
        cached_inner = []
        for inner_val_local in inner_folds_local:
            inner_val_global = train_indices[inner_val_local]
            inner_train_mask = np.ones(len(train_indices), dtype=bool)
            inner_train_mask[inner_val_local] = False
            inner_train_global = train_indices[inner_train_mask]

            inner_train_counts = [term_counts[i] for i in inner_train_global]
            inner_val_counts = [term_counts[i] for i in inner_val_global]
            x_inner_train, x_inner_val = _fit_fold_features(
                inner_train_counts, inner_val_counts, stopwords
            )
            x_inner_train, x_inner_val = _with_extra_features(
                x_inner_train, x_inner_val, extra_features, inner_train_global, inner_val_global
            )
            cached_inner.append(
                (x_inner_train, y[inner_train_global], x_inner_val, y[inner_val_global])
            )

        best_score = -np.inf
        best_params = grid[0]
        for params in grid:
            inner_scores = []
            for x_it, y_it, x_iv, y_iv in cached_inner:
                model = build_fn(RANDOM_SEED, **params)
                model.fit(x_it, y_it)
                preds = model.predict_proba(x_iv).argmax(axis=1)
                inner_scores.append(_macro_f1(y_iv, preds))
            mean_score = float(np.mean(inner_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        chosen_params_per_fold.append(best_params)

        # --- outer: refit on the FULL outer-train with the winning params,
        # score once on the never-touched outer validation fold ---
        train_counts = [term_counts[i] for i in train_indices]
        val_counts = [term_counts[i] for i in val_indices]
        x_train, x_val = _fit_fold_features(train_counts, val_counts, stopwords)
        x_train, x_val = _with_extra_features(
            x_train, x_val, extra_features, train_indices, val_indices
        )

        model = build_fn(RANDOM_SEED + fold_id, **best_params)
        model.fit(x_train, y[train_indices])
        if list(model.classes_) != list(range(len(VALID_LABELS))):
            raise AssertionError(
                f"Fold {fold_id}: unexpected class order {list(model.classes_)}"
            )
        probabilities[val_indices] = model.predict_proba(x_val)
        print(
            f"    outer fold {fold_id}/{resolved_splits}: best={best_params} "
            f"(inner macro_f1={best_score:.3f})"
        )

    predictions = probabilities.argmax(axis=1)
    return probabilities, predictions, fold_of_row, chosen_params_per_fold


def select_final_params(
    build_fn,
    grid: list[dict],
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    extra_features: np.ndarray | None = None,
    n_splits: int = N_SPLITS,
) -> dict:
    """Plain (non-nested) K-fold grid search on the WHOLE tune set - not used
    to report a performance number (that would be the exact optimism nested
    CV exists to avoid), only to pick the single value to hardcode."""
    y = np.asarray(y, dtype=int)
    n_rows = len(y)
    all_indices = np.arange(n_rows)
    resolved_splits = resolve_n_splits(y, n_splits)
    folds = build_stratified_folds(y, n_splits=resolved_splits)

    best_score = -np.inf
    best_params = grid[0]
    for params in grid:
        fold_scores = []
        for val_indices in folds:
            train_mask = np.ones(n_rows, dtype=bool)
            train_mask[val_indices] = False
            train_indices = all_indices[train_mask]
            train_counts = [term_counts[i] for i in train_indices]
            val_counts = [term_counts[i] for i in val_indices]
            x_train, x_val = _fit_fold_features(train_counts, val_counts, stopwords)
            x_train, x_val = _with_extra_features(
                x_train, x_val, extra_features, train_indices, val_indices
            )
            model = build_fn(RANDOM_SEED, **params)
            model.fit(x_train, y[train_indices])
            preds = model.predict_proba(x_val).argmax(axis=1)
            fold_scores.append(_macro_f1(y[val_indices], preds))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score


def holdout_check(
    ground_truth_csv: Path,
    checks: dict[str, tuple],
    n_repeats: int = 10,
) -> pd.DataFrame:
    """One-time confirmatory check on a split never used to pick these
    params - same discipline as ``experiment_Lexicon_features/``: only run
    this for a candidate that already cleared the tune-set bar, and only
    once. ``checks``: ``{model_name: (build_fn, default_params, tuned_params)}``.
    """
    from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
        load_frame_from_csv,
    )

    frame = load_frame_from_csv(ground_truth_csv)
    print(f"Holdout: {ground_truth_csv} ({len(frame)} rows)")
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()

    rows = []
    for name, (build_fn, default_params, tuned_params) in checks.items():
        default_factory = lambda seed, p=default_params: build_fn(seed, **p)
        tuned_factory = lambda seed, p=tuned_params: build_fn(seed, **p)

        print(f"\n[{name}] holdout default={default_params} ({n_repeats} repeats) ...", flush=True)
        default_repeat, default_oof = run_repeated_cv(
            default_factory, term_counts, y, stopwords, n_repeats=n_repeats
        )
        print(f"[{name}] holdout tuned={tuned_params} ({n_repeats} repeats) ...", flush=True)
        tuned_repeat, tuned_oof = run_repeated_cv(
            tuned_factory, term_counts, y, stopwords, n_repeats=n_repeats
        )

        boot = bootstrap_samples(
            np.asarray(y), {"default": default_oof, "tuned": tuned_oof}, n_boot=2000
        )
        delta_sample = boot["tuned"] - boot["default"]
        ci_low, ci_high = ci(delta_sample)
        p_value = two_sided_p(delta_sample)
        default_mean = float(default_repeat["macro_f1"].mean())
        tuned_mean = float(tuned_repeat["macro_f1"].mean())
        rows.append(
            {
                "model": name,
                "default_params": default_params,
                "tuned_params": tuned_params,
                "default_macro_f1": default_mean,
                "tuned_macro_f1": tuned_mean,
                "delta": tuned_mean - default_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
            }
        )
        print(
            f"    {name}: default {default_mean:.3f} -> tuned {tuned_mean:.3f}  "
            f"delta={tuned_mean - default_mean:+.3f}  CI=[{ci_low:+.3f},{ci_high:+.3f}]  "
            f"p={p_value:.3f}  -> {'EXCLUDES 0' if not (ci_low <= 0.0 <= ci_high) else 'overlaps 0'}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=SCRIPT_DIR / "tune_holdout" / "ground_truth_tune.csv",
    )
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument(
        "--select-final",
        action="store_true",
        help="also run the plain whole-tune-set grid search to pick one final value per model",
    )
    parser.add_argument(
        "--holdout-csv",
        type=Path,
        default=None,
        help="one-time confirmatory check: logistic_regression and svm, "
        "default C=1.0 vs the C=0.1 nested-CV winner, on this (never-tuned-"
        "against) split. Only run this for a candidate that already cleared "
        "the tune-set bar.",
    )
    args = parser.parse_args()

    if args.holdout_csv is not None:
        checks = {
            "logistic_regression": (build_logistic_regression, {"C": 1.0}, {"C": 0.1}),
            "svm": (build_svm, {"C": 1.0}, {"C": 0.1}),
        }
        result_df = holdout_check(args.holdout_csv, checks)
        out_dir = SCRIPT_DIR / "tune_hyperparameters_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_dir / "holdout_check.csv", index=False, encoding="utf-8-sig")
        print(f"\nWritten: {out_dir / 'holdout_check.csv'}")
        return

    from News.Build_sentiment_label.Traditional_ML.Common.prepare_ground_truth import (
        load_frame_from_csv,
    )

    frame = load_frame_from_csv(args.ground_truth_csv)
    print(f"Ground truth: {args.ground_truth_csv} ({len(frame)} rows)")
    term_counts = build_document_term_counts(frame)
    y = encode_labels(frame["ground_truth_label"])
    stopwords = load_stopword_set()
    lexicon_matrix = build_lexicon_feature_matrix(frame["Tokenize_content"].tolist())
    label_counts = frame["ground_truth_label"].value_counts().to_dict()
    print("Label counts:", label_counts)

    rows = []
    for name in args.models:
        spec = MODELS[name]
        extra_features = lexicon_matrix if spec["uses_lexicon"] else None

        print(f"\n[{name}] default (single 5-fold CV) ...", flush=True)
        default_probabilities, default_predictions, _fold = run_cross_validation(
            spec["build_default"], term_counts, y, stopwords, extra_features=extra_features
        )
        default_macro_f1 = _macro_f1(y, default_predictions)
        print(f"    default macro_f1={default_macro_f1:.3f}  params={spec['default_params']}")

        print(f"[{name}] nested CV tuning ({N_SPLITS} outer x {INNER_SPLITS} inner) ...", flush=True)
        tuned_probabilities, tuned_predictions, _fold, chosen = nested_cv_tune(
            spec["build_fn"], spec["grid"], term_counts, y, stopwords, extra_features=extra_features
        )
        tuned_macro_f1 = _macro_f1(y, tuned_predictions)
        print(f"    tuned (nested) macro_f1={tuned_macro_f1:.3f}")
        print(f"    params chosen per outer fold: {chosen}")

        # macro_f1_per_repeat (inside bootstrap_samples) expects label-id
        # arrays shaped (n_repeats, n_rows), not probabilities - only one
        # "repeat" here (one nested-CV pass), so wrap each in [None, :].
        oof_by_arm = {
            "default": default_predictions[None, :],
            "tuned": tuned_predictions[None, :],
        }
        boot = bootstrap_samples(np.asarray(y), oof_by_arm, n_boot=2000)
        delta_sample = boot["tuned"] - boot["default"]
        ci_low, ci_high = ci(delta_sample)
        p_value = two_sided_p(delta_sample)
        delta = tuned_macro_f1 - default_macro_f1

        final_params = None
        final_score = None
        if args.select_final:
            print(f"[{name}] final param selection (plain {N_SPLITS}-fold on whole tune set) ...")
            final_params, final_score = select_final_params(
                spec["build_fn"], spec["grid"], term_counts, y, stopwords, extra_features
            )
            print(f"    final params: {final_params} (macro_f1={final_score:.3f})")

        rows.append(
            {
                "model": name,
                "default_params": spec["default_params"],
                "default_macro_f1": default_macro_f1,
                "tuned_macro_f1": tuned_macro_f1,
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "crosses_zero": bool(ci_low <= 0.0 <= ci_high),
                "chosen_params_per_outer_fold": chosen,
                "final_params": final_params,
                "final_macro_f1": final_score,
            }
        )

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("NESTED-CV HYPERPARAMETER TUNING - SUMMARY")
    print("=" * 72)
    for row in rows:
        verdict = "RELIABLE improvement" if (row["delta"] > 0 and not row["crosses_zero"]) else "not distinguishable from noise"
        print(
            f"  {row['model']:<22} default={row['default_macro_f1']:.3f}  "
            f"tuned={row['tuned_macro_f1']:.3f}  delta={row['delta']:+.3f}  "
            f"CI=[{row['ci_low']:+.3f},{row['ci_high']:+.3f}]  p={row['p_value']:.3f}  -> {verdict}"
        )
        if row["final_params"] is not None:
            print(f"      final params (whole-tune-set search): {row['final_params']}")

    out_dir = SCRIPT_DIR / "tune_hyperparameters_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
