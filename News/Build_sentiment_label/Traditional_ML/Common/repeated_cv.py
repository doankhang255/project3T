"""M2.2 - repeated stratified 5-fold CV for the Traditional_ML models.

Self-contained, in the spirit of ``../experiment_vocab/``: it only *imports*
pure helpers from the main pipeline (TF-IDF fit/transform, the leak-free
feature recipe, the metric functions) and does its own CV wiring so the one
added knob - the number of repeats - is explicit. It does NOT edit or
monkeypatch ``model/common.py``, ``TF_IDF.py`` or ``model/*.py``.

Why: a single 5-fold CV run moves by ~0.01-0.02 macro-F1 with the split
alone, so one value cannot tell two models apart. This runs the whole
leak-free 5-fold CV ``n_repeats`` times with a different split each time and
reports mean +/- std. It is also THE production CV: ``production_cv.py``
runs it once per model and persists the out-of-fold probabilities, which
every comparison/calibration script then reuses instead of re-running CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)
from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import (
    REMOVE_STOPWORDS,
    fit_tfidf_vocabulary,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    MAX_FEATURES,
    RANDOM_SEED,
    VALID_LABELS,
    compute_metrics,
    resolve_n_splits,
    select_top_features,
)

N_SPLITS = 5
N_REPEATS = 10


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """Round-robin stratification identical to
    ``model/common.build_stratified_folds``, but the RNG seed is an argument so
    the CV can be repeated. ``seed == RANDOM_SEED`` reproduces the main-pipeline
    folds exactly (same ``np.random.default_rng`` call sequence).
    """
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label_id in range(len(VALID_LABELS)):
        label_indices = np.flatnonzero(y == label_id)
        rng.shuffle(label_indices)
        for position, row_index in enumerate(label_indices):
            folds[position % n_splits].append(int(row_index))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def run_single_cv(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    fold_seed: int,
    n_splits: int = N_SPLITS,
    max_features: int = MAX_FEATURES,
    extra_features: np.ndarray | None = None,
) -> np.ndarray:
    """One leak-free 5-fold CV pass -> out-of-fold predicted label id per row
    (argmax of ``run_single_cv_proba``)."""
    return run_single_cv_proba(
        estimator_factory,
        term_counts,
        y,
        stopwords,
        fold_seed,
        n_splits=n_splits,
        max_features=max_features,
        extra_features=extra_features,
    ).argmax(axis=1)


def run_single_cv_proba(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    fold_seed: int,
    n_splits: int = N_SPLITS,
    max_features: int = MAX_FEATURES,
    extra_features: np.ndarray | None = None,
) -> np.ndarray:
    """One leak-free 5-fold CV pass -> out-of-fold class probabilities,
    shape ``(n_rows, n_labels)``.

    TF-IDF vocabulary / idf / top-feature cut are fit on the training rows of
    each fold only, exactly like ``model/common.run_cross_validation``.

    ``extra_features`` (optional): a ``(n_rows, k)`` array of precomputed,
    non-fitted feature columns (e.g. lexicon-category hit ratios) to hstack
    onto the TF-IDF matrix after feature selection. "Precomputed" means the
    *values* are not fit from the data (no leakage risk in that sense) - but
    they are still standardized (mean/std) using the TRAIN fold only before
    being hstacked, same discipline as the TF-IDF idf: without this, columns
    on a much smaller scale than the TF-IDF weights (e.g. proportions in
    [0, 0.06] next to TF-IDF weights of ~0.5) get an effectively-zero
    coefficient from any linear model with fixed regularization, silently
    contributing nothing. Default ``None`` keeps this function byte-identical
    to the pre-existing TF-IDF-only behaviour.
    """
    y = np.asarray(y, dtype=int)
    n_splits = resolve_n_splits(y, n_splits)
    n_rows = len(y)
    probabilities = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    all_indices = np.arange(n_rows)

    for fold_id, validation_indices in enumerate(
        stratified_folds(y, n_splits, fold_seed), start=1
    ):
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[validation_indices] = False
        train_indices = all_indices[train_mask]

        train_counts = [term_counts[i] for i in train_indices]
        val_counts = [term_counts[i] for i in validation_indices]

        vocabulary_df = fit_tfidf_vocabulary(
            train_counts, total_documents=len(train_indices), stopwords=stopwords
        )
        x_train, _ = transform_tfidf(train_counts, vocabulary_df)
        x_val, _ = transform_tfidf(val_counts, vocabulary_df)
        x_train_selected, _, selected_indices = select_top_features(
            x_train, vocabulary_df, max_features=max_features
        )
        x_val_selected = x_val[:, selected_indices]

        if extra_features is not None:
            train_extra = extra_features[train_indices]
            val_extra = extra_features[validation_indices]
            extra_mean = train_extra.mean(axis=0)
            extra_std = train_extra.std(axis=0)
            extra_std = np.where(extra_std > 1e-8, extra_std, 1.0)  # guard constant columns
            x_train_selected = np.hstack(
                [x_train_selected, (train_extra - extra_mean) / extra_std]
            )
            x_val_selected = np.hstack(
                [x_val_selected, (val_extra - extra_mean) / extra_std]
            )

        model = estimator_factory(RANDOM_SEED + fold_seed * 100 + fold_id)
        model.fit(x_train_selected, y[train_indices])
        if list(model.classes_) != list(range(len(VALID_LABELS))):
            raise AssertionError(
                f"fold {fold_id} (seed {fold_seed}): estimator class order "
                f"{list(model.classes_)} != {list(range(len(VALID_LABELS)))} - "
                "a training fold is missing a class; predict_proba columns would "
                "misalign (same guard as model/common.run_cross_validation)"
            )
        probabilities[validation_indices] = model.predict_proba(x_val_selected)

    return probabilities


def run_repeated_cv_proba(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    extra_features: np.ndarray | None = None,
) -> np.ndarray:
    """``n_repeats`` independent CV passes (fold seed = repeat index) ->
    out-of-fold probabilities, shape ``(n_repeats, n_rows, n_labels)``.
    Repeat ``r`` uses the same split for every model, so these are paired
    across models."""
    y = np.asarray(y, dtype=int)
    return np.stack(
        [
            run_single_cv_proba(
                estimator_factory,
                term_counts,
                y,
                stopwords,
                fold_seed=repeat_id,
                n_splits=n_splits,
                extra_features=extra_features,
            )
            for repeat_id in range(n_repeats)
        ]
    )


def per_repeat_metrics(y: np.ndarray, oof_predictions: np.ndarray) -> pd.DataFrame:
    """One row per repeat, columns
    ``repeat / macro_f1 / accuracy / f1_negative / f1_neutral / f1_positive``."""
    y = np.asarray(y, dtype=int)
    rows: list[dict] = []
    for repeat_id, predictions in enumerate(oof_predictions):
        metrics = compute_metrics(y, predictions)
        overall = metrics.loc[metrics["metric_scope"].eq("overall")].iloc[0]
        per_class = metrics.set_index("metric_scope")["f1"]
        rows.append(
            {
                "repeat": repeat_id,
                "macro_f1": float(overall["f1"]),
                "accuracy": float(overall["accuracy"]),
                "f1_negative": float(per_class["negative"]),
                "f1_neutral": float(per_class["neutral"]),
                "f1_positive": float(per_class["positive"]),
            }
        )
    return pd.DataFrame(rows)


def run_repeated_cv(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    extra_features: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """``n_repeats`` independent CV passes (fold seed = repeat index).

    ``extra_features``: see ``run_single_cv`` - passed straight through,
    ``None`` by default (unchanged behaviour for every existing caller).

    Returns
      - per_repeat_df : see ``per_repeat_metrics``
      - oof_predictions : shape ``(n_repeats, n_rows)``, out-of-fold label id of
        each row in each repeat. Repeat ``r`` uses the same split for every
        model, so these are paired across models for the McNemar test.
    """
    probabilities = run_repeated_cv_proba(
        estimator_factory,
        term_counts,
        y,
        stopwords,
        n_repeats=n_repeats,
        n_splits=n_splits,
        extra_features=extra_features,
    )
    oof_predictions = probabilities.argmax(axis=2)
    return per_repeat_metrics(y, oof_predictions), oof_predictions


def load_stopword_set(stopwords: set[str] | None = None) -> set[str]:
    if stopwords is not None:
        return stopwords
    return load_stopwords(DEFAULT_STOPWORDS_PATH) if REMOVE_STOPWORDS else set()
