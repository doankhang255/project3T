"""M2.2 - repeated stratified 5-fold CV for the Traditional_ML models.

Self-contained, in the spirit of ``../experiment_vocab/``: it only *imports*
pure helpers from the main pipeline (TF-IDF fit/transform, the leak-free
feature recipe, the metric functions) and does its own CV wiring so the one
added knob - the number of repeats - is explicit. It does NOT edit or
monkeypatch ``model/common.py``, ``TF_IDF.py`` or ``model/*.py``.

Why: the single-run pipeline reports one macro-F1 per model. With 152 rows the
stratified split alone moves that number by ~0.02, so a single value cannot
tell two models apart. This runs the whole leak-free 5-fold CV ``n_repeats``
times with a different split each time and reports mean +/- std.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)
from News.Build_sentiment_label.Traditional_ML.TF_IDF import (
    REMOVE_STOPWORDS,
    fit_tfidf_vocabulary,
    transform_tfidf,
)
from News.Build_sentiment_label.Traditional_ML.model.common import (
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
) -> np.ndarray:
    """One leak-free 5-fold CV pass -> out-of-fold predicted label id per row.

    TF-IDF vocabulary / idf / top-feature cut are fit on the training rows of
    each fold only, exactly like ``model/common.run_cross_validation``.
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

    return probabilities.argmax(axis=1)


def run_repeated_cv(
    estimator_factory,
    term_counts,
    y: np.ndarray,
    stopwords: set[str],
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """``n_repeats`` independent CV passes (fold seed = repeat index).

    Returns
      - per_repeat_df : one row per repeat, columns
        ``repeat / macro_f1 / accuracy / f1_negative / f1_neutral / f1_positive``
      - oof_predictions : shape ``(n_repeats, n_rows)``, out-of-fold label id of
        each row in each repeat. Repeat ``r`` uses the same split for every
        model, so these are paired across models for the McNemar test.
    """
    y = np.asarray(y, dtype=int)
    rows: list[dict] = []
    oof_predictions = np.zeros((n_repeats, len(y)), dtype=int)

    for repeat_id in range(n_repeats):
        predictions = run_single_cv(
            estimator_factory,
            term_counts,
            y,
            stopwords,
            fold_seed=repeat_id,
            n_splits=n_splits,
        )
        oof_predictions[repeat_id] = predictions
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

    return pd.DataFrame(rows), oof_predictions


def load_stopword_set(stopwords: set[str] | None = None) -> set[str]:
    if stopwords is not None:
        return stopwords
    return load_stopwords(DEFAULT_STOPWORDS_PATH) if REMOVE_STOPWORDS else set()
