from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Common.stopword_utils import (
    DEFAULT_STOPWORDS_PATH,
    load_stopwords,
)
from News.Build_sentiment_label.Traditional_ML.TF_IDF import (
    REMOVE_STOPWORDS,
    build_document_term_counts,
    fit_tfidf_vocabulary,
    load_tokenized_ground_truth as _load_tokenized_ground_truth,
    transform_tfidf,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

LABEL_COLUMN = "sentiment"
VALID_LABELS = ["negative", "neutral", "positive"]
LABEL_ALIASES = {
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "pos": "positive",
    "positive": "positive",
}

N_SPLITS = 5
RANDOM_SEED = 42

# Cap kept on purpose: the ground-truth set is expected to grow to a few
# thousand rows, at which point the fitted vocabulary can exceed this and the
# top-feature cut starts doing real work. Below the cap every feature is kept,
# so today it is a no-op. The cut is applied per fold on the training rows
# only (see run_cross_validation) so it never leaks the held-out fold.
MAX_FEATURES = 5000


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return LABEL_ALIASES.get(str(value).strip().casefold())


def load_ground_truth_frame() -> pd.DataFrame:
    df = _load_tokenized_ground_truth()
    df = df.copy()
    df["ground_truth_label"] = df[LABEL_COLUMN].apply(normalize_label)
    valid_mask = df["ground_truth_label"].isin(VALID_LABELS)
    if not valid_mask.any():
        raise ValueError("No valid labels found for Traditional ML models.")
    return df.loc[valid_mask].reset_index(drop=True)


def encode_labels(labels: pd.Series) -> np.ndarray:
    label_to_id = {label: index for index, label in enumerate(VALID_LABELS)}
    return labels.map(label_to_id).to_numpy(dtype=int)


def build_stratified_folds(labels: np.ndarray, n_splits: int = N_SPLITS) -> list[np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    folds = [[] for _ in range(n_splits)]

    for label_id in range(len(VALID_LABELS)):
        label_indices = np.flatnonzero(labels == label_id)
        rng.shuffle(label_indices)
        for position, row_index in enumerate(label_indices):
            folds[position % n_splits].append(int(row_index))

    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def resolve_n_splits(y: np.ndarray, n_splits: int = N_SPLITS) -> int:
    """One rule for every model (used to differ: LR shrank n_splits, the rest
    hardcoded 5). Cannot have more folds than the smallest class has rows.
    """
    min_class_count = int(np.bincount(y, minlength=len(VALID_LABELS)).min())
    resolved = min(n_splits, min_class_count)
    if resolved < 2:
        raise ValueError(
            "Each class needs at least 2 rows for cross-validation; "
            f"class counts = {np.bincount(y, minlength=len(VALID_LABELS)).tolist()}"
        )
    return resolved


def select_top_features(
    x: np.ndarray,
    vocabulary: pd.DataFrame,
    max_features: int = MAX_FEATURES,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if x.shape[1] <= max_features:
        selected_indices = np.arange(x.shape[1], dtype=int)
    else:
        nonzero_df = (x > 0).sum(axis=0)
        total_weight = x.sum(axis=0)
        feature_rank = np.lexsort((-total_weight, -nonzero_df))
        selected_indices = np.sort(feature_rank[:max_features])

    selected_vocabulary = vocabulary.iloc[selected_indices].reset_index(drop=True).copy()
    selected_vocabulary["selected_feature_id"] = np.arange(len(selected_vocabulary))
    return x[:, selected_indices], selected_vocabulary, selected_indices


def _load_stopwords(stopwords: set[str] | None) -> set[str]:
    if stopwords is not None:
        return stopwords
    return load_stopwords(DEFAULT_STOPWORDS_PATH) if REMOVE_STOPWORDS else set()


def run_cross_validation(
    estimator_factory,
    term_counts_by_document: list[Counter[str]],
    y: np.ndarray,
    stopwords: set[str] | None = None,
    n_splits: int = N_SPLITS,
    max_features: int = MAX_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    stopwords = _load_stopwords(stopwords)
    resolved_splits = resolve_n_splits(y, n_splits)

    n_rows = len(y)
    probabilities = np.zeros((n_rows, len(VALID_LABELS)), dtype=np.float64)
    fold_of_row = np.full(n_rows, -1, dtype=int)
    folds = build_stratified_folds(y, n_splits=resolved_splits)
    all_indices = np.arange(n_rows)

    for fold_id, validation_indices in enumerate(folds, start=1):
        fold_of_row[validation_indices] = fold_id
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[validation_indices] = False
        train_indices = all_indices[train_mask]

        train_counts = [term_counts_by_document[i] for i in train_indices]
        val_counts = [term_counts_by_document[i] for i in validation_indices]

        vocabulary_df = fit_tfidf_vocabulary(
            train_counts,
            total_documents=len(train_indices),
            stopwords=stopwords,
        )
        x_train, _ = transform_tfidf(train_counts, vocabulary_df)
        x_val, _ = transform_tfidf(val_counts, vocabulary_df)

        x_train_selected, _, selected_indices = select_top_features(
            x_train, vocabulary_df, max_features=max_features
        )
        x_val_selected = x_val[:, selected_indices]

        model = estimator_factory(RANDOM_SEED + fold_id)
        model.fit(x_train_selected, y[train_indices])
        if list(model.classes_) != list(range(len(VALID_LABELS))):
            raise AssertionError(
                f"Fold {fold_id}: unexpected class order {list(model.classes_)}"
            )
        probabilities[validation_indices] = model.predict_proba(x_val_selected)
        print(
            f"Fold {fold_id}/{resolved_splits}: train={len(train_indices)}, "
            f"validation={len(validation_indices)}, vocab={len(vocabulary_df)}, "
            f"features={x_train_selected.shape[1]}"
        )

    if (fold_of_row < 0).any():
        raise AssertionError("Some rows were never assigned to a validation fold.")

    predictions = probabilities.argmax(axis=1)
    return probabilities, predictions, fold_of_row


def build_full_fit_features(
    term_counts_by_document: list[Counter[str]],
    stopwords: set[str] | None = None,
    max_features: int = MAX_FEATURES,
) -> tuple[np.ndarray, pd.DataFrame]:
    stopwords = _load_stopwords(stopwords)
    vocabulary_df = fit_tfidf_vocabulary(
        term_counts_by_document,
        total_documents=len(term_counts_by_document),
        stopwords=stopwords,
    )
    x, _ = transform_tfidf(term_counts_by_document, vocabulary_df)
    x_selected, selected_vocabulary, _ = select_top_features(
        x, vocabulary_df, max_features=max_features
    )
    return x_selected, selected_vocabulary


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(VALID_LABELS), len(VALID_LABELS)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[true_label, pred_label] += 1
    return matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    f1_values = []
    for label_id, label in enumerate(VALID_LABELS):
        true_positive = int(((y_true == label_id) & (y_pred == label_id)).sum())
        false_positive = int(((y_true != label_id) & (y_pred == label_id)).sum())
        false_negative = int(((y_true == label_id) & (y_pred != label_id)).sum())
        support = int((y_true == label_id).sum())

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1_score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1_values.append(f1_score)
        rows.append(
            {
                "metric_scope": label,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "support": support,
            }
        )

    rows.append(
        {
            "metric_scope": "overall",
            "precision": np.nan,
            "recall": np.nan,
            "f1": float(np.mean(f1_values)),
            "support": int(len(y_true)),
            "accuracy": float((y_true == y_pred).mean()),
        }
    )
    return pd.DataFrame(rows)


def build_prediction_output(
    document_frame: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    fold_of_row: np.ndarray | None = None,
) -> pd.DataFrame:
    keep_columns = [
        column
        for column in ["id", "source_row_id", "title", "ground_truth_label"]
        if column in document_frame.columns
    ]
    out = document_frame[keep_columns].reset_index(drop=True).copy()
    if fold_of_row is not None:
        out["validation_fold"] = np.asarray(fold_of_row, dtype=int)
    out["predicted_label"] = [VALID_LABELS[index] for index in predictions]
    for label_id, label in enumerate(VALID_LABELS):
        out[f"prob_{label}"] = probabilities[:, label_id]
    out["sentiment_score_ml"] = out["prob_positive"] - out["prob_negative"]
    if "ground_truth_label" in out.columns:
        out["is_correct"] = out["ground_truth_label"].eq(out["predicted_label"])
    return out
