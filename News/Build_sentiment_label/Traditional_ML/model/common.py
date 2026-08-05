from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

INPUT_CSR_NPZ_PATH = DATA_DIR / "tfidf_matrix_csr.npz"
INPUT_DOCUMENT_INDEX_PATH = DATA_DIR / "tfidf_document_index.csv"
INPUT_VOCAB_PATH = DATA_DIR / "tfidf_vocabulary.csv"

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
MAX_FEATURES = 5000


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    return LABEL_ALIASES.get(str(value).strip().casefold())


def load_csr_as_dense(path: Path = INPUT_CSR_NPZ_PATH) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"TF-IDF CSR matrix not found: {path}")

    csr_data = np.load(path)
    data = csr_data["data"]
    indices = csr_data["indices"]
    indptr = csr_data["indptr"]
    shape = tuple(int(value) for value in csr_data["shape"])

    dense = np.zeros(shape, dtype=np.float32)
    for row_index in range(shape[0]):
        start = indptr[row_index]
        end = indptr[row_index + 1]
        dense[row_index, indices[start:end]] = data[start:end]
    return dense


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    x = load_csr_as_dense(INPUT_CSR_NPZ_PATH)
    document_index = pd.read_csv(INPUT_DOCUMENT_INDEX_PATH, encoding="utf-8-sig")
    vocabulary = pd.read_csv(INPUT_VOCAB_PATH, encoding="utf-8-sig")

    if len(document_index) != x.shape[0]:
        raise ValueError(
            "Document index row count does not match matrix row count: "
            f"{len(document_index)} != {x.shape[0]}"
        )
    if len(vocabulary) != x.shape[1]:
        raise ValueError(
            "Vocabulary row count does not match matrix column count: "
            f"{len(vocabulary)} != {x.shape[1]}"
        )
    if LABEL_COLUMN not in document_index.columns:
        raise ValueError(f"Document index is missing label column: {LABEL_COLUMN!r}")

    document_index = document_index.copy()
    document_index["ground_truth_label"] = document_index[LABEL_COLUMN].apply(
        normalize_label
    )
    valid_mask = document_index["ground_truth_label"].isin(VALID_LABELS).to_numpy()
    if not valid_mask.any():
        raise ValueError("No valid labels found for Traditional ML models.")

    return x[valid_mask], document_index.loc[valid_mask].reset_index(drop=True), vocabulary


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
    document_index: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    out = document_index.copy()
    out["predicted_label"] = [VALID_LABELS[index] for index in predictions]
    for label_id, label in enumerate(VALID_LABELS):
        out[f"prob_{label}"] = probabilities[:, label_id]
    out["sentiment_score_ml"] = out["prob_positive"] - out["prob_negative"]
    out["is_correct"] = out["ground_truth_label"].eq(out["predicted_label"])
    return out
