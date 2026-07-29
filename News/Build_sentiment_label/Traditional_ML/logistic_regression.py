from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

INPUT_CSR_NPZ_PATH = DATA_DIR / "tfidf_matrix_csr.npz"
INPUT_DOCUMENT_INDEX_PATH = DATA_DIR / "tfidf_document_index.csv"
INPUT_VOCAB_PATH = DATA_DIR / "tfidf_vocabulary.csv"

OUTPUT_METRICS_PATH = DATA_DIR / "logistic_regression_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "logistic_regression_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "logistic_regression_confusion_matrix.csv"
OUTPUT_TOP_FEATURES_PATH = DATA_DIR / "logistic_regression_top_features.csv"

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
EPOCHS = 800
LEARNING_RATE = 0.05
L2_REGULARIZATION = 0.1


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
        raise ValueError("No valid labels found for Logistic Regression.")

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


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((len(labels), num_classes), dtype=np.float64)
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Each class must appear in the training fold.")
    return len(labels) / (num_classes * counts)


def fit_multinomial_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    l2_regularization: float = L2_REGULARIZATION,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    num_samples, num_features = x.shape
    num_classes = len(VALID_LABELS)

    weights = rng.normal(0.0, 0.001, size=(num_features, num_classes))
    bias = np.zeros(num_classes, dtype=np.float64)
    y_one_hot = one_hot(y, num_classes)
    class_weights = compute_class_weights(y, num_classes)
    sample_weights = class_weights[y]
    sample_weight_sum = sample_weights.sum()

    adam_m_w = np.zeros_like(weights)
    adam_v_w = np.zeros_like(weights)
    adam_m_b = np.zeros_like(bias)
    adam_v_b = np.zeros_like(bias)
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    for step in range(1, epochs + 1):
        probabilities = softmax(x @ weights + bias)
        error = (probabilities - y_one_hot) * sample_weights[:, None]

        grad_w = (x.T @ error) / sample_weight_sum
        grad_w += l2_regularization * weights / num_samples
        grad_b = error.sum(axis=0) / sample_weight_sum

        adam_m_w = beta_1 * adam_m_w + (1.0 - beta_1) * grad_w
        adam_v_w = beta_2 * adam_v_w + (1.0 - beta_2) * (grad_w * grad_w)
        adam_m_b = beta_1 * adam_m_b + (1.0 - beta_1) * grad_b
        adam_v_b = beta_2 * adam_v_b + (1.0 - beta_2) * (grad_b * grad_b)

        m_w_hat = adam_m_w / (1.0 - beta_1**step)
        v_w_hat = adam_v_w / (1.0 - beta_2**step)
        m_b_hat = adam_m_b / (1.0 - beta_1**step)
        v_b_hat = adam_v_b / (1.0 - beta_2**step)

        weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return weights, bias


def predict_proba(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return softmax(x @ weights + bias)


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


def cross_validate_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_class_count = int(np.bincount(y, minlength=len(VALID_LABELS)).min())
    n_splits = min(N_SPLITS, min_class_count)
    if n_splits < 2:
        raise ValueError("Each class needs at least 2 rows for cross-validation.")

    probabilities = np.zeros((len(y), len(VALID_LABELS)), dtype=np.float64)
    folds = build_stratified_folds(y, n_splits=n_splits)
    all_indices = np.arange(len(y))

    for fold_id, validation_indices in enumerate(folds, start=1):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[validation_indices] = False
        train_indices = all_indices[train_mask]

        weights, bias = fit_multinomial_logistic_regression(
            x[train_indices],
            y[train_indices],
            random_seed=RANDOM_SEED + fold_id,
        )
        probabilities[validation_indices] = predict_proba(
            x[validation_indices],
            weights,
            bias,
        )
        print(
            f"Fold {fold_id}/{n_splits}: "
            f"train={len(train_indices)}, validation={len(validation_indices)}"
        )

    predictions = probabilities.argmax(axis=1)
    return probabilities, predictions


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


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    weights, bias = fit_multinomial_logistic_regression(
        x,
        y,
        random_seed=RANDOM_SEED + 999,
    )

    rows = []
    for label_id, label in enumerate(VALID_LABELS):
        top_indices = np.argsort(weights[:, label_id])[-top_n:][::-1]
        for rank, feature_index in enumerate(top_indices, start=1):
            vocab_row = vocabulary.iloc[int(feature_index)]
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "selected_feature_id": int(feature_index),
                    "term_id": int(vocab_row["term_id"]),
                    "term": vocab_row["term"],
                    "ngram_n": int(vocab_row["ngram_n"]),
                    "coefficient": float(weights[feature_index, label_id]),
                    "bias": float(bias[label_id]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    x, document_index, vocabulary = load_inputs()
    x_selected, selected_vocabulary, selected_indices = select_top_features(
        x,
        vocabulary,
        max_features=MAX_FEATURES,
    )
    y = encode_labels(document_index["ground_truth_label"])

    print("Input matrix:", INPUT_CSR_NPZ_PATH)
    print("Documents:", x_selected.shape[0])
    print("Original features:", x.shape[1])
    print("Selected features:", x_selected.shape[1])
    print("Label counts:")
    print(document_index["ground_truth_label"].value_counts().to_string())

    probabilities, predictions = cross_validate_logistic_regression(x_selected, y)
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(document_index, probabilities, predictions)

    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    top_features_df = build_top_features(x_selected, y, selected_vocabulary)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(
        OUTPUT_CONFUSION_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    top_features_df.to_csv(OUTPUT_TOP_FEATURES_PATH, index=False, encoding="utf-8-sig")

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print("\nOutput metrics:", OUTPUT_METRICS_PATH)
    print("Output predictions:", OUTPUT_PREDICTIONS_PATH)
    print("Output confusion matrix:", OUTPUT_CONFUSION_MATRIX_PATH)
    print("Output top features:", OUTPUT_TOP_FEATURES_PATH)
    print("Selected feature source ids:", selected_indices[:10].tolist(), "...")


if __name__ == "__main__":
    main()
