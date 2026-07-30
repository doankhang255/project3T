from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.logistic_regression import (
    DATA_DIR,
    INPUT_CSR_NPZ_PATH,
    MAX_FEATURES,
    N_SPLITS,
    RANDOM_SEED,
    VALID_LABELS,
    build_prediction_output,
    build_stratified_folds,
    compute_metrics,
    confusion_matrix,
    encode_labels,
    load_inputs,
    select_top_features,
)


OUTPUT_METRICS_PATH = DATA_DIR / "naive_bayes_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "naive_bayes_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "naive_bayes_confusion_matrix.csv"
OUTPUT_TOP_FEATURES_PATH = DATA_DIR / "naive_bayes_top_features.csv"


def cross_validate_naive_bayes(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((len(y), len(VALID_LABELS)), dtype=np.float64)
    folds = build_stratified_folds(y, n_splits=N_SPLITS)
    all_indices = np.arange(len(y))

    for fold_id, validation_indices in enumerate(folds, start=1):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[validation_indices] = False
        train_indices = all_indices[train_mask]

        model = MultinomialNB()
        model.fit(x[train_indices], y[train_indices])
        assert list(model.classes_) == list(range(len(VALID_LABELS)))
        probabilities[validation_indices] = model.predict_proba(x[validation_indices])
        print(
            f"Fold {fold_id}/{N_SPLITS}: "
            f"train={len(train_indices)}, validation={len(validation_indices)}"
        )

    predictions = probabilities.argmax(axis=1)
    return probabilities, predictions


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    model = MultinomialNB()
    model.fit(x, y)

    rows = []
    for label_id, label in enumerate(VALID_LABELS):
        top_indices = np.argsort(model.feature_log_prob_[label_id])[-top_n:][::-1]
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
                    "feature_log_prob": float(
                        model.feature_log_prob_[label_id, feature_index]
                    ),
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

    print("Model: Multinomial Naive Bayes")
    print("Input matrix:", INPUT_CSR_NPZ_PATH)
    print("Documents:", x_selected.shape[0])
    print("Selected features:", x_selected.shape[1])
    print("Label counts:")
    print(document_index["ground_truth_label"].value_counts().to_string())

    probabilities, predictions = cross_validate_naive_bayes(x_selected, y)
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
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, index=False, encoding="utf-8-sig")
    top_features_df.to_csv(OUTPUT_TOP_FEATURES_PATH, index=False, encoding="utf-8-sig")

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print("\nOutput metrics:", OUTPUT_METRICS_PATH)
    print("Output predictions:", OUTPUT_PREDICTIONS_PATH)
    print("Output confusion matrix:", OUTPUT_CONFUSION_MATRIX_PATH)
    print("Output top features:", OUTPUT_TOP_FEATURES_PATH)


if __name__ == "__main__":
    main()
