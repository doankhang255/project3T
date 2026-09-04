from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.model.common import (
    DATA_DIR,
    RANDOM_SEED,
    VALID_LABELS,
    build_full_fit_features,
    build_prediction_output,
    compute_metrics,
    confusion_matrix,
    encode_labels,
    load_ground_truth_frame,
    run_cross_validation,
)


OUTPUT_METRICS_PATH = DATA_DIR / "logistic_regression_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "logistic_regression_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "logistic_regression_confusion_matrix.csv"
OUTPUT_TOP_FEATURES_PATH = DATA_DIR / "logistic_regression_top_features.csv"

LOGISTIC_C = 1.0
MAX_ITER = 5000


def build_estimator(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        max_iter=MAX_ITER,
        class_weight="balanced",
        random_state=random_state,
    )


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    model = build_estimator(RANDOM_SEED + 999)
    model.fit(x, y)
    assert list(model.classes_) == list(range(len(VALID_LABELS)))

    rows = []
    for label_id, label in enumerate(VALID_LABELS):
        top_indices = np.argsort(model.coef_[label_id])[-top_n:][::-1]
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
                    "coefficient": float(model.coef_[label_id, feature_index]),
                    "bias": float(model.intercept_[label_id]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])

    print("Model: Logistic Regression (scikit-learn)")
    print("Documents:", len(df))
    print("Label counts:")
    print(df["ground_truth_label"].value_counts().to_string())

    probabilities, predictions, fold_of_row = run_cross_validation(
        build_estimator, term_counts, y
    )
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(df, probabilities, predictions, fold_of_row)

    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    x_full, vocabulary_full = build_full_fit_features(term_counts)
    top_features_df = build_top_features(x_full, y, vocabulary_full)

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
