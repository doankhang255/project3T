"""Production run: logistic_regression, naive_bayes, svm - TF-IDF only.

    python News/Build_sentiment_label/Traditional_ML/experiment_only_TF_IDF/run_model.py
    python .../run_model.py --models svm naive_bayes   # subset

Writes the CSVs run_pipeline.py / compare_models.py consume (one run of
leak-free CV per model: metrics / predictions / confusion_matrix /
top_features). Model definitions (build_estimator, build_top_features) live
in ``../Common/model/*.py`` and stay untouched - this file is only the
runnable step. To add a feature block or change how a model is scored here,
edit ``run_one()`` below and wire it into the imported model, rather than
editing ``Common/model/*.py`` (which is shared - the lexicon experiment and
any future one also import from it).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    DATA_DIR,
    VALID_LABELS,
    build_full_fit_features,
    build_prediction_output,
    compute_metrics,
    confusion_matrix,
    encode_labels,
    load_ground_truth_frame,
    run_cross_validation,
)
from News.Build_sentiment_label.Traditional_ML.Common.model import (
    logistic_regression,
    naive_bayes,
    svm,
)

# name -> (build_estimator, build_top_features, display label). Add a new
# TF-IDF-only model here (and to Common/model/) to have it run + reported
# alongside these three - no other file needs to change.
MODEL_SPECS = {
    "logistic_regression": (
        logistic_regression.build_estimator,
        logistic_regression.build_top_features,
        "Logistic Regression (scikit-learn)",
    ),
    "naive_bayes": (
        naive_bayes.build_estimator,
        naive_bayes.build_top_features,
        "Multinomial Naive Bayes",
    ),
    "svm": (
        svm.build_estimator,
        svm.build_top_features,
        "Linear SVM (LinearSVC, margin-softmax - NOT a calibrated probability)",
    ),
}


def run_one(name, build_estimator, build_top_features_fn, display_label, df, term_counts, y) -> None:
    print(f"\nModel: {display_label}")
    print("Documents:", len(df))

    probabilities, predictions, fold_of_row = run_cross_validation(
        build_estimator, term_counts, y
    )
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(df, probabilities, predictions, fold_of_row)
    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{lbl}" for lbl in VALID_LABELS],
        columns=[f"pred_{lbl}" for lbl in VALID_LABELS],
    ).reset_index(names="true_label")

    x_full, vocabulary_full = build_full_fit_features(term_counts)
    top_features_df = build_top_features_fn(x_full, y, vocabulary_full)

    metrics_path = DATA_DIR / f"{name}_metrics.csv"
    predictions_path = DATA_DIR / f"{name}_predictions.csv"
    confusion_path = DATA_DIR / f"{name}_confusion_matrix.csv"
    top_features_path = DATA_DIR / f"{name}_top_features.csv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    top_features_df.to_csv(top_features_path, index=False, encoding="utf-8-sig")

    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print("Confusion matrix:")
    print(confusion_df.to_string(index=False))
    print("Output metrics:", metrics_path)
    print("Output predictions:", predictions_path)
    print("Output confusion matrix:", confusion_path)
    print("Output top features:", top_features_path)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="subset of models to run (default: all)",
    )
    args = parser.parse_args()

    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    print("Documents:", len(df))
    print("Label counts:")
    print(df["ground_truth_label"].value_counts().to_string())

    for name in args.models:
        build_estimator, build_top_features_fn, display_label = MODEL_SPECS[name]
        run_one(name, build_estimator, build_top_features_fn, display_label, df, term_counts, y)


if __name__ == "__main__":
    main()
