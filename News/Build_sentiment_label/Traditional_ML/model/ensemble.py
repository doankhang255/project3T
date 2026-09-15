"""Average-probability ensemble of logistic_regression + naive_bayes +
random_forest (IMPROVEMENTS.md section D).

Each base model already runs the same leak-free 5-fold CV
(``run_cross_validation``) with folds determined only by ``y`` and
``RANDOM_SEED`` - identical across models regardless of which estimator is
passed in. So calling it once per base model and averaging the three
out-of-fold probability matrices row-wise is a valid paired average, with no
new CV wiring needed here.

``svm`` is excluded: since ``model/svm.py`` dropped Platt calibration (see
its module docstring / ``improve/calibration_check.py``), its
``predict_proba`` is a margin-softmax, not a calibrated probability -
averaging it in would let an arbitrarily-scaled number pull the ensemble
mean off of what the other three models actually believe.

``random_forest``'s member run also gets the lexicon feature block (see
``model/lexicon_features.py``), same as its standalone run in
``model/random_forest.py`` - passed through ``extra_features_by_model`` so
the ensemble does not silently score a different (TF-IDF-only) version of RF
than what ``random_forest_metrics.csv`` reports.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.model.common import (
    DATA_DIR,
    VALID_LABELS,
    build_prediction_output,
    compute_metrics,
    confusion_matrix,
    encode_labels,
    load_ground_truth_frame,
    run_cross_validation,
)
from News.Build_sentiment_label.Traditional_ML.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.model.naive_bayes import (
    build_estimator as build_naive_bayes,
)
from News.Build_sentiment_label.Traditional_ML.model.random_forest import (
    build_estimator as build_random_forest,
)
from News.Build_sentiment_label.Traditional_ML.model.lexicon_features import (
    build_lexicon_feature_matrix,
)


OUTPUT_METRICS_PATH = DATA_DIR / "ensemble_metrics.csv"
OUTPUT_PREDICTIONS_PATH = DATA_DIR / "ensemble_predictions.csv"
OUTPUT_CONFUSION_MATRIX_PATH = DATA_DIR / "ensemble_confusion_matrix.csv"

MEMBER_FACTORIES = {
    "logistic_regression": build_logistic_regression,
    "naive_bayes": build_naive_bayes,
    "random_forest": build_random_forest,
}


def run_ensemble_cv(
    term_counts,
    y: np.ndarray,
    stopwords: set[str] | None = None,
    extra_features_by_model: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run each member's own leak-free CV, then average the probabilities.

    ``extra_features_by_model``: optional ``{model_name: extra_features}`` -
    lets a specific member (e.g. ``random_forest``, which now takes the
    lexicon feature block - see ``model/random_forest.py``) use the same
    ``extra_features`` it uses standalone, so the ensemble's RF component
    matches its standalone behaviour instead of silently reverting to
    TF-IDF-only. Members with no entry get ``extra_features=None``.

    Returns the same ``(probabilities, predictions, fold_of_row)`` shape as
    ``run_cross_validation`` so it drops into ``build_prediction_output``
    unchanged.
    """
    extra_features_by_model = extra_features_by_model or {}
    member_probabilities = []
    fold_of_row_reference = None
    for name, factory in MEMBER_FACTORIES.items():
        probabilities, _predictions, fold_of_row = run_cross_validation(
            factory,
            term_counts,
            y,
            stopwords,
            extra_features=extra_features_by_model.get(name),
        )
        member_probabilities.append(probabilities)
        if fold_of_row_reference is None:
            fold_of_row_reference = fold_of_row
        elif not np.array_equal(fold_of_row_reference, fold_of_row):
            raise AssertionError(
                f"{name}'s validation folds differ from the first member's - "
                "ensemble members must share the same fold assignment for "
                "the average to be a paired average."
            )

    averaged_probabilities = np.mean(member_probabilities, axis=0)
    predictions = averaged_probabilities.argmax(axis=1)
    return averaged_probabilities, predictions, fold_of_row_reference


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    lexicon_matrix = build_lexicon_feature_matrix(df["Tokenize_content"].tolist())

    print(f"Model: Ensemble (average probability of {', '.join(MEMBER_FACTORIES)})")
    print("svm excluded - its predict_proba is an uncalibrated margin-softmax")
    print("random_forest member uses +lexicon features, matching its standalone run")
    print("Documents:", len(df))
    print("Label counts:")
    print(df["ground_truth_label"].value_counts().to_string())

    probabilities, predictions, fold_of_row = run_ensemble_cv(
        term_counts, y, extra_features_by_model={"random_forest": lexicon_matrix}
    )
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(df, probabilities, predictions, fold_of_row)

    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{label}" for label in VALID_LABELS],
        columns=[f"pred_{label}" for label in VALID_LABELS],
    ).reset_index(names="true_label")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION_MATRIX_PATH, index=False, encoding="utf-8-sig")

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))
    print("\nConfusion matrix:")
    print(confusion_df.to_string(index=False))
    print("\nOutput metrics:", OUTPUT_METRICS_PATH)
    print("Output predictions:", OUTPUT_PREDICTIONS_PATH)
    print("Output confusion matrix:", OUTPUT_CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()
