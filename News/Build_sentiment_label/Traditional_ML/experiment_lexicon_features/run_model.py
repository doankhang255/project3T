"""Production run: random_forest and ensemble - both get the lexicon feature
block (random_forest directly, ensemble through its random_forest member).

    python News/Build_sentiment_label/Traditional_ML/experiment_Lexicon_features/run_model.py
    python .../run_model.py --models random_forest   # subset

Writes the CSVs run_pipeline.py / compare_models.py consume. Model/ensemble
definitions (build_estimator, run_ensemble_cv) live in
``../Common/model/{random_forest,ensemble}.py`` and stay untouched - this is
the one place that computes the lexicon feature matrix and wires it in via
``extra_features``. To try a different feature block (or a different model
here), edit ``build_feature_matrix()`` / the per-model run function below and
wire it into the imported model - not ``Common/model/*.py`` (shared with
experiment_only_TF_IDF/).
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
from News.Build_sentiment_label.Traditional_ML.Common.model import ensemble, random_forest
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    build_lexicon_feature_matrix,
)

MODEL_NAMES = ["random_forest", "ensemble"]


def build_feature_matrix(df: pd.DataFrame):
    """The one place this folder's runs add a feature on top of TF-IDF.
    Swap this out (and the ``extra_features=`` calls below) to try a
    different feature block without touching Common/model/*.py."""
    return build_lexicon_feature_matrix(df["Tokenize_content"].tolist())


def _write_and_report(name: str, metrics_df, prediction_df, confusion_df, top_features_df=None) -> None:
    metrics_path = DATA_DIR / f"{name}_metrics.csv"
    predictions_path = DATA_DIR / f"{name}_predictions.csv"
    confusion_path = DATA_DIR / f"{name}_confusion_matrix.csv"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    prediction_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(confusion_path, index=False, encoding="utf-8-sig")

    print("Metrics:")
    print(metrics_df.to_string(index=False))
    print("Confusion matrix:")
    print(confusion_df.to_string(index=False))
    print("Output metrics:", metrics_path)
    print("Output predictions:", predictions_path)
    print("Output confusion matrix:", confusion_path)

    if top_features_df is not None:
        top_features_path = DATA_DIR / f"{name}_top_features.csv"
        top_features_df.to_csv(top_features_path, index=False, encoding="utf-8-sig")
        print("Output top features:", top_features_path)


def run_random_forest(df, term_counts, y, lexicon_matrix) -> None:
    print("\nModel: Random Forest (isotonic-calibrated probabilities, +lexicon features)")
    print("Documents:", len(df))
    print("Trees:", random_forest.N_ESTIMATORS)

    probabilities, predictions, fold_of_row = run_cross_validation(
        random_forest.build_estimator, term_counts, y, extra_features=lexicon_matrix
    )
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(df, probabilities, predictions, fold_of_row)
    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{lbl}" for lbl in VALID_LABELS],
        columns=[f"pred_{lbl}" for lbl in VALID_LABELS],
    ).reset_index(names="true_label")

    # TF-IDF vocabulary only (no lexicon columns) - top_features maps each
    # row to a vocabulary term, which the 9 lexicon columns don't have.
    x_full, vocabulary_full = build_full_fit_features(term_counts)
    top_features_df = random_forest.build_top_features(x_full, y, vocabulary_full)

    _write_and_report("random_forest", metrics_df, prediction_df, confusion_df, top_features_df)


def run_ensemble(df, term_counts, y, lexicon_matrix) -> None:
    print(f"\nModel: Ensemble (average probability of {', '.join(ensemble.MEMBER_FACTORIES)})")
    print("svm excluded - its predict_proba is an uncalibrated margin-softmax")
    print("random_forest member uses +lexicon features, matching its standalone run")
    print("Documents:", len(df))

    probabilities, predictions, fold_of_row = ensemble.run_ensemble_cv(
        term_counts, y, extra_features_by_model={"random_forest": lexicon_matrix}
    )
    metrics_df = compute_metrics(y, predictions)
    prediction_df = build_prediction_output(df, probabilities, predictions, fold_of_row)
    confusion_df = pd.DataFrame(
        confusion_matrix(y, predictions),
        index=[f"true_{lbl}" for lbl in VALID_LABELS],
        columns=[f"pred_{lbl}" for lbl in VALID_LABELS],
    ).reset_index(names="true_label")

    _write_and_report("ensemble", metrics_df, prediction_df, confusion_df)


RUNNERS = {
    "random_forest": run_random_forest,
    "ensemble": run_ensemble,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="subset to run (default: both)",
    )
    args = parser.parse_args()

    df = load_ground_truth_frame()
    term_counts = build_document_term_counts(df)
    y = encode_labels(df["ground_truth_label"])
    lexicon_matrix = build_feature_matrix(df)
    print("Documents:", len(df))
    print("Label counts:")
    print(df["ground_truth_label"].value_counts().to_string())
    print("Lexicon feature matrix:", lexicon_matrix.shape)

    for name in args.models:
        RUNNERS[name](df, term_counts, y, lexicon_matrix)


if __name__ == "__main__":
    main()
