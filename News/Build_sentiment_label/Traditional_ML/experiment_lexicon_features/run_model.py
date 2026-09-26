"""Production run: random_forest and ensemble - both get the lexicon feature
block (random_forest directly, ensemble through its random_forest member).

    python News/Build_sentiment_label/Traditional_ML/experiment_Lexicon_features/run_model.py
    python .../run_model.py --models random_forest   # subset

Writes the CSVs run_pipeline.py / compare_models.py consume, from one run of
leak-free 10 x 5-fold repeated CV (see ``Common/production_cv.py``). Model /
ensemble definitions live in ``../Common/model/{random_forest,ensemble}.py``
and stay untouched - this is the one place that computes the lexicon
feature matrix and wires it in via ``extra_features``. To try a different
feature block, edit ``build_feature_matrix()`` below - not
``Common/model/*.py`` (shared with experiment_only_TF_IDF/).

The ensemble has no CV of its own: it averages the saved out-of-fold
probabilities of logistic_regression + naive_bayes (from
``experiment_only_TF_IDF/run_model.py``, which run_pipeline.py runs first)
and random_forest (from this file's own run, so it goes first below).
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
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.Common.model import ensemble, random_forest
from News.Build_sentiment_label.Traditional_ML.Common.model.lexicon_features import (
    build_lexicon_feature_matrix,
)
from News.Build_sentiment_label.Traditional_ML.Common.production_cv import (
    run_production_cv,
    write_production_outputs,
)

MODEL_NAMES = ["random_forest", "ensemble"]


def build_feature_matrix(df: pd.DataFrame):
    """The one place this folder's runs add a feature on top of TF-IDF.
    Swap this out (and the ``extra_features=`` call below) to try a
    different feature block without touching Common/model/*.py."""
    return build_lexicon_feature_matrix(df["Tokenize_content"].tolist())


def run_random_forest(df, term_counts, y) -> None:
    print("\nModel: Random Forest (isotonic-calibrated probabilities, +lexicon features)")
    print("Documents:", len(df))
    print("Trees:", random_forest.N_ESTIMATORS)

    lexicon_matrix = build_feature_matrix(df)
    print("Lexicon feature matrix:", lexicon_matrix.shape)
    probabilities = run_production_cv(
        "random_forest", random_forest.build_estimator, df, term_counts, y,
        extra_features=lexicon_matrix,
    )
    write_production_outputs("random_forest", y, probabilities)


def run_ensemble(df, term_counts, y) -> None:
    print(f"\nModel: Ensemble (average probability of {', '.join(ensemble.MEMBER_NAMES)})")
    print("svm excluded - its predict_proba is an uncalibrated margin-softmax")
    print("members' saved production OOF probabilities are reused (random_forest = +lexicon)")
    print("Documents:", len(df))

    probabilities = ensemble.ensemble_probabilities(df["source_row_id"].to_numpy(), y)
    write_production_outputs("ensemble", y, probabilities)


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
    print("Documents:", len(df))
    print("Label counts:")
    print(df["ground_truth_label"].value_counts().to_string())

    for name in MODEL_NAMES:
        if name in args.models:
            RUNNERS[name](df, term_counts, y)


if __name__ == "__main__":
    main()
