"""Production run: logistic_regression, naive_bayes, complement_nb, svm - TF-IDF only.

    python News/Build_sentiment_label/Traditional_ML/experiment_only_TF_IDF/run_model.py
    python .../run_model.py --models svm naive_bayes   # subset

Writes the files run_pipeline.py / compare_models.py consume (one run of
leak-free 10 x 5-fold repeated CV per model, see ``Common/production_cv.py``:
oof_probabilities.npz / metrics / confusion_matrix). Model definitions
(build_estimator) live in ``../Common/model/*.py`` and stay untouched - this
file is only the runnable step. To add a feature block or change how a model
is scored here, edit ``run_one()`` below and wire it into the imported model,
rather than editing ``Common/model/*.py`` (which is shared - the lexicon
experiment and any future one also import from it).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.TF_IDF import build_document_term_counts
from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    encode_labels,
    load_ground_truth_frame,
)
from News.Build_sentiment_label.Traditional_ML.Common.production_cv import (
    run_production_cv,
    write_production_outputs,
)
from News.Build_sentiment_label.Traditional_ML.Common.model import (
    complement_nb,
    logistic_regression,
    naive_bayes,
    svm,
)

# name -> (build_estimator, display label). Add a new TF-IDF-only model here
# (and to Common/model/) to have it run + reported alongside these - no other
# file needs to change.
MODEL_SPECS = {
    "logistic_regression": (
        logistic_regression.build_estimator,
        "Logistic Regression (scikit-learn)",
    ),
    "naive_bayes": (
        naive_bayes.build_estimator,
        "Multinomial Naive Bayes",
    ),
    "complement_nb": (
        complement_nb.build_estimator,
        "Complement Naive Bayes (Rennie et al. 2003, promoted from M2.1 comparison)",
    ),
    "svm": (
        svm.build_estimator,
        "Linear SVM (LinearSVC, margin-softmax - NOT a calibrated probability)",
    ),
}


def run_one(name, build_estimator, display_label, df, term_counts, y) -> None:
    print(f"\nModel: {display_label}")
    print("Documents:", len(df))

    probabilities = run_production_cv(name, build_estimator, df, term_counts, y)
    write_production_outputs(name, y, probabilities)


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
        build_estimator, display_label = MODEL_SPECS[name]
        run_one(name, build_estimator, display_label, df, term_counts, y)


if __name__ == "__main__":
    main()
