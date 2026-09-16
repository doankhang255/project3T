"""Shared model-factory registry, reused by every comparison/tuning script
(experiment_only_TF_IDF/, experiment_Lexicon_features/, sanity_checks.py,
run_nadeau_bengio.py, tune_hyperparameters.py) so the set of models being
compared - and their exact ``build_estimator`` - is defined in exactly one
place, not re-declared per script.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.naive_bayes import ComplementNB

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.naive_bayes import (
    build_estimator as build_multinomial_nb,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.random_forest import (
    build_estimator as build_random_forest,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.svm import (
    build_estimator as build_svm,
)

METRIC_COLUMNS = ["macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"]

# Single-split numbers from the committed RESULTS_SUMMARY.txt, shown for
# reference only (one fold seed, MultinomialNB, pre-lexicon/pre-tuning).
SINGLE_RUN_REFERENCE = {
    "random_forest": (0.634, 0.658),
    "naive_bayes": (0.605, 0.632),
    "logistic_regression": (0.593, 0.632),
    "svm": (0.562, 0.638),
}


def build_complement_nb(random_state: int) -> ComplementNB:
    del random_state  # ComplementNB has no randomness; uniform factory signature
    return ComplementNB()


# TF-IDF-only factories - none of these pass a lexicon extra_features matrix.
# random_forest here is deliberately the plain (no-lexicon) variant: this
# registry is what experiment_only_TF_IDF/ and the general-purpose diagnostics
# (sanity_checks.py, run_nadeau_bengio.py, tune_hyperparameters.py) compare -
# the production random_forest+lexicon combination lives in
# experiment_Lexicon_features/ instead, not here.
MODEL_FACTORIES = {
    "logistic_regression": build_logistic_regression,
    "multinomial_nb": build_multinomial_nb,
    "complement_nb": build_complement_nb,
    "random_forest": build_random_forest,
    "svm": build_svm,
}


def summarize(per_repeat: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        column: (
            float(per_repeat[column].mean()),
            float(per_repeat[column].std(ddof=1)),
        )
        for column in METRIC_COLUMNS
    }
