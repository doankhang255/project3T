"""Average-probability ensemble of logistic_regression + naive_bayes +
random_forest (IMPROVEMENTS.md section D).

Each base model already runs the same leak-free 5-fold CV
(``run_cross_validation``) with folds determined only by ``y`` and
``RANDOM_SEED`` - identical across models regardless of which estimator is
passed in. So calling it once per base model and averaging the three
out-of-fold probability matrices row-wise is a valid paired average, with no
new CV wiring needed here.

``svm`` is excluded: since ``model/svm.py`` dropped Platt calibration (see
its module docstring / ``Common/calibration_check.py``), its
``predict_proba`` is a margin-softmax, not a calibrated probability -
averaging it in would let an arbitrarily-scaled number pull the ensemble
mean off of what the other three models actually believe.

``random_forest``'s member run also gets the lexicon feature block (see
``model/lexicon_features.py``), same as its standalone run in
``../../experiment_Lexicon_features/random_forest.py`` - passed through
``extra_features_by_model`` so the ensemble does not silently score a
different (TF-IDF-only) version of RF than production. The actual production
run (that writes ``ensemble_metrics.csv``) lives in
``../../experiment_Lexicon_features/ensemble.py``, since its RF member uses
the lexicon block. No ``main()`` here on purpose: this module is a library,
not a runnable step.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    run_cross_validation,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.logistic_regression import (
    build_estimator as build_logistic_regression,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.naive_bayes import (
    build_estimator as build_naive_bayes,
)
from News.Build_sentiment_label.Traditional_ML.Common.model.random_forest import (
    build_estimator as build_random_forest,
)


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
