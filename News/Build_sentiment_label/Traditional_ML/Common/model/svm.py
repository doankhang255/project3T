"""Linear SVM - build_estimator() + build_top_features() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/svm.py`` for the
production run - this model never uses the lexicon feature block). No
``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    RANDOM_SEED,
    VALID_LABELS,
)


MAX_ITER = 5000
SVM_C = 0.1
# Tuned via Common/tune_hyperparameters.py (nested 5-fold outer x 3-fold
# inner CV, grid C in {0.1,0.3,1,3,10}) - C=0.1 (stronger L2) beat the
# default C=1.0 across every outer fold, tune-set macro-F1 0.572 -> 0.617
# (bootstrap CI [+0.022,+0.071], p=0.000), confirmed on a never-tuned-against
# holdout: 0.539 -> 0.563 (CI [+0.009,+0.041], p=0.002).

# Used to calibrate via Platt scaling (CalibratedClassifierCV(cv=3), i.e. an
# inner 3-fold split of an already ~120-row training fold to fit the sigmoid
# A,B). Common/calibration_check.py measured that directly: Platt was
# overconfident relative to the observed frequency in several probability
# bins (see RESULTS.txt), consistent with too little data per inner fold for
# a one-vs-rest calibration where the positive class is a minority to begin
# with. Removed - see MarginSoftmaxSVC below.


class MarginSoftmaxSVC:
    """``LinearSVC`` + a row-wise softmax of ``decision_function`` in place of
    ``predict_proba`` - NOT a calibrated probability.

    Exists only so this model satisfies the same ``fit`` / ``classes_`` /
    ``predict_proba`` interface ``run_cross_validation`` uses for every
    model. ``argmax(predict_proba(x))`` is identical to
    ``argmax(decision_function(x))`` (softmax is monotonic), so the hard
    label prediction is unaffected - only the numeric "probability" values
    are not to be trusted as calibrated:

    - ``model/ensemble.py`` excludes this model from probability averaging.
    - ``sentiment_score_ml`` (``prob_positive - prob_negative``) for this
      model is a margin-derived score, not a calibrated probability gap;
      treat it as ranking-only, like the raw ``decision_function`` it comes
      from.
    """

    def __init__(self, random_state: int, C: float = SVM_C) -> None:
        self._svc = build_base_svm(random_state, C=C)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MarginSoftmaxSVC":
        self._svc.fit(x, y)
        self.classes_ = self._svc.classes_
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self._svc.decision_function(x)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def build_base_svm(random_state: int, C: float = SVM_C) -> LinearSVC:
    return LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=random_state,
        max_iter=MAX_ITER,
    )


def build_estimator(random_state: int) -> MarginSoftmaxSVC:
    return MarginSoftmaxSVC(random_state=random_state)


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    model = build_base_svm(RANDOM_SEED + 999)
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
                }
            )
    return pd.DataFrame(rows)
