"""Linear SVM - build_estimator() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/run_model.py`` for the
production run - this model never uses the lexicon feature block). No
``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC


MAX_ITER = 5000
SVM_C = 0.1
# Nested-CV tuning (Common/tune_hyperparameters.py, grid C in
# {0.1,0.3,1,3,10}) on the 1044-row tune split shows NO reliable gain over the
# default C=1.0: 0.666 -> 0.662, Delta=-0.003 CI[-0.014,+0.008] p=0.590.
# C=0.1 is kept only until that decision is made - see ML_SUMMARY.qmd
# section 5.3.

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
