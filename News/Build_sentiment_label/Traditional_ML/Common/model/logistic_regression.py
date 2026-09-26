"""Logistic Regression - build_estimator() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/run_model.py`` for the
production run - this model never uses the lexicon feature block).
No ``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


LOGISTIC_C = 0.1
# Nested-CV tuning (Common/tune_hyperparameters.py, grid C in
# {0.1,0.3,1,3,10}) on the 1044-row tune split shows NO reliable gain over the
# default C=1.0: 0.689 -> 0.681, Delta=-0.008 CI[-0.019,+0.002] p=0.124, and
# the chosen C varies by outer fold (0.1 / 0.3). C=0.1 is kept only until
# that decision is made - see ML_SUMMARY.qmd section 5.3.
MAX_ITER = 5000


def build_estimator(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        max_iter=MAX_ITER,
        class_weight="balanced",
        random_state=random_state,
    )
