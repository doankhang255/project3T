"""Average-probability ensemble of logistic_regression + naive_bayes +
random_forest (ML_SUMMARY.qmd section 6.1).

No CV of its own: every member's production run
(``Common/production_cv.py``) already saved its out-of-fold probabilities
for the same 10 repeats x 5-fold splits (fold seed = repeat index,
identical across models regardless of estimator). Averaging the members'
saved probabilities repeat by repeat is therefore a valid paired average -
and it scores exactly the production members (random_forest WITH the
lexicon feature block) rather than re-training copies of them.

``svm`` is excluded: since ``model/svm.py`` dropped Platt calibration (see
its module docstring / ``Common/calibration_check.py``), its
``predict_proba`` is a margin-softmax, not a calibrated probability -
averaging it in would let an arbitrarily-scaled number pull the ensemble
mean off of what the other three models actually believe.

The runnable step that writes ``ensemble_metrics.csv`` lives in
``../../experiment_Lexicon_features/run_model.py`` (run after every member).
No ``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

import numpy as np

from News.Build_sentiment_label.Traditional_ML.Common.production_cv import (
    load_production_oof,
)

MEMBER_NAMES = ["logistic_regression", "naive_bayes", "random_forest"]


def ensemble_probabilities(source_row_id: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Members' saved ``(n_repeats, n_rows, n_labels)`` probabilities,
    averaged across members."""
    member_probabilities = [load_production_oof(name, source_row_id, y) for name in MEMBER_NAMES]
    shapes = {probabilities.shape for probabilities in member_probabilities}
    if len(shapes) != 1:
        raise AssertionError(
            f"ensemble members were saved with different repeat counts / shapes: {shapes}"
        )
    return np.mean(member_probabilities, axis=0)
