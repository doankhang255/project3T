"""Random Forest - build_estimator() only.

Pure model-definition module. The actual production run (isotonic-calibrated
+ the lexicon feature block) lives in
``../../experiment_Lexicon_features/run_model.py`` - this is the only one of
the 4 base models that uses the lexicon block, so its runnable step belongs
there, not in experiment_only_TF_IDF/. No ``main()`` here on purpose: this
module is a library, not a runnable step.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


N_ESTIMATORS = 300

# RandomForestClassifier.predict_proba is a vote fraction across trees, not a
# fitted probability - it tends to be pulled toward the middle (rarely near 0
# or 1 even on confident rows). Common/calibration_check.py measured this
# directly (reliability diagram + Brier score, raw vs isotonic) - confirmed at
# both 599 and 1044 rows; isotonic calibration is used here because there is
# enough data per inner fold for it (it needs more than Platt/sigmoid does -
# too little data makes isotonic overfit the calibration curve).
CALIBRATION_CV = 3


def build_base_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_estimator(random_state: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        build_base_random_forest(random_state),
        method="isotonic",
        cv=CALIBRATION_CV,
    )
