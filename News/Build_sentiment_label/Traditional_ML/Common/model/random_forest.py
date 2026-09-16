"""Random Forest - build_estimator() + build_top_features() only.

Pure model-definition module. The actual production run (isotonic-calibrated
+ the lexicon feature block) lives in
``../../experiment_Lexicon_features/random_forest.py`` - this is the only
one of the 4 base models that uses the lexicon block, so its runnable step
belongs there, not in experiment_only_TF_IDF/. No ``main()`` here on
purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    RANDOM_SEED,
    VALID_LABELS,
)


N_ESTIMATORS = 300

# RandomForestClassifier.predict_proba is a vote fraction across trees, not a
# fitted probability - it tends to be pulled toward the middle (rarely near 0
# or 1 even on confident rows). Common/calibration_check.py measured this
# directly (reliability diagram + Brier score, raw vs isotonic) - confirmed at
# both 599 and 1064 rows; isotonic calibration is used here because there is
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


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    # feature_importances_ is a single global ranking (impurity decrease),
    # not per-class like the coefficient-based models. CalibratedClassifierCV
    # does not expose it directly, so fit the uncalibrated forest here - same
    # pattern as model/svm.py's build_top_features using build_base_svm.
    model = build_base_random_forest(RANDOM_SEED + 999)
    model.fit(x, y)

    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_n:][::-1]

    rows = []
    for rank, feature_index in enumerate(top_indices, start=1):
        vocab_row = vocabulary.iloc[int(feature_index)]
        rows.append(
            {
                "rank": rank,
                "selected_feature_id": int(feature_index),
                "term_id": int(vocab_row["term_id"]),
                "term": vocab_row["term"],
                "ngram_n": int(vocab_row["ngram_n"]),
                "feature_importance": float(importances[feature_index]),
            }
        )
    return pd.DataFrame(rows)
