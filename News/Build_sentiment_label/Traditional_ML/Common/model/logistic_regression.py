"""Logistic Regression - build_estimator() + build_top_features() only.

Pure model-definition module, shared by whichever script actually runs the
CV and writes output (``../../experiment_only_TF_IDF/logistic_regression.py``
for the production run - this model never uses the lexicon feature block).
No ``main()`` here on purpose: this module is a library, not a runnable step.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from News.Build_sentiment_label.Traditional_ML.Common.model.common import (
    RANDOM_SEED,
    VALID_LABELS,
)


LOGISTIC_C = 0.1
# Tuned via Common/tune_hyperparameters.py (nested 5-fold outer x 3-fold
# inner CV, grid C in {0.1,0.3,1,3,10}) - C=0.1 (stronger L2) beat the
# default C=1.0 across every outer fold, tune-set macro-F1 0.614 -> 0.644
# (bootstrap CI [+0.011,+0.052], p=0.000), confirmed on a never-tuned-against
# holdout: 0.568 -> 0.595 (CI [+0.011,+0.044], p=0.001).
MAX_ITER = 5000


def build_estimator(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        max_iter=MAX_ITER,
        class_weight="balanced",
        random_state=random_state,
    )


def build_top_features(
    x: np.ndarray,
    y: np.ndarray,
    vocabulary: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    model = build_estimator(RANDOM_SEED + 999)
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
                    "bias": float(model.intercept_[label_id]),
                }
            )
    return pd.DataFrame(rows)
